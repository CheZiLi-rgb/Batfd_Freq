from typing import Dict, Sequence

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import MaskedFrameLoss, MaskedBMLoss, MaskedContrastLoss
from model.visualencoder import VisualEncoderModel
from model.audioEncoder import audioEncoderDet
from model.boundary_module import BoundaryModulePlus
from model.frame_classifier import FrameLogisticRegression
from model.fusion_module import ModalFeatureAttnBoundaryMapFusion
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from model.fusion import ModalFusionModule
from model.FPN_Attention import FPN_CA_MSBlock_Fusion
from model.freq_encoder import FAD_MultiBranch
import torch.nn as nn


@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


class Batfd_Freq(LightningModule):
    def __init__(self,
                 frame_classifier: str = "lr", v_cla_feature_in=256, a_cla_feature_in=256,
                 boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
                 weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99, weight_giou_loss=0.1,
                 weight_decay=0.0001, learning_rate=0.0002, distributed=False,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.cla_feature_in = v_cla_feature_in
        self.temporal_dim = temporal_dim

        self.video_encoder = VisualEncoderModel()
        self.audio_encoder = audioEncoderDet(layers=[3, 4, 6, 3],  num_filters=[16, 32, 64, 128])

        self.video_attention = FPN_CA_MSBlock_Fusion(in_channels=256, out_channels=256)
        self.audio_attention = FPN_CA_MSBlock_Fusion(in_channels=256, out_channels=256)

        if frame_classifier == "lr":
            self.video_frame_classifier = FrameLogisticRegression(n_features=v_cla_feature_in)
            self.audio_frame_classifier = FrameLogisticRegression(n_features=a_cla_feature_in)

        assert self.video_encoder and self.audio_encoder and self.video_frame_classifier and self.audio_frame_classifier
        assert v_cla_feature_in == a_cla_feature_in

        v_bm_in = v_cla_feature_in
        a_bm_in = a_cla_feature_in
        fusion_bm_in = (a_bm_in + v_bm_in) // 2

        self.fusion_model = ModalFusionModule(audio_channels=a_bm_in,
                                              video_channels=v_bm_in,
                                              out_channels=fusion_bm_in)

        self.video_audio_boundary_module = BoundaryModulePlus(v_bm_in, boundary_features, boundary_samples, temporal_dim, max_duration)

        self.fusion = ModalFeatureAttnBoundaryMapFusion(v_bm_in, a_bm_in, fusion_bm_in, max_duration)
        self.freq_extractor = FAD_MultiBranch(size=96)
        self.frame_loss = MaskedFrameLoss(BCEWithLogitsLoss())
        self.contrast_loss = MaskedContrastLoss(margin=contrast_loss_margin)
        self.bm_loss = MaskedBMLoss(MSELoss())
        self.weight_frame_loss = weight_frame_loss
        self.weight_modal_bm_loss = weight_modal_bm_loss
        self.weight_contrastive_loss = weight_contrastive_loss / (v_cla_feature_in * temporal_dim)
        self.weight_giou_loss = weight_giou_loss
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.consistency_criterion = nn.L1Loss()
        self.consistency_weight = 0.5
        self.distributed = distributed

    def _extract_frequency_features(self, video: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, T, H, W)
        B, C, T, H, W = video.shape

        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = video.view(-1, C, H, W)

        freq_branches = self.freq_extractor(video)

        fused_frames = freq_branches.view(B, T, 4*C, H, W)
        # Output shape: (B, C, T, H, W)
        fused_frames = fused_frames.permute(0, 2, 1, 3, 4).contiguous()

        return fused_frames

    def forward(self, video: Tensor, audio: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # encoders
        freq = self._extract_frequency_features(video)

        v_features = self.video_encoder(video, freq)
        a_features = self.audio_encoder(audio)

        v_features = self.video_attention(v_features)
        a_features = self.audio_attention(a_features)

        # frame classifiers
        v_frame_cla = self.video_frame_classifier(v_features)
        a_frame_cla = self.audio_frame_classifier(a_features)

        v_bm_in = v_features
        a_bm_in = a_features

        fusion_bm_in = self.fusion_model(v_bm_in, a_bm_in)

        # modal boundary module
        v_bm_map, a_bm_map = self.video_audio_boundary_module(v_bm_in, a_bm_in)

        # boundary map modal attention fusion
        fusion_bm_map = self.fusion(v_bm_in, a_bm_in, fusion_bm_in, v_bm_map, a_bm_map)

        return fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features, a_features

    def loss_fn(self, fusion_bm_map: Tensor, v_bm_map: Tensor, a_bm_map: Tensor,
                v_frame_cla: Tensor, a_frame_cla: Tensor, label: Tensor, n_frames: Tensor,
                v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features, a_features
                ) -> Dict[str, Tensor]:
        fusion_bm_loss = self.bm_loss(fusion_bm_map, label, n_frames)

        v_bm_loss = self.bm_loss(v_bm_map, v_bm_label, n_frames)
        a_bm_loss = self.bm_loss(a_bm_map, a_bm_label, n_frames)

        v_frame_loss = self.frame_loss(v_frame_cla.squeeze(1), v_frame_label, n_frames)
        a_frame_loss = self.frame_loss(a_frame_cla.squeeze(1), a_frame_label, n_frames)

        contrast_loss = torch.clip(self.contrast_loss(v_features, a_features, contrast_label, n_frames)
                                   / (self.cla_feature_in * self.temporal_dim), max=1.)

        loss = fusion_bm_loss + \
               self.weight_modal_bm_loss * (a_bm_loss + v_bm_loss) / 2 + \
               self.weight_frame_loss * (a_frame_loss + v_frame_loss) / 2 + \
               self.weight_contrastive_loss * contrast_loss

        return {
            "loss": loss, "fusion_bm_loss": fusion_bm_loss, "v_bm_loss": v_bm_loss, "a_bm_loss": a_bm_loss,
            "v_frame_loss": v_frame_loss, "a_frame_loss": a_frame_loss, "contrast_loss": contrast_loss,
        }

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                      optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
                      ) -> Tensor:
        video, audio, label, n_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label = batch

        fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features, a_features = self(video, audio)
        loss_dict = self.loss_fn(fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, label, n_frames,
                                 v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features,
                                 a_features
                                 )

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
                        dataloader_idx: Optional[int] = None
                        ) -> Tensor:
        video, audio, label, n_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label = batch

        fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, v_features, a_features = self(video, audio)
        loss_dict = self.loss_fn(fusion_bm_map, v_bm_map, a_bm_map, v_frame_cla, a_frame_cla, label, n_frames,
                                 v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label, v_features,
                                 a_features
                                 )

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
                      prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
                     ) -> Tuple[Tensor, Tensor, Tensor]:
        video, audio, *_ = batch
        fusion_bm_map, v_bm_map, a_bm_map, *_ = self(video, audio)
        return fusion_bm_map, v_bm_map, a_bm_map

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }

    @staticmethod
    def get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor):
        label_fake = label
        label_real = torch.zeros(label.size(), dtype=label.dtype, device=label.device)

        v_bm_label = label_fake if meta.modify_video else label_real
        a_bm_label = label_fake if meta.modify_audio else label_real

        frame_label_real = torch.zeros(512)
        frame_label_fake = torch.zeros(512)
        for begin, end in meta.fake_periods:
            begin = int(begin * 25)
            end = int(end * 25)
            frame_label_fake[begin: end] = 1

        v_frame_label = frame_label_fake if meta.modify_video else frame_label_real
        a_frame_label = frame_label_fake if meta.modify_audio else frame_label_real

        contrast_label = 0 if meta.modify_audio or meta.modify_video else 1

        return [meta.video_frames, v_bm_label, a_bm_label, v_frame_label, a_frame_label, contrast_label]
