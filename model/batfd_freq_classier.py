from typing import Dict, Sequence

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import MaskedFrameLoss, MaskedBMLoss, MaskedContrastLoss
from model.visualencoder import VisualEncoderModel
from model.audioEncoder import audioEncoderDet
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


class Batfd_Classifier(LightningModule):
    def __init__(self,
                 v_cla_feature_in=256, a_cla_feature_in=256,
                 weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
                 learning_rate=0.0002, weight_decay=0.0001, distributed=False
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.video_encoder = VisualEncoderModel()
        self.audio_encoder = audioEncoderDet(layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128])
        self.freq_extractor = FAD_MultiBranch(size=96)

        self.video_attention = FPN_CA_MSBlock_Fusion(in_channels=256, out_channels=256)
        self.audio_attention = FPN_CA_MSBlock_Fusion(in_channels=256, out_channels=256)

        v_bm_in = v_cla_feature_in
        a_bm_in = a_cla_feature_in
        fusion_bm_in = (a_bm_in + v_bm_in) // 2

        self.fusion_model = ModalFusionModule(audio_channels=a_bm_in,
                                              video_channels=v_bm_in,
                                              out_channels=fusion_bm_in)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(fusion_bm_in, 1)

        self.criterion = BCEWithLogitsLoss()
        self.contrast_loss = MaskedContrastLoss(margin=contrast_loss_margin)
        self.weight_contrastive_loss = weight_contrastive_loss

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.distributed = distributed

    def _extract_frequency_features(self, video: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        freq_branches = self.freq_extractor(video)
        fused_frames = freq_branches.view(B, T, 4 * C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return fused_frames

    def forward(self, video: Tensor, audio: Tensor):
        freq = self._extract_frequency_features(video)
        v_features = self.video_encoder(video, freq)  # (B, 256, T)
        a_features = self.audio_encoder(audio)  # (B, 256, T)

        v_features = self.video_attention(v_features)
        a_features = self.audio_attention(a_features)

        fusion_features = self.fusion_model(v_features, a_features)  # (B, Fusion_C, T)

        pooled_features = self.temporal_pool(fusion_features).squeeze(2)
        logits = self.classifier(pooled_features)

        return logits, v_features, a_features

    def training_step(self, batch, batch_idx):
        video, audio, label = batch
        logits, v_feat, a_feat = self(video, audio)

        cls_loss = self.criterion(logits, label)

        loss = cls_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", ((torch.sigmoid(logits) > 0.5) == label).float().mean(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, audio, label = batch
        logits, _, _ = self(video, audio)

        loss = self.criterion(logits, label)
        preds = torch.sigmoid(logits)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", ((preds > 0.5) == label).float().mean(), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
