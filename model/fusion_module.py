import torch
from torch import Tensor
from torch.nn import Sigmoid, Module
import torch.nn as nn
from utils import Conv1d


class ModalMapAttnBlock(Module):
    def __init__(self, n_self_features, n_another_features, max_duration=40):
        super().__init__()
        self.attn_from_self_features = Conv1d(n_self_features, max_duration, kernel_size=1)
        self.attn_from_another_features = Conv1d(n_another_features, max_duration, kernel_size=1)
        self.attn_from_bm = Conv1d(max_duration, max_duration, kernel_size=1)
        self.attn_from_fusion_features = Conv1d(n_self_features, max_duration, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, self_bm, self_features, another_features, fusion_features):
        w_bm = self.attn_from_bm(self_bm)
        w_self_feat = self.attn_from_self_features(self_features)
        w_another_feat = self.attn_from_another_features(another_features)
        w_fusion_feat = self.attn_from_fusion_features(fusion_features)
        w_stack = torch.stack((w_bm, w_self_feat, w_another_feat, w_fusion_feat), dim=3)

        w = w_stack.mean(dim=3)
        return self.sigmoid(w)


class ModalCbgAttnBlock(ModalMapAttnBlock):
    def __init__(self, n_self_features: int, n_another_features: int):
        super().__init__(n_self_features, n_another_features, 1)


class ModalFeatureAttnBoundaryMapFusion(Module):
    """
    融合模块
    Input:
        video_feature: (B, C_f, T)
        audio_feature: (B, C_f, T)
        fusion_feature:(B, C_f, T)
        video_bm: (B, D, T)
        audio_bm: (B, D, T)
    Output:
        fusion_bm: (B, D, T)
    """
    def __init__(self, n_video_features=257, n_audio_features=257, n_fusion_features=257, max_duration=40):
        super().__init__()
        self.a_attn_block = ModalMapAttnBlock(n_audio_features, n_video_features, max_duration)
        self.v_attn_block = ModalMapAttnBlock(n_video_features, n_audio_features, max_duration)
        self.fusion_attn_block = ModalMapAttnBlock(n_fusion_features, n_video_features, max_duration)

    def forward(self, video_feature, audio_feature, fusion_feature, video_bm, audio_bm):
        a_attn = self.a_attn_block(audio_bm, audio_feature, video_feature, fusion_feature)
        v_attn = self.v_attn_block(video_bm, video_feature, audio_feature, fusion_feature)

        sum_attn = a_attn + v_attn

        a_w = a_attn / sum_attn
        v_w = v_attn / sum_attn

        fusion_bm = video_bm * v_w + audio_bm * a_w
        return fusion_bm
