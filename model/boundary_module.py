import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, Sigmoid, Module

from utils import Conv3d, Conv2d
import torch.nn as nn
import torch.nn.functional as F

class PositionAwareAttentionModule(Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=None, dim=2):
        super(PositionAwareAttentionModule, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dim = dim

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dim == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2,))
            bn = nn.BatchNorm1d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        if self.sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        # value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=2)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)

        z = y + x
        return z


class ChannelAwareAttentionModule(Module):
    def __init__(self, in_channels, inter_channels=None, dim=2):
        super(ChannelAwareAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dim = dim

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=2)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)

        z = y + x
        return z


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, bn_layer=False, activate=False):
    module_list = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1)]
    if bn_layer:
        module_list.append(nn.BatchNorm2d(out_ch))
        module_list.append(nn.ReLU(inplace=True))
    if activate:
        module_list.append(nn.Sigmoid())
    conv = nn.Sequential(*module_list)
    return conv


class ProposalRelationBlock(Module):
    def __init__(self, in_channels, inter_channels=128, out_channels=2, sub_sample=False):
        super(ProposalRelationBlock, self).__init__()
        self.p_net = PositionAwareAttentionModule(in_channels, inter_channels=inter_channels, sub_sample=sub_sample, dim=2)
        self.c_net = ChannelAwareAttentionModule(in_channels, inter_channels=inter_channels, dim=2)
        self.conv0_0 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv0_1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)

        self.conv1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv2 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)
        self.conv3 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)
        self.conv4 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv5 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)

    def forward(self, x):
        x_p = self.conv0_0(x)
        x_c = self.conv0_1(x)

        x_p = self.p_net(x_p)
        x_c = self.c_net(x_c)

        x_p_0 = self.conv1(x_p)
        x_p_1 = self.conv2(x_p_0)

        x_c_0 = self.conv4(x_c)
        x_c_1 = self.conv5(x_c_0)

        x_p_c = self.conv3(x_p_0 + x_c_0)
        x_out = (x_p_1 + x_c_1 + x_p_c) / 3
        return x_out


class CrossModalGuidance(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn_v2a = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_a2v = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate_v = nn.Linear(dim, dim)
        self.gate_a = nn.Linear(dim, dim)

    def forward(self, v_feat, a_feat):
        v_feat = v_feat.permute(0, 2, 1)  # (B, T, C)
        a_feat = a_feat.permute(0, 2, 1)  # (B, T, C)

        a_guided_by_v, _ = self.attn_v2a(a_feat, v_feat, v_feat)
        v_guided_by_a, _ = self.attn_a2v(v_feat, a_feat, a_feat)

        gate_v = torch.sigmoid(self.gate_v(v_feat))  # (B, T, C)
        gate_a = torch.sigmoid(self.gate_a(a_feat))  # (B, T, C)
        v_fused = gate_v * v_guided_by_a + (1 - gate_v) * v_feat
        a_fused = gate_a * a_guided_by_v + (1 - gate_a) * a_feat

        return v_fused.permute(0, 2, 1), a_fused.permute(0, 2, 1)


class ProposalBlock(nn.Module):
    def __init__(self, in_channels=768, mid_channels=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, mid_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.score_head = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.feature = None

    def forward(self, x):
        # x: (B, 768, D, T)
        x = self.conv_layers(x)  # (B, 256, D, T)
        self.feature = x
        score = self.score_head(x)  # (B, 1, D, T)
        return score


class BoundaryModulePlus(Module):
    """
    Boundary matching module for video or audio features.
    Input:
        F_v or F_a: (B, C_f, T)
    Output:
        M_v^ or M_a^: (B, D, T)
    """

    def __init__(self, n_feature_in, n_features=(512, 128), num_samples: int = 10, temporal_dim: int = 512,
        max_duration: int = 40
    ):
        super().__init__()

        dim0, dim1 = n_features

        # (B, n_feature_in, temporal_dim) -> (B, n_feature_in, sample, max_duration, temporal_dim)
        self.bm_layer = BMLayer(temporal_dim, num_samples, max_duration)

        # (B, n_feature_in, sample, max_duration, temporal_dim) -> (B, dim0, max_duration, temporal_dim)
        self.block0 = Sequential(
            Conv3d(n_feature_in, dim0, kernel_size=(num_samples, 1, 1), stride=(num_samples, 1, 1),
                build_activation=LeakyReLU
            ),
            Rearrange("b c n d t -> b c (n d) t")
        )

        self.block1 = Sequential(
            Conv2d(dim0, dim1, kernel_size=1, build_activation=LeakyReLU),
            Conv2d(dim1, dim1, kernel_size=3, padding=1, build_activation=LeakyReLU)
        )
        self.proposal_block = ProposalRelationBlock(dim1, dim1, 1, sub_sample=True)
        self.cross_modal_guidance = CrossModalGuidance(dim=n_feature_in)
        self.offset_head = nn.Conv2d(dim1, 2, kernel_size=1)
        self.out = Rearrange("b c d t -> b (c d) t")

    def single_model_forward(self, feature: Tensor) -> Tensor:
        feature = self.bm_layer(feature)  # (B, 256, 512) -> (B, 256, 10, 40, 512)
        feature = self.block0(feature)    # (B, 256, 10, 40, 512) -> (B, 512, 40, 512)
        feature = self.block1(feature)    # (B, 512, 40, 512) -> (B, 128, 40, 512)
        feature = self.proposal_block(feature) # (B, 128, 40, 512) -> (B, 1, 40, 512)
        feature = self.out(feature)            # (B, 1, 40, 512) -> (B, 40, 512)
        return feature

    def forward(self, v_feat: Tensor, a_feat: Tensor) -> Tensor:
        v_feat, a_feat = self.cross_modal_guidance(v_feat, a_feat)
        v_proposal = self.single_model_forward(v_feat)
        a_proposal = self.single_model_forward(a_feat)

        return v_proposal, a_proposal


class BMLayer(Module):
    """BM Layer"""

    def __init__(self, temporal_dim: int, num_sample: int, max_duration: int, roi_expand_ratio: float = 0.5):
        super().__init__()
        self.temporal_dim = temporal_dim
        # self.feat_dim = opt['bmn_feat_dim']
        self.num_sample = num_sample
        self.duration = max_duration
        self.roi_expand_ratio = roi_expand_ratio
        self.smp_weight = self.get_pem_smp_weight()

    def get_pem_smp_weight(self):
        T = self.temporal_dim
        N = self.num_sample
        D = self.duration
        w = torch.zeros([T, N, D, T])  # T * N * D * T
        # In each temporal location i, there are D predefined proposals,
        # with length ranging between 1 and D
        # the j-th proposal is [i, i+j+1], 0<=j<D
        # however, a valid proposal should meet i+j+1 < T
        for i in range(T - 1):
            for j in range(min(T - 1 - i, D)):
                xmin = i
                xmax = (j + 1)
                # proposals[j, i, :] = [xmin, xmax]
                length = xmax - xmin
                xmin_ext = xmin - length * self.roi_expand_ratio
                xmax_ext = xmax + length * self.roi_expand_ratio
                bin_size = (xmax_ext - xmin_ext) / (N - 1)
                points = [xmin_ext + ii *
                          bin_size for ii in range(N)]
                for k, xp in enumerate(points):
                    if xp < 0 or xp > T - 1:
                        continue
                    left, right = int(np.floor(xp)), int(np.ceil(xp))
                    left_weight = 1 - (xp - left)
                    right_weight = 1 - (right - xp)
                    w[left, k, j, i] += left_weight
                    w[right, k, j, i] += right_weight
        return w.view(T, -1).float()

    def _apply(self, fn):
        self.smp_weight = fn(self.smp_weight)

    def forward(self, X):
        input_size = X.size()   # (1,257,512)
        assert (input_size[-1] == self.temporal_dim)
        # assert(len(input_size) == 3 and
        X_view = X.reshape(-1, input_size[-1])  # (1,257,512) -> (1*257,512)
        # feature [bs*C, T]
        # smp_w    [T, N*D*T]
        # out      [bs*C, N*D*T] --> [bs, C, N, D, T]
        result = torch.matmul(X_view, self.smp_weight)  # (1*257,512)*(512, 204800) = (257, 204800)
        return result.reshape(-1, input_size[1], self.num_sample, self.duration, self.temporal_dim) # (257, 204800) -> (1, 257, num_samples=10, max_during=40, T=512)


if __name__ == "__main__":
    n_feature_in = 256
    temporal_dim = 512
    max_duration = 40
    num_samples = 10
    boundary_module = BoundaryModulePlus(n_feature_in=n_feature_in, n_features=(512, 128), num_samples=num_samples,
                                     temporal_dim=temporal_dim, max_duration=max_duration)

    B = 1
    video_tensor = torch.randn(B, n_feature_in, temporal_dim)
    audio_tensor = torch.randn(B, n_feature_in, temporal_dim)

    v_proposal, a_proposal = boundary_module(video_tensor, audio_tensor)
    print("v_proposal tensor shape:", v_proposal.shape)
    print("a_proposal tensor shape:", a_proposal.shape)

