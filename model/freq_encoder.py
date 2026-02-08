from typing import Literal
import numpy as np
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU, MaxPool3d, Module, Linear
from torchvision.models.video.mvit import MSBlockConfig, _mvit
from utils import Conv3d, Conv1d
import torch
import os
import sys
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    def __init__(self, size,
                 band_start,
                 band_end,
                 use_learnable=True,
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(self.generate_filter(band_start, band_end, size)),requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(self.generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

    @staticmethod
    def generate_filter(start, end, size):
        return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]


class FAD_MultiBranch(nn.Module):
    def __init__(self, size=96):
        super().__init__()
        self._DCT_all = nn.Parameter(torch.tensor(self.DCT_mat(size), dtype=torch.float32), requires_grad=False)  # DCT matrix
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(self.DCT_mat(size)).float(), 0, 1), requires_grad=False)
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    @staticmethod
    def DCT_mat(size):
        m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
              range(size)] for i in range(size)]
        return m

    def forward(self, x):
        # x：[B, 3, H, W]
        # output：low/middle/high [B, 3, H, W] ×3
        # x_freq = self.dct_mat @ x @ self.dct_mat.T  # DCT trans to Frequency domain
        # return [filt(x_freq) for filt in self.filters]  # Three frequency band characteristics

        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 299, 299]
        return out

class CrossFrequencyAttention(nn.Module):
    def __init__(self, in_channels=3, reduction=3):
        super().__init__()
        # Global average pooling is used to obtain the statistical characteristics of the frequency band
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels * 3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Softmax(dim=1)  # Weight normalization
        )

    def forward(self, freq_branches):
        # freq_branches：[low, middle, high] list, every x shape is [B, C, H, W]
        low, middle, high = freq_branches
        concat = torch.cat([low, middle, high], dim=1)  # [B, 3C, H, W]
        # learning weight：[B, 3, 1, 1]
        weights = self.attn(self.global_pool(concat))  # [B, 3, 1, 1]
        # weighted fusion：(low * w0 + middle * w1 + high * w2)
        fused = low * weights[:, 0:1] + middle * weights[:, 1:2] + high * weights[:, 2:3]
        return fused
