# -*- coding: utf-8 -*-
"""Common layers: LayerNorm2D, DropPath, RMSNorm, DownsampleLayer, StageTransition."""

import torch
import torch.nn as nn
from .qtensor import QTensorTools


def _gn(ch):
    """GroupNorm group count: largest divisor of ch among (16, 8, 4, 2)."""
    for g in (16, 8, 4, 2):
        if ch % g == 0:
            return g
    return 1


class LayerNorm2D(nn.Module):
    """LayerNorm for (B, C, H, W) tensors."""
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Stochastic depth (drop path)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * random_tensor / keep_prob


class DownsampleLayer(nn.Module):
    """Downsample with overlapping convolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = LayerNorm2D(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class StageTransition(nn.Module):
    """Feature + Q synchronized downsample."""
    def __init__(self, feat_in, feat_out):
        super().__init__()
        self.feat_down = DownsampleLayer(feat_in, feat_out)

    def forward(self, feat, q):
        feat_next = self.feat_down(feat)
        q_next = QTensorTools.downsample_q(q[:, 0:1], q[:, 1:2], feat_next.shape[2:])
        return feat_next, q_next
