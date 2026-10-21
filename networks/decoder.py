# -*- coding: utf-8 -*-
"""UPerNet decoder for NFAM-Net segmentation.

Internal decoder pipeline:
  1. PPM extracts multi-scale context
  2. FPN upsampling + skip connections
  3. Segmentation head output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module."""
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.pool_scales = pool_scales
        ppm_inner = out_channels // len(pool_scales)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, ppm_inner, 1, bias=False),
                nn.GroupNorm(8, ppm_inner),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + ppm_inner * len(pool_scales), out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outs = [x]
        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(out)
        return self.fuse(torch.cat(outs, dim=1))


class Decoder(nn.Module):
    """UPerNet-style decoder.
    """
    def __init__(self, encoder_dims, channels=512, num_classes=2,
                 dropout_ratio=0.1, pool_scales=(1, 2, 3, 6), **kwargs):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.num_levels = len(encoder_dims)

        # PPM on bottleneck -> outputs channels
        self.ppm = PPM(encoder_dims[-1], channels, pool_scales)

        # Lateral convs: project each encoder stage to channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, channels, 1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(inplace=True),
            )
            for dim in encoder_dims[:-1]
        ])

        # FPN convs
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_levels)
        ])

        # Segmentation head
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.seg_head = nn.Conv2d(channels, num_classes, 3, padding=1)

    def forward(self, stage_features, x_input):
        # PPM on bottleneck
        ppm_feat = self.ppm(stage_features[-1])

        # Lateral projections for non-bottleneck stages
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, stage_features[:-1])]
        laterals.append(ppm_feat)

        # Top-down FPN
        for i in range(self.num_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False
            )

        # FPN convs
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

        # Upsample all to same size and fuse
        target_size = fpn_outs[0].shape[2:]
        upsampled = [
            F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            for feat in fpn_outs
        ]
        fused = sum(upsampled) / len(upsampled)

        # Dropout + segmentation
        fused = self.dropout(fused)
        logits = self.seg_head(fused)

        # Upsample to input resolution
        logits = F.interpolate(logits, size=x_input.shape[2:],
                              mode='bilinear', align_corners=False)

        return logits
