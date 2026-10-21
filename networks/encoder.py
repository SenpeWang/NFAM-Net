# -*- coding: utf-8 -*-
"""NFAM-Net encoder with unified 4-stage StageBackbone, NIBS, and ablation support."""

import torch
import torch.nn as nn
from .nofm import NOFMBlock
from .layer import StageTransition, _gn
from .dfe import DFEBlock
from .dfr import DFR
from .qtensor import QTensorTools
from .nibs import NematicInformationBottleneckSelection


class StageBackbone(nn.Module):
    """N-stage backbone with block-level DFR and deep-stage NIBS."""

    def __init__(
        self,
        dims,
        depths,
        block_cls,
        drop_path_rates,
        q_updaters=None,
        chunk_size=64,
        nibs_rho=0.70,
        nibs_floor=0.70,
        enable_nibs=True,
        block_kwargs=None,
    ):
        super().__init__()
        self.dims = list(dims)
        self.depths = list(depths)
        self.q_updaters = q_updaters
        self.enable_nibs = enable_nibs
        block_kwargs = block_kwargs or [{} for _ in range(len(depths))]

        # ---- blocks per stage ----
        self.stages = nn.ModuleList()
        idx = 0
        for stage_i in range(len(self.depths)):
            blocks = nn.ModuleList()
            cls = block_cls[stage_i]
            b_kw = block_kwargs[stage_i]
            for b in range(self.depths[stage_i]):
                kwargs = dict(drop_path=drop_path_rates[idx])
                if cls is NOFMBlock:
                    kwargs.update(chunk_size=chunk_size)
                kwargs.update(b_kw)
                blocks.append(cls(self.dims[stage_i], **kwargs))
                idx += 1
            self.stages.append(blocks)

        # ---- deep stage NIBS selectors ----
        self.nibs2 = NematicInformationBottleneckSelection(self.dims[2], rho=nibs_rho, floor=nibs_floor, enable_nibs=enable_nibs)
        self.nibs3 = NematicInformationBottleneckSelection(self.dims[3], rho=nibs_rho, floor=nibs_floor, enable_nibs=enable_nibs)

        # ---- inter-stage transitions ----
        self.transitions = nn.ModuleList([
            StageTransition(self.dims[i], self.dims[i + 1])
            for i in range(len(self.depths) - 1)
        ])

    def forward(self, x, q=None, s_raw=None):
        stage_features, stage_q = [], []
        cr_losses = {}

        for stage_i, blocks in enumerate(self.stages):
            # NIBS deep feature channel selection before Stage 2 & 3 VSSBlocks
            if stage_i == 2 and self.nibs2 is not None:
                x, loss2 = self.nibs2(x, s_raw)
                cr_losses.update({f"{k}2": v for k, v in loss2.items()})
            elif stage_i == 3 and self.nibs3 is not None:
                x, loss3 = self.nibs3(x, s_raw)
                cr_losses.update({f"{k}3": v for k, v in loss3.items()})

            for block in blocks:
                if self.q_updaters is not None and self.q_updaters[stage_i] is not None and q is not None:
                    q, _ = self.q_updaters[stage_i](x, q)
                x = block(x, q)

            stage_features.append(x)
            if q is not None:
                stage_q.append(q)

            if stage_i < len(self.transitions):
                if q is not None:
                    x, q = self.transitions[stage_i](x, q)
                else:
                    x, _ = self.transitions[stage_i](x, torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device))

        return x, q, stage_features, stage_q, cr_losses


class Encoder(nn.Module):
    """NFAM-Net Encoder with explicit stage dimensions, NIBS, and ablation support.

    Ablation modes:
        'full': Full NFAM-Net with all nematic mechanisms.
        'baseline': Pure hybrid Conv + SSM without any nematic prior, Q-field, or NIBS.
        'wo_dfe': Stage 0 & 1 bypass NematicConv (pure ConvBlock).
        'wo_dfr': DFR dynamically disabled (statically frozen q_prior).
        'wo_nibs': NIBS feature modulation bypassed and no CR loss.
        'wo_sgate': NOFMBlock disables scalar order parameter S injection.
    """

    def __init__(self, in_chans, config, stats_logger=None, ablation='full'):
        super().__init__()
        self.ablation = ablation.lower()

        dims = [96, 192, 384, 768]
        depths = list(config.encoder_depths)
        self.dims = dims
        self.depths = depths

        is_baseline = (self.ablation == 'baseline')
        enable_dfe = self.ablation not in ('baseline', 'wo_dfe')
        enable_dfr = self.ablation not in ('baseline', 'wo_dfr')
        enable_nibs = self.ablation not in ('baseline', 'wo_nibs')
        enable_sgate = self.ablation not in ('baseline', 'wo_sgate')
        nibs_floor = config.nibs_floor
        nibs_rho = config.nibs_rho

        # ---- stem ----
        mid_ch = dims[0] // 2
        stem_in = in_chans if is_baseline else (in_chans + 3)
        self.stem = nn.Sequential(
            nn.GroupNorm(1, stem_in), nn.GELU(),
            nn.Conv2d(stem_in, mid_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_gn(mid_ch), mid_ch), nn.GELU(),
            nn.Conv2d(mid_ch, dims[0], 3, stride=2, padding=1, bias=False),
        )

        # ---- drop path rates ----
        dpr_all = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(depths))]

        # ---- Q updaters (one per stage) ----
        if is_baseline:
            self.q_updaters = None
        else:
            self.q_updaters = nn.ModuleList([
                DFR(dims[i], stats_logger=stats_logger, name=f'stage{i}', enable_update=enable_dfr)
                for i in range(len(dims))
            ])

        # Per-stage block configuration
        block_kwargs = [
            {'enable_nematic': enable_dfe},
            {'enable_nematic': enable_dfe},
            {'enable_sgate': enable_sgate},
            {'enable_sgate': enable_sgate},
        ]

        # ---- backbone ----
        self.backbone = StageBackbone(
            dims=dims, depths=depths,
            block_cls=[DFEBlock, DFEBlock, NOFMBlock, NOFMBlock],
            drop_path_rates=dpr_all,
            q_updaters=list(self.q_updaters) if self.q_updaters is not None else None,
            chunk_size=64,
            nibs_rho=nibs_rho,
            nibs_floor=nibs_floor,
            enable_nibs=enable_nibs,
            block_kwargs=block_kwargs,
        )

    def forward(self, x, q_prior=None):
        if self.ablation == 'baseline' or q_prior is None:
            h = self.stem(x)
            h, _, stage_features, _, _ = self.backbone(h, q=None, s_raw=None)
            return stage_features, [], None, {}

        h = self.stem(torch.cat([x, q_prior], dim=1))
        q = QTensorTools.downsample_q(q_prior[:, 0:1], q_prior[:, 1:2], size=h.shape[2:])
        q_stem = q
        s_raw = q_prior[:, 2:3]
        h, q, stage_features, stage_q, cr_losses = self.backbone(h, q, s_raw=s_raw)
        return stage_features, stage_q, q_stem, cr_losses
