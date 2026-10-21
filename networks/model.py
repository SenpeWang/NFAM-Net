# -*- coding: utf-8 -*-
"""NFAM-Net top-level model with NIBS and ablation study support."""

import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .qtensor import QImagePrior, QTensorTools
from datasets.dataset_config import get_dataset_config


class NFAMNet(nn.Module):
    """NFAMNet: Nematic Field-Aware Modeling network with NIBS and ablation support.

    Args:
        config: Experiment config (encoder_depths, drop_path_rate).
        in_chans: Input image channels (from dataset registry).
        stats_logger: Optional tensorboard logger.
        ablation: Ablation study mode ('full', 'baseline', 'wo_dfe', 'wo_dfr', 'wo_nibs', 'wo_sgate').
    """

    def __init__(self, config, in_chans, stats_logger=None, ablation='full'):
        super().__init__()
        self.ablation = ablation.lower()
        self.input_norm = nn.GroupNorm(1, in_chans)

        if self.ablation == 'baseline':
            self.q_image_prior = None
        else:
            self.q_image_prior = QImagePrior(tangent=True)

        self.encoder = Encoder(in_chans, config, stats_logger, ablation=self.ablation)
        self.decoder = Decoder(self.encoder.dims)

    @torch.no_grad()
    def _normalize_for_q_prior(self, x):
        x_float = x.float()
        x_min = x_float.amin(dim=(-2, -1), keepdim=True)
        x_max = x_float.amax(dim=(-2, -1), keepdim=True)
        span = (x_max - x_min).clamp_min(1e-6)
        return (x_float - x_min) / span

    def _stage_prior(self, x):
        x_q = self._normalize_for_q_prior(x)
        q1, q2, _ = self.q_image_prior(x_q)
        q_s = QTensorTools.get_nematic(q1, q2)
        return q_s

    def forward(self, x):
        x_c = self.input_norm(x)

        if self.ablation == 'baseline' or self.q_image_prior is None:
            stage_features, stage_q, _, cr_losses = self.encoder(x_c, q_prior=None)
            logits = self.decoder(stage_features, x_c)
            return {
                "logits": logits,
                "stage_Q": [],
                "cr_losses": {},
            }

        q_image_prior = self._stage_prior(x)
        stage_features, stage_q, q_stem, cr_losses = self.encoder(x_c, q_image_prior)
        logits = self.decoder(stage_features, x_c)
        return {
            "logits": logits,
            "stage_Q": stage_q,
            "cr_losses": cr_losses,
        }


def build_model(config, dataset_name='PDAC', stats_logger=None, ablation='full'):
    """Build NFAMNet from config with ablation support.

    input_channels comes from dataset registry, not from config.
    """
    ds = get_dataset_config(dataset_name)
    return NFAMNet(config, ds['channels'], stats_logger, ablation=ablation)
