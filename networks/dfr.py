# -*- coding: utf-8 -*-
"""DFR: Dynamic Field Refinement — Q-tensor evolution module.

Predicts ideal Q-tensor state conditioned on current semantic features and Q-field,
and applies soft residual gated update.
"""

import torch
import torch.nn as nn
from .qtensor import QTensorTools
from .layer import _gn


class DFR(nn.Module):
    """DFR Q-tensor dynamic refinement.

    Args:
        dim: Feature channel dimension.
        stats_logger: Optional logger for Q update statistics.
        name: Identifier name for the stage.
        enable_update: Whether to dynamically refine Q-tensor. Defaults to True.
            When False, statically passes through Q unchanged (ablation: w/o DFR).
    """

    def __init__(
        self,
        dim: int,
        stats_logger=None,
        name: str = "",
        enable_update: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.stats_logger = stats_logger
        self.name = name
        self.enable_update = enable_update

        # Predictor for ideal Q state
        self.q_pred = nn.Sequential(
            nn.GroupNorm(_gn(dim + 2), dim + 2),
            nn.GELU(),
            nn.Conv2d(dim + 2, dim // 4, 1, bias=False),
            nn.GroupNorm(_gn(dim // 4), dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 2, 1),
            nn.Tanh(),
        )

        # Global step size
        self.scale = nn.Parameter(torch.tensor(0.0))

        # Local semantic gate
        self.conv = nn.Sequential(
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, q: torch.Tensor):
        """Update Q-tensor conditioned on semantic feature x and current field q.

        Args:
            x: Semantic feature tensor [B, dim, H, W].
            q: Current physical field tensor [B, 2, H, W].

        Returns:
            Tuple[torch.Tensor, dict]: (Updated Q field [B, 2, H, W], empty dict or stats).
        """
        if not self.enable_update:
            # Ablation: w/o DFR (statically frozen q_prior, no dynamic refinement)
            return q, {}

        # 1. Concatenate physical field Q into semantic feature
        inputs = torch.cat([x, q], dim=1)

        # 2. Predict ideal Q state
        q_pred = self.q_pred(inputs)

        # 3. Cap norm to 1: S = sqrt(q1^2 + q2^2) <= 1
        q_pred = QTensorTools.get_qtensor(q_pred[:, 0:1], q_pred[:, 1:2])

        # 4. Compute adaptive update gate
        gate = torch.sigmoid(self.scale) * self.conv(x)

        # 5. Soft gated convex update
        q_new = q + gate * (q_pred - q)

        # 6. Optional statistics logging
        if self.stats_logger is not None:
            s_after = QTensorTools.get_s(q_new[:, 0:1], q_new[:, 1:2]).mean().item()
            self.stats_logger.log_q_update(
                self.name,
                gate.mean().item(),
                torch.sigmoid(self.scale).item(),
                s_after,
            )

        return q_new, {}
