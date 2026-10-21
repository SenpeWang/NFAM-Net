# -*- coding: utf-8 -*-
"""DFE: Directional Feature Extraction block with dual-branch nematic projection.

Extracts directional representations using parallel and perpendicular projections
against the local director field.
"""

import torch
import torch.nn as nn
from .qtensor import QTensorTools
from .layer import _gn, DropPath


class ConvBlock(nn.Module):
    """Standard residual convolutional block for shallow stage representation."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.conv(x))


class NematicConv(nn.Module):
    """Nematic convolution decomposing feature gradients onto nematic director bases."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Sobel gradient extractor
        self.grad = nn.Conv2d(dim, dim * 2, 3, padding=1, groups=dim, bias=False)
        self._init_sobel()
        self.grad.weight.requires_grad = False

        # Parallel branch
        self.branch_par = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

        # Perpendicular branch
        self.branch_perp = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

        # Cross-branch adaptive fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False),
            nn.GroupNorm(_gn(dim * 2), dim * 2),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
        )

    @staticmethod
    def _sobel_kernels():
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        return sx, sy

    def _init_sobel(self):
        sx, sy = self._sobel_kernels()
        with torch.no_grad():
            for i in range(self.dim):
                self.grad.weight.data[2 * i, 0] = sx
                self.grad.weight.data[2 * i + 1, 0] = sy

    def forward(self, f_sem: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_sem.shape

        q1_hat, q2_hat = QTensorTools.get_unit_q(q[:, 0:1], q[:, 1:2])

        r = self.grad(f_sem).view(b, c, 2, h, w)
        r_x = r[:, :, 0]
        r_y = r[:, :, 1]

        r_par_x = 0.5 * ((1 + q1_hat) * r_x + q2_hat * r_y)
        r_par_y = 0.5 * (q2_hat * r_x + (1 - q1_hat) * r_y)
        r_perp_x = 0.5 * ((1 - q1_hat) * r_x - q2_hat * r_y)
        r_perp_y = 0.5 * (-q2_hat * r_x + (1 + q1_hat) * r_y)

        f_par = self.branch_par(torch.cat([r_par_x, r_par_y], dim=1))
        f_perp = self.branch_perp(torch.cat([r_perp_x, r_perp_y], dim=1))

        f_fused = self.fusion(torch.cat([f_par, f_perp], dim=1))
        return f_sem + f_fused


class DFEBlock(nn.Module):
    """Directional Feature Extraction Block for shallow stages.

    Args:
        dim: Feature channel dimension.
        drop_path: Stochastic depth rate. Defaults to 0.0.
        enable_nematic: Whether to apply NematicConv directional decomposition.
            Defaults to True. When False, acts as pure ConvBlock for ablation study.
    """

    def __init__(self, dim: int, drop_path: float = 0.0, enable_nematic: bool = True, **kwargs):
        super().__init__()
        self.conv = ConvBlock(dim, drop_path=drop_path)
        self.enable_nematic = enable_nematic
        self.nematic = NematicConv(dim)

    def forward(self, x: torch.Tensor, q: torch.Tensor = None) -> torch.Tensor:
        x = self.conv(x)
        if self.enable_nematic and q is not None:
            x = self.nematic(x, q)
        return x
