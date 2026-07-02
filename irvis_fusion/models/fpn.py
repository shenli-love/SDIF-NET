from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class FeaturePyramid(nn.Module):
    """Top-down FPN with lateral connections."""

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels) for _ in in_channels]
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1, c2, c3 = features
        p3 = self.lateral[2](c3)
        p2 = self.lateral[1](c2) + F.interpolate(
            p3,
            size=c2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.lateral[0](c1) + F.interpolate(
            p2,
            size=c1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.smooth[0](p1), self.smooth[1](p2), self.smooth[2](p3)
