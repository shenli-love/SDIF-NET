from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class FeaturePyramid(nn.Module):
    """Top-down FPN with lateral connections.

    Inputs are ordered from high resolution to low resolution (C2-C5), and the
    outputs keep the same order (P2-P5).
    """

    def __init__(
        self,
        in_channels: tuple[int, ...],
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
        features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(features) != len(self.lateral):
            raise ValueError("FeaturePyramid expects one feature per lateral layer.")

        laterals = [lateral(feature) for lateral, feature in zip(self.lateral, features)]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return tuple(smooth(feature) for smooth, feature in zip(self.smooth, laterals))
