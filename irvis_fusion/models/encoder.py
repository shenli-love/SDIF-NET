from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBNAct, ResidualConvBlock


class CNNEncoder(nn.Module):
    """Three-stage CNN encoder.

    Stage 1 keeps the input spatial size for detail preservation.
    Stage 2 downsamples by 2.
    Stage 3 downsamples by 4 relative to input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int, int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, c1, stride=1),
            ResidualConvBlock(c1),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(c1, c2, stride=2),
            ResidualConvBlock(c2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(c2, c3, stride=2),
            ResidualConvBlock(c3),
        )
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3
