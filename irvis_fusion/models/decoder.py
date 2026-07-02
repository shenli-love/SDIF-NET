from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class FusionDecoder(nn.Module):
    """FPN-style decoder that reconstructs the fused image."""

    def __init__(self, channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        self.up3 = ConvBNAct(channels, channels)
        self.up2 = ConvBNAct(channels, channels)
        self.refine1 = ConvBNAct(channels * 2, channels)
        self.refine2 = ConvBNAct(channels * 2, channels)
        self.out = nn.Sequential(
            ConvBNAct(channels, channels // 2),
            nn.Conv2d(channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        f1, f2, f3 = features
        p2 = self.up3(
            F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        )
        p2 = self.refine2(torch.cat([p2, f2], dim=1))
        p1 = self.up2(
            F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        )
        p1 = self.refine1(torch.cat([p1, f1], dim=1))
        fused = self.out(p1)
        if output_size is not None and fused.shape[-2:] != output_size:
            fused = F.interpolate(
                fused,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return fused
