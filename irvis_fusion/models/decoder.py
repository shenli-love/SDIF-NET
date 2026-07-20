from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class FusionDecoder(nn.Module):
    """FPN-style decoder that reconstructs the fused image from P2-P5."""

    def __init__(self, channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        self.up_p5 = ConvBNAct(channels, channels)
        self.up_p4 = ConvBNAct(channels, channels)
        self.up_p3 = ConvBNAct(channels, channels)
        self.refine_p4 = ConvBNAct(channels * 2, channels)
        self.refine_p3 = ConvBNAct(channels * 2, channels)
        self.refine_p2 = ConvBNAct(channels * 2, channels)
        self.out = nn.Sequential(
            ConvBNAct(channels, channels // 2),
            nn.Conv2d(channels // 2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("FusionDecoder expects P2-P5 fused features.")
        p2, p3, p4, p5 = features
        x4 = self.up_p5(
            F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        )
        x4 = self.refine_p4(torch.cat([x4, p4], dim=1))
        x3 = self.up_p4(
            F.interpolate(x4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        )
        x3 = self.refine_p3(torch.cat([x3, p3], dim=1))
        x2 = self.up_p3(
            F.interpolate(x3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        )
        x2 = self.refine_p2(torch.cat([x2, p2], dim=1))
        fused = self.out(x2)
        if output_size is not None and fused.shape[-2:] != output_size:
            fused = F.interpolate(
                fused,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return fused
