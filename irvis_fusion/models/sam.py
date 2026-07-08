from __future__ import annotations

import torch
from torch import nn


class SAMPriorEncoder(nn.Module):
    """Encode a SAM mask into a spatial attention prior."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, sam_mask: torch.Tensor) -> torch.Tensor:
        if sam_mask.dim() == 3:
            sam_mask = sam_mask.unsqueeze(1)
        sam_mask = sam_mask.float().clamp(0.0, 1.0)
        return self.net(sam_mask) * sam_mask
