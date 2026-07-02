from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class SDIFScaleFusion(nn.Module):
    """Single-scale SDIF fusion with SAM as an attention-logit bias.

    SAM is not used as a hard feature gate. The soft prior only shifts the
    cross-modal attention logits, allowing the model to learn when and how much
    the prior should matter.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ir_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.vis_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.attn_logits = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.sam_bias = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.mix = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1),
            ConvBNAct(channels, channels),
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ir_feat = self.ir_proj(ir)
        vis_feat = self.vis_proj(vis)
        logits = self.attn_logits(torch.cat([ir_feat, vis_feat], dim=1))
        if sam_attention is not None:
            sam = F.interpolate(
                sam_attention,
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits = logits + self.sam_bias(sam)
        alpha = torch.sigmoid(logits)
        cross_modal = alpha * ir_feat + (1.0 - alpha) * vis_feat
        complement = (1.0 - alpha) * ir_feat + alpha * vis_feat
        contrast = torch.abs(ir_feat - vis_feat)
        return self.mix(torch.cat([cross_modal, complement, contrast], dim=1))


class SDIFUnifiedFusion(nn.Module):
    """Unified three-scale SDIF fusion function.

    F_fuse = f(F_ir, F_vis, A_sam)

    Detection feedback is intentionally absent from this forward path. It is
    handled by the training objective as a dynamic detection-loss weight.
    """

    def __init__(self, channels: int = 128, num_scales: int = 3) -> None:
        super().__init__()
        self.scales = nn.ModuleList([SDIFScaleFusion(channels) for _ in range(num_scales)])

    def forward(
        self,
        ir_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        vis_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sam_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(ir_features) != len(vis_features) or len(ir_features) != len(self.scales):
            raise ValueError("SDIFUnifiedFusion expects three aligned FPN scales.")
        fused = [
            block(ir, vis, sam_attention)
            for block, ir, vis in zip(self.scales, ir_features, vis_features)
        ]
        return tuple(fused)  # type: ignore[return-value]
