from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct, DepthwiseSeparableConv


class DetailFusionBlock(nn.Module):
    """Level-1 detail fusion that highlights texture and local contrast."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 3, channels),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        detail = self.edge_proj(torch.abs(ir - vis))
        return self.fuse(torch.cat([ir, vis, detail], dim=1))


class CrossModalInteraction(nn.Module):
    """Lightweight spatial/channel interaction between two modalities."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(channels * 2, channels),
            ConvBNAct(channels, channels),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([query, context, torch.abs(query - context)], dim=1))
        attended = query + gate * context
        return self.refine(torch.cat([attended, context], dim=1))


class SemanticFusionBlock(nn.Module):
    """Level-2 bidirectional semantic interaction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ir_to_vis = CrossModalInteraction(channels)
        self.vis_to_ir = CrossModalInteraction(channels)
        self.out = ConvBNAct(channels * 2, channels)

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        a = self.ir_to_vis(ir, vis)
        b = self.vis_to_ir(vis, ir)
        return self.out(torch.cat([a, b], dim=1))


class TargetFusionBlock(nn.Module):
    """Level-3 target fusion with SAM and detection feedback guidance."""

    def __init__(self, channels: int, init_feedback_lambda: float = 0.5) -> None:
        super().__init__()
        self.ir_to_vis = CrossModalInteraction(channels)
        self.vis_to_ir = CrossModalInteraction(channels)
        self.refine = ConvBNAct(channels, channels)
        self.feedback_lambda = nn.Parameter(
            torch.tensor(float(init_feedback_lambda), dtype=torch.float32)
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_attention: torch.Tensor | None = None,
        feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self.ir_to_vis(ir, vis) + self.vis_to_ir(vis, ir)
        weight = torch.ones(
            fused.shape[0],
            1,
            fused.shape[-2],
            fused.shape[-1],
            device=fused.device,
            dtype=fused.dtype,
        )
        if sam_attention is not None:
            weight = F.interpolate(
                sam_attention,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if feedback is not None:
            feedback = F.interpolate(
                feedback,
                size=fused.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            weight = weight + self.feedback_lambda.clamp(min=0.0) * feedback
        return self.refine(weight * fused)


class MultiLevelFusion(nn.Module):
    """Three-level fusion: details, semantics, and target-aware fusion."""

    def __init__(self, channels: int = 128, init_feedback_lambda: float = 0.5) -> None:
        super().__init__()
        self.level1 = DetailFusionBlock(channels)
        self.level2 = SemanticFusionBlock(channels)
        self.level3 = TargetFusionBlock(channels, init_feedback_lambda)

    def forward(
        self,
        ir_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        vis_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sam_attention: torch.Tensor | None = None,
        feedback: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.level1(ir_features[0], vis_features[0])
        f2 = self.level2(ir_features[1], vis_features[1])
        f3 = self.level3(ir_features[2], vis_features[2], sam_attention, feedback)
        return f1, f2, f3
