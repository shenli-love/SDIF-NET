from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CrossModalQKVFusion(nn.Module):
    """Bidirectional cross-modal QKV attention for one FPN level.

    Both directions are preserved explicitly:

    - IR-to-VIS: IR query retrieves VIS key/value detail.
    - VIS-to-IR: VIS query retrieves IR key/value detail.

    The output projection receives the two cross-attention responses and the
    original modal FPN features through a residual fusion path, which prevents
    one direction from suppressing modality-specific evidence.
    """

    def __init__(self, channels: int, local_windows: tuple[int, ...] = (3, 7, 11)) -> None:
        super().__init__()
        self.scale = channels ** -0.5
        self.local_windows = local_windows
        self.q_ir_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_ir_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_ir_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.q_vis_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_vis_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_vis_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_global_mix = nn.Parameter(torch.tensor(1.0))
        self.modality_gate = nn.Sequential(
            nn.Conv2d(channels * 4 + 1, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, kernel_size=1),
        )
        self.base_proj = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # Preserve absolute response levels for the decoder; instance norm was
        # washing out modality brightness and made fused images drift gray.
        self.norm = nn.Identity()

    def _hybrid_attention(self, score: torch.Tensor) -> torch.Tensor:
        local_maps = []
        for window in self.local_windows:
            local_score = F.max_pool2d(
                score,
                kernel_size=window,
                stride=1,
                padding=window // 2,
            )
            local_maps.append(torch.sigmoid(local_score))

        local_attn = torch.stack(local_maps, dim=0).mean(dim=0)
        global_attn = torch.sigmoid(score.mean(dim=(-2, -1), keepdim=True))
        local_weight = torch.sigmoid(self.local_global_mix)

        return local_weight * local_attn + (1.0 - local_weight) * global_attn

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
    ) -> torch.Tensor:
        ir_cond = ir
        vis_cond = vis

        q_ir = self.q_ir_proj(ir_cond)
        k_vis = self.k_vis_proj(vis_cond)
        v_vis = self.v_vis_proj(vis_cond)

        q_vis = self.q_vis_proj(vis_cond)
        k_ir = self.k_ir_proj(ir_cond)
        v_ir = self.v_ir_proj(ir_cond)

        score_ir_to_vis = (q_ir * k_vis).sum(dim=1, keepdim=True) * self.scale
        score_vis_to_ir = (q_vis * k_ir).sum(dim=1, keepdim=True) * self.scale
        attn_ir_to_vis = self._hybrid_attention(score_ir_to_vis)
        attn_vis_to_ir = self._hybrid_attention(score_vis_to_ir)

        cross_vis = attn_ir_to_vis * v_vis
        cross_ir = attn_vis_to_ir * v_ir
        difference_energy = (ir - vis).abs().mean(dim=1, keepdim=True)
        fusion_sources = torch.cat([ir, vis, cross_ir, cross_vis], dim=1)
        gates = torch.softmax(
            self.modality_gate(torch.cat([fusion_sources, difference_energy], dim=1)),
            dim=1,
        )
        gated_sources = torch.cat(
            [
                gates[:, 0:1] * ir,
                gates[:, 1:2] * vis,
                gates[:, 2:3] * cross_ir,
                gates[:, 3:4] * cross_vis,
            ],
            dim=1,
        )
        fused = self.base_proj(gated_sources)
        fused = fused + 0.5 * (ir + vis)
        fused = self.norm(fused)
        return self.out(fused)


class CrossModalQKVUnifiedFusion(nn.Module):
    """Multi-scale wrapper around bidirectional cross-modal QKV fusion.

    F_fuse = f(F_ir, F_vis)

    Detection feedback is intentionally absent from this forward path. It is
    handled by the training objective as a dynamic detection-loss weight.
    """

    def __init__(self, channels: int = 128, num_scales: int = 4) -> None:
        super().__init__()
        self.operators = nn.ModuleList(
            [CrossModalQKVFusion(channels) for _ in range(num_scales)]
        )

    def forward(
        self,
        ir_features: tuple[torch.Tensor, ...],
        vis_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(ir_features) != len(vis_features) or len(ir_features) != len(self.operators):
            raise ValueError("CrossModalQKVUnifiedFusion expects aligned P2-P5 FPN scales.")
        fused = [
            operator(ir, vis)
            for operator, ir, vis in zip(self.operators, ir_features, vis_features)
        ]
        return tuple(fused)  # type: ignore[return-value]
