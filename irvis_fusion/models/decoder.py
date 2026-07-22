from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class DetailPreservingDecoder(nn.Module):
    """带原始图像跳跃连接的解码器，防止高频信息丢失。

    融合特征提供语义/目标信息，原始图像跳跃提供纹理/边缘。
    """

    def __init__(self, channels: int = 128, out_channels: int = 1) -> None:
        super().__init__()
        # 特征上采样路径
        self.up_p5_to_p4 = ConvBNAct(channels * 2, channels)
        self.up_p4_to_p3 = ConvBNAct(channels * 2, channels)
        self.up_p3_to_p2 = ConvBNAct(channels * 2, channels)

        # 原始图像梯度引导分支 (从原图提取浅层特征)
        self.ir_detail = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )
        self.vis_detail = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
        )

        # 最终融合输出
        self.final_fuse = nn.Sequential(
            ConvBNAct(channels + 64, channels // 2),
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1),
            nn.GroupNorm(8, channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        ir_image: torch.Tensor,
        vis_image: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("DetailPreservingDecoder expects P2-P5 fused features.")
        p2, p3, p4, p5 = features

        # 逐级上采样融合
        x = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_p5_to_p4(torch.cat([x, p4], dim=1))
        x = F.interpolate(x, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_p4_to_p3(torch.cat([x, p3], dim=1))
        x = F.interpolate(x, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_p3_to_p2(torch.cat([x, p2], dim=1))

        # 原始图像细节跳跃 (全分辨率 → /4 分辨率)
        target_h, target_w = p2.shape[-2:]
        ir_detail = self.ir_detail(ir_image)
        vis_detail = self.vis_detail(vis_image)
        if ir_detail.shape[-2:] != (target_h, target_w):
            ir_detail = F.interpolate(
                ir_detail, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            vis_detail = F.interpolate(
                vis_detail, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        # 拼接语义特征 + 原始细节
        x = torch.cat([x, ir_detail, vis_detail], dim=1)
        fused = self.final_fuse(x)

        if output_size is not None and fused.shape[-2:] != output_size:
            fused = F.interpolate(
                fused, size=output_size, mode="bilinear", align_corners=False
            )
        return fused


# 保持向后兼容
FusionDecoder = DetailPreservingDecoder
