from __future__ import annotations

import torch
from torch import nn

from .decoder import DetailPreservingDecoder
from .detector import YOLOLikeHead
from .encoder import ResNet50Encoder
from .feedback_modulator import DetectionFeedbackModulator
from .fpn import FeaturePyramid
from .fusion import CrossModalQKVUnifiedFusion, DetectionGuidedFusion
from .saliency import TaskSaliencyPredictor


class IRVISFusionDetectionNet(nn.Module):
    """端到端融合检测网络（旧版，保留兼容）。

    架构: 双ResNet编码 → 双FPN → 通道交叉注意力融合 → 细节保持解码 → 检测头
    """

    def __init__(
        self,
        ir_channels: int = 1,
        vis_channels: int = 1,
        fused_channels: int = 1,
        encoder_channels: tuple[int, ...] | None = None,
        resnet_base_channels: int = 64,
        fpn_channels: int = 128,
        num_classes: int = 6,
        use_feedback: bool = True,
        anchor_sizes: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0),
        anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        self.use_feedback = use_feedback
        self.ir_encoder = ResNet50Encoder(ir_channels, base_channels=resnet_base_channels)
        self.vis_encoder = ResNet50Encoder(vis_channels, base_channels=resnet_base_channels)
        encoder_channels = encoder_channels or self.ir_encoder.out_channels
        self.ir_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.vis_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.fusion = CrossModalQKVUnifiedFusion(fpn_channels, num_scales=4, num_heads=4)
        self.decoder = DetailPreservingDecoder(fpn_channels, fused_channels)
        self.detector = YOLOLikeHead(
            fused_channels=fpn_channels,
            use_fused_features=True,
            num_classes=num_classes,
            strides=(4, 8, 16, 32),
            anchor_sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
        use_feedback: bool | None = None,
        return_logs: bool = True,
    ) -> dict[str, object]:
        if vis.shape[-2:] != ir.shape[-2:]:
            raise ValueError("IR and VIS tensors must share the same spatial size.")

        ir_features = self.ir_fpn(self.ir_encoder(ir))
        vis_features = self.vis_fpn(self.vis_encoder(vis))

        fused_features = self.fusion(ir_features, vis_features)

        fused_image = self.decoder(
            fused_features,
            ir_image=ir,
            vis_image=vis,
            output_size=ir.shape[-2:],
        )

        detection_outputs = self.detector(image=None, fused_features=fused_features)

        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "loss_feedback_enabled": self.use_feedback if use_feedback is None else use_feedback,
        }
        if return_logs:
            output["forward_logs"] = {
                "pipeline": "encoder->fpn->cross_attention_fusion->detail_decoder->detector",
                "fpn_levels": "P2-P5",
                "detector": "anchor_dense",
                "mode": "single_pass",
                "feedback_path": "loss_only",
            }
        return output


class IRVISFusionDetectionNetV2(nn.Module):
    """闭环检测引导融合网络 V2。

    核心改进 (相比 V1):
    1. TaskSaliencyPredictor 提供空间级任务先验 → 直接参与融合决策
    2. DetectionGuidedFusion 在特征层面被显著性图调制
    3. DetectionFeedbackModulator 将检测中间特征回传到融合层 (训练时闭环)
    4. 显著性预测器有独立的监督损失，确保学到检测语义

    架构: 双ResNet编码 → 双FPN → 显著性预测 → 检测引导融合
          → [训练: 检测反馈调制] → 细节保持解码 → 检测头
    """

    def __init__(
        self,
        ir_channels: int = 1,
        vis_channels: int = 1,
        fused_channels: int = 1,
        encoder_channels: tuple[int, ...] | None = None,
        resnet_base_channels: int = 64,
        fpn_channels: int = 128,
        num_classes: int = 6,
        num_scales: int = 4,
        num_heads: int = 4,
        use_feedback_loop: bool = True,
        anchor_sizes: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0),
        anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        self.use_feedback_loop = use_feedback_loop
        self.num_scales = num_scales

        # 双模态编码器
        self.ir_encoder = ResNet50Encoder(ir_channels, base_channels=resnet_base_channels)
        self.vis_encoder = ResNet50Encoder(vis_channels, base_channels=resnet_base_channels)
        encoder_channels = encoder_channels or self.ir_encoder.out_channels

        # 双模态 FPN
        self.ir_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.vis_fpn = FeaturePyramid(encoder_channels, fpn_channels)

        # 任务显著性预测器
        self.saliency_predictor = TaskSaliencyPredictor(fpn_channels, num_scales)

        # 检测引导融合模块
        self.fusion = DetectionGuidedFusion(fpn_channels, num_scales, num_heads)

        # 检测反馈调制器 (训练时闭环)
        if use_feedback_loop:
            self.feedback_modulator = DetectionFeedbackModulator(
                fpn_channels, fpn_channels, num_scales
            )

        # 细节保持解码器
        self.decoder = DetailPreservingDecoder(fpn_channels, fused_channels)

        # 检测头
        self.detector = YOLOLikeHead(
            fused_channels=fpn_channels,
            use_fused_features=True,
            num_classes=num_classes,
            strides=(4, 8, 16, 32),
            anchor_sizes=anchor_sizes,
            aspect_ratios=anchor_ratios,
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
        return_logs: bool = True,
    ) -> dict[str, object]:
        if vis.shape[-2:] != ir.shape[-2:]:
            raise ValueError("IR and VIS tensors must share the same spatial size.")

        # 1. 双模态编码 + FPN
        ir_features = self.ir_fpn(self.ir_encoder(ir))
        vis_features = self.vis_fpn(self.vis_encoder(vis))

        # 2. 任务显著性预测
        saliency_maps, modal_weights = self.saliency_predictor(ir_features, vis_features)

        # 3. 检测引导融合
        fused_features = self.fusion(ir_features, vis_features, saliency_maps, modal_weights)

        # 4. 检测头第一次前向
        detection_outputs = self.detector(image=None, fused_features=fused_features)

        # 5. 训练时: 检测反馈调制 (特征级闭环)
        if self.use_feedback_loop and self.training:
            det_intermediate = self._extract_det_intermediate(detection_outputs, fused_features)
            fused_features = self.feedback_modulator(fused_features, det_intermediate)

        # 6. 细节保持解码
        fused_image = self.decoder(
            fused_features,
            ir_image=ir,
            vis_image=vis,
            output_size=ir.shape[-2:],
        )

        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "saliency_maps": saliency_maps,
            "modal_weights": modal_weights,
            "ir_features": ir_features,
            "vis_features": vis_features,
        }
        if return_logs:
            output["forward_logs"] = {
                "pipeline": "encoder->fpn->saliency->guided_fusion->feedback->decoder->detector",
                "fpn_levels": "P2-P5",
                "detector": "anchor_dense",
                "mode": "closed_loop" if (self.use_feedback_loop and self.training) else "single_pass",
                "feedback_path": "feature_level_closed_loop",
            }
        return output

    def _extract_det_intermediate(
        self,
        detection_outputs: dict[str, object],
        fused_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """从检测头的 raw 输出中提取中间特征用于反馈调制。

        检测头的 raw 输出是 [B, num_anchors * pred_dim, H, W]，
        我们取前 fusion_channels 个通道作为中间特征。
        """
        raw_outputs = detection_outputs["raw"]
        intermediates = []
        for i, raw in enumerate(raw_outputs):
            target_channels = fused_features[i].shape[1]
            if raw.shape[1] >= target_channels:
                intermediates.append(raw[:, :target_channels])
            else:
                # 如果通道不足，用插值补齐
                intermediates.append(
                    torch.nn.functional.interpolate(
                        raw, size=fused_features[i].shape[-2:],
                        mode="bilinear", align_corners=False,
                    ).expand(-1, target_channels, -1, -1)
                )
        return tuple(intermediates)
