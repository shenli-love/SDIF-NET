from __future__ import annotations

import torch
from torch import nn

from .decoder import FusionDecoder
from .detector import YOLOLikeHead
from .encoder import ResNet50Encoder
from .fpn import FeaturePyramid
from .fusion import CrossModalQKVUnifiedFusion


class IRVISFusionDetectionNet(nn.Module):
    """End-to-end fusion network: dual ResNet encoders -> FPN -> QKV fusion -> dual heads."""

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
        self.fusion = CrossModalQKVUnifiedFusion(fpn_channels, num_scales=4)
        self.decoder = FusionDecoder(fpn_channels, fused_channels)
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

        fused_features = self.fusion(
            ir_features,
            vis_features,
        )
        
        fused_image = self.decoder(fused_features, output_size=ir.shape[-2:])
        
        # Detector directly consumes multi-scale fused features.
        detection_outputs = self.detector(image=None, fused_features=fused_features)

        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "loss_feedback_enabled": self.use_feedback if use_feedback is None else use_feedback,
        }
        if return_logs:
            output["forward_logs"] = {
                "pipeline": "encoder->fpn->cross_modal_qkv_fusion->decoder->detector",
                "fpn_levels": "P2-P5",
                "detector": "anchor_dense_small_object",
                "mode": "single_pass",
                "feedback_path": "loss_only",
            }
        return output
