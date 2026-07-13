from __future__ import annotations

import torch
from torch import nn

from .decoder import FusionDecoder
from .detector import YOLOLikeHead
from .encoder import CNNEncoder
from .fpn import FeaturePyramid
from .fusion import SAMQKVUnifiedFusion
from .sam import SAMPriorEncoder


class IRVISFusionDetectionNet(nn.Module):
    """End-to-end fusion network: encoder -> FPN -> SQUA fusion -> decoder -> detector."""

    def __init__(
        self,
        ir_channels: int = 1,
        vis_channels: int = 1,
        fused_channels: int = 1,
        encoder_channels: tuple[int, int, int] = (32, 64, 128),
        fpn_channels: int = 128,
        num_classes: int = 6,
        use_sam: bool = True,
        use_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.use_sam = use_sam
        self.use_feedback = use_feedback
        self.ir_encoder = CNNEncoder(ir_channels, encoder_channels)
        self.vis_encoder = CNNEncoder(vis_channels, encoder_channels)
        self.ir_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.vis_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.sam_encoder = SAMPriorEncoder()
        self.fusion = SAMQKVUnifiedFusion(fpn_channels)
        self.decoder = FusionDecoder(fpn_channels, fused_channels)
        self.detector = YOLOLikeHead(
            fused_channels=fpn_channels,
            use_fused_features=True,
            num_classes=num_classes,
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_mask: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        use_sam: bool | None = None,
        use_feedback: bool | None = None,
        return_logs: bool = True,
    ) -> dict[str, object]:
        if vis.shape[-2:] != ir.shape[-2:]:
            raise ValueError("IR and VIS tensors must share the same spatial size.")
        run_sam = self.use_sam if use_sam is None else use_sam

        ir_features = self.ir_fpn(self.ir_encoder(ir))
        vis_features = self.vis_fpn(self.vis_encoder(vis))
        sam_attention = None
        if run_sam and sam_mask is not None:
            sam_attention = self.sam_encoder(sam_mask)

        fused_features = self.fusion(
            ir_features,
            vis_features,
            sam_attention=sam_attention,
        )
        
        # Decoder generates fused image for visualization and fusion loss
        fused_image = self.decoder(fused_features, output_size=ir.shape[-2:])
        
        # Detector directly consumes multi-scale fused features.
        detection_outputs = self.detector(image=None, fused_features=fused_features)

        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "sam_attention": sam_attention,
            "loss_feedback_enabled": self.use_feedback if use_feedback is None else use_feedback,
        }
        if return_logs:
            output["forward_logs"] = {
                "pipeline": "encoder->fpn->squa_fusion->decoder->detector",
                "detector": "yolo_like",
                "sam_prior": sam_attention is not None,
                "mode": "single_pass",
                "feedback_path": "loss_only",
            }
        return output
