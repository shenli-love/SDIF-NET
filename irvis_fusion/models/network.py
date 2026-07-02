from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .decoder import FusionDecoder
from .detector import UltralyticsYOLODetector, YOLOLikeHead
from .encoder import CNNEncoder
from .fpn import FeaturePyramid
from .fusion import SDIFUnifiedFusion
from .sam import SAMPriorEncoder


class IRVISFusionDetectionNet(nn.Module):
    """End-to-end SDIF-Net: encoder -> FPN -> SDIF fusion -> decoder -> detector."""

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
        detector_backend: str = "yolo_like",
        yolo_weights: str | None = None,
        yolo_imgsz: int = 640,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.7,
        yolo_max_det: int = 300,
        yolo_classes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.use_sam = use_sam
        self.use_feedback = use_feedback
        self.ir_encoder = CNNEncoder(ir_channels, encoder_channels)
        self.vis_encoder = CNNEncoder(vis_channels, encoder_channels)
        self.ir_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.vis_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.sam_encoder = SAMPriorEncoder()
        self.fusion = SDIFUnifiedFusion(fpn_channels)
        self.decoder = FusionDecoder(fpn_channels, fused_channels)
        if detector_backend == "ultralytics":
            default_weights = Path(__file__).with_name("yolo11n.pt")
            self.detector = UltralyticsYOLODetector(
                yolo_weights or default_weights,
                num_classes=num_classes,
                imgsz=yolo_imgsz,
                conf_threshold=yolo_conf,
                iou_threshold=yolo_iou,
                max_det=yolo_max_det,
                classes=yolo_classes,
            )
        elif detector_backend == "yolo_like":
            self.detector = YOLOLikeHead(fused_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported detector backend: {detector_backend}")

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
        fused_image = self.decoder(fused_features, output_size=ir.shape[-2:])
        detection_outputs = self.detector(fused_image)

        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "sam_attention": sam_attention,
            "loss_feedback_enabled": self.use_feedback if use_feedback is None else use_feedback,
        }
        if return_logs:
            output["forward_logs"] = {
                "pipeline": "encoder->fpn->sdif_fusion->decoder->detector",
                "sam_prior": sam_attention is not None,
                "mode": "single_pass",
                "feedback_path": "loss_only",
            }
        return output
