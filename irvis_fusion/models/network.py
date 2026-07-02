from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .decoder import FusionDecoder
from .detector import UltralyticsYOLODetector, YOLOLikeHead
from .encoder import CNNEncoder
from .feedback import DetectionFeedback
from .fpn import FeaturePyramid
from .fusion import MultiLevelFusion
from .sam import SAMPriorEncoder


class IRVISFusionDetectionNet(nn.Module):
    """SAM-guided detection-feedback iterative IR/VIS fusion network."""

    def __init__(
        self,
        ir_channels: int = 1,
        vis_channels: int = 1,
        fused_channels: int = 1,
        encoder_channels: tuple[int, int, int] = (32, 64, 128),
        fpn_channels: int = 128,
        num_classes: int = 6,
        max_iterations: int = 3,
        use_sam: bool = True,
        use_feedback: bool = True,
        detector_backend: str = "ultralytics",
        yolo_weights: str | None = None,
        yolo_imgsz: int = 640,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.7,
        yolo_max_det: int = 300,
        yolo_classes: list[int] | None = None,
        feedback_iou_threshold: float = 0.5,
        feedback_score_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_sam = use_sam
        self.use_feedback = use_feedback
        self.max_iterations = max(1, int(max_iterations))
        self.ir_encoder = CNNEncoder(ir_channels, encoder_channels)
        self.vis_encoder = CNNEncoder(vis_channels, encoder_channels)
        self.ir_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.vis_fpn = FeaturePyramid(encoder_channels, fpn_channels)
        self.sam_encoder = SAMPriorEncoder()
        self.fusion = MultiLevelFusion(fpn_channels)
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
        self.feedback = DetectionFeedback(
            iou_threshold=feedback_iou_threshold,
            score_threshold=feedback_score_threshold,
        )

    def forward(
        self,
        ir: torch.Tensor,
        vis: torch.Tensor,
        sam_mask: torch.Tensor | None = None,
        targets: list[dict[str, torch.Tensor]] | None = None,
        max_iterations: int | None = None,
        use_sam: bool | None = None,
        use_feedback: bool | None = None,
        return_logs: bool = True,
    ) -> dict[str, object]:
        if vis.shape[-2:] != ir.shape[-2:]:
            raise ValueError("IR and VIS tensors must share the same spatial size.")
        run_sam = self.use_sam if use_sam is None else use_sam
        run_feedback = self.use_feedback if use_feedback is None else use_feedback
        iterations = self.max_iterations if max_iterations is None else max(1, max_iterations)

        ir_features = self.ir_fpn(self.ir_encoder(ir))
        vis_features = self.vis_fpn(self.vis_encoder(vis))
        sam_attention = None
        if run_sam and sam_mask is not None:
            sam_attention = self.sam_encoder(sam_mask)

        feedback_mask = None
        logs = []
        fused_image = None
        fused_features = None
        detection_outputs = None
        feedback_outputs = None
        for iteration in range(iterations):
            fused_features = self.fusion(
                ir_features,
                vis_features,
                sam_attention=sam_attention,
                feedback=feedback_mask if run_feedback else None,
            )
            fused_image = self.decoder(fused_features, output_size=ir.shape[-2:])
            detection_outputs = self.detector(fused_image)

            feedback_outputs = self.feedback(
                detection_outputs["decoded"],
                targets,
                image_size=ir.shape[-2:],
            )
            logs.append(
                {
                    "iteration": iteration,
                    "recall": feedback_outputs["recall"],
                    "mean_confidence": feedback_outputs["mean_confidence"],
                    "feedback_mean": float(feedback_outputs["G_fb"].mean().item()),
                }
            )

            if not run_feedback or iteration == iterations - 1:
                break
            feedback_mask = feedback_outputs["G_fb"].detach()

        assert fused_image is not None
        assert fused_features is not None
        assert detection_outputs is not None
        output = {
            "I_fused": fused_image,
            "detections": detection_outputs,
            "fused_features": fused_features,
            "sam_attention": sam_attention,
            "feedback": feedback_outputs,
        }
        if return_logs:
            output["iteration_logs"] = logs
        return output
