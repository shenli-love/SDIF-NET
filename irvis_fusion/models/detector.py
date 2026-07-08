from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct
from ..utils.boxes import cxcywh_to_xyxy


class UltralyticsYOLODetector(nn.Module):
    """Official Ultralytics YOLO detector wrapper.

    This wrapper keeps the same decoded output contract as YOLOLikeHead:
    boxes are normalized xyxy, scores are confidences, and class_logits is a
    padded tensor compatible with the existing feedback code.
    """

    def __init__(
        self,
        weights_path: str | Path,
        num_classes: int | None = None,
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 300,
        classes: list[int] | None = None,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for UltralyticsYOLODetector. "
                "Install it with `pip install ultralytics`."
            ) from exc

        self.weights_path = str(weights_path)
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

        self.yolo = YOLO(self.weights_path)
        model_nc = int(getattr(getattr(self.yolo, "model", None), "nc", 0) or 0)
        self.num_classes = max(int(num_classes or 0), model_nc, 1)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.classes = classes

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> dict[str, object]:
        image_for_yolo, restore_scale = self._prepare_image(image)
        results = self.yolo.predict(
            source=image_for_yolo,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
            device=str(image.device),
        )
        decoded = self._results_to_decoded(
            results,
            image.device,
            image.dtype,
            restore_scale,
        )
        return {"raw": results, "decoded": decoded, "trainable": False}

    def _prepare_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float]]:
        image = image.detach().float().clamp(0.0, 1.0)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] > 3:
            image = image[:, :3]
        _, _, height, width = image.shape
        pad_h = (-height) % 32
        pad_w = (-width) % 32
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        padded_h, padded_w = image.shape[-2:]
        restore_scale = (padded_w / width, padded_h / height)
        return image, restore_scale

    def _results_to_decoded(
        self,
        results: list[object],
        device: torch.device,
        dtype: torch.dtype,
        restore_scale: tuple[float, float],
    ) -> dict[str, torch.Tensor]:
        batch_size = len(results)
        boxes = torch.zeros(
            batch_size,
            self.max_det,
            4,
            device=device,
            dtype=dtype,
        )
        scores = torch.zeros(batch_size, self.max_det, device=device, dtype=dtype)
        class_logits = torch.full(
            (batch_size, self.max_det, self.num_classes),
            -20.0,
            device=device,
            dtype=dtype,
        )
        for batch_idx, result in enumerate(results):
            result_boxes = result.boxes
            if result_boxes is None or result_boxes.xyxyn.numel() == 0:
                continue
            n = min(result_boxes.xyxyn.shape[0], self.max_det)
            cur_boxes = result_boxes.xyxyn[:n].to(device=device, dtype=dtype)
            scale = cur_boxes.new_tensor(
                [restore_scale[0], restore_scale[1], restore_scale[0], restore_scale[1]]
            )
            cur_boxes = cur_boxes * scale
            cur_scores = result_boxes.conf[:n].to(device=device, dtype=dtype)
            cur_labels = result_boxes.cls[:n].to(device=device, dtype=torch.long)
            boxes[batch_idx, :n] = cur_boxes.clamp(0.0, 1.0)
            scores[batch_idx, :n] = cur_scores
            valid = (cur_labels >= 0) & (cur_labels < self.num_classes)
            if valid.any():
                row_idx = torch.arange(n, device=device)[valid]
                class_logits[batch_idx, row_idx, cur_labels[valid]] = 20.0
        return {
            "boxes": boxes,
            "scores": scores,
            "class_logits": class_logits,
        }

    @torch.no_grad()
    def postprocess(
        self,
        detections: dict[str, torch.Tensor],
        conf_threshold: float | None = None,
        topk: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        threshold = self.conf_threshold if conf_threshold is None else conf_threshold
        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["class_logits"].argmax(dim=-1)
        results: list[dict[str, torch.Tensor]] = []
        for batch_idx in range(boxes.shape[0]):
            keep = scores[batch_idx] >= threshold
            cur_scores = scores[batch_idx][keep]
            cur_boxes = boxes[batch_idx][keep]
            cur_labels = labels[batch_idx][keep]
            if cur_scores.numel() > topk:
                top_scores, top_idx = torch.topk(cur_scores, k=topk)
                cur_scores = top_scores
                cur_boxes = cur_boxes[top_idx]
                cur_labels = cur_labels[top_idx]
            results.append(
                {"boxes": cur_boxes, "scores": cur_scores, "labels": cur_labels}
            )
        return results


class YOLOLikeHead(nn.Module):
    """A compact YOLO-style dense detection interface.

    The head predicts one anchor-free box per grid cell:
    [tx, ty, tw, th, objectness, class logits...].
    It is intentionally small so the project can train end-to-end now while a
    full YOLO/Ultralytics detector can later be swapped behind the same API.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        width: int = 64,
        strides: tuple[int, int, int] = (8, 16, 32),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, width),
            ConvBNAct(width, width, stride=2),
            ConvBNAct(width, width * 2, stride=2),
        )
        self.p3 = nn.Sequential(ConvBNAct(width * 2, width * 2, stride=2))
        self.p4 = nn.Sequential(ConvBNAct(width * 2, width * 4, stride=2))
        self.p5 = nn.Sequential(ConvBNAct(width * 4, width * 4, stride=2))
        pred_dim = 5 + num_classes
        self.heads = nn.ModuleList(
            [
                nn.Conv2d(width * 2, pred_dim, kernel_size=1),
                nn.Conv2d(width * 4, pred_dim, kernel_size=1),
                nn.Conv2d(width * 4, pred_dim, kernel_size=1),
            ]
        )

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        base = self.stem(image)
        p3 = self.p3(base)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        raw = [head(feat) for head, feat in zip(self.heads, (p3, p4, p5))]
        decoded = self.decode(raw)
        return {"raw": raw, "decoded": decoded, "trainable": True}

    def decode(self, raw_outputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes_all: list[torch.Tensor] = []
        obj_all: list[torch.Tensor] = []
        cls_all: list[torch.Tensor] = []
        for raw in raw_outputs:
            b, _, h, w = raw.shape
            pred = raw.permute(0, 2, 3, 1).contiguous()
            device = pred.device
            dtype = pred.dtype
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((gx, gy), dim=-1).view(1, h, w, 2)
            xy = (torch.sigmoid(pred[..., 0:2]) + grid) / pred.new_tensor([w, h])
            cell_size = pred.new_tensor([1.0 / w, 1.0 / h])
            wh = (F.softplus(pred[..., 2:4]) + 1e-4) * cell_size
            boxes = cxcywh_to_xyxy(torch.cat([xy, wh], dim=-1).view(b, -1, 4))
            boxes = boxes.clamp(0.0, 1.0)
            obj = torch.sigmoid(pred[..., 4]).view(b, -1)
            cls = pred[..., 5:].view(b, -1, self.num_classes)
            boxes_all.append(boxes)
            obj_all.append(obj)
            cls_all.append(cls)
        return {
            "boxes": torch.cat(boxes_all, dim=1),
            "scores": torch.cat(obj_all, dim=1),
            "class_logits": torch.cat(cls_all, dim=1),
        }

    @torch.no_grad()
    def postprocess(
        self,
        detections: dict[str, torch.Tensor],
        conf_threshold: float = 0.25,
        topk: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        boxes = detections["boxes"]
        scores = detections["scores"]
        class_logits = detections["class_logits"]
        probs = F.softmax(class_logits, dim=-1)
        class_scores, labels = probs.max(dim=-1)
        final_scores = scores * class_scores
        results: list[dict[str, torch.Tensor]] = []
        for batch_idx in range(boxes.shape[0]):
            keep = final_scores[batch_idx] >= conf_threshold
            cur_scores = final_scores[batch_idx][keep]
            cur_boxes = boxes[batch_idx][keep]
            cur_labels = labels[batch_idx][keep]
            if cur_scores.numel() > topk:
                top_scores, top_idx = torch.topk(cur_scores, k=topk)
                cur_scores = top_scores
                cur_boxes = cur_boxes[top_idx]
                cur_labels = cur_labels[top_idx]
            results.append(
                {"boxes": cur_boxes, "scores": cur_scores, "labels": cur_labels}
            )
        return results
