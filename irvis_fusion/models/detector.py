from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct
from ..utils.boxes import box_iou, cxcywh_to_xyxy


class YOLOLikeHead(nn.Module):
    """Anchor-based dense detector over fused P2-P5 features.

    The class name is kept for backward imports, but the implementation now
    uses small-object anchors. P2 receives the smallest base size by default.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        width: int = 64,
        strides: tuple[int, ...] = (4, 8, 16, 32),
        use_fused_features: bool = False,
        fused_channels: int = 128,
        anchor_sizes: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0),
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        if len(strides) != len(anchor_sizes):
            raise ValueError("strides and anchor_sizes must describe the same FPN levels.")

        self.num_classes = num_classes
        self.strides = strides
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(aspect_ratios)
        self.use_fused_features = use_fused_features

        if not use_fused_features:
            self.stem = nn.Sequential(
                ConvBNAct(in_channels, width),
                ConvBNAct(width, width, stride=2),
                ConvBNAct(width, width * 2, stride=2),
            )
            self.p3 = nn.Sequential(ConvBNAct(width * 2, width * 2, stride=2))
            self.p4 = nn.Sequential(ConvBNAct(width * 2, width * 4, stride=2))
            self.p5 = nn.Sequential(ConvBNAct(width * 4, width * 4, stride=2))
            head_in_channels = [width * 2, width * 4, width * 4]
            self.level_strides = (8, 16, 32)
            self.level_anchor_sizes = anchor_sizes[-3:]
        else:
            self.stem = nn.Identity()
            self.p3 = nn.Identity()
            self.p4 = nn.Identity()
            self.p5 = nn.Identity()
            head_in_channels = [fused_channels for _ in strides]
            self.level_strides = strides
            self.level_anchor_sizes = anchor_sizes

        pred_dim = 5 + num_classes
        self.pred_dim = pred_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(ch, ch),
                    nn.Conv2d(ch, self.num_anchors * pred_dim, kernel_size=1),
                )
                for ch in head_in_channels
            ]
        )

    def forward(
        self,
        image: torch.Tensor | None = None,
        fused_features: tuple[torch.Tensor, ...] | None = None,
    ) -> dict[str, object]:
        if self.use_fused_features:
            if fused_features is None:
                raise ValueError("fused_features must be provided when use_fused_features=True")
            features = fused_features
        else:
            if image is None:
                raise ValueError("image must be provided when use_fused_features=False")
            base = self.stem(image)
            p3 = self.p3(base)
            p4 = self.p4(p3)
            p5 = self.p5(p4)
            features = (p3, p4, p5)

        if len(features) != len(self.heads):
            raise ValueError("Detector received an unexpected number of FPN features.")
        raw = [head(feat) for head, feat in zip(self.heads, features)]
        decoded = self.decode(raw)
        return {"raw": raw, "decoded": decoded, "trainable": True}

    def _anchor_wh(
        self,
        base_size: float,
        stride: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_h = max(grid_h * stride, 1)
        image_w = max(grid_w * stride, 1)
        anchors = []
        for ratio in self.aspect_ratios:
            ratio_root = math.sqrt(ratio)
            anchor_w = base_size * ratio_root / image_w
            anchor_h = base_size / ratio_root / image_h
            anchors.append([anchor_w, anchor_h])
        return torch.tensor(anchors, device=device, dtype=dtype)

    def decode(self, raw_outputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes_all: list[torch.Tensor] = []
        obj_all: list[torch.Tensor] = []
        cls_all: list[torch.Tensor] = []
        levels_all: list[torch.Tensor] = []
        for level_idx, raw in enumerate(raw_outputs):
            b, _, h, w = raw.shape
            pred = raw.view(b, self.num_anchors, self.pred_dim, h, w)
            pred = pred.permute(0, 3, 4, 1, 2).contiguous()
            device = pred.device
            dtype = pred.dtype
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((gx, gy), dim=-1).view(1, h, w, 1, 2)
            xy = (torch.sigmoid(pred[..., 0:2]) + grid) / pred.new_tensor([w, h])
            anchor_wh = self._anchor_wh(
                self.level_anchor_sizes[level_idx],
                self.level_strides[level_idx],
                h,
                w,
                device,
                dtype,
            ).view(1, 1, 1, self.num_anchors, 2)
            wh = torch.exp(pred[..., 2:4].clamp(min=-4.0, max=4.0)) * anchor_wh
            boxes = cxcywh_to_xyxy(torch.cat([xy, wh], dim=-1).view(b, -1, 4))
            boxes_all.append(boxes.clamp(0.0, 1.0))
            obj_all.append(torch.sigmoid(pred[..., 4]).view(b, -1))
            cls_all.append(pred[..., 5:].view(b, -1, self.num_classes))
            levels_all.append(
                torch.full(
                    (b, h * w * self.num_anchors),
                    float(level_idx),
                    device=device,
                    dtype=dtype,
                )
            )
        return {
            "boxes": torch.cat(boxes_all, dim=1),
            "scores": torch.cat(obj_all, dim=1),
            "class_logits": torch.cat(cls_all, dim=1),
            "feature_levels": torch.cat(levels_all, dim=1),
        }

    @torch.no_grad()
    def postprocess(
        self,
        detections: dict[str, torch.Tensor],
        conf_threshold: float = 0.25,
        topk: int = 100,
        nms_iou_threshold: float = 0.5,
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
            if cur_scores.numel() > 0:
                keep_idx = self._batched_nms(
                    cur_boxes,
                    cur_scores,
                    cur_labels,
                    nms_iou_threshold,
                )
                cur_scores = cur_scores[keep_idx]
                cur_boxes = cur_boxes[keep_idx]
                cur_labels = cur_labels[keep_idx]
            if cur_scores.numel() > topk:
                top_scores, top_idx = torch.topk(cur_scores, k=topk)
                cur_scores = top_scores
                cur_boxes = cur_boxes[top_idx]
                cur_labels = cur_labels[top_idx]
            results.append(
                {"boxes": cur_boxes, "scores": cur_scores, "labels": cur_labels}
            )
        return results

    @staticmethod
    def _batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        iou_threshold: float,
    ) -> torch.Tensor:
        keep_all = []
        for label in labels.unique():
            idx = torch.where(labels == label)[0]
            order = idx[scores[idx].argsort(descending=True)]
            kept = []
            while order.numel() > 0:
                current = order[0]
                kept.append(current)
                if order.numel() == 1:
                    break
                ious = box_iou(boxes[current].view(1, 4), boxes[order[1:]]).view(-1)
                order = order[1:][ious <= iou_threshold]
            if kept:
                keep_all.append(torch.stack(kept))
        if not keep_all:
            return torch.zeros((0,), device=boxes.device, dtype=torch.long)
        keep = torch.cat(keep_all)
        return keep[scores[keep].argsort(descending=True)]
