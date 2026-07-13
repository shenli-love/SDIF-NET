from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct
from ..utils.boxes import cxcywh_to_xyxy


class YOLOLikeHead(nn.Module):
    """A compact YOLO-style dense detection interface.

    The head predicts one anchor-free box per grid cell:
    [tx, ty, tw, th, objectness, class logits...].
    It is intentionally small so the project can train end-to-end now.
    
    Supports two modes:
    - Legacy mode: Extract features from single-channel fused image (use_fused_features=False)
    - Feature mode: Directly consume multi-scale fused features (use_fused_features=True)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        width: int = 64,
        strides: tuple[int, int, int] = (8, 16, 32),
        use_fused_features: bool = False,
        fused_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.use_fused_features = use_fused_features
        
        if not use_fused_features:
            # Legacy mode: extract features from single-channel image
            self.stem = nn.Sequential(
                ConvBNAct(in_channels, width),
                ConvBNAct(width, width, stride=2),
                ConvBNAct(width, width * 2, stride=2),
            )
            self.p3 = nn.Sequential(ConvBNAct(width * 2, width * 2, stride=2))
            self.p4 = nn.Sequential(ConvBNAct(width * 2, width * 4, stride=2))
            self.p5 = nn.Sequential(ConvBNAct(width * 4, width * 4, stride=2))
            head_in_channels = [width * 2, width * 4, width * 4]
        else:
            # Feature mode: directly receive multi-scale fused features
            self.stem = nn.Identity()
            self.p3 = nn.Identity()
            self.p4 = nn.Identity()
            self.p5 = nn.Identity()
            head_in_channels = [fused_channels, fused_channels, fused_channels]

        pred_dim = 5 + num_classes
        self.heads = nn.ModuleList(
            [
                nn.Conv2d(ch, pred_dim, kernel_size=1) for ch in head_in_channels
            ]
        )

    def forward(self, image: torch.Tensor | None = None, fused_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> dict[str, object]:
        if self.use_fused_features:
            # Feature mode: directly use fused features
            if fused_features is None:
                raise ValueError("fused_features must be provided when use_fused_features=True")
            p3, p4, p5 = fused_features
        else:
            # Legacy mode: extract features from image
            if image is None:
                raise ValueError("image must be provided when use_fused_features=False")
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
