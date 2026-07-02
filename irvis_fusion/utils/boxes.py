from __future__ import annotations

import torch


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=0.0)
    h = (y2 - y1).clamp(min=0.0)
    return torch.stack([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def draw_normalized_boxes(
    boxes: torch.Tensor,
    values: torch.Tensor | float,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw normalized xyxy boxes into a single-channel mask."""

    if device is None:
        device = boxes.device
    mask = torch.zeros((1, height, width), device=device, dtype=dtype)
    if boxes.numel() == 0:
        return mask
    if not torch.is_tensor(values):
        values = torch.full((boxes.shape[0],), float(values), device=device, dtype=dtype)
    boxes_px = boxes.to(device=device, dtype=dtype).clone()
    boxes_px[:, [0, 2]] *= width
    boxes_px[:, [1, 3]] *= height
    boxes_px = boxes_px.round().long()
    for box, value in zip(boxes_px, values.to(device=device, dtype=dtype)):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[:, y1:y2, x1:x2] = torch.maximum(mask[:, y1:y2, x1:x2], value)
    return mask
