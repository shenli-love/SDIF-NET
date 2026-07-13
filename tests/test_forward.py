from __future__ import annotations

import torch

from irvis_fusion.models import IRVISFusionDetectionNet
from irvis_fusion.utils.losses import JointFusionDetectionLoss


def test_forward_backward_dynamic_shape() -> None:
    batch_size = 2
    height, width = 95, 133
    ir = torch.rand(batch_size, 1, height, width)
    vis = torch.rand(batch_size, 1, height, width)
    sam = torch.rand(batch_size, 1, height, width)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long),
            "box_format": "cxcywh",
        }
        for _ in range(batch_size)
    ]
    model = IRVISFusionDetectionNet(num_classes=6)
    criterion = JointFusionDetectionLoss(num_classes=6)
    outputs = model(ir, vis, sam_mask=sam, targets=targets)
    losses = criterion(outputs, ir, vis, sam, targets)
    losses["loss"].backward()
    assert outputs["I_fused"].shape == ir.shape
    assert len(outputs["fused_features"]) == 3
    assert outputs["detections"]["decoded"]["boxes"].shape[0] == batch_size
