# SDIF-Net: SAM-guided Detection-aware IR-VIS Fusion

This project implements a trainable PyTorch pipeline for infrared-visible image
fusion and detection-guided optimization.

## What Is Included

- independent IR and VIS CNN encoders
- FPN top-down multi-scale enhancement
- `SAMPriorEncoder` for a soft SAM attention prior
- unified three-scale SDIF fusion over FPN features
- FPN-style decoder that reconstructs `I_fused`
- compact YOLO-like dense detection head
- optional Ultralytics YOLO11 inference wrapper
- detection feedback as a dynamic detection-loss weight
- ablation switches: `use_sam`, `use_feedback`, `detector_backend`

## Data Layout

The default reader targets the current M3FD-style structure:

```text
datasets/M3FD_Detection
|__ ir
|__ vi
|__ sam_masks
|__ labels
|__ meta/train.txt
|__ meta/val.txt
```

Labels are YOLO normalized `class cx cy w h`.

## Quick Check

```bash
python -m irvis_fusion.smoke_test
```

The smoke test runs dynamic-size forward, fusion loss, detection loss, dynamic
detection weighting, and backward propagation.

## Train

```bash
python -m irvis_fusion.train ^
  --data-root datasets/M3FD_Detection ^
  --image-size 256 320 ^
  --batch-size 2 ^
  --epochs 20 ^
  --num-classes 6
```

Disable modules for ablation:

```bash
python -m irvis_fusion.train --no-sam
python -m irvis_fusion.train --no-feedback
python -m irvis_fusion.train --detector-backend ultralytics
```

## Main Output Keys

`IRVISFusionDetectionNet.forward()` returns:

- `I_fused`: reconstructed fused image
- `detections`: raw and decoded YOLO-like predictions
- `fused_features`: level1, level2, level3 fused features
- `sam_attention`: soft SAM prior `A_sam`
- `forward_logs`: confirms the single-pass SDIF pipeline

## Detector Backends

The default `yolo_like` backend is trainable end-to-end. The `ultralytics`
backend uses `irvis_fusion/models/yolo11n.pt` by default and is suitable for
inference-time detection/feedback metrics, but Ultralytics NMS is not used as a
differentiable detection loss.

Both backends keep this return contract:

```python
{
    "raw": ...,
    "decoded": {
        "boxes": Tensor[B, N, 4],       # normalized xyxy
        "scores": Tensor[B, N],
        "class_logits": Tensor[B, N, C],
    },
}
```

The loss consumes decoded fields and computes:

```text
L_total = L_fusion + lambda_det * L_detection
```

When `use_feedback=True`, `lambda_det` increases for low recall or low detection
confidence. Detection feedback is used only in the loss path and does not enter
the forward feature path.
