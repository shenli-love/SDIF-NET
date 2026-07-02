# IR-VIS Fusion with SAM-guided Detection Feedback Iterative Network

This project implements a trainable PyTorch pipeline for infrared-visible image
fusion and YOLO-style detection feedback.

## What Is Included

- independent IR and VIS CNN encoders
- FPN top-down multi-scale enhancement
- `SAMPriorEncoder` for SAM mask attention
- three-level fusion: detail, semantic interaction, and target-aware feedback
- FPN-style decoder that reconstructs `I_fused`
- compact YOLO-like dense detection head
- detection feedback loop that builds `M_miss`, uncertainty `U`, and `G_fb`
- ablation switches: `use_sam`, `use_feedback`, `max_iterations`

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

The smoke test runs dynamic-size forward, two feedback iterations, detection
loss, fusion loss, and backward propagation.

## Train

```bash
python -m irvis_fusion.train ^
  --data-root datasets/M3FD_Detection ^
  --image-size 256 320 ^
  --batch-size 2 ^
  --epochs 20 ^
  --num-classes 6 ^
  --max-iterations 3
```

Disable modules for ablation:

```bash
python -m irvis_fusion.train --no-sam
python -m irvis_fusion.train --no-feedback
python -m irvis_fusion.train --max-iterations 1
```

## Main Output Keys

`IRVISFusionDetectionNet.forward()` returns:

- `I_fused`: reconstructed fused image
- `detections`: raw and decoded YOLO-like predictions
- `fused_features`: level1, level2, level3 fused features
- `feedback`: `M_miss`, `U`, `G_fb`, recall, and confidence
- `iteration_logs`: per-iteration recall/confidence summaries

## Replacing The Detector

`irvis_fusion/models/detector.py` is intentionally a small YOLO-style head. For
paper experiments with a full detector, keep its return contract:

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

The feedback module and detection loss consume only the decoded fields.
