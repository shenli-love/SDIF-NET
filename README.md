# SQUA-Net: SAM-QKV Detection-aware IR-VIS Fusion

This project implements a trainable PyTorch pipeline for infrared-visible image
fusion and detection-guided optimization.

## What Is Included

- independent IR and VIS CNN encoders
- FPN top-down multi-scale enhancement
- `SAMPriorEncoder` for a soft SAM attention prior
- small-target-aware three-scale SQUA fusion over FPN features
- FPN-style decoder that reconstructs `I_fused`
- compact YOLO-like dense detection head for differentiable joint training
- detection feedback as a dynamic detection-loss weight
- ablation switches: `use_sam`, `use_feedback`

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

The smoke test runs dynamic-size forward, detection-aware fusion loss,
differentiable detection loss, feedback weighting, and backward propagation
through the fusion and YOLO-like detection branches.

## Train

```bash
python -m irvis_fusion.train ^
  --data-root datasets/M3FD_Detection ^
  --image-size 768 1024 ^
  --batch-size 1 ^
  --epochs 20 ^
  --num-classes 6
```

Disable modules for ablation:

```bash
python -m irvis_fusion.train --no-sam
python -m irvis_fusion.train --no-feedback
```

## Inference

```bash
python -m irvis_fusion.infer ^
  --data-root datasets/M3FD_Detection ^
  --split val ^
  --checkpoint runs/irvis_sdif_feedback/epoch_020.pt ^
  --output-dir runs/infer
```

Inference saves:

- `fused/*.png`: fused images
- `detections/*.txt`: `class score x1 y1 x2 y2` normalized detections
- `visualizations/*.png`: fused images with detection boxes

## Main Output Keys

`IRVISFusionDetectionNet.forward()` returns:

- `I_fused`: reconstructed fused image
- `detections`: raw and decoded detector predictions
- `fused_features`: level1, level2, level3 fused features
- `sam_attention`: soft SAM prior `A_sam`
- `forward_logs`: confirms the single-pass SQUA pipeline

## Detector

The built-in YOLO-like detector is trainable and keeps the fusion/detection
objective fully differentiable.

SQUA avoids full-image softmax suppression by mixing local-window contrast
attention with global context. It also learns a region-adaptive IR/VIS modality
gate, giving IR stronger influence in locally salient thermal regions.

The detector keeps this return contract:

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

The loss consumes decoded fields, applies GT-box spatial weighting to the
fusion image, and computes:

```text
L_total = L_fusion + lambda_det * L_detection
```

When `use_feedback=True`, `lambda_det` increases for low recall or low detection
confidence. Detection feedback is used only in the loss path and does not enter
the forward feature path.
