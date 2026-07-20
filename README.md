# SDIF-Net: Cross-modal QKV Detection-aware IR-VIS Fusion

This project implements a trainable PyTorch pipeline for infrared-visible image
fusion and detection-guided optimization.

## What Is Included

- independent IR and VIS ResNet-50 style bottleneck encoders
- P2-P5 FPN top-down multi-scale enhancement
- four-scale bidirectional cross-modal QKV fusion over FPN features
- FPN-style decoder that reconstructs `I_fused`
- anchor-based dense detection head with small P2 anchors for differentiable joint training
- detection feedback as a dynamic detection-loss weight
- ablation switch: `use_feedback`

## Data Layout

The default reader targets the current M3FD-style structure:

```text
datasets/M3FD_Detection
|__ ir
|__ vi
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
  --num-classes 6 ^
  --anchor-sizes 8 16 32 64
```

Disable modules for ablation:

```bash
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
- `fused_features`: P2, P3, P4, P5 fused features
- `forward_logs`: confirms the single-pass bidirectional cross-modal QKV pipeline

## Enhanced Architecture

The network now follows the requested dual-stream, FPN, bidirectional QKV, dual-head
architecture. IR and VIS are encoded by independent ResNet-50 style bottleneck
branches and converted into P2-P5 pyramids. Each pyramid level runs two explicit
cross-modal QKV paths: IR queries VIS keys/values and VIS queries IR keys/values.
The fusion projection receives both cross-attention responses plus the original
IR/VIS FPN features through a gated residual path, preserving modality-specific
thermal and texture evidence instead of collapsing everything into a one-way
attention map.

The detector consumes the fused P2-P5 features directly. P2 is configured with
an 8-pixel base anchor by default, followed by 16, 32, and 64 pixels on deeper
levels, with configurable aspect ratios. This keeps the training path compact
while explicitly biasing the model toward far small pedestrians and vehicles.

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
fusion image, and computes a compact objective:

```text
L_total = L_fusion + lambda_det * L_detection
```

`L_fusion` is grouped into four terms: intensity preservation, gradient
preservation, SSIM structure, and modal-specific information preservation.

When `use_feedback=True`, `lambda_det` increases for low recall or low detection
confidence. Detection feedback is used only in the loss path and does not enter
the forward feature path.
