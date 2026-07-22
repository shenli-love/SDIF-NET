"""可视化网络编码器 C2-C5 和 FPN P2-P5 特征图。

用法:
    python -m tests.vis_features \
        --ir-image datasets/M3FD_Detection/ir/00061.png \
        --vis-image datasets/M3FD_Detection/vi/00061.png \
        --checkpoint checkpoints/latest.pt \
        --output-dir feature_vis

输出:
    output_dir/
        IR_C2.png ~ IR_C5.png   (红外编码器特征)
        VIS_C2.png ~ VIS_C5.png (可见光编码器特征)
        IR_P2.png ~ IR_P5.png   (红外 FPN 特征)
        VIS_P2.png ~ VIS_P5.png (可见光 FPN 特征)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from irvis_fusion.models import IRVISFusionDetectionNet


# ─────────────────────────── 工具函数 ───────────────────────────


def load_gray_tensor(path: str | Path, image_size: tuple[int, int]) -> torch.Tensor:
    """加载灰度图并归一化到 [0,1]，返回 (1,1,H,W) 张量。"""
    image = Image.open(path).convert("L")
    h, w = image_size
    if image.size != (w, h):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((w, h), resample)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def feature_map_to_image(feat: torch.Tensor, target_size: tuple[int, int] | None = None) -> Image.Image:
    """将单张特征图 (C,H,W) 转为灰度 PIL Image。

    策略: 对所有通道取均值 → min-max 归一化 → 映射到 [0,255]。
    可选上采样到 target_size 以便对比。
    """
    feat = feat.detach().float().cpu()
    # 通道均值
    heatmap = feat.mean(dim=0)  # (H, W)
    # min-max 归一化
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin > 1e-8:
        heatmap = (heatmap - vmin) / (vmax - vmin)
    else:
        heatmap = torch.zeros_like(heatmap)
    arr = (heatmap.numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if target_size is not None and img.size != (target_size[1], target_size[0]):
        resample = getattr(Image, "Resampling", Image).BILINEAR
        img = img.resize((target_size[1], target_size[0]), resample)
    return img


def make_grid_image(images: list[Image.Image], titles: list[str], cols: int = 4) -> Image.Image:
    """将多张灰度图拼成网格，附带标题。"""
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        ImageDraw = None

    pad = 4
    title_h = 20
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    rows = (len(images) + cols - 1) // cols

    canvas_w = cols * (max_w + pad) + pad
    canvas_h = rows * (max_h + title_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas) if ImageDraw else None

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + title_h + pad)
        # 标题
        if draw:
            draw.text((x + 2, y), title, fill=(0, 0, 0))
        # 图像
        canvas.paste(img.convert("RGB"), (x, y + title_h))

    return canvas


# ─────────────────────────── 主逻辑 ───────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化编码器 C2-C5 和 FPN P2-P5 特征图")
    parser.add_argument("--ir-image", type=str, required=True, help="红外图像路径")
    parser.add_argument("--vis-image", type=str, required=True, help="可见光图像路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型权重路径 (可选)")
    parser.add_argument("--output-dir", type=str, default="feature_vis", help="输出目录")
    parser.add_argument("--image-size", nargs=2, type=int, default=[768, 1024], help="输入尺寸 H W")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--resnet-base-channels", type=int, default=64)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--anchor-sizes", nargs=4, type=float, default=[8.0, 16.0, 32.0, 64.0])
    parser.add_argument("--anchor-ratios", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--upsample-to-input", action="store_true",
                        help="将特征图上采样到输入图像尺寸 (便于对比)")
    parser.add_argument("--save-grid", action="store_true", default=True,
                        help="额外保存拼接网格图")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    image_size = tuple(args.image_size)  # (H, W)

    # ── 1. 构建模型 ──
    model = IRVISFusionDetectionNet(
        num_classes=args.num_classes,
        resnet_base_channels=args.resnet_base_channels,
        fpn_channels=args.fpn_channels,
        anchor_sizes=tuple(args.anchor_sizes),
        anchor_ratios=tuple(args.anchor_ratios),
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        print(f"[INFO] 已加载权重: {args.checkpoint}")
    else:
        print("[INFO] 未指定 checkpoint，使用随机初始化权重")

    model.eval()

    # ── 2. 加载图像 ──
    ir = load_gray_tensor(args.ir_image, image_size).to(device)
    vis = load_gray_tensor(args.vis_image, image_size).to(device)
    print(f"[INFO] 输入尺寸: IR={list(ir.shape)}, VIS={list(vis.shape)}")

    # ── 3. 提取 C2-C5 (编码器输出) ──
    ir_c2, ir_c3, ir_c4, ir_c5 = model.ir_encoder(ir)
    vis_c2, vis_c3, vis_c4, vis_c5 = model.vis_encoder(vis)

    # ── 4. 提取 P2-P5 (FPN 输出) ──
    ir_p2, ir_p3, ir_p4, ir_p5 = model.ir_fpn((ir_c2, ir_c3, ir_c4, ir_c5))
    vis_p2, vis_p3, vis_p4, vis_p5 = model.vis_fpn((vis_c2, vis_c3, vis_c4, vis_c5))

    # ── 5. 可视化 ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = image_size if args.upsample_to_input else None

    # 收集所有特征图
    feature_groups = {
        "IR_C": [("IR_C2", ir_c2), ("IR_C3", ir_c3), ("IR_C4", ir_c4), ("IR_C5", ir_c5)],
        "VIS_C": [("VIS_C2", vis_c2), ("VIS_C3", vis_c3), ("VIS_C4", vis_c4), ("VIS_C5", vis_c5)],
        "IR_P": [("IR_P2", ir_p2), ("IR_P3", ir_p3), ("IR_P4", ir_p4), ("IR_P5", ir_p5)],
        "VIS_P": [("VIS_P2", vis_p2), ("VIS_P3", vis_p3), ("VIS_P4", vis_p4), ("VIS_P5", vis_p5)],
    }

    all_images: list[Image.Image] = []
    all_titles: list[str] = []

    for group_name, feats in feature_groups.items():
        for name, feat_tensor in feats:
            # feat_tensor shape: (1, C, H, W) → 取 batch 0
            img = feature_map_to_image(feat_tensor[0], target_size=target_size)
            save_path = output_dir / f"{name}.png"
            img.save(save_path)
            print(f"  [保存] {save_path}  (原始特征尺寸: {list(feat_tensor.shape[2:])}, 通道数: {feat_tensor.shape[1]})")
            all_images.append(img)
            all_titles.append(f"{name}\n{feat_tensor.shape[1]}ch {feat_tensor.shape[2]}x{feat_tensor.shape[3]}")

    # ── 6. 保存网格总览图 ──
    if args.save_grid:
        grid = make_grid_image(all_images, all_titles, cols=4)
        grid_path = output_dir / "feature_grid_all.png"
        grid.save(grid_path)
        print(f"\n[INFO] 网格总览已保存: {grid_path}")

        # 分组网格
        for group_name, feats in feature_groups.items():
            group_imgs = []
            group_titles = []
            for name, feat_tensor in feats:
                img = feature_map_to_image(feat_tensor[0], target_size=target_size)
                group_imgs.append(img)
                group_titles.append(f"{name} ({feat_tensor.shape[1]}ch)")
            group_grid = make_grid_image(group_imgs, group_titles, cols=4)
            group_path = output_dir / f"feature_grid_{group_name}.png"
            group_grid.save(group_path)
            print(f"[INFO] {group_name} 网格已保存: {group_path}")

    print(f"\n[完成] 所有特征图已保存至: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
