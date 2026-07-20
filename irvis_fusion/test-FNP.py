from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 修正为绝对导入，确保在 -m 模式下正确运行
from irvis_fusion.models.encoder import CNNEncoder
from irvis_fusion.models.fpn import FeaturePyramid


def load_grayscale_image(path: str | Path, target_size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("L")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor


def save_feature_map(tensor: torch.Tensor, save_path: Path, layer_name: str) -> None:
    """
    将多通道特征图转成灰度图并保存。
    方法：对所有通道取均值，再归一化到 [0, 1]
    """
    # 1. 必须添加 .detach() 去掉梯度，才能转 numpy
    feat_map = tensor.squeeze(0).mean(dim=0).detach()  # (H, W)

    # 归一化到 0~1
    min_val = feat_map.min()
    max_val = feat_map.max()
    if max_val - min_val > 1e-6:
        feat_map = (feat_map - min_val) / (max_val - min_val)
    else:
        feat_map = torch.zeros_like(feat_map)

    # 转为 numpy 并保存
    arr = (feat_map.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    img.save(save_path)
    print(f"  ✅ 已保存: {save_path.name} (原特征图通道数: {tensor.shape[1]})")


def main() -> None:
    parser = argparse.ArgumentParser(description="拆解 Encoder 和 FPN，并保存各层特征图为图片")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--height", type=int, default=768, help="目标高度")
    parser.add_argument("--width", type=int, default=1024, help="目标宽度")
    parser.add_argument("--output-dir", type=str, default="feature_vis", help="输出目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 可视化结果将保存至: {out_dir.absolute()}\n")

    # 1. 加载输入
    image_tensor = load_grayscale_image(args.image, (args.height, args.width)).to(device)
    print("📥 输入图片信息:")
    print(f"  形状: {tuple(image_tensor.shape)}")

    # 2. 实例化模型 (模型默认带有 requires_grad=True)
    encoder = CNNEncoder(in_channels=1, base_channels=32).to(device)
    fpn = FeaturePyramid(in_channels=encoder.out_channels, out_channels=128).to(device)

    # 设置为 eval 模式，模型内部不再跟踪梯度（但输出的张量依然会带有梯度历史，所以需要 detach）
    encoder.eval()
    fpn.eval()

    # 3. Encoder 前向
    print("\n🔧 [1] 运行 CNNEncoder ...")
    with torch.no_grad():  # 加上这个上下文管理器更安全，直接禁用梯度计算
        f1, f2, f3, f4 = encoder(image_tensor)

    # 保存 Encoder 输出
    save_feature_map(f1, out_dir / "01_Stage1_f1.png", "Stage1 (f1)")
    save_feature_map(f2, out_dir / "02_Stage2_f2.png", "Stage2 (f2)")
    save_feature_map(f3, out_dir / "03_Stage3_f3.png", "Stage3 (f3)")
    save_feature_map(f4, out_dir / "04_Stage4_f4.png", "Stage4 (f4)")

    # 4. FPN 前向
    print("\n🔧 [2] 运行 FeaturePyramid ...")
    with torch.no_grad():
        p1, p2, p3, p4 = fpn((f1, f2, f3, f4))

    # 保存 FPN 输出
    save_feature_map(p1, out_dir / "05_FPN_P2.png", "FPN P2")
    save_feature_map(p2, out_dir / "06_FPN_P3.png", "FPN P3")
    save_feature_map(p3, out_dir / "07_FPN_P4.png", "FPN P4")
    save_feature_map(p4, out_dir / "08_FPN_P5.png", "FPN P5")

    print("\n🎉 可视化拆解完成！")
    print("💡 观察建议：")
    print("   - Stage 1 (f1): 尺寸与原图一致，云的细节最清晰。")
    print("   - Stage 2 (f2): 下采样了2倍，云朵纹理开始变模糊，但保留了轮廓。")
    print("   - Stage 3 (f3): 下采样了4倍，云朵纹理几乎消失，变成了抽象的'块状语义'。")
    print("   - FPN P1/P2/P3: 经过自顶向下融合后，底层细节和高层语义进行了结合。")


if __name__ == "__main__":
    main()
