from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBNAct


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block with the canonical 1x1-3x3-1x1 topology."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = bottleneck_channels * self.expansion
        self.conv1 = ConvBNAct(in_channels, bottleneck_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBNAct(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv3 = ConvBNAct(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            act=False,
        )
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                act=False,
            )
        else:
            self.downsample = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        return self.act(out + identity)


class ResNet50Encoder(nn.Module):
    """Independent ResNet-50 style encoder that emits C2-C5.

    The stage depths are the ResNet-50 bottleneck depths (3, 4, 6, 3). The
    initial stem accepts single-channel IR or VIS tensors directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        layers: tuple[int, int, int, int] = (3, 4, 6, 3),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
        )
        self.inplanes = base_channels
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)
        self.out_channels = (
            base_channels * BottleneckBlock.expansion,
            base_channels * 2 * BottleneckBlock.expansion,
            base_channels * 4 * BottleneckBlock.expansion,
            base_channels * 8 * BottleneckBlock.expansion,
        )

    def _make_layer(
        self,
        bottleneck_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BottleneckBlock(self.inplanes, bottleneck_channels, stride=stride)]
        self.inplanes = bottleneck_channels * BottleneckBlock.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.inplanes, bottleneck_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class CNNEncoder(ResNet50Encoder):
    """Backward-compatible alias for older imports.

    The old encoder exposed three stages. The enhanced system emits C2-C5, so
    callers should now consume four returned tensors.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] | None = None,
        base_channels: int | None = None,
    ) -> None:
        if base_channels is None:
            base_channels = channels[0] if channels else 64
        super().__init__(in_channels=in_channels, base_channels=base_channels)
