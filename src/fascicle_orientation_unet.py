"""
U-Net style orientation-field model for fascicle direction prediction.

Input
-----
Grayscale ultrasound image:
    [B, 1, H, W]

Output
------
Raw 2-channel orientation vectors:
    [B, 2, H, W]

The two output channels represent an unconstrained prediction of:
    (cos(2 theta), sin(2 theta))

The model intentionally does NOT normalize the final vectors inside forward().
Normalization should be handled in the loss / evaluation code with an epsilon
for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two consecutive 3x3 convolutions with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Encoder stage:
        max-pool -> ConvBlock
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    Decoder stage:
        transpose-conv upsample
        -> concatenate encoder skip connection
        -> ConvBlock
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class FascicleOrientationUNet(nn.Module):
    """
    U-Net for dense local fascicle orientation prediction.

    Parameters
    ----------
    in_channels:
        Number of image channels. Default 1 for grayscale ultrasound.
    out_channels:
        Number of orientation channels. Default 2 for
        (cos(2 theta), sin(2 theta)).
    base_channels:
        Width of the first encoder level. Default 32.

    Notes
    -----
    The final layer returns raw vectors. Use `normalize_orientation_field()`
    before angular comparison or visualization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)
        self.enc4 = DownBlock(c3, c4)

        self.bottleneck_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.bottleneck = ConvBlock(c4, c5)

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.head = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        xb = self.bottleneck_pool(x4)
        xb = self.bottleneck(xb)

        x = self.up4(xb, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        return self.head(x)


def normalize_orientation_field(
    field: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize a raw 2-channel orientation field to unit vectors.

    Parameters
    ----------
    field:
        Tensor of shape [B, 2, H, W].
    eps:
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Unit-length orientation vectors with the same shape.
    """
    if field.ndim != 4:
        raise ValueError(
            f"`field` must have shape [B, 2, H, W], "
            f"got {tuple(field.shape)}."
        )

    if field.shape[1] != 2:
        raise ValueError(
            f"`field` must have exactly 2 channels, "
            f"got {field.shape[1]}."
        )

    norm = torch.sqrt(
        field[:, 0:1] ** 2
        + field[:, 1:2] ** 2
        + eps
    )

    return field / norm


if __name__ == "__main__":
    model = FascicleOrientationUNet()

    x = torch.randn(
        2,
        1,
        512,
        512,
    )

    y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
