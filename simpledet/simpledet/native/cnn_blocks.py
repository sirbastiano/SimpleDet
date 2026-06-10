"""Reusable CNN blocks for native detector composition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..detectors._deps import require_dependency
from ..errors import TensorContractError

require_dependency("torch", "native CNN blocks")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


_CONVNEXT_PRESETS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "convnext_tiny": ((96, 192, 384, 768), (3, 3, 9, 3)),
}


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_channels = _positive_int(num_channels, "num_channels")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.num_channels))
        self.bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_nchw(x, channels=self.num_channels, block_name=self.__class__.__name__)
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Per-sample stochastic depth used by ConvNeXt-style residual blocks."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        if self.drop_prob < 0.0 or self.drop_prob >= 1.0:
            raise ValueError("drop_prob must be in the range [0, 1).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (int(x.shape[0]),) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class ConvNeXtBlock(nn.Module):
    """ConvNeXt residual block with depthwise convolution and pointwise MLP."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        hidden_channels = self.channels * _positive_int(expansion, "expansion")
        if int(kernel_size) % 2 != 1:
            raise ValueError("ConvNeXtBlock requires an odd kernel_size.")
        padding = int(kernel_size) // 2
        self.depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=int(kernel_size),
            padding=padding,
            groups=self.channels,
        )
        self.norm = LayerNorm2d(self.channels)
        self.pointwise1 = nn.Conv2d(self.channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pointwise2 = nn.Conv2d(hidden_channels, self.channels, kernel_size=1)
        self.drop_path = DropPath(drop_path)
        self.layer_scale = (
            nn.Parameter(float(layer_scale_init_value) * torch.ones(self.channels))
            if float(layer_scale_init_value) > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_nchw(x, channels=self.channels, block_name=self.__class__.__name__)
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)
        if self.layer_scale is not None:
            x = x * self.layer_scale.view(1, -1, 1, 1)
        return residual + self.drop_path(x)


class ConvNeXtStage(nn.Module):
    """A ConvNeXt feature stage with optional spatial downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        downsample: bool,
        drop_path_rates: Iterable[float] | None = None,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.depth = _positive_int(depth, "depth")
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(self.in_channels),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=2, stride=2),
            )
        elif self.in_channels != self.out_channels:
            self.downsample = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()
        rates = list(drop_path_rates or ())
        if not rates:
            rates = [0.0] * self.depth
        if len(rates) != self.depth:
            raise ValueError("drop_path_rates length must match stage depth.")
        self.blocks = nn.Sequential(
            *(
                ConvNeXtBlock(
                    self.out_channels,
                    drop_path=float(rates[index]),
                    layer_scale_init_value=layer_scale_init_value,
                )
                for index in range(self.depth)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class ConvNeXtFeatureBackbone(nn.Module):
    """Native ConvNeXt feature backbone assembled from reusable CNN blocks."""

    def __init__(
        self,
        *,
        model_name: str = "convnext_tiny",
        in_channels: int = 3,
        out_indices: tuple[int, ...] | list[int] | str | None = None,
        pretrained: bool = False,
        stage_channels: tuple[int, ...] | list[int] | None = None,
        depths: tuple[int, ...] | list[int] | None = None,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name or "convnext_tiny")
        self.pretrained = bool(pretrained)
        default_channels, default_depths = _CONVNEXT_PRESETS.get(
            self.model_name,
            _CONVNEXT_PRESETS["convnext_tiny"],
        )
        self.stage_channels = _positive_int_tuple(stage_channels or default_channels, "stage_channels")
        self.depths = _positive_int_tuple(depths or default_depths, "depths")
        if len(self.stage_channels) != len(self.depths):
            raise ValueError("ConvNeXt stage_channels and depths must have the same length.")
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_indices = _normalize_out_indices(
            out_indices,
            num_stages=len(self.stage_channels),
        )
        self.feature_channels = tuple(self.stage_channels[index] for index in self.out_indices)

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.stage_channels[0], kernel_size=4, stride=4),
            LayerNorm2d(self.stage_channels[0]),
        )
        drop_rates = torch.linspace(0, float(drop_path_rate), sum(self.depths)).tolist()
        offset = 0
        stages: list[nn.Module] = []
        for index, (channels, depth) in enumerate(zip(self.stage_channels, self.depths)):
            stage_rates = drop_rates[offset : offset + depth]
            offset += depth
            stages.append(
                ConvNeXtStage(
                    self.stage_channels[index - 1] if index else channels,
                    channels,
                    depth=depth,
                    downsample=index > 0,
                    drop_path_rates=stage_rates,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        outputs = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in self.out_indices:
                outputs.append(x)
        return tuple(outputs)


def _positive_int(value: int, field_name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return resolved


def _positive_int_tuple(values: Iterable[int], field_name: str) -> tuple[int, ...]:
    resolved = tuple(_positive_int(int(value), field_name) for value in values)
    if not resolved:
        raise ValueError(f"{field_name} must not be empty.")
    return resolved


def _normalize_out_indices(
    out_indices: tuple[int, ...] | list[int] | str | None,
    *,
    num_stages: int,
) -> tuple[int, ...]:
    if out_indices is None:
        values = tuple(range(num_stages))
    elif isinstance(out_indices, str):
        values = tuple(int(item.strip()) for item in out_indices.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in out_indices)
    if not values:
        raise ValueError("out_indices must contain at least one stage.")
    zero_based = 0 in values
    normalized = values if zero_based else tuple(index - 1 for index in values)
    invalid = [index for index in normalized if index < 0 or index >= num_stages]
    if invalid:
        available = ", ".join(str(index) for index in range(1, num_stages + 1))
        raise ValueError(f"ConvNeXt out_indices must be in: {available}.")
    return normalized


def _validate_nchw(x: torch.Tensor, *, channels: int, block_name: str) -> None:
    if x.ndim != 4:
        raise TensorContractError(f"{block_name} expected a 4D NCHW tensor, got {tuple(x.shape)}.")
    if int(x.shape[1]) != int(channels):
        raise TensorContractError(
            f"{block_name} expected {int(channels)} channels, got {int(x.shape[1])}."
        )
