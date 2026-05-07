"""Tests for Hugging Face vision helpers."""

from types import SimpleNamespace

import torch

from excvish.hf.vision import (
    requires_model_positional_interpolation,
    resolve_effective_input_size,
    resolve_model_forward_kwargs,
    resolve_processor_input_size,
)


class PlainForwardModel(torch.nn.Module):
    """Expose a model without positional-interpolation support."""

    def __init__(self) -> None:
        """Store a minimal config object with an image_size attribute."""
        super().__init__()
        self.config = SimpleNamespace(image_size=32)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged so signature inspection remains the behavior under test."""
        return pixel_values


class InterpolationForwardModel(torch.nn.Module):
    """Expose the forward argument used by DeiT-style positional interpolation."""

    def __init__(self) -> None:
        """Store a minimal config object with an image_size attribute."""
        super().__init__()
        self.config = SimpleNamespace(image_size=32)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """Return the input unchanged while exposing interpolation support in the signature."""
        return pixel_values


def test_resolve_model_forward_kwargs_enables_interpolation_for_larger_inputs() -> None:
    """Detect larger incoming images than the checkpoint config and request interpolation."""
    model = InterpolationForwardModel()

    forward_kwargs = resolve_model_forward_kwargs(model, torch.randn(2, 3, 48, 48))

    assert forward_kwargs == {"interpolate_pos_encoding": True}


def test_requires_model_positional_interpolation_ignores_unsupported_models() -> None:
    """Return False for models whose forward path does not expose an interpolation switch."""
    model = PlainForwardModel()

    assert requires_model_positional_interpolation(model, 48) is False


def test_resolve_processor_input_size_reads_crop_then_size_metadata() -> None:
    """Resolve square preprocessing size from common image processor metadata layouts."""
    processor = SimpleNamespace(
        crop_size={"height": 224, "width": 256}, size={"shortest_edge": 384}
    )
    resize_only_processor = SimpleNamespace(crop_size=None, size={"shortest_edge": 384})

    assert resolve_processor_input_size(processor) == 256
    assert resolve_processor_input_size(resize_only_processor) == 384


def test_resolve_effective_input_size_prefers_explicit_override() -> None:
    """Return explicit override when provided and checkpoint size otherwise."""
    assert resolve_effective_input_size(384, checkpoint_input_size=224) == 384
    assert resolve_effective_input_size(None, checkpoint_input_size=224) == 224
