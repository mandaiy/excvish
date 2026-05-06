"""Hugging Face vision model and processor helpers."""

import inspect
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import torch


def resolve_model_forward_kwargs(
    model: torch.nn.Module, pixel_values: torch.Tensor
) -> dict[str, Any]:
    """Inspect tensor and config image sizes, then enable positional interpolation when needed."""
    if not requires_model_positional_interpolation(model, pixel_values):
        return {}
    return {"interpolate_pos_encoding": True}


def requires_model_positional_interpolation(model: torch.nn.Module, image_size: Any) -> bool:
    """Report whether requested image size differs from config.image_size."""
    if not _supports_interpolate_pos_encoding(type(model)):
        return False

    configured_image_size = _normalize_image_size(
        getattr(getattr(model, "config", None), "image_size", None)
    )
    if configured_image_size is None:
        return False

    requested_image_size = _normalize_image_size(image_size)
    if requested_image_size is None and isinstance(image_size, torch.Tensor):
        requested_image_size = _normalize_tensor_image_size(image_size)
    if requested_image_size is None:
        return False
    return requested_image_size != configured_image_size


def resolve_processor_input_size(processor: Any, fallback: int = 224) -> int:
    """Read preferred square size from processor crop metadata first and resize metadata second."""
    crop_size = getattr(processor, "crop_size", None)
    if isinstance(crop_size, Mapping):
        if "height" in crop_size and "width" in crop_size:
            return int(max(crop_size["height"], crop_size["width"]))
        if "shortest_edge" in crop_size:
            return int(crop_size["shortest_edge"])
    elif isinstance(crop_size, int):
        return int(crop_size)

    size = getattr(processor, "size", None)
    if isinstance(size, Mapping):
        if "height" in size and "width" in size:
            return int(max(size["height"], size["width"]))
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
    elif isinstance(size, int):
        return int(size)

    return fallback


def resolve_effective_input_size(
    requested_input_size: int | None, checkpoint_input_size: int
) -> int:
    """Prefer an explicit override and otherwise reuse the checkpoint processor size."""
    if requested_input_size is None:
        return checkpoint_input_size
    return requested_input_size


@cache
def _supports_interpolate_pos_encoding(model_type: type[torch.nn.Module]) -> bool:
    """Cache whether one model class exposes interpolate_pos_encoding on its forward signature."""
    try:
        signature = inspect.signature(model_type.forward)
    except (TypeError, ValueError):
        return False
    return "interpolate_pos_encoding" in signature.parameters


def _normalize_image_size(image_size: Any) -> tuple[int, int] | None:
    """Convert supported config layouts into one height-width tuple."""
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, Mapping):
        if "height" in image_size and "width" in image_size:
            return (int(image_size["height"]), int(image_size["width"]))
        return None
    if (
        isinstance(image_size, Sequence)
        and not isinstance(image_size, str)
        and len(image_size) == 2
    ):
        height, width = image_size
        if isinstance(height, int) and isinstance(width, int):
            return (height, width)
    return None


def _normalize_tensor_image_size(pixel_values: torch.Tensor) -> tuple[int, int] | None:
    """Read trailing height and width dimensions from one image batch tensor when available."""
    if pixel_values.ndim < 4:
        return None
    return (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
