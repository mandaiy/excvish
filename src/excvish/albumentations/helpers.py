"""Helpers for Albumentations transform inspection and PIL/RGBA application."""

from collections.abc import Callable
from typing import Any

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformType
from PIL import Image

from excvish.cv.colors import apply_random_background_np, apply_solid_background_np


def has_dual_transform(transform: TransformType) -> bool:
    """Report whether a transform mutates spatial targets with the image.

    Args:
        transform: Albumentations transform or composition to inspect.

    Returns:
        True if the transform itself (or any child transform in a composition)
        is a ``DualTransform``.

    Raises:
        ValueError: If ``transform`` is not an Albumentations transform type.
    """
    if not isinstance(transform, A.BasicTransform | A.BaseCompose):
        raise ValueError(f"Unexpected transform type: {type(transform)}")

    if isinstance(transform, A.BasicTransform):
        return isinstance(transform, A.DualTransform)

    return any(has_dual_transform(t) for t in transform.transforms)


def apply_rgba_augmentations(
    pil_img: Image.Image,
    transform: A.Compose,
    feed_metadata: Callable[[], dict[str, Any]] | None = None,
    rng: np.random.Generator | None = None,
    background_color: tuple[int, int, int] | None = None,
    random_background_seed: int | None = None,
) -> np.ndarray:
    """Apply an Albumentations transform to a PIL image while preserving alpha as mask.

    Args:
        pil_img: Input PIL image.
        transform: Albumentations composition to apply.
        feed_metadata: Optional callback that returns additional transform targets.
        rng: Optional generator for random background colors.
        background_color: Optional fixed RGB background for RGBA inputs.
        random_background_seed: Seed used when ``rng`` is not provided and a random
            background is needed.

    Returns:
        Transformed RGB image array. RGBA inputs are split into RGB and alpha mask,
        transformed together, and composited onto either a fixed or random background.
    """
    np_img = np.array(pil_img)

    if np_img.ndim == 3 and np_img.shape[2] == 4:
        rgb = np_img[..., :3]
        mask = np_img[..., 3]
    else:
        rgb = np_img
        mask = None

    metadata = feed_metadata() if feed_metadata else {}
    transformed = transform(image=rgb, mask=mask, **metadata)
    rgb = transformed["image"]
    mask = transformed.get("mask")

    if mask is None:
        return rgb

    rgba = np.dstack([rgb, mask])
    if background_color is not None:
        return apply_solid_background_np(rgba, background_color=background_color)

    effective_rng = rng
    if effective_rng is None and random_background_seed is not None:
        effective_rng = np.random.default_rng(random_background_seed)
    return apply_random_background_np(rgba, rng=effective_rng)


def build_fixed_crop_transform(
    *,
    x_min: int,
    y_min: int,
    resize_size: int,
    crop_size: int,
) -> A.Compose:
    """Resize to a square canvas, then crop a fixed square window.

    Args:
        x_min: Left crop coordinate after resize.
        y_min: Top crop coordinate after resize.
        resize_size: Intermediate square size.
        crop_size: Final square crop size.

    Returns:
        Albumentations composition that emits a deterministic crop.
    """
    return A.Compose(
        [
            A.Resize(height=resize_size, width=resize_size),
            A.Crop(x_min=x_min, y_min=y_min, x_max=x_min + crop_size, y_max=y_min + crop_size),
        ]
    )
