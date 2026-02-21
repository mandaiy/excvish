"""Helpers for Albumentations transform inspection."""

import albumentations as A
from albumentations.core.composition import TransformType


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
