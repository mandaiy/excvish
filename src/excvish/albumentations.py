"""Albumentations helpers for excvish pipelines.

The utilities defined here provide lightweight introspection helpers that are
used by Ultralytics adapters to determine whether a transform manipulates
spatial targets (e.g., bounding boxes or keypoints) alongside the image.
"""

import albumentations as A
from albumentations.core.composition import TransformType


def has_dual_transform(transform: TransformType) -> bool:
    """Report whether the transform updates spatial targets alongside the image.

    Args:
        transform: Albumentations transform or composition to inspect.

    Returns:
        bool: True when the transform inherits from ``A.DualTransform`` or any
        child within a composition does.

    Raises:
        ValueError: If ``transform`` is neither a ``BasicTransform`` instance nor
            an ``albumentations`` composition.
    """
    if not isinstance(transform, (A.BasicTransform, A.BaseCompose)):
        raise ValueError(f"Unexpected transform type: {type(transform)}")

    if isinstance(transform, A.BasicTransform):
        return isinstance(transform, A.DualTransform)

    return any(has_dual_transform(t) for t in transform.transforms)
