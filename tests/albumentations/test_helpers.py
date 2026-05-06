"""Tests for Albumentations helper functions."""

from typing import cast

import albumentations as A
import numpy as np
from PIL import Image

from excvish.albumentations.helpers import apply_rgba_augmentations, build_fixed_crop_transform


def test_apply_rgba_augmentations_uses_fixed_background_for_alpha() -> None:
    """RGBA inputs should use the alpha channel as mask and composite onto the fixed color."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    pil_img = Image.fromarray(rgba)
    transform = A.Compose([A.Resize(height=2, width=2)])

    result = apply_rgba_augmentations(
        pil_img,
        transform,
        background_color=(128, 128, 128),
    )

    np.testing.assert_array_equal(result, np.full((2, 2, 3), 128, dtype=np.uint8))


def test_apply_rgba_augmentations_passes_feed_metadata_to_transform() -> None:
    """Metadata callbacks should provide extra Albumentations inputs."""
    image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))

    class MetadataTransform:
        """Record metadata by adding it to the image and forwarding mask."""

        def __call__(self, **kwargs: object) -> dict[str, object]:
            image_array = kwargs["image"]
            assert isinstance(image_array, np.ndarray)
            value = kwargs["value"]
            assert isinstance(value, int)
            return {"image": image_array + value, "mask": kwargs.get("mask")}

    transform = cast(A.Compose, MetadataTransform())

    result = apply_rgba_augmentations(image, transform, feed_metadata=lambda: {"value": 7})

    np.testing.assert_array_equal(result, np.full((2, 2, 3), 7, dtype=np.uint8))


def test_build_fixed_crop_transform_resizes_then_crops() -> None:
    """The fixed crop builder should emit the requested final spatial dimensions."""
    image = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
    transform = build_fixed_crop_transform(x_min=1, y_min=2, resize_size=6, crop_size=3)

    result = transform(image=image)["image"]

    assert result.shape == (3, 3, 3)
    np.testing.assert_array_equal(result, image[2:5, 1:4])
