"""Tests for color and alpha compositing utilities."""

import numpy as np
import pytest

import excvish.cv.geometry as geometry
from excvish.cv.colors import (
    adjust_brightness_contrast,
    adjust_saturation,
    align_crop_and_fill_rgba,
    apply_solid_background_np,
)


def test_apply_solid_background_np_composites_uint8_rgba() -> None:
    """Alpha blending should preserve uint8 dtype and use rounded channel values."""
    rgba = np.array([[[200, 100, 0, 128], [10, 20, 30, 255]]], dtype=np.uint8)

    composited = apply_solid_background_np(rgba, background_color=(100, 100, 100))

    expected = np.array([[[150, 100, 50], [10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_array_equal(composited, expected)


def test_apply_solid_background_np_composites_normalized_float_rgba() -> None:
    """Normalized float arrays should blend with normalized background values."""
    rgba = np.array([[[1.0, 0.0, 0.0, 0.25]]], dtype=np.float32)

    composited = apply_solid_background_np(rgba, background_color=(0, 0, 255))

    expected = np.array([[[0.25, 0.0, 0.75]]], dtype=np.float32)
    np.testing.assert_allclose(composited, expected)
    assert composited.dtype == np.float32


def test_adjust_brightness_contrast_clips_to_uint8() -> None:
    """Linear color adjustment should clip values into image channel bounds."""
    image = np.array([[[10, 100, 250]]], dtype=np.uint8)

    adjusted = adjust_brightness_contrast(image, brightness_offset=20.0, contrast_scale=1.1)

    np.testing.assert_array_equal(adjusted, np.array([[[31, 130, 255]]], dtype=np.uint8))


def test_adjust_saturation_returns_original_shape_for_non_rgb() -> None:
    """Non-RGB inputs should be passed through without channel reinterpretation."""
    image = np.zeros((2, 2), dtype=np.uint8)

    adjusted = adjust_saturation(image, saturation_scale=0.5)

    assert adjusted is image


def test_adjust_saturation_moves_rgb_toward_grayscale() -> None:
    """A zero saturation scale should replace every channel with luminance."""
    image = np.array([[[255, 0, 0]]], dtype=np.uint8)

    adjusted = adjust_saturation(image, saturation_scale=0.0)

    np.testing.assert_array_equal(adjusted, np.array([[[76, 76, 76]]], dtype=np.uint8))


def test_align_crop_and_fill_rgba_crops_to_alpha_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opaque alpha bounds should drive the crop and remove transparent surroundings."""
    image = np.zeros((5, 6, 4), dtype=np.uint8)
    image[1:4, 2:5, :3] = (10, 20, 30)
    image[1:4, 2:5, 3] = 255
    monkeypatch.setattr(geometry, "rotate_rgba_by_mask_long_edge", lambda img: (img, img[..., 3]))

    aligned = align_crop_and_fill_rgba(image)

    assert aligned.shape == (3, 3, 3)
    np.testing.assert_array_equal(aligned, np.full((3, 3, 3), (10, 20, 30), dtype=np.uint8))
