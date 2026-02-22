"""Tests for rotation-related geometry utilities."""

import cv2
import numpy as np
import pytest

from excvish.cv import geometry


def test_compute_horizontal_rotation_angle_for_vertical_rect() -> None:
    """Vertical long edge should add 90 degrees."""
    rect = ((10.0, 20.0), (30.0, 80.0), -15.0)

    angle = geometry.compute_horizontal_rotation_angle(rect)

    assert angle == 75.0


def test_compute_horizontal_rotation_angle_for_horizontal_rect() -> None:
    """Horizontal long edge should keep original angle."""
    rect = ((10.0, 20.0), (80.0, 30.0), -15.0)

    angle = geometry.compute_horizontal_rotation_angle(rect)

    assert angle == -15.0


@pytest.mark.parametrize(
    ("height", "width", "angle", "expected"),
    [
        (4, 7, 0.0, (4, 7)),
        (4, 7, 90.0, (7, 4)),
    ],
)
def test_compute_rotated_image_size(
    height: int,
    width: int,
    angle: float,
    expected: tuple[int, int],
) -> None:
    """Output size should follow geometric bounds of rotation."""
    assert geometry.compute_rotated_image_size(height, width, angle) == expected


def test_rotate_image_without_expand_keeps_shape() -> None:
    """When expand=False output shape should match input shape."""
    image = np.zeros((6, 10), dtype=np.uint8)

    rotated = geometry.rotate_image(image, angle=90.0, expand=False)

    assert rotated.shape == image.shape


def test_rotate_image_with_expand_uses_rotated_size() -> None:
    """When expand=True output shape should be computed from angle."""
    image = np.zeros((6, 10), dtype=np.uint8)
    expected = geometry.compute_rotated_image_size(6, 10, 90.0)

    rotated = geometry.rotate_image(image, angle=90.0, expand=True)

    assert rotated.shape == expected


def test_rotate_image_constant_border_value() -> None:
    """Constant border value should appear in outer pixels."""
    image = np.full((5, 5), 255, dtype=np.uint8)

    rotated = geometry.rotate_image(
        image,
        angle=45.0,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=7,
        expand=True,
    )

    assert rotated[0, 0] == 7


def test_rotate_image_with_nearest_keeps_binary_mask_values() -> None:
    """Nearest interpolation should keep binary masks as binary values."""
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 255

    rotated = geometry.rotate_image(mask, angle=33.0, interpolation=cv2.INTER_NEAREST, expand=True)

    assert set(np.unique(rotated)).issubset({0, 255})


def test_rotate_rgba_by_mask_long_edge_returns_original_when_no_foreground() -> None:
    """Empty alpha mask should return original image and zero mask."""
    image = np.zeros((4, 4, 4), dtype=np.uint8)

    rotated_image, rotated_mask = geometry.rotate_rgba_by_mask_long_edge(image)

    assert np.array_equal(rotated_image, image)
    assert np.array_equal(rotated_mask, np.zeros((4, 4), dtype=np.uint8))


def test_rotate_rgba_by_mask_long_edge_skips_rotate_when_angle_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Angle zero path should not call rotate_image."""
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1:3, 1:3, 3] = 255

    monkeypatch.setattr(
        geometry, "compute_min_area_rect", lambda _mask: ((0.0, 0.0), (2.0, 2.0), 0.0)
    )
    monkeypatch.setattr(geometry, "compute_horizontal_rotation_angle", lambda _rect: 0.0)

    def fail_rotate(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("rotate_image should not be called when angle is zero")

    monkeypatch.setattr(geometry, "rotate_image", fail_rotate)

    rotated_image, rotated_mask = geometry.rotate_rgba_by_mask_long_edge(image)

    assert np.array_equal(rotated_image, image)
    assert np.array_equal(rotated_mask, image[..., 3])


def test_rotate_rgba_by_mask_long_edge_rotates_image_and_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero angle path should rotate both image and mask."""
    image = np.zeros((2, 3, 4), dtype=np.uint8)
    image[..., 3] = 255

    rotated_rgba = np.ones((3, 2, 4), dtype=np.uint8)
    rotated_mask = np.full((3, 2), 255, dtype=np.uint8)

    monkeypatch.setattr(
        geometry, "compute_min_area_rect", lambda _mask: ((0.0, 0.0), (1.0, 2.0), 0.0)
    )
    monkeypatch.setattr(geometry, "compute_horizontal_rotation_angle", lambda _rect: 90.0)

    calls: list[tuple[float, int | tuple[int, ...], bool, int]] = []

    def fake_rotate_image(
        array: np.ndarray,
        angle: float,
        border_mode: int,
        border_value: int | tuple[int, ...],
        expand: bool,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        calls.append((angle, border_value, expand, interpolation))
        if array.ndim == 3:
            return rotated_rgba
        return rotated_mask

    monkeypatch.setattr(geometry, "rotate_image", fake_rotate_image)

    actual_image, actual_mask = geometry.rotate_rgba_by_mask_long_edge(image)

    assert np.array_equal(actual_image, rotated_rgba)
    assert np.array_equal(actual_mask, rotated_mask)
    assert calls == [
        (90.0, (0, 0, 0, 0), True, cv2.INTER_LINEAR),
        (90.0, 0, True, cv2.INTER_NEAREST),
    ]
