"""Tests for Albumentations custom transforms."""

import albumentations as A
import numpy as np
import pytest

from excvish.albumentations.transforms import AlignLongEdgeHorizontal, ShadesOfGray


def test_align_long_edge_horizontal_raises_for_bboxes_input() -> None:
    """Bboxes are explicitly unsupported for this transform."""
    transform = AlignLongEdgeHorizontal()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(NotImplementedError, match="does not support bboxes"):
        transform.get_params_dependent_on_data({}, {"image": image, "mask": mask, "bboxes": []})


def test_align_long_edge_horizontal_raises_for_keypoints_input() -> None:
    """Keypoints are explicitly unsupported for this transform."""
    transform = AlignLongEdgeHorizontal()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(NotImplementedError, match="does not support keypoints"):
        transform.get_params_dependent_on_data({}, {"image": image, "mask": mask, "keypoints": []})


def test_shades_of_gray_reproducible_with_compose_seed() -> None:
    """Compose seed makes custom transform sampling reproducible."""
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    transform = A.Compose([ShadesOfGray(p_norm_range=(4.0, 8.0), p=1.0)], seed=137)

    out1 = transform(image=image)["image"]
    out2 = transform(image=image)["image"]

    transform_replayed = A.Compose([ShadesOfGray(p_norm_range=(4.0, 8.0), p=1.0)], seed=137)
    out1_replayed = transform_replayed(image=image)["image"]
    out2_replayed = transform_replayed(image=image)["image"]

    np.testing.assert_array_equal(out1, out1_replayed)
    np.testing.assert_array_equal(out2, out2_replayed)
