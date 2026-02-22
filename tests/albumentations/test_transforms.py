"""Tests for Albumentations custom transforms."""

import numpy as np
import pytest

from excvish.albumentations.transforms import AlignLongEdgeHorizontal


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
