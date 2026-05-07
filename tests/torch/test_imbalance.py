"""Tests for PyTorch imbalance helpers."""

import torch

from excvish.torch.imbalance import build_class_weights, build_weighted_sampler


def test_build_class_weights_uses_normalized_inverse_frequency() -> None:
    """Normalize inverse-frequency weights by total sample count over class count."""
    weights = build_class_weights([0, 0, 1, 2, 2, 2], num_classes=3)

    expected = torch.tensor([1.0, 2.0, 2.0 / 3.0], dtype=torch.float32)
    torch.testing.assert_close(weights, expected)


def test_build_weighted_sampler_contains_one_weight_per_label() -> None:
    """Create a replacement sampler whose sampled population length follows the input labels."""
    sampler = build_weighted_sampler([0, 0, 1, 2, 2, 2], seed=13)

    assert sampler.num_samples == 6
    assert sampler.replacement is True
