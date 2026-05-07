"""Tests for PyTorch test-time augmentation helpers."""

import math

import pytest
import torch

from excvish.torch.tta import aggregate_tta_logits


def test_aggregate_tta_logits_supports_mean_prob_mode() -> None:
    """Average normalized per-view probabilities in log space."""
    logits = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [3.0, 1.0, -2.0]],
            [[0.5, 1.5, -0.5], [1.5, 0.5, -1.5]],
        ],
        dtype=torch.float32,
    )

    aggregated = aggregate_tta_logits(logits, tta_merge_mode="mean_prob")

    expected = torch.logsumexp(torch.log_softmax(logits, dim=-1), dim=1) - math.log(logits.size(1))
    torch.testing.assert_close(aggregated, expected)


def test_aggregate_tta_logits_supports_mean_logit_mode() -> None:
    """Average raw logits across views when the mean-logit merge mode is configured."""
    logits = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [3.0, 1.0, -2.0]],
            [[0.5, 1.5, -0.5], [1.5, 0.5, -1.5]],
        ],
        dtype=torch.float32,
    )

    aggregated = aggregate_tta_logits(logits, tta_merge_mode="mean_logit")

    expected = torch.tensor(
        [
            [2.0, 0.5, -1.5],
            [1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(aggregated, expected)


def test_aggregate_tta_logits_rejects_unknown_mode() -> None:
    """Raise a clear error for unsupported merge policies."""
    logits = torch.zeros((2, 3, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="Unknown TTA merge mode"):
        aggregate_tta_logits(logits, tta_merge_mode="median")
