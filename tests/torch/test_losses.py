"""Tests for PyTorch classification losses."""

import torch

from excvish.torch.losses import FocalLoss


def test_focal_loss_returns_finite_scalar() -> None:
    """Compute a finite scalar focal loss for integer labels."""
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss_fn = FocalLoss(alpha=torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32), gamma=2.0)

    loss = loss_fn(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
