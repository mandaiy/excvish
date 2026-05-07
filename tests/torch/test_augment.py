"""Tests for PyTorch batch-level augmentations."""

import torch

from excvish.torch.augment import BatchMixConfig, BatchMixer


def _build_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deterministic toy batch for batch-mixing tests."""
    pixel_values = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    return pixel_values, labels


def test_batch_mixer_returns_original_batch_when_disabled() -> None:
    """Return unchanged tensors when the mode is none."""
    pixel_values, labels = _build_batch()
    mixer = BatchMixer(BatchMixConfig(mode="none"), num_classes=3, seed=7)

    mixed_images, mixed_labels = mixer(pixel_values, labels)

    assert torch.equal(mixed_images, pixel_values)
    assert torch.equal(mixed_labels, labels)


def test_batch_mixer_mixup_outputs_soft_labels() -> None:
    """Convert hard labels to probability targets when mixup is enabled."""
    pixel_values, labels = _build_batch()
    mixer = BatchMixer(
        BatchMixConfig(mode="mixup", mix_prob=1.0, mixup_alpha=0.4),
        num_classes=3,
        seed=7,
    )

    mixed_images, mixed_labels = mixer(pixel_values, labels)

    assert mixed_images.shape == pixel_values.shape
    assert mixed_labels.shape == (pixel_values.size(0), 3)
    torch.testing.assert_close(mixed_labels.sum(dim=1), torch.ones(pixel_values.size(0)))


def test_batch_mixer_cutmix_outputs_soft_labels() -> None:
    """Produce cutmix labels that keep per-row probability mass normalized."""
    pixel_values, labels = _build_batch()
    mixer = BatchMixer(
        BatchMixConfig(mode="cutmix", mix_prob=1.0, cutmix_alpha=0.8),
        num_classes=3,
        seed=11,
    )

    mixed_images, mixed_labels = mixer(pixel_values, labels)

    assert mixed_images.shape == pixel_values.shape
    assert mixed_labels.shape == (pixel_values.size(0), 3)
    torch.testing.assert_close(mixed_labels.sum(dim=1), torch.ones(pixel_values.size(0)))
