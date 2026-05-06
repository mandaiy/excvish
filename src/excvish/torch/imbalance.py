"""Class-imbalance utilities for PyTorch training."""

from typing import cast

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels: list[int], seed: int | None = None) -> WeightedRandomSampler:
    """Build a replacement sampler from inverse-frequency sample weights."""
    counts = np.maximum(np.bincount(np.array(labels)), 1)
    class_weights = 1.0 / counts
    sample_weights = class_weights[np.array(labels)]
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=cast(list[float], torch.tensor(sample_weights, dtype=torch.double).tolist()),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def build_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights for a fixed class count."""
    counts = np.bincount(np.array(labels), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights * (counts.sum() / len(counts))
    return torch.tensor(weights, dtype=torch.float)
