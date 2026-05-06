"""Test-time augmentation utilities for PyTorch tensors."""

import math

import torch
import torch.nn.functional as F

TTA_MERGE_MODES: tuple[str, ...] = ("mean_prob", "mean_logit")


def aggregate_tta_logits(logits: torch.Tensor, tta_merge_mode: str) -> torch.Tensor:
    """Reduce per-view logits by averaging logits or normalized probabilities."""
    if tta_merge_mode == "mean_logit":
        return logits.mean(dim=1)
    if tta_merge_mode == "mean_prob":
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.logsumexp(log_probs, dim=1) - math.log(logits.size(1))
    raise ValueError(f"Unknown TTA merge mode: {tta_merge_mode}")
