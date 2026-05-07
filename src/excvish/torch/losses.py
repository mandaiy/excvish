"""Loss functions for PyTorch classification models."""

import torch


class FocalLoss(torch.nn.Module):
    """Apply focal loss with optional per-class alpha weighting."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        """Store weighting, focusing, and smoothing parameters for later loss computation."""
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss from logits and integer class labels."""
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()
