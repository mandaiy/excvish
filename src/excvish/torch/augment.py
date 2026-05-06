"""Batch-level augmentation utilities for PyTorch image tensors."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

BatchMixMode = Literal["none", "mixup", "cutmix", "mixup_cutmix"]


@dataclass(frozen=True)
class BatchMixConfig:
    """Store batch-level mixing settings used during train-time regularization."""

    mode: BatchMixMode = "none"
    mix_prob: float = 1.0
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_switch_prob: float = 0.5

    def __post_init__(self) -> None:
        """Validate ranges so sampled configuration values fail fast when invalid."""
        if self.mode not in {"none", "mixup", "cutmix", "mixup_cutmix"}:
            raise ValueError(f"Unknown batch mix mode: {self.mode}")
        if not 0.0 <= self.mix_prob <= 1.0:
            raise ValueError("mix_prob must be in [0.0, 1.0].")
        if self.mixup_alpha <= 0.0:
            raise ValueError("mixup_alpha must be > 0.")
        if self.cutmix_alpha <= 0.0:
            raise ValueError("cutmix_alpha must be > 0.")
        if not 0.0 <= self.cutmix_switch_prob <= 1.0:
            raise ValueError("cutmix_switch_prob must be in [0.0, 1.0].")


class BatchMixer:
    """Apply Mixup/CutMix to a training batch and return mixed images and soft labels."""

    def __init__(
        self,
        config: BatchMixConfig,
        num_classes: int,
        seed: int | None = None,
    ) -> None:
        """Store configuration and initialize a deterministic RNG for batch mixing."""
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for batch mixing.")
        self.config = config
        self.num_classes = num_classes
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply configured batch mixing and return transformed tensors."""
        if self.config.mode == "none":
            return pixel_values, labels
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape (batch, channel, height, width).")
        if pixel_values.size(0) < 2:
            return pixel_values, labels
        if float(self.rng.random()) > self.config.mix_prob:
            return pixel_values, labels

        labels_one_hot = self._to_soft_labels(
            labels,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        mode = self._select_mode()
        if mode == "mixup":
            return self._apply_mixup(pixel_values, labels_one_hot)
        if mode == "cutmix":
            return self._apply_cutmix(pixel_values, labels_one_hot)
        raise ValueError(f"Unknown resolved batch mix mode: {mode}")

    def _select_mode(self) -> Literal["mixup", "cutmix"]:
        """Resolve mixed mode to either mixup or cutmix for the current batch."""
        if self.config.mode == "mixup_cutmix":
            is_cutmix = float(self.rng.random()) < self.config.cutmix_switch_prob
            return "cutmix" if is_cutmix else "mixup"
        return "mixup" if self.config.mode == "mixup" else "cutmix"

    def _apply_mixup(
        self,
        pixel_values: torch.Tensor,
        labels_one_hot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend each sample with a shuffled partner using a beta-sampled ratio."""
        lam = self._sample_beta(self.config.mixup_alpha)
        indices = self._sample_permutation(pixel_values.size(0), device=pixel_values.device)
        mixed_images = lam * pixel_values + (1.0 - lam) * pixel_values[indices]
        mixed_labels = lam * labels_one_hot + (1.0 - lam) * labels_one_hot[indices]
        return mixed_images, mixed_labels

    def _apply_cutmix(
        self,
        pixel_values: torch.Tensor,
        labels_one_hot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace a random patch with another sample and adjust soft labels by area ratio."""
        lam = self._sample_beta(self.config.cutmix_alpha)
        indices = self._sample_permutation(pixel_values.size(0), device=pixel_values.device)
        x1, y1, x2, y2 = self._sample_cutmix_box(
            height=pixel_values.size(2),
            width=pixel_values.size(3),
            lam=lam,
        )

        mixed_images = pixel_values.clone()
        mixed_images[:, :, y1:y2, x1:x2] = pixel_values[indices, :, y1:y2, x1:x2]

        area = float((x2 - x1) * (y2 - y1))
        total = float(pixel_values.size(2) * pixel_values.size(3))
        lam_adjusted = 1.0 - (area / total)
        mixed_labels = (
            lam_adjusted * labels_one_hot + (1.0 - lam_adjusted) * labels_one_hot[indices]
        )
        return mixed_images, mixed_labels

    def _to_soft_labels(
        self,
        labels: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert integer labels to one-hot probabilities and preserve soft labels."""
        if labels.ndim == 2:
            return labels.to(device=device, dtype=dtype)
        return F.one_hot(labels.to(dtype=torch.long), num_classes=self.num_classes).to(
            device=device,
            dtype=dtype,
        )

    def _sample_permutation(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample a shuffled index tensor for pairing examples within a batch."""
        return torch.as_tensor(self.rng.permutation(batch_size), device=device, dtype=torch.long)

    def _sample_beta(self, alpha: float) -> float:
        """Sample beta(alpha, alpha) by normalizing two gamma draws."""
        x = float(self.rng.gamma(alpha, 1.0))
        y = float(self.rng.gamma(alpha, 1.0))
        return x / (x + y)

    def _sample_cutmix_box(
        self,
        height: int,
        width: int,
        lam: float,
    ) -> tuple[int, int, int, int]:
        """Sample a rectangular region sized according to lambda and clipped to image bounds."""
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)
        center_x = int(self.rng.integers(0, width))
        center_y = int(self.rng.integers(0, height))

        x1 = max(center_x - cut_w // 2, 0)
        x2 = min(center_x + cut_w // 2, width)
        y1 = max(center_y - cut_h // 2, 0)
        y2 = min(center_y + cut_h // 2, height)
        return x1, y1, x2, y2
