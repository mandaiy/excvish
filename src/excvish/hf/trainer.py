"""Hugging Face Trainer extensions for imbalanced vision classification."""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from excvish.hf.vision import resolve_model_forward_kwargs
from excvish.torch.augment import BatchMixConfig, BatchMixer
from excvish.torch.imbalance import build_class_weights, build_weighted_sampler
from excvish.torch.losses import FocalLoss
from excvish.torch.tta import TTA_MERGE_MODES, aggregate_tta_logits


class ImbalanceAwareTrainer(Trainer):
    """Combine weighted sampling, focal loss, batch mixing, and TTA-aware prediction."""

    def __init__(
        self,
        *args: Any,
        train_labels: list[int],
        use_weighted_sampler: bool = True,
        use_focal_loss: bool = True,
        use_focal_alpha: bool = True,
        focal_gamma: float = 2.0,
        alpha_clip: float | None = None,
        batch_mix_config: BatchMixConfig | None = None,
        batch_mix_seed: int | None = None,
        tta_merge_mode: str = "mean_prob",
        **kwargs: Any,
    ) -> None:
        """Initialize imbalance handling, optional batch mixing, and class-dependent settings."""
        super().__init__(*args, **kwargs)
        self._train_labels = train_labels
        num_classes = max(train_labels) + 1
        self._class_weights = build_class_weights(train_labels, num_classes)

        self._use_weighted_sampler = use_weighted_sampler
        self._use_focal_loss = use_focal_loss
        self._use_focal_alpha = use_focal_alpha
        self._focal_gamma = focal_gamma
        self._alpha_clip = alpha_clip
        if tta_merge_mode not in TTA_MERGE_MODES:
            raise ValueError(f"Unknown TTA merge mode: {tta_merge_mode}")
        self._tta_merge_mode = tta_merge_mode

        self._loss_fn: FocalLoss | None = None
        self._batch_mixer: BatchMixer | None = None
        if batch_mix_config is not None and batch_mix_config.mode != "none":
            effective_seed = (
                batch_mix_seed
                if batch_mix_seed is not None
                else (self.args.data_seed if self.args.data_seed is not None else self.args.seed)
            )
            self._batch_mixer = BatchMixer(
                config=batch_mix_config,
                num_classes=num_classes,
                seed=effective_seed,
            )

    def add_callbacks(self, callbacks: list[TrainerCallback]) -> None:
        """Register callbacks while wiring trainer references when supported."""
        for callback in callbacks:
            if hasattr(callback, "set_trainer"):
                cast(Any, callback).set_trainer(self)

            self.add_callback(callback)

    def get_train_dataloader(self) -> DataLoader:
        """Build a DataLoader, optionally attaching a weighted sampler."""
        if self.train_dataset is None:
            raise ValueError("Trainer: train_dataset is required for training.")
        sampler = None
        if self._use_weighted_sampler:
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            sampler = build_weighted_sampler(self._train_labels, seed=seed)
        dataloader_kwargs: dict[str, Any] = {}
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(
            cast(torch.utils.data.Dataset, self.train_dataset),
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            **dataloader_kwargs,
        )

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute loss with optional batch mixing while handling both hard and soft labels."""
        labels = cast(torch.Tensor, inputs["labels"])
        pixel_values = cast(torch.Tensor, inputs["pixel_values"])
        if self._batch_mixer is not None and model.training:
            pixel_values, labels = self._batch_mixer(pixel_values, labels)
        if model.training:
            self._notify_loss_batch(pixel_values=pixel_values, labels=labels)

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        model_inputs["pixel_values"] = pixel_values
        model_inputs.update(resolve_model_forward_kwargs(model, pixel_values))
        outputs = model(**model_inputs)
        label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
        if labels.ndim == 2:
            loss = self._compute_soft_target_loss(
                outputs.logits, labels, label_smoothing=label_smoothing
            )
            return (loss, outputs) if return_outputs else loss

        if self._loss_fn is None and self._use_focal_loss:
            alpha = None
            if self._use_focal_alpha:
                alpha = self._class_weights.to(outputs.logits.device)
                if self._alpha_clip is not None:
                    alpha = torch.clamp(alpha, max=self._alpha_clip)
            self._loss_fn = FocalLoss(
                alpha=alpha,
                gamma=self._focal_gamma,
                label_smoothing=label_smoothing,
            )

        if self._use_focal_loss:
            loss = cast(FocalLoss, self._loss_fn)(outputs.logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                reduction="mean",
                label_smoothing=label_smoothing,
            )
        return (loss, outputs) if return_outputs else loss

    def _compute_soft_target_loss(
        self,
        logits: torch.Tensor,
        soft_labels: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Apply soft-target cross entropy with optional post-mix label smoothing."""
        if label_smoothing > 0.0:
            num_classes = soft_labels.size(-1)
            smoothed = label_smoothing / num_classes
            soft_labels = soft_labels * (1.0 - label_smoothing) + smoothed
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_labels * log_probs).sum(dim=-1).mean()

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Flatten multi-view eval batches, merge logits, and preserve single-view behavior."""
        pixel_values = inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
            return cast(
                tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
                super().prediction_step(
                    model,
                    inputs,
                    prediction_loss_only=prediction_loss_only,
                    ignore_keys=ignore_keys,
                ),
            )

        inputs = cast(dict[str, Any], self._prepare_inputs(inputs))
        pixel_values = cast(torch.Tensor, inputs["pixel_values"])
        labels = inputs.get("labels")
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")

        batch_size, tta_views, channels, height, width = pixel_values.shape
        flat_inputs = {
            key: value for key, value in inputs.items() if key not in {"pixel_values", "labels"}
        }
        flat_inputs["pixel_values"] = pixel_values.reshape(
            batch_size * tta_views, channels, height, width
        )
        flat_inputs.update(resolve_model_forward_kwargs(model, flat_inputs["pixel_values"]))

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**flat_inputs)

        aggregated_logits = self._aggregate_tta_logits(
            outputs.logits.reshape(batch_size, tta_views, -1)
        )
        loss = self._compute_eval_loss_from_scores(aggregated_logits, labels)
        if prediction_loss_only:
            return loss, None, None

        detached_labels = labels.detach() if labels is not None else None
        return loss, aggregated_logits.detach(), detached_labels

    def _aggregate_tta_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Delegate TTA logit reduction to the shared tensor utility."""
        return aggregate_tta_logits(logits, tta_merge_mode=self._tta_merge_mode)

    def _compute_eval_loss_from_scores(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute an evaluation loss from aggregated scores."""
        if labels is None:
            return None
        if self._tta_merge_mode == "mean_prob":
            return F.nll_loss(scores, labels).detach()
        return F.cross_entropy(scores, labels).detach()

    def _notify_loss_batch(self, pixel_values: torch.Tensor, labels: torch.Tensor) -> None:
        """Forward the effective loss batch to callbacks that expose an on_loss_batch hook."""
        callback_handler = getattr(self, "callback_handler", None)
        control = getattr(self, "control", None)
        if callback_handler is None or control is None:
            return
        for callback in callback_handler.callbacks:
            hook = getattr(callback, "on_loss_batch", None)
            if hook is None:
                continue
            hook(
                args=self.args,
                state=self.state,
                control=control,
                pixel_values=pixel_values.detach(),
                labels=labels.detach(),
            )
