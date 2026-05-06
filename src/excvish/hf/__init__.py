"""Hugging Face integration utilities for excvish."""

from .datasets import (
    LabelScheme,
    build_identity_label_scheme,
    ensure_label_scheme_targets,
    filter_dataset_by_labels,
    load_dataset_with_split,
    remap_dataset_labels,
    split_dataset_by_split_column,
    stratified_group_split,
    stratified_group_train_val_test_split,
)
from .optuna import TrainerPruningCallback
from .trainer import ImbalanceAwareTrainer
from .vision import (
    requires_model_positional_interpolation,
    resolve_effective_input_size,
    resolve_model_forward_kwargs,
    resolve_processor_input_size,
)

__all__ = [
    "LabelScheme",
    "ImbalanceAwareTrainer",
    "TrainerPruningCallback",
    "build_identity_label_scheme",
    "ensure_label_scheme_targets",
    "filter_dataset_by_labels",
    "load_dataset_with_split",
    "remap_dataset_labels",
    "requires_model_positional_interpolation",
    "resolve_effective_input_size",
    "resolve_model_forward_kwargs",
    "resolve_processor_input_size",
    "split_dataset_by_split_column",
    "stratified_group_split",
    "stratified_group_train_val_test_split",
]
