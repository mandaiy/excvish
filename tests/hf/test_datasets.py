"""Tests for Hugging Face Datasets helpers."""

from typing import cast

import pytest
from datasets import Dataset, DatasetDict

from excvish.hf.datasets import (
    LabelScheme,
    ensure_label_scheme_targets,
    filter_dataset_by_labels,
    remap_dataset_labels,
    split_dataset_by_split_column,
)


def test_label_scheme_rejects_unknown_target_labels() -> None:
    """Reject mappings that point to labels outside the declared target label space."""
    with pytest.raises(ValueError, match="unknown target labels"):
        LabelScheme(
            name="invalid",
            label_mapping={"source": "other"},
            target_labels=["target"],
        )


def test_remap_dataset_labels_merges_classes_and_preserves_target_order() -> None:
    """Map source labels into merged targets while keeping the declared ClassLabel ordering."""
    dataset = Dataset.from_dict({"label": ["a", "b", "c", "d"]})
    scheme = LabelScheme(
        name="binary",
        label_mapping={"a": "x", "b": "y", "c": "y"},
        target_labels=["x", "y"],
    )

    remapped = cast(Dataset, remap_dataset_labels(dataset, label_scheme=scheme))

    assert remapped.features["label"].names == ["x", "y"]
    assert remapped["label"] == [0, 1, 1]


def test_ensure_label_scheme_targets_requires_exact_order() -> None:
    """Fail when a checkpoint label order differs from the scheme order even if labels match."""
    scheme = LabelScheme(
        name="ordered",
        label_mapping={"a": "a", "b": "b"},
        target_labels=["a", "b"],
    )

    with pytest.raises(ValueError, match="expects target labels"):
        ensure_label_scheme_targets(scheme, ["b", "a"])


def test_split_dataset_by_split_column_partitions_valid_splits() -> None:
    """Partition a dataset into train/val/test subsets using a split column."""
    dataset = Dataset.from_dict(
        {
            "label": ["a", "b", "c", "d"],
            "split": ["train", "val", "test", "train"],
        }
    )

    split = split_dataset_by_split_column(dataset)

    assert list(split) == ["train", "val", "test"]
    assert split["train"]["label"] == ["a", "d"]
    assert split["val"]["label"] == ["b"]
    assert split["test"]["label"] == ["c"]


def test_filter_dataset_by_labels_supports_dataset_dict() -> None:
    """Apply label allowlist filtering to every split in a DatasetDict."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": ["a", "b"]}),
            "val": Dataset.from_dict({"label": ["b", "c"]}),
        }
    )

    filtered = cast(DatasetDict, filter_dataset_by_labels(dataset, ["b"]))

    assert filtered["train"]["label"] == ["b"]
    assert filtered["val"]["label"] == ["b"]
