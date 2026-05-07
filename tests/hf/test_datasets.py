"""Tests for Hugging Face Datasets helpers."""

from typing import cast

from datasets import Dataset, DatasetDict

from excvish.hf.datasets import (
    filter_dataset_by_labels,
    split_dataset_by_split_column,
)


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
