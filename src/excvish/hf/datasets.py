"""Hugging Face Datasets helpers for image classification workflows."""

from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_train_val_test_split(
    y: np.ndarray,
    groups: np.ndarray,
    train_ratio: int,
    val_ratio: int,
    test_ratio: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test with stratified group k-fold in two stages."""
    total = train_ratio + val_ratio + test_ratio
    indices = np.arange(len(y))

    test_splits = max(2, round(total / test_ratio))
    test_splitter = StratifiedGroupKFold(
        n_splits=test_splits,
        shuffle=True,
        random_state=random_state,
    )
    trainval_idx, test_idx = next(test_splitter.split(indices, y, groups=groups))

    trainval_total = train_ratio + val_ratio
    val_splits = max(2, round(trainval_total / val_ratio))
    val_splitter = StratifiedGroupKFold(
        n_splits=val_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_local_idx, val_local_idx = next(
        val_splitter.split(trainval_idx, y[trainval_idx], groups=groups[trainval_idx])
    )

    train_idx = trainval_idx[train_local_idx]
    val_idx = trainval_idx[val_local_idx]
    return train_idx, val_idx, test_idx


def stratified_group_split(
    ds: Dataset,
    group_column: str,
    label_column: str,
    train_ratio: int = 70,
    val_ratio: int = 15,
    test_ratio: int = 15,
) -> DatasetDict:
    """Create a stratified group split from a dataset using group and label columns."""
    groups = np.array(
        [str(value) if value is not None else "unknown" for value in ds[group_column]]
    )
    labels = np.array(ds[label_column])

    train_idx, val_idx, test_idx = stratified_group_train_val_test_split(
        labels,
        groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return DatasetDict(
        {
            "train": ds.select(train_idx.tolist()),
            "val": ds.select(val_idx.tolist()),
            "test": ds.select(test_idx.tolist()),
        }
    )


def load_dataset_with_split(data_dir: str | Path) -> DatasetDict:
    """Load an imagefolder dataset and return a stratified train/val/test split."""
    ds = load_dataset("imagefolder", data_dir=str(data_dir))["train"]
    assert isinstance(ds, Dataset)

    split = ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    valtest_split = split["test"].train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="label",
    )
    return DatasetDict(
        {
            "train": split["train"],
            "val": valtest_split["train"],
            "test": valtest_split["test"],
        }
    )


def split_dataset_by_split_column(ds: Dataset) -> DatasetDict:
    """Partition a dataset into train/val/test splits based on its split column values."""
    if "split" not in ds.column_names:
        raise ValueError("split column is required.")

    valid_splits = {"train", "val", "test"}
    split_values = set(ds["split"])
    invalid_splits = split_values - valid_splits
    if invalid_splits:
        invalid_display = ", ".join(sorted(map(str, invalid_splits)))
        raise ValueError(f"Invalid split values: {invalid_display}")

    split_indices: dict[str, list[int]] = {split: [] for split in ["train", "val", "test"]}
    for idx, split in enumerate(ds["split"]):
        split_indices[split].append(idx)

    return DatasetDict(
        {split: ds.select(indices) for split, indices in split_indices.items() if indices}
    )


def filter_dataset_by_labels(
    ds: Dataset | DatasetDict, labels: list[str]
) -> Dataset | DatasetDict:
    """Filter datasets to keep only records whose label is in the allowlist."""
    allowed_labels = set(labels)

    def _filter(dataset: Dataset) -> Dataset:
        if "label" not in dataset.column_names:
            raise ValueError("label column is required.")
        return dataset.filter(lambda record: record["label"] in allowed_labels)

    if isinstance(ds, DatasetDict):
        return DatasetDict({split: _filter(dataset) for split, dataset in ds.items()})
    if isinstance(ds, Dataset):
        return _filter(ds)
    raise TypeError("ds must be a Dataset or DatasetDict.")
