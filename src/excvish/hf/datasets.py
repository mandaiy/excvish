"""Hugging Face Datasets helpers for image classification workflows."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class LabelScheme:
    """Describe a stable target label space and how source labels are projected into it."""

    name: str
    label_mapping: dict[str, str]
    target_labels: list[str]

    def __post_init__(self) -> None:
        """Validate the mapping once so callers can rely on consistent scheme metadata."""
        if not self.name:
            raise ValueError("Label scheme name must not be empty.")
        if not self.target_labels:
            raise ValueError("Label scheme must define at least one target label.")
        if len(set(self.target_labels)) != len(self.target_labels):
            raise ValueError("Label scheme target labels must be unique.")
        unknown_targets = sorted(set(self.label_mapping.values()) - set(self.target_labels))
        if unknown_targets:
            unknown_display = ", ".join(unknown_targets)
            raise ValueError(
                f"Label scheme '{self.name}' maps to unknown target labels: {unknown_display}"
            )

    def remap_label(self, source_label: str) -> str | None:
        """Return the mapped target label or None when the row should be dropped."""
        return self.label_mapping.get(source_label)


def build_identity_label_scheme(target_labels: Sequence[str], name: str) -> LabelScheme:
    """Create a scheme that keeps a selected label subset while preserving the requested order."""
    labels = [str(label) for label in target_labels]
    return LabelScheme(
        name=name,
        label_mapping={label: label for label in labels},
        target_labels=labels,
    )


def ensure_label_scheme_targets(label_scheme: LabelScheme, target_labels: Sequence[str]) -> None:
    """Check that expected target labels exactly match the scheme order used by a model."""
    expected = list(label_scheme.target_labels)
    actual = [str(label) for label in target_labels]
    if expected != actual:
        raise ValueError(
            f"Label scheme '{label_scheme.name}' expects target labels "
            f"{expected}, but received {actual}."
        )


def remap_dataset_labels(
    ds: Dataset | DatasetDict, label_scheme: LabelScheme
) -> Dataset | DatasetDict:
    """Rewrite labels, drop unmapped rows, and attach a deterministic ClassLabel schema."""

    def _remap(dataset: Dataset) -> Dataset:
        remapped = dataset.map(
            lambda record: {"label": label_scheme.remap_label(str(record["label"]))}
        )
        filtered = remapped.filter(lambda record: record["label"] is not None)
        return filtered.cast_column("label", ClassLabel(names=label_scheme.target_labels))

    if isinstance(ds, DatasetDict):
        return DatasetDict({split: _remap(dataset) for split, dataset in ds.items()})
    if isinstance(ds, Dataset):
        return _remap(ds)
    raise TypeError("ds must be a Dataset or DatasetDict.")


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
