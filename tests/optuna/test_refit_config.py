"""Tests for Optuna refit metadata helpers."""

from pathlib import Path
from typing import Any, cast

import pytest

from excvish.optuna import (
    REFIT_CONFIG_USER_ATTR,
    build_refit_trial_config,
    build_results_dir,
    load_refit_trial_config,
    record_refit_trial_config,
)


class StubTrial:
    """Store Optuna-like user attrs in memory."""

    def __init__(self) -> None:
        """Initialize an empty user-attribute store."""
        self.user_attrs: dict[str, object] = {}

    def set_user_attr(self, key: str, value: object) -> None:
        """Persist one attribute key and value."""
        self.user_attrs[key] = value


def test_build_results_dir_places_study_artifacts_under_results_directory() -> None:
    """Normalize the study name and place the artifact directory below results."""
    assert build_results_dir("optuna-skipjack-vs-rest") == Path("results/optuna-skipjack-vs-rest")


def test_refit_trial_config_round_trips_through_trial_user_attrs(tmp_path: Path) -> None:
    """Serialize one refit payload, store it on a trial, and load the same values."""
    payload = build_refit_trial_config(
        search_script="scripts/02_hyp_svr.py",
        trial_params={"checkpoint": "facebook/convnext-tiny-224", "train_batch_size": 32},
        dataset_path=tmp_path / "dataset.csv",
        fda_reference_index_path=tmp_path / "refs.csv",
        label_scheme_name="skipjack-vs-rest",
        label_mapping={
            "Skipjack-tuna": "Skipjack-tuna",
            "Yellowfin-tuna": "Other-tuna",
            "Bigeye-tuna": "Other-tuna",
        },
        target_labels=["Skipjack-tuna", "Other-tuna"],
        ignore_label=(),
        seed=42,
        objective_metric="min_class_recall",
        tta_policy="none",
        tta_merge_mode="mean_logit",
        normalized_tta_merge_mode="single_view",
        requested_input_size=384,
        effective_input_size=384,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
        full_determinism=False,
        train_transform_factory_type="example.Factory",
        source_trial_output_dir=tmp_path / "results" / "trial-001",
        source_best_checkpoint_dir=tmp_path / "results" / "trial-001" / "checkpoint-best",
        best_epoch=7,
    )
    trial = StubTrial()

    record_refit_trial_config(cast(Any, trial), payload)

    loaded = load_refit_trial_config(cast(Any, trial))

    assert loaded == payload
    assert trial.user_attrs[REFIT_CONFIG_USER_ATTR] == payload


def test_load_refit_trial_config_returns_none_without_payload() -> None:
    """Return None when older studies do not contain refit metadata."""
    trial = StubTrial()

    loaded = load_refit_trial_config(cast(Any, trial))

    assert loaded is None


def test_load_refit_trial_config_rejects_unknown_schema_versions() -> None:
    """Fail fast when the saved payload uses an unsupported schema version."""
    trial = StubTrial()
    trial.user_attrs[REFIT_CONFIG_USER_ATTR] = {"schema_version": 999}

    with pytest.raises(ValueError, match="Unsupported trial refit config schema version"):
        load_refit_trial_config(cast(Any, trial))
