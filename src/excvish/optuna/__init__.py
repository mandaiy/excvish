"""Optuna storage, artifact, metric, and study-policy helpers."""

import hashlib
import json
import shutil
import sqlite3
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import optuna

DEFAULT_SQLITE_TIMEOUT_SECONDS = 30.0
OBJECTIVE_METRIC_CHOICES = ("macro_f1", "min_class_recall", "min_class_f1")
OBJECTIVE_METRIC_USER_ATTR = "objective_metric"
EVAL_POLICY_USER_ATTR = "eval_policy"
REFIT_CONFIG_USER_ATTR = "refit_config_v1"
TTA_MERGE_MODE_CHOICES = ("mean_prob", "mean_logit")


def get_git_short_sha() -> str:
    """Return the current repository short SHA and fall back to unknown outside git worktrees."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_hyperparameter_tag(params: Mapping[str, Any]) -> str:
    """Serialize the parameter mapping, hash the payload, and return a directory-safe tag."""
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).hexdigest()
    return f"hp-{digest}"


def build_run_id(trial_number: int) -> str:
    """Combine trial number, current timestamp, and git SHA into a sortable run identifier."""
    return f"t{trial_number:03d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{get_git_short_sha()}"


def build_results_dir(study_name: str) -> Path:
    """Normalize the study name into a filesystem-safe directory below results."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in study_name.strip())
    return Path("results") / (safe or "study")


def build_storage_path(study_name: str) -> str:
    """Resolve the study sqlite filename and format it as an Optuna sqlite URL."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in study_name.strip())
    filename = f"optuna-{safe or 'study'}.db"
    return f"sqlite:///{build_results_dir(study_name) / filename}"


def build_sqlite_file_path(study_name: str) -> Path:
    """Resolve the sqlite file location for one study and return its absolute path."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in study_name.strip())
    filename = f"optuna-{safe or 'study'}.db"
    return (build_results_dir(study_name) / filename).resolve()


def build_sqlite_storage_path(sqlite_path: Path) -> str:
    """Expand the sqlite path and format it as a sqlite URL accepted by Optuna."""
    return f"sqlite:///{sqlite_path.expanduser().resolve()}"


def configure_sqlite_wal(sqlite_path: Path, timeout_seconds: float) -> None:
    """Create the parent directory, enable WAL mode, and reduce durability overhead."""
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path, timeout=timeout_seconds) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.commit()


def build_storage(
    study_name: str,
    timeout_seconds: float = DEFAULT_SQLITE_TIMEOUT_SECONDS,
) -> optuna.storages.RDBStorage:
    """Prepare sqlite WAL settings and return an Optuna storage bound to that file."""
    sqlite_path = build_sqlite_file_path(study_name)
    configure_sqlite_wal(sqlite_path=sqlite_path, timeout_seconds=timeout_seconds)
    return optuna.storages.RDBStorage(
        url=build_storage_path(study_name),
        engine_kwargs={"connect_args": {"timeout": timeout_seconds}},
    )


def build_storage_from_sqlite_path(
    sqlite_path: Path,
    timeout_seconds: float = DEFAULT_SQLITE_TIMEOUT_SECONDS,
) -> optuna.storages.RDBStorage:
    """Open an Optuna storage from an explicit sqlite file with the configured timeout."""
    return optuna.storages.RDBStorage(
        url=build_sqlite_storage_path(sqlite_path),
        engine_kwargs={"connect_args": {"timeout": timeout_seconds}},
    )


def create_best_checkpoint_symlink(output_dir: Path, best_checkpoint: str | None) -> None:
    """Resolve the best checkpoint path, replace stale links, and create checkpoint-best."""
    if not best_checkpoint:
        click.echo("best checkpoint is not available; skip creating checkpoint-best symlink.")
        return

    candidate = Path(best_checkpoint).expanduser()
    if candidate.is_absolute() or candidate.exists():
        best_path = candidate.resolve()
    else:
        best_path = (output_dir / candidate).resolve()

    link_path = output_dir / "checkpoint-best"
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"{link_path} already exists and is not a symlink.")

    link_path.symlink_to(best_path, target_is_directory=True)
    click.echo(f"created symlink: {link_path} -> {best_path}")


def record_trial_artifact_paths(trial: optuna.Trial, output_dir: Path) -> None:
    """Persist the trial output directory in Optuna user attributes for later cleanup."""
    trial.set_user_attr("output_dir", str(output_dir.resolve()))


def resolve_optional_path_string(path: Path | str | None) -> str | None:
    """Convert an optional filesystem path into one resolved absolute string."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def build_refit_trial_config(
    *,
    search_script: str,
    trial_params: Mapping[str, Any],
    dataset_path: Path | str,
    fda_reference_index_path: Path | str | None,
    label_scheme_name: str,
    label_mapping: Mapping[str, str],
    target_labels: list[str],
    ignore_label: tuple[str, ...],
    seed: int,
    objective_metric: str,
    tta_policy: str,
    tta_merge_mode: str,
    normalized_tta_merge_mode: str,
    requested_input_size: int | None,
    effective_input_size: int,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_persistent_workers: bool,
    full_determinism: bool,
    train_transform_factory_type: str,
    source_trial_output_dir: Path | str,
    source_best_checkpoint_dir: Path | str | None,
    best_epoch: int,
) -> dict[str, Any]:
    """Assemble versioned metadata that lets refit reconstruct one finished trial."""
    return {
        "schema_version": 1,
        "search_script": search_script,
        "trial_params": dict(trial_params),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "fda_reference_index_path": resolve_optional_path_string(fda_reference_index_path),
        "label_scheme_name": label_scheme_name,
        "label_mapping": dict(label_mapping),
        "target_labels": list(target_labels),
        "ignore_label": list(ignore_label),
        "seed": seed,
        "objective_metric": objective_metric,
        "tta_policy": tta_policy,
        "tta_merge_mode": tta_merge_mode,
        "normalized_tta_merge_mode": normalized_tta_merge_mode,
        "requested_input_size": requested_input_size,
        "effective_input_size": effective_input_size,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_prefetch_factor": dataloader_prefetch_factor,
        "dataloader_persistent_workers": dataloader_persistent_workers,
        "full_determinism": full_determinism,
        "train_transform_factory_type": train_transform_factory_type,
        "source_trial_output_dir": str(Path(source_trial_output_dir).expanduser().resolve()),
        "source_best_checkpoint_dir": resolve_optional_path_string(source_best_checkpoint_dir),
        "best_epoch": best_epoch,
    }


def record_refit_trial_config(trial: optuna.Trial, payload: Mapping[str, Any]) -> None:
    """Store one finished-trial refit payload in Optuna user attributes."""
    trial.set_user_attr(REFIT_CONFIG_USER_ATTR, dict(payload))


def load_refit_trial_config(trial: optuna.trial.FrozenTrial) -> dict[str, Any] | None:
    """Return the saved refit payload and reject incompatible or malformed schemas."""
    raw_payload = trial.user_attrs.get(REFIT_CONFIG_USER_ATTR)
    if raw_payload is None:
        return None
    if not isinstance(raw_payload, dict):
        raise ValueError(
            f"Trial refit config must be a dict, but received {type(raw_payload).__name__}."
        )
    schema_version = raw_payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported trial refit config schema version: {schema_version}")
    return dict(raw_payload)


def remove_trial_output_dir(output_dir: Path) -> None:
    """Delete the recorded trial directory and quietly skip already-removed paths."""
    if not output_dir.exists():
        return
    shutil.rmtree(output_dir)
    click.echo(f"removed non-best trial artifacts: {output_dir}")


def cleanup_non_best_trial_dirs(study: optuna.study.Study) -> None:
    """Walk finished trials, keep the best artifact directory, and remove older outputs."""
    try:
        best_trial_number = study.best_trial.number
    except ValueError:
        best_trial_number = None
    for frozen_trial in study.trials:
        if not frozen_trial.state.is_finished():
            continue
        if frozen_trial.number == best_trial_number:
            continue
        output_dir_value = frozen_trial.user_attrs.get("output_dir")
        if not output_dir_value:
            continue
        remove_trial_output_dir(Path(output_dir_value))


def build_non_best_trial_dir_cleanup_callback() -> Callable[
    [optuna.study.Study, optuna.trial.FrozenTrial], None
]:
    """Create an Optuna callback that prunes artifact directories after each update."""

    def callback(study: optuna.study.Study, _frozen_trial: optuna.trial.FrozenTrial) -> None:
        """Remove non-best finished trial outputs after Optuna records a result."""
        cleanup_non_best_trial_dirs(study)

    return callback


def normalize_metric_label(label_name: str) -> str:
    """Sanitize label text so per-class metric names stay stable."""
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label_name)
    return normalized.strip("_") or "unknown"


def build_optuna_metrics(
    metric_f1: Any,
    metric_recall: Any,
    label_names: list[str] | None = None,
    metric_acc: Any = None,
) -> Callable[[Any], dict[str, float] | None]:
    """Aggregate macro and per-class metrics from logits with label-aware metric names."""

    def compute_metrics(prediction: Any) -> dict[str, float] | None:
        """Convert logits to predictions and emit macro, minimum, and per-class metrics."""
        preds = np.argmax(prediction.predictions, axis=1)
        num_labels = int(prediction.predictions.shape[1])
        labels = list(range(num_labels))
        has_valid_label_names = label_names is not None and len(label_names) == num_labels
        if has_valid_label_names:
            assert label_names is not None
            resolved_label_names = label_names
        else:
            resolved_label_names = [str(label_id) for label_id in labels]
        macro_f1 = metric_f1.compute(
            predictions=preds, references=prediction.label_ids, average="macro"
        )["f1"]
        f1_per_class_raw = metric_f1.compute(
            predictions=preds,
            references=prediction.label_ids,
            average=None,
            labels=labels,
        )["f1"]
        recalls_raw = metric_recall.compute(
            predictions=preds,
            references=prediction.label_ids,
            average=None,
            labels=labels,
        )["recall"]
        f1_per_class = np.asarray(f1_per_class_raw, dtype=np.float64)
        recalls = np.asarray(recalls_raw, dtype=np.float64)
        metrics = {
            "macro_f1": float(macro_f1),
            "min_class_recall": float(np.min(recalls)),
            "min_class_f1": float(np.min(f1_per_class)),
        }
        for index, f1_value in enumerate(f1_per_class):
            label_tag = normalize_metric_label(resolved_label_names[index])
            key = f"f1_class_{index}_{label_tag}" if has_valid_label_names else f"f1_class_{index}"
            metrics[key] = float(f1_value)
        for index, recall in enumerate(recalls):
            label_tag = normalize_metric_label(resolved_label_names[index])
            key = (
                f"recall_class_{index}_{label_tag}"
                if has_valid_label_names
                else f"recall_class_{index}"
            )
            metrics[key] = float(recall)
        if metric_acc is not None:
            metrics["accuracy"] = metric_acc.compute(
                predictions=preds, references=prediction.label_ids
            )["accuracy"]
        return metrics

    return compute_metrics


def validate_study_objective_metric(
    study: optuna.study.Study, objective_metric: str, role: str
) -> None:
    """Initialize empty studies with objective metadata and reject mismatched reuse."""
    registered_metric = study.user_attrs.get(OBJECTIVE_METRIC_USER_ATTR)
    if registered_metric is None:
        if len(study.trials) == 0:
            study.set_user_attr(OBJECTIVE_METRIC_USER_ATTR, objective_metric)
            return
        raise click.UsageError(
            f"{role} study '{study.study_name}' has no '{OBJECTIVE_METRIC_USER_ATTR}' metadata. "
            "Please use a dedicated study for this objective metric."
        )
    if registered_metric != objective_metric:
        raise click.UsageError(
            f"{role} study '{study.study_name}' uses objective metric '{registered_metric}', "
            f"but current run specifies '{objective_metric}'."
        )


def normalize_tta_merge_mode(tta_policy: str, tta_merge_mode: str) -> str:
    """Collapse disabled-TTA runs into one policy key for equivalent single-view studies."""
    return "single_view" if tta_policy == "none" else tta_merge_mode


def build_eval_policy(tta_policy: str, tta_merge_mode: str) -> str:
    """Serialize evaluation settings into one canonical JSON string."""
    payload = {
        "tta_policy": tta_policy,
        "tta_merge_mode": normalize_tta_merge_mode(tta_policy, tta_merge_mode),
    }
    return json.dumps(payload, sort_keys=True)


def validate_study_eval_policy(
    study: optuna.study.Study,
    tta_policy: str,
    tta_merge_mode: str,
    role: str,
) -> None:
    """Initialize empty studies with evaluation metadata and reject incompatible reuse."""
    eval_policy = build_eval_policy(tta_policy=tta_policy, tta_merge_mode=tta_merge_mode)
    registered_policy = study.user_attrs.get(EVAL_POLICY_USER_ATTR)
    if registered_policy is None:
        if len(study.trials) == 0:
            study.set_user_attr(EVAL_POLICY_USER_ATTR, eval_policy)
            return
        raise click.UsageError(
            f"{role} study '{study.study_name}' has no '{EVAL_POLICY_USER_ATTR}' metadata. "
            "Please use a dedicated study for this evaluation policy."
        )
    if registered_policy != eval_policy:
        raise click.UsageError(
            f"{role} study '{study.study_name}' uses evaluation policy '{registered_policy}', "
            f"but current run specifies '{eval_policy}'."
        )


def resolve_eval_batch_size(
    base_eval_batch_size: int, tta_policy: Any, image_size: int = 224
) -> int:
    """Scale evaluation batch size by the number of TTA views at the requested image size."""
    return max(1, base_eval_batch_size // int(tta_policy.get_view_count(image_size=image_size)))


__all__ = [
    "DEFAULT_SQLITE_TIMEOUT_SECONDS",
    "EVAL_POLICY_USER_ATTR",
    "OBJECTIVE_METRIC_CHOICES",
    "OBJECTIVE_METRIC_USER_ATTR",
    "REFIT_CONFIG_USER_ATTR",
    "TTA_MERGE_MODE_CHOICES",
    "build_eval_policy",
    "build_hyperparameter_tag",
    "build_non_best_trial_dir_cleanup_callback",
    "build_optuna_metrics",
    "build_refit_trial_config",
    "build_results_dir",
    "build_run_id",
    "build_sqlite_file_path",
    "build_sqlite_storage_path",
    "build_storage",
    "build_storage_from_sqlite_path",
    "cleanup_non_best_trial_dirs",
    "configure_sqlite_wal",
    "create_best_checkpoint_symlink",
    "load_refit_trial_config",
    "normalize_metric_label",
    "normalize_tta_merge_mode",
    "record_refit_trial_config",
    "record_trial_artifact_paths",
    "remove_trial_output_dir",
    "resolve_eval_batch_size",
    "resolve_optional_path_string",
    "validate_study_eval_policy",
    "validate_study_objective_metric",
]
