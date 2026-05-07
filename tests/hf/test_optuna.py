"""Tests for Hugging Face Trainer Optuna callbacks."""

from typing import Any, cast

import optuna
from transformers import TrainerControl, TrainerState, TrainingArguments

from excvish.hf.optuna import TrainerPruningCallback


class StubTrial:
    """Capture reported values and expose a configurable pruning decision."""

    def __init__(self, should_prune_result: bool) -> None:
        """Store the desired pruning outcome and initialize an empty report log."""
        self.should_prune_result = should_prune_result
        self.reports: list[tuple[float, int]] = []

    def report(self, value: float, step: int) -> None:
        """Record the metric value and step."""
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        """Return the configured pruning decision."""
        return self.should_prune_result


def test_trainer_pruning_callback_reports_metric_without_pruning() -> None:
    """Forward the selected evaluation metric to the trial."""
    trial = StubTrial(should_prune_result=False)
    callback = TrainerPruningCallback(trial=cast(Any, trial), metric_name="eval_macro_f1")
    state = TrainerState(global_step=12)

    callback.on_evaluate(
        args=TrainingArguments(output_dir="tmp/test-optuna-callback"),
        state=state,
        control=TrainerControl(),
        metrics={"eval_macro_f1": 0.73, "eval_loss": 0.42},
    )

    assert trial.reports == [(0.73, 12)]


def test_trainer_pruning_callback_raises_when_trial_should_prune() -> None:
    """Raise Optuna's pruning signal after reporting the score."""
    trial = StubTrial(should_prune_result=True)
    callback = TrainerPruningCallback(trial=cast(Any, trial), metric_name="eval_macro_f1")
    state = TrainerState(global_step=7)

    try:
        callback.on_evaluate(
            args=TrainingArguments(output_dir="tmp/test-optuna-callback"),
            state=state,
            control=TrainerControl(),
            metrics={"eval_macro_f1": 0.51},
        )
    except optuna.TrialPruned:
        pass
    else:
        raise AssertionError("Expected optuna.TrialPruned to be raised.")

    assert trial.reports == [(0.51, 7)]
