"""Optuna callbacks for Hugging Face Trainer."""

import optuna
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class TrainerPruningCallback(TrainerCallback):
    """Report a chosen evaluation metric to Optuna and stop when pruning is requested."""

    def __init__(self, trial: optuna.Trial, metric_name: str) -> None:
        """Store the active trial and metric key for evaluation reports."""
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        """Report the configured metric to Optuna and raise when pruning is triggered."""
        if metrics is None:
            return
        score = metrics.get(self.metric_name)
        if score is None:
            return
        self.trial.report(score, step=state.global_step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at step {state.global_step} with {self.metric_name}={score:.4f}"
            )
