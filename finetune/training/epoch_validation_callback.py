"""Epoch validation callback."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Any

from transformers import TrainerCallback

from finetune.workflows.selection import (
    TrialSpec,
    discover_validation_scores,
    early_stopping_from_scores,
    select_trial_checkpoint,
    should_stop_after_epoch,
)
from finetune.workflows.validation import (
    BackendValidationPredictor,
    GENERATION_CONFIG,
    QwenValidationPredictor,
    VALIDATION_COMPLETE,
    ValidationIdentity,
    run_validation,
)

from .checkpoint_manager import atomic_write_json
from .signals import signal_requested


CONTROLLED_FAILURE_USED = "CONTROLLED_FAILURE_USED.json"


class ControlledSmallJobFailure(RuntimeError):
    pass


class EpochValidationCallback(TrainerCallback):
    def __init__(
        self,
        *,
        adapter,
        validation_root: str | Path,
        learning_rate: float,
        seed: int,
        resolution_candidate: str,
        patience: int,
        min_delta: float,
        initial_max_epochs: int,
        extended_max_epochs: int,
        small_job: bool = False,
        small_job_max_epochs: int = 3,
        hard_max_epochs: int | None = None,
        controlled_failure_after_epoch_one: bool = False,
        controlled_failure_after_epoch: int | None = None,
        stop_after_epoch: int | None = None,
        validation_predictor_factory: Callable | None = None,
        validation_backend: Any | None = None,
        generation: Mapping[str, Any] = GENERATION_CONFIG,
    ) -> None:
        self.adapter = adapter
        self.validation_root = Path(validation_root).expanduser().resolve()
        self.learning_rate = learning_rate
        self.seed = seed
        self.parameter_dir = TrialSpec(learning_rate, seed).directory_name
        self.resolution_candidate = resolution_candidate
        self.patience = patience
        self.min_delta = min_delta
        self.initial_max_epochs = initial_max_epochs
        self.extended_max_epochs = extended_max_epochs
        self.small_job = small_job
        self.small_job_max_epochs = small_job_max_epochs
        if hard_max_epochs is not None and hard_max_epochs <= 0:
            raise ValueError("hard_max_epochs must be positive when provided")
        self.hard_max_epochs = hard_max_epochs
        if controlled_failure_after_epoch_one:
            if controlled_failure_after_epoch not in (None, 1):
                raise ValueError("controlled failure epoch arguments disagree")
            controlled_failure_after_epoch = 1
        if controlled_failure_after_epoch is not None and controlled_failure_after_epoch <= 0:
            raise ValueError("controlled_failure_after_epoch must be positive")
        if stop_after_epoch is not None and stop_after_epoch <= 0:
            raise ValueError("stop_after_epoch must be positive")
        self.controlled_failure_after_epoch = controlled_failure_after_epoch
        self.stop_after_epoch = stop_after_epoch
        self.validation_predictor_factory = validation_predictor_factory
        self.validation_backend = validation_backend
        self.generation = dict(generation)

    def _validation_dir(self, epoch: int) -> Path:
        return self.validation_root / self.parameter_dir / f"epoch-{epoch:04d}"

    def _scores(self):
        return tuple(
            score
            for score in discover_validation_scores(
                self.validation_root / self.parameter_dir
            )
            if score.learning_rate == self.learning_rate and score.seed == self.seed
        )

    def _validate_current(self, args, state, control, model, processing_class):
        if not state.global_step or state.epoch is None:
            return control
        epoch_value = float(state.epoch)
        epoch = int(round(epoch_value))
        if abs(epoch_value - epoch) > 1e-6 or epoch <= 0:
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not (checkpoint / "CHECKPOINT_COMPLETE.json").is_file():
            return control
        output = self._validation_dir(epoch)
        if not (output / VALIDATION_COMPLETE).is_file():
            identity = ValidationIdentity(
                task=self.adapter.config.task,
                checkpoint_dir=str(checkpoint.resolve()),
                epoch=epoch,
                learning_rate=self.learning_rate,
                seed=self.seed,
                resolution_candidate=self.resolution_candidate,
                generation=self.generation,
            )
            predictor = (
                self.validation_predictor_factory(epoch, model, processing_class)
                if self.validation_predictor_factory is not None
                else (
                    BackendValidationPredictor(
                        self.validation_backend,
                        model,
                        processing_class,
                        self.generation,
                    )
                    if self.validation_backend is not None
                    else QwenValidationPredictor(model, processing_class, self.generation)
                )
            )
            run_validation(
                adapter=self.adapter,
                checkpoint_dir=checkpoint,
                output_dir=output,
                identity=identity,
                predictor=predictor,
                should_interrupt=signal_requested,
            )

        scores = self._scores()
        early = early_stopping_from_scores(
            scores, patience=self.patience, min_delta=self.min_delta
        )
        best = select_trial_checkpoint(scores)
        atomic_write_json(
            Path(args.output_dir) / "early_stopping_state.json",
            {
                "state": asdict(early),
                "best_checkpoint": asdict(best),
                "selection_source": "canonical_validation_only",
            },
        )

        if self.controlled_failure_after_epoch == epoch:
            marker = Path(args.output_dir) / CONTROLLED_FAILURE_USED
            if not marker.exists():
                atomic_write_json(
                    marker,
                    {"epoch": epoch, "checkpoint": str(checkpoint.resolve())},
                )
                raise ControlledSmallJobFailure(
                    f"intentional failure after epoch-{epoch} checkpoint and validation"
                )

        if self.stop_after_epoch is not None and epoch >= self.stop_after_epoch:
            control.should_training_stop = True
        elif self.hard_max_epochs is not None:
            if early.should_stop or epoch >= self.hard_max_epochs:
                control.should_training_stop = True
        elif self.small_job:
            if early.should_stop or epoch >= self.small_job_max_epochs:
                control.should_training_stop = True
        elif should_stop_after_epoch(
            scores,
            patience=self.patience,
            min_delta=self.min_delta,
            initial_max_epochs=self.initial_max_epochs,
            extended_max_epochs=self.extended_max_epochs,
        ):
            control.should_training_stop = True
        return control

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        return self._validate_current(args, state, control, model, processing_class)

    def on_save(self, args, state, control, model=None, processing_class=None, **kwargs):
        return self._validate_current(args, state, control, model, processing_class)
