"""Selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from finetune.config.schema import Direction, SearchConfig
from finetune.training.checkpoint_manager import (
    CheckpointError,
    load_checkpoint_manifest,
    validate_checkpoint_files,
)
from .validation import (
    VALIDATION_COMPLETE,
    VALIDATION_CONFIG,
    VALIDATION_METRICS,
    ValidationCompleteMarker,
    ValidationMetrics,
    has_forbidden_evaluation_component,
    prediction_index_manifest,
    reject_test_path,
)


SELECTION_FILE = "selection.json"
SELECTION_COMPLETE = "SELECTION_COMPLETE.json"


class SelectionError(RuntimeError):
    pass


def learning_rate_slug(value: float) -> str:
    decimal = Decimal(str(value)).normalize()
    exponent = decimal.adjusted()
    coefficient = decimal.scaleb(-exponent).normalize()
    coefficient_text = format(coefficient, "f").rstrip("0").rstrip(".")
    return f"{coefficient_text}e{exponent}"


@dataclass(frozen=True)
class TrialSpec:
    learning_rate: float
    seed: int
    expansion_round: int = 0

    @property
    def directory_name(self) -> str:
        return f"lr-{learning_rate_slug(self.learning_rate)}-seed-{self.seed}"


@dataclass(frozen=True)
class TrialDurationPolicy:
    initial_max_epochs: int
    extended_max_epochs: int
    scheduler_horizon_epochs: int
    patience: int
    min_delta: float
    scheduler_continuous_across_epoch_5_gate: bool = True

    def __post_init__(self) -> None:
        if self.initial_max_epochs <= 0:
            raise SelectionError("initial_max_epochs must be positive")
        if self.extended_max_epochs < self.initial_max_epochs:
            raise SelectionError("extended_max_epochs must not be below initial_max_epochs")
        if self.scheduler_horizon_epochs != self.extended_max_epochs:
            raise SelectionError("scheduler horizon must equal extended_max_epochs")
        if self.patience < 0 or self.min_delta < 0:
            raise SelectionError("patience and min_delta must be non-negative")
        if not self.scheduler_continuous_across_epoch_5_gate:
            raise SelectionError("scheduler must remain continuous across the epoch-5 gate")

    @classmethod
    def from_search(cls, search: SearchConfig) -> "TrialDurationPolicy":
        return cls(
            initial_max_epochs=search.duration.initial_max_epochs,
            extended_max_epochs=search.duration.extended_max_epochs,
            scheduler_horizon_epochs=search.duration.extended_max_epochs,
            patience=search.early_stopping.patience_validation_events,
            min_delta=search.early_stopping.min_delta,
        )


@dataclass(frozen=True)
class CheckpointScore:
    task: str
    checkpoint_dir: str
    validation_dir: str
    epoch: int
    learning_rate: float
    seed: int
    metric_name: str
    direction: str
    primary_metric: float
    parse_failure_rate: float
    resolution_candidate: str = ""
    generation: Mapping[str, Any] | None = None
    validation_scope: str = "canonical_full"

    def __post_init__(self) -> None:
        if self.direction not in {Direction.MINIMIZE.value, Direction.MAXIMIZE.value}:
            raise SelectionError(f"invalid metric direction: {self.direction}")
        if not math.isfinite(self.primary_metric):
            raise SelectionError("primary metric must be finite")
        if not 0.0 <= self.parse_failure_rate <= 1.0:
            raise SelectionError("parse failure rate must be in [0, 1]")
        if self.epoch <= 0 or self.learning_rate <= 0:
            raise SelectionError("score epoch and learning rate must be positive")
        if self.seed < 0:
            raise SelectionError("score seed must be non-negative")
        if self.validation_scope not in {"canonical_full", "screening_subset"}:
            raise SelectionError("invalid validation scope")

    @property
    def trial_key(self) -> tuple[float, int]:
        return (self.learning_rate, self.seed)


@dataclass(frozen=True)
class EarlyStoppingState:
    direction: str
    patience: int
    min_delta: float
    best_primary_metric: float | None = None
    best_epoch: int | None = None
    validation_events: int = 0
    events_without_improvement: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {Direction.MINIMIZE.value, Direction.MAXIMIZE.value}:
            raise SelectionError(f"invalid early-stopping direction: {self.direction}")
        if self.patience < 0 or self.min_delta < 0:
            raise SelectionError("patience and min_delta must be non-negative")

    def observe(self, primary_metric: float, epoch: int) -> "EarlyStoppingState":
        if not math.isfinite(primary_metric) or epoch <= 0:
            raise SelectionError("early-stopping observations must be finite and positive epoch")
        improved = self.best_primary_metric is None
        if self.best_primary_metric is not None:
            if self.direction == Direction.MINIMIZE.value:
                improved = primary_metric < self.best_primary_metric - self.min_delta
            else:
                improved = primary_metric > self.best_primary_metric + self.min_delta
        misses = 0 if improved else self.events_without_improvement + 1
        return replace(
            self,
            best_primary_metric=primary_metric if improved else self.best_primary_metric,
            best_epoch=epoch if improved else self.best_epoch,
            validation_events=self.validation_events + 1,
            events_without_improvement=misses,
            should_stop=(not improved and misses >= self.patience),
        )


def early_stopping_from_scores(
    scores: Sequence[CheckpointScore], *, patience: int, min_delta: float
) -> EarlyStoppingState:
    if not scores:
        raise SelectionError("cannot reconstruct early stopping without validation scores")
    ordered = sorted(scores, key=lambda score: score.epoch)
    if len({score.epoch for score in ordered}) != len(ordered):
        raise SelectionError("a trial has duplicate validation epochs")
    epochs = [score.epoch for score in ordered]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise SelectionError(f"a trial has non-contiguous validation epochs: {epochs}")
    direction = ordered[0].direction
    state = EarlyStoppingState(direction, patience, min_delta)
    for score in ordered:
        if score.direction != direction:
            raise SelectionError("a trial mixes metric directions")
        state = state.observe(score.primary_metric, score.epoch)
    return state


def should_stop_after_epoch(
    scores: Sequence[CheckpointScore],
    *,
    patience: int,
    min_delta: float,
    initial_max_epochs: int,
    extended_max_epochs: int,
) -> bool:
    if not scores:
        raise SelectionError("epoch stop decision requires validation scores")
    current_epoch = max(score.epoch for score in scores)
    state = early_stopping_from_scores(scores, patience=patience, min_delta=min_delta)
    if current_epoch < initial_max_epochs:
        return state.should_stop
    if current_epoch == initial_max_epochs:
        if initial_max_epochs == extended_max_epochs:
            return True
        return select_trial_checkpoint(scores).epoch != initial_max_epochs
    if state.should_stop:
        return True
    return current_epoch >= extended_max_epochs


def trial_completion_reason(
    scores: Sequence[CheckpointScore], policy: TrialDurationPolicy
) -> str:
    if not should_stop_after_epoch(
        scores,
        patience=policy.patience,
        min_delta=policy.min_delta,
        initial_max_epochs=policy.initial_max_epochs,
        extended_max_epochs=policy.extended_max_epochs,
    ):
        raise SelectionError("trial has not reached a compliant stop condition")
    state = early_stopping_from_scores(scores, patience=policy.patience, min_delta=policy.min_delta)
    current_epoch = max(score.epoch for score in scores)
    if state.should_stop:
        return "early_stopping"
    if current_epoch == policy.initial_max_epochs:
        return "initial_gate_complete"
    if current_epoch == policy.extended_max_epochs:
        return "extended_hard_max"
    raise SelectionError(f"trial stopped at unsupported epoch {current_epoch}")


def _score_key(score: CheckpointScore, *, include_learning_rate: bool) -> tuple[float, ...]:
    primary = -score.primary_metric if score.direction == Direction.MAXIMIZE.value else score.primary_metric
    key: tuple[float, ...] = (primary, score.parse_failure_rate, float(score.epoch))
    if include_learning_rate:
        key += (score.learning_rate, float(score.seed))
    return key


def _validate_score_cohort(scores: Sequence[CheckpointScore]) -> None:
    if not scores:
        raise SelectionError("validation score cohort is empty")
    cohort_fields = {
        "task": {score.task for score in scores},
        "metric_name": {score.metric_name for score in scores},
        "direction": {score.direction for score in scores},
        "resolution_candidate": {score.resolution_candidate for score in scores},
        "generation": {
            json.dumps(score.generation, sort_keys=True) for score in scores
        },
        "validation_ids": {
            json.dumps(
                prediction_index_manifest(
                    Path(score.validation_dir) / "predictions.jsonl"
                ),
                sort_keys=True,
            )
            for score in scores
        },
        "validation_scope": {score.validation_scope for score in scores},
    }
    drift = sorted(field for field, values in cohort_fields.items() if len(values) != 1)
    if drift:
        raise SelectionError(f"validation candidates do not belong to one cohort: {drift}")
    score_keys = [(score.learning_rate, score.seed, score.epoch) for score in scores]
    if len(set(score_keys)) != len(score_keys):
        raise SelectionError("validation candidates contain duplicate parameter/epoch entries")


def _scores_for_trial(
    scores: Sequence[CheckpointScore], learning_rate: float, seed: int
) -> tuple[CheckpointScore, ...]:
    return tuple(
        score
        for score in scores
        if score.learning_rate == learning_rate and score.seed == seed
    )


def verify_training_completion_marker(
    *,
    learning_rate: float,
    seed: int,
    scores: Sequence[CheckpointScore],
    marker_path: str | Path,
    duration_policy: TrialDurationPolicy | None = None,
) -> dict[str, Any]:
    parameter_scores = _scores_for_trial(scores, learning_rate, seed)
    if not parameter_scores:
        raise SelectionError(
            f"training completion has no validation scores: lr={learning_rate}, seed={seed}"
        )
    _validate_score_cohort(parameter_scores)
    epochs = sorted(score.epoch for score in parameter_scores)
    if epochs != list(range(1, epochs[-1] + 1)):
        raise SelectionError(f"validation epochs are not contiguous: {epochs}")
    marker_file = reject_test_path(marker_path, "trial_completion")
    if marker_file.name != "TRAINING_COMPLETE.json":
        raise SelectionError(f"training completion path has wrong name: {marker_file}")
    if (marker_file.parent / "TRAINING_FAILED.json").exists():
        raise SelectionError(f"training is marked both complete and failed: {marker_file.parent}")
    manifest_path = marker_file.parent / "training_manifest.json"
    try:
        marker = json.loads(marker_file.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"invalid training artifact near {marker_file}") from exc
    checks = {
        "learning_rate": learning_rate,
        "seed": seed,
        "selection_source": "canonical_validation_only",
        "validation_epochs": epochs,
        "best_checkpoint": asdict(select_trial_checkpoint(parameter_scores)),
    }
    for field, expected in checks.items():
        if marker.get(field) != expected:
            raise SelectionError(f"training completion {field} mismatch: {marker_file}")
    if not isinstance(manifest, dict):
        raise SelectionError(f"training manifest must be an object: {manifest_path}")
    if duration_policy is not None:
        expected_duration = asdict(duration_policy)
        if manifest.get("duration_policy") != expected_duration:
            raise SelectionError("trial manifest duration policy is not compliant")
        if marker.get("duration_policy") != expected_duration:
            raise SelectionError("trial completion duration policy is not compliant")
        if marker.get("completion_reason") != trial_completion_reason(parameter_scores, duration_policy):
            raise SelectionError("training completion reason does not match validation history")
        for score in parameter_scores:
            checkpoint = Path(score.checkpoint_dir).expanduser().resolve()
            if checkpoint.parent != marker_file.parent:
                raise SelectionError("checkpoint is outside its parameter training directory")
            try:
                compatibility = load_checkpoint_manifest(checkpoint).get("compatibility")
            except CheckpointError as exc:
                raise SelectionError(f"invalid checkpoint manifest: {checkpoint}") from exc
            if not isinstance(compatibility, dict):
                raise SelectionError(f"checkpoint compatibility is missing: {checkpoint}")
            expected_checkpoint = {
                "duration_policy": expected_duration,
                "scheduler_horizon_epochs": duration_policy.scheduler_horizon_epochs,
                "learning_rate": learning_rate,
                "seed": seed,
            }
            for field, expected in expected_checkpoint.items():
                if compatibility.get(field) != expected:
                    raise SelectionError(f"checkpoint {field} is not compliant: {checkpoint}")
    return {"path": str(marker_file), "learning_rate": learning_rate, "seed": seed}


def select_trial_checkpoint(scores: Sequence[CheckpointScore]) -> CheckpointScore:
    if not scores:
        raise SelectionError("a trial has no completed validation checkpoints")
    _validate_score_cohort(scores)
    if len({score.trial_key for score in scores}) != 1:
        raise SelectionError("trial checkpoint selection requires one LR/seed pair")
    return min(scores, key=lambda score: _score_key(score, include_learning_rate=False))


def select_best_checkpoint(scores: Sequence[CheckpointScore]) -> CheckpointScore:
    if not scores:
        raise SelectionError("no completed validation checkpoints are available")
    _validate_score_cohort(scores)
    per_trial = [
        select_trial_checkpoint(_scores_for_trial(scores, learning_rate, seed))
        for learning_rate, seed in sorted({score.trial_key for score in scores})
    ]
    return min(per_trial, key=lambda score: _score_key(score, include_learning_rate=True))


def plan_initial_trials(search: SearchConfig, seed: int) -> tuple[TrialSpec, ...]:
    return tuple(TrialSpec(lr, seed) for lr in search.learning_rate.initial_values)


def rank_scores(scores: Sequence[CheckpointScore]) -> tuple[CheckpointScore, ...]:
    """Rank candidates by the primary validation metric, then by the lower
    parse-failure/refusal rate, earlier epoch, and lower learning rate."""
    _validate_score_cohort(scores)
    return tuple(
        sorted(scores, key=lambda score: _score_key(score, include_learning_rate=True))
    )


def plan_boundary_expansion(
    selected_learning_rate: float, search: SearchConfig, seed: int
) -> TrialSpec | None:
    """Return the single expansion candidate for a boundary winner.

    If successive halving selects the lowest initial learning rate (1e-5), a
    5e-6 candidate is added; if it selects the highest (5e-5), a 1e-4 candidate
    is added. The candidate is trained for the last-rung budget (128 steps) and
    compared with the incumbent. No expansion follows an interior winner.
    """
    lr_search = search.learning_rate
    if not lr_search.maximum_expansion_rounds:
        return None
    if selected_learning_rate not in lr_search.initial_values:
        raise SelectionError("boundary expansion requires an initial-grid winner")
    if selected_learning_rate == lr_search.lower_boundary_value:
        return TrialSpec(lr_search.lower_expansion_value, seed, expansion_round=1)
    if selected_learning_rate == lr_search.upper_boundary_value:
        return TrialSpec(lr_search.upper_expansion_value, seed, expansion_round=1)
    return None


def load_checkpoint_score(marker_path: str | Path) -> CheckpointScore:
    marker_file = Path(marker_path).expanduser().resolve()
    if marker_file.name != VALIDATION_COMPLETE:
        raise SelectionError(f"selection requires {VALIDATION_COMPLETE}: {marker_file}")
    validation_dir = reject_test_path(marker_file.parent, "selection.validation_dir")
    metrics_path = validation_dir / VALIDATION_METRICS
    config_path = validation_dir / VALIDATION_CONFIG
    try:
        marker = ValidationCompleteMarker(**json.loads(marker_file.read_text(encoding="utf-8")))
        metrics = ValidationMetrics(**json.loads(metrics_path.read_text(encoding="utf-8")))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise SelectionError(f"invalid validation artifact: {validation_dir}") from exc
    if not isinstance(config, dict):
        raise SelectionError(f"validation config must be an object: {config_path}")
    if marker.canonical_split != "validation" or metrics.canonical_split != "validation":
        raise SelectionError("selection input is not canonical validation")
    prediction_index_manifest(validation_dir / "predictions.jsonl")
    common_fields = (
        "task", "checkpoint_dir", "epoch", "learning_rate", "seed",
        "canonical_split", "validation_scope",
    )
    for field in common_fields:
        expected = getattr(metrics, field)
        if getattr(marker, field) != expected or config.get(field) != expected:
            raise SelectionError(f"validation artifact mismatch for {field}: {validation_dir}")
    checkpoint = Path(metrics.checkpoint_dir).expanduser().resolve()
    validate_checkpoint_files(checkpoint)
    return CheckpointScore(
        task=metrics.task,
        checkpoint_dir=str(checkpoint),
        validation_dir=str(validation_dir),
        epoch=metrics.epoch,
        learning_rate=metrics.learning_rate,
        seed=metrics.seed,
        metric_name=metrics.metric_name,
        direction=metrics.direction,
        primary_metric=metrics.primary_metric,
        parse_failure_rate=metrics.parse_failure_rate,
        resolution_candidate=str(config.get("resolution_candidate", "")),
        generation=dict(config.get("generation", {})),
        validation_scope=metrics.validation_scope,
    )


def discover_validation_scores(validation_root: str | Path) -> tuple[CheckpointScore, ...]:
    root = reject_test_path(validation_root, "selection.validation_root")
    if not root.is_dir():
        return ()
    markers = [
        path
        for path in sorted(root.rglob(VALIDATION_COMPLETE))
        if not has_forbidden_evaluation_component(path.parent)
    ]
    return tuple(load_checkpoint_score(path) for path in markers)


