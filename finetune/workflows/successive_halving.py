"""Successive halving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

from finetune.config.schema import SuccessiveHalvingConfig
from finetune.datasets import (
    AdapterExample,
    CanonicalSplit,
    DatasetAdapter,
    ValidationPrediction,
)
from finetune.training.checkpoint_manager import checkpoint_step, validate_checkpoint_files
from finetune.workflows.selection import (
    CheckpointScore,
    load_checkpoint_score,
)
from finetune.workflows.validation import (
    VALIDATION_COMPLETE,
    VALIDATION_PREDICTIONS,
    prediction_index_manifest,
)


RUNG_COMPLETE_TEMPLATE = "RUNG_{rung}_COMPLETE.json"
FINAL_TRAINING_COMPLETE = "TRAINING_COMPLETE.json"


class SuccessiveHalvingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RungPlan:
    rung: int
    budget_steps: int
    survivors: int
    validation_scope: str


@dataclass(frozen=True)
class RungResult:
    task: str
    learning_rate: float
    seed: int
    rung: int
    budget_steps: int
    scheduler_horizon_steps: int
    validation_scope: str
    validation_index: tuple[tuple[str, str], ...]
    score: CheckpointScore
    marker_path: str


def require_policy(config: SuccessiveHalvingConfig | None) -> SuccessiveHalvingConfig:
    if config is None or not config.enabled:
        raise SuccessiveHalvingError("successive-halving policy is not enabled")
    return config


def rung_plans(policy: SuccessiveHalvingConfig) -> tuple[RungPlan, ...]:
    return tuple(
        RungPlan(
            rung=index,
            budget_steps=steps,
            survivors=policy.survivors_after_rung[index - 1],
            validation_scope="screening_subset",
        )
        for index, steps in enumerate(policy.rung_steps, start=1)
    )


def _shuffled(values: Sequence[Any], seed: str) -> list[Any]:
    ordered = list(values)
    random.Random(seed).shuffle(ordered)
    return ordered


def _training_strata(examples: Sequence[AdapterExample]) -> tuple[str, ...]:
    cities = tuple(
        str(example.metadata.get("city") or example.metadata.get("source_qualifier") or "").strip()
        for example in examples
    )
    if cities and all(cities):
        return cities
    labels = tuple(
        example.ground_truth if isinstance(example.ground_truth, str) else None
        for example in examples
    )
    distinct = {label for label in labels if label is not None}
    if labels and all(label is not None for label in labels) and len(distinct) <= 256:
        return tuple(str(label) for label in labels)
    return tuple("__all__" for _ in examples)


def screening_validation_subset(
    examples: Sequence[AdapterExample], *, maximum: int
) -> tuple[AdapterExample, ...]:
    """Fixed screening subset: the first ``maximum`` canonical validation
    samples in a deterministic ordering (sorted by sample ID)."""
    if maximum <= 0:
        raise ValueError("maximum screening subset size must be positive")
    if not examples:
        raise SuccessiveHalvingError("canonical validation examples are empty")
    if any(example.canonical_split is not CanonicalSplit.VALIDATION for example in examples):
        raise SuccessiveHalvingError("screening subset contains a non-validation example")
    if len({example.sample_id for example in examples}) != len(examples):
        raise SuccessiveHalvingError("validation candidates contain duplicate sample IDs")
    ordered = sorted(examples, key=lambda example: example.sample_id)
    return tuple(ordered[:maximum])


def deterministic_training_subset(
    examples: Sequence[AdapterExample], *, maximum: int, seed: int
) -> tuple[AdapterExample, ...]:
    if maximum <= 0:
        raise ValueError("maximum training subset size must be positive")
    if not examples:
        raise SuccessiveHalvingError("canonical training examples are empty")
    if any(example.canonical_split is not CanonicalSplit.TRAIN for example in examples):
        raise SuccessiveHalvingError("final training subset contains a non-train example")
    if len({example.sample_id for example in examples}) != len(examples):
        raise SuccessiveHalvingError("training candidates contain duplicate sample IDs")
    if len(examples) <= maximum:
        return tuple(examples)
    strata: dict[str, list[AdapterExample]] = {}
    for example, stratum in zip(examples, _training_strata(examples)):
        strata.setdefault(stratum, []).append(example)
    total = len(examples)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, members in strata.items():
        exact = maximum * len(members) / total
        quotas[stratum] = min(int(exact), len(members))
        remainders.append((exact - int(exact), stratum))
    remaining = maximum - sum(quotas.values())
    candidates = sorted(remainders, key=lambda item: (-item[0], item[1]))
    for _, stratum in candidates[:remaining]:
        quotas[stratum] += 1
    if sum(quotas.values()) != maximum:
        raise SuccessiveHalvingError("could not allocate the final training subset budget")
    selected_ids: set[str] = set()
    for stratum, members in strata.items():
        canonical = sorted(members, key=lambda example: example.sample_id)
        chosen = _shuffled(canonical, f"train:{seed}:{stratum}")[: quotas[stratum]]
        selected_ids.update(example.sample_id for example in chosen)
    return tuple(example for example in examples if example.sample_id in selected_ids)


class FixedValidationSubsetAdapter(DatasetAdapter[Any, Any]):

    def __init__(self, base: DatasetAdapter[Any, Any], examples: Sequence[AdapterExample]) -> None:
        super().__init__(base.config)
        self.base = base
        self.examples = tuple(examples)
        if not self.examples or any(
            example.canonical_split is not CanonicalSplit.VALIDATION
            for example in self.examples
        ):
            raise SuccessiveHalvingError("validation screening subset is empty or invalid")

    def iter_raw_records(self, split: CanonicalSplit):
        raise NotImplementedError("subset adapter exposes materialized validation only")

    def source_id(self, record: Any, source_split: str, index: int) -> str:
        raise NotImplementedError("subset adapter exposes materialized validation only")

    def build_example(self, record: Any, split: CanonicalSplit) -> AdapterExample:
        raise NotImplementedError("subset adapter exposes materialized validation only")

    def parse_prediction(self, text: str) -> Any:
        return self.base.parse_prediction(text)

    def parse_example_prediction(self, text: str, example: AdapterExample) -> Any:
        return self.base.parse_example_prediction(text, example)

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]):
        return self.base.compute_validation_metric(predictions)

    def materialize_split(self, split: CanonicalSplit) -> tuple[AdapterExample, ...]:
        if split is not CanonicalSplit.VALIDATION:
            raise SuccessiveHalvingError("screening adapter cannot materialize train or test")
        self._materialized[split] = self.examples
        return self.examples


def rung_marker_name(rung: int) -> str:
    if rung <= 0:
        raise ValueError("rung must be positive")
    return RUNG_COMPLETE_TEMPLATE.format(rung=rung)


def load_rung_result(path: str | Path) -> RungResult:
    marker_path = Path(path).expanduser().resolve()
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuccessiveHalvingError(f"invalid rung marker: {marker_path}") from exc
    if not isinstance(marker, dict):
        raise SuccessiveHalvingError(f"rung marker must be an object: {marker_path}")
    rung = int(marker.get("rung", 0))
    if marker_path.name != rung_marker_name(rung):
        raise SuccessiveHalvingError(f"rung marker filename/content mismatch: {marker_path}")
    validation_marker = Path(str(marker.get("validation_marker", ""))).expanduser().resolve()
    if validation_marker.name != VALIDATION_COMPLETE or not validation_marker.is_file():
        raise SuccessiveHalvingError("rung marker does not point to validation completion")
    score = load_checkpoint_score(validation_marker)
    if marker.get("score") != asdict(score):
        raise SuccessiveHalvingError("rung score changed after completion")
    checks = {
        "task": score.task,
        "learning_rate": score.learning_rate,
        "seed": score.seed,
        "validation_scope": score.validation_scope,
    }
    for field, expected in checks.items():
        if marker.get(field) != expected:
            raise SuccessiveHalvingError(f"rung marker {field} mismatch")
    if marker.get("selection_source") != "canonical_validation_only":
        raise SuccessiveHalvingError("rung marker is not validation-only")
    if marker.get("test_data_used") is not False:
        raise SuccessiveHalvingError("rung marker indicates test data usage")
    checkpoint = Path(score.checkpoint_dir).expanduser().resolve()
    validate_checkpoint_files(checkpoint)
    if checkpoint_step(checkpoint) != int(marker["budget_steps"]):
        raise SuccessiveHalvingError("rung checkpoint step does not equal its budget")
    index = prediction_index_manifest(validation_marker.parent / VALIDATION_PREDICTIONS)
    validation_index = tuple((item["sample_id"], item["source_id"]) for item in index)
    return RungResult(
        task=score.task,
        learning_rate=score.learning_rate,
        seed=score.seed,
        rung=rung,
        budget_steps=int(marker["budget_steps"]),
        scheduler_horizon_steps=int(marker["scheduler_horizon_steps"]),
        validation_scope=score.validation_scope,
        validation_index=validation_index,
        score=score,
        marker_path=str(marker_path),
    )


