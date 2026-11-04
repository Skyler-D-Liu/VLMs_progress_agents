"""Validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from finetune.datasets import (
    AdapterExample,
    CanonicalSplit,
    DatasetAdapter,
    ValidationPrediction,
)
from finetune.model_backends import get_model_backend
from finetune.training.checkpoint_manager import atomic_write_json, validate_checkpoint_files


VALIDATION_COMPLETE = "VALIDATION_COMPLETE.json"
VALIDATION_METRICS = "metrics.json"
VALIDATION_PREDICTIONS = "predictions.jsonl"
VALIDATION_CONFIG = "validation_config.json"
GENERATION_CONFIG = {"max_new_tokens": 128, "do_sample": False}
FORBIDDEN_PATH_PARTS = frozenset({"test", "tests", "evaluation"})


class ValidationArtifactError(RuntimeError):
    pass


class ValidationInterrupted(RuntimeError):
    pass


def _jsonable(value: Any, field: str) -> Any:
    try:
        payload = json.dumps(value, ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValidationArtifactError(f"{field} is not JSON serializable") from exc
    return json.loads(payload)


def validation_index_manifest(examples: Sequence[AdapterExample]) -> list[dict[str, str]]:
    return [
        {"sample_id": example.sample_id, "source_id": example.source_id}
        for example in examples
    ]


def prediction_index_manifest(path: str | Path) -> list[dict[str, str]]:
    prediction_path = Path(path).expanduser().resolve()
    try:
        lines = prediction_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationArtifactError(
            f"cannot read validation predictions: {prediction_path}"
        ) from exc
    index: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    source_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationArtifactError(
                f"invalid prediction JSONL at {prediction_path}:{line_number}"
            ) from exc
        sample_id = record.get("sample_id")
        source_id = record.get("source_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValidationArtifactError(
                f"prediction at {prediction_path}:{line_number} has invalid sample_id"
            )
        if not isinstance(source_id, str) or not source_id:
            raise ValidationArtifactError(f"prediction {sample_id} has invalid source_id")
        if record.get("canonical_split") != CanonicalSplit.VALIDATION.value:
            raise ValidationArtifactError(f"prediction {sample_id} is not canonical validation")
        if sample_id in sample_ids or source_id in source_ids:
            raise ValidationArtifactError(
                f"duplicate validation prediction identity: {sample_id}/{source_id}"
            )
        sample_ids.add(sample_id)
        source_ids.add(source_id)
        index.append({"sample_id": sample_id, "source_id": source_id})
    return index


def has_forbidden_evaluation_component(path: str | Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    return any(
        FORBIDDEN_PATH_PARTS.intersection(
            token for token in re.split(r"[^a-z0-9]+", part.casefold()) if token
        )
        for part in resolved.parts
    )


def reject_test_path(path: str | Path, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    forbidden = [
        part
        for part in resolved.parts
        if FORBIDDEN_PATH_PARTS.intersection(
            token for token in re.split(r"[^a-z0-9]+", part.casefold()) if token
        )
    ]
    if forbidden:
        raise ValidationArtifactError(
            f"{field} must be validation-only; forbidden path component {forbidden[0]!r}: {resolved}"
        )
    return resolved


@dataclass(frozen=True)
class ValidationIdentity:

    task: str
    checkpoint_dir: str
    epoch: int
    learning_rate: float
    seed: int
    resolution_candidate: str
    generation: Mapping[str, Any]
    canonical_split: str = "validation"
    validation_scope: str = "canonical_full"

    def __post_init__(self) -> None:
        if self.canonical_split != CanonicalSplit.VALIDATION.value:
            raise ValidationArtifactError("validation config must use canonical_split=validation")
        generation = dict(self.generation)
        if (
            set(generation) != {"max_new_tokens", "do_sample"}
            or type(generation.get("max_new_tokens")) is not int
            or generation["max_new_tokens"] not in {128, 1024, 2048}
            or generation.get("do_sample") is not False
        ):
            raise ValidationArtifactError(
                "validation generation requires 128, 1024, or 2048 greedy tokens"
            )
        if self.epoch <= 0 or self.learning_rate <= 0 or self.seed < 0:
            raise ValidationArtifactError("epoch and learning rate must be positive; seed non-negative")
        if self.validation_scope not in {"canonical_full", "screening_subset"}:
            raise ValidationArtifactError("unsupported validation_scope")


@dataclass(frozen=True)
class ValidationMetrics:
    task: str
    checkpoint_dir: str
    epoch: int
    learning_rate: float
    seed: int
    metric_name: str
    direction: str
    primary_metric: float
    parse_failure_rate: float
    details: Mapping[str, Any]
    canonical_split: str = "validation"
    validation_scope: str = "canonical_full"

    def __post_init__(self) -> None:
        if self.canonical_split != CanonicalSplit.VALIDATION.value:
            raise ValidationArtifactError("metrics must use canonical_split=validation")
        if self.direction not in {"minimize", "maximize"}:
            raise ValidationArtifactError("metric direction must be minimize or maximize")
        if not math.isfinite(self.primary_metric):
            raise ValidationArtifactError("primary validation metric must be finite")
        if not 0.0 <= self.parse_failure_rate <= 1.0:
            raise ValidationArtifactError("parse failure rate must be in [0, 1]")
        if self.validation_scope not in {"canonical_full", "screening_subset"}:
            raise ValidationArtifactError("unsupported validation_scope")


@dataclass(frozen=True)
class ValidationCompleteMarker:
    task: str
    checkpoint_dir: str
    epoch: int
    learning_rate: float
    seed: int
    canonical_split: str = "validation"
    validation_scope: str = "canonical_full"


@dataclass(frozen=True)
class ValidationArtifact:
    output_dir: Path
    identity: ValidationIdentity
    metrics: ValidationMetrics
    marker: ValidationCompleteMarker


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_prediction_records(
    path: Path,
    *,
    identity: ValidationIdentity,
    examples_by_id: Mapping[str, AdapterExample],
    parse_prediction: Callable[[str, AdapterExample], Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        raw = raw[: last_newline + 1] if last_newline >= 0 else b""
        with path.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationArtifactError(
                f"invalid complete JSONL record at {path}:{line_number}"
            ) from exc
        sample_id = record.get("sample_id")
        if sample_id not in examples_by_id:
            raise ValidationArtifactError(f"non-validation sample in predictions: {sample_id!r}")
        if sample_id in records:
            raise ValidationArtifactError(f"duplicate validation sample in predictions: {sample_id}")
        example = examples_by_id[sample_id]
        required = {
            "canonical_split": CanonicalSplit.VALIDATION.value,
            "source_id": example.source_id,
            "task": identity.task,
            "epoch": identity.epoch,
            "learning_rate": identity.learning_rate,
            "seed": identity.seed,
        }
        for field, expected in required.items():
            if field not in record and field == "seed":
                continue
            if record.get(field) != expected:
                raise ValidationArtifactError(
                    f"prediction {sample_id} has incompatible {field}: "
                    f"{record.get(field)!r} != {expected!r}"
                )
        response = record.get("response_text")
        if not isinstance(response, str):
            raise ValidationArtifactError(f"prediction {sample_id} response_text must be a string")
        if type(record.get("parse_failed")) is not bool:
            raise ValidationArtifactError(f"prediction {sample_id} parse_failed must be boolean")
        expected_ground_truth = _jsonable(
            example.ground_truth, f"prediction {sample_id} ground_truth"
        )
        if record.get("ground_truth") != expected_ground_truth:
            raise ValidationArtifactError(
                f"prediction {sample_id} ground_truth does not match canonical validation"
            )
        try:
            expected_parsed = _jsonable(
                parse_prediction(response, example), f"prediction {sample_id} parsed_prediction"
            )
            expected_parse_failed = False
        except (TypeError, ValueError):
            expected_parsed = None
            expected_parse_failed = True
        if (
            record.get("parse_failed") is not expected_parse_failed
            or record.get("parsed_prediction") != expected_parsed
        ):
            raise ValidationArtifactError(
                f"prediction {sample_id} parser result does not match stored record"
            )
        records[sample_id] = record
    return records


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationArtifactError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationArtifactError(f"{label} must be a JSON object: {path}")
    return value


def _load_completed(
    output_dir: Path,
    identity: ValidationIdentity,
    *,
    examples: Sequence[AdapterExample],
    parse_prediction: Callable[[str, AdapterExample], Any],
) -> ValidationArtifact | None:
    marker_path = output_dir / VALIDATION_COMPLETE
    if not marker_path.is_file():
        return None
    marker_data = _load_json_object(marker_path, "validation marker")
    metrics_data = _load_json_object(output_dir / VALIDATION_METRICS, "validation metrics")
    marker_data.setdefault("seed", identity.seed)
    metrics_data.setdefault("seed", identity.seed)
    marker = ValidationCompleteMarker(
        **{key: marker_data[key] for key in ValidationCompleteMarker.__dataclass_fields__ if key in marker_data}
    )
    metrics = ValidationMetrics(
        **{key: metrics_data[key] for key in ValidationMetrics.__dataclass_fields__ if key in metrics_data}
    )
    config_data = _load_json_object(output_dir / VALIDATION_CONFIG, "validation config")
    expected_config = asdict(identity)
    if any(field in config_data and config_data.get(field) != value for field, value in expected_config.items()):
        raise ValidationArtifactError(f"completed validation config mismatch: {output_dir}")
    for field in (
        "task",
        "checkpoint_dir",
        "epoch",
        "learning_rate",
        "seed",
        "canonical_split",
        "validation_scope",
    ):
        expected = getattr(identity, field)
        if getattr(marker, field) != expected or getattr(metrics, field) != expected:
            raise ValidationArtifactError(
                f"completed validation {field} mismatch: {output_dir}"
            )
    examples_by_id = {example.sample_id: example for example in examples}
    records = _read_prediction_records(
        output_dir / VALIDATION_PREDICTIONS,
        identity=identity,
        examples_by_id=examples_by_id,
        parse_prediction=parse_prediction,
    )
    if set(records) != set(examples_by_id):
        raise ValidationArtifactError(
            f"completed validation does not cover canonical validation IDs: {output_dir}"
        )
    if prediction_index_manifest(output_dir / VALIDATION_PREDICTIONS) != validation_index_manifest(examples):
        raise ValidationArtifactError(
            f"completed validation prediction order/identity mismatch: {output_dir}"
        )
    validate_checkpoint_files(Path(identity.checkpoint_dir))
    return ValidationArtifact(output_dir, identity, metrics, marker)


def run_validation(
    *,
    adapter: DatasetAdapter,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    identity: ValidationIdentity,
    predictor: Callable[[AdapterExample], str],
    should_interrupt: Callable[[], bool] | None = None,
) -> ValidationArtifact:
    output = reject_test_path(output_dir, "validation.output_dir")
    checkpoint = reject_test_path(checkpoint_dir, "validation.checkpoint_dir")
    if checkpoint != Path(identity.checkpoint_dir).expanduser().resolve():
        raise ValidationArtifactError("validation config checkpoint path does not match checkpoint_dir")
    validate_checkpoint_files(checkpoint)
    source_split = adapter.config.splits.validation.source_split.casefold()
    if source_split in {"test", "evaluation"}:
        raise ValidationArtifactError(f"validation source split is forbidden: {source_split}")

    examples = adapter.materialize_split(CanonicalSplit.VALIDATION)
    examples_by_id = {example.sample_id: example for example in examples}
    if not examples or len(examples_by_id) != len(examples):
        raise ValidationArtifactError("canonical validation sample IDs are empty or not unique")
    existing = _load_completed(
        output,
        identity,
        examples=examples,
        parse_prediction=adapter.parse_example_prediction,
    )
    if existing is not None:
        return existing

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / VALIDATION_CONFIG
    if config_path.exists():
        if _load_json_object(config_path, "validation config") != asdict(identity):
            raise ValidationArtifactError(f"validation config drift: {output}")
    else:
        atomic_write_json(config_path, asdict(identity))
    predictions_path = output / VALIDATION_PREDICTIONS
    records = _read_prediction_records(
        predictions_path,
        identity=identity,
        examples_by_id=examples_by_id,
        parse_prediction=adapter.parse_example_prediction,
    )

    for example in examples:
        if example.sample_id in records:
            continue
        if should_interrupt is not None and should_interrupt():
            raise ValidationInterrupted("validation interrupted before the next sample")
        response = predictor(example)
        if not isinstance(response, str):
            raise ValidationArtifactError(
                f"predictor response for {example.sample_id} must be a string"
            )
        try:
            parsed = adapter.parse_example_prediction(response, example)
            parse_failed = False
        except (TypeError, ValueError):
            parsed = None
            parse_failed = True
        record = {
            "task": identity.task,
            "canonical_split": CanonicalSplit.VALIDATION.value,
            "sample_id": example.sample_id,
            "source_id": example.source_id,
            "epoch": identity.epoch,
            "learning_rate": identity.learning_rate,
            "seed": identity.seed,
            "response_text": response,
            "parsed_prediction": _jsonable(parsed, "parsed_prediction"),
            "parse_failed": parse_failed,
            "ground_truth": _jsonable(example.ground_truth, "ground_truth"),
        }
        _append_jsonl(predictions_path, record)
        records[example.sample_id] = record
        if should_interrupt is not None and should_interrupt():
            raise ValidationInterrupted(
                f"validation interrupted after durable sample {example.sample_id}"
            )

    if set(records) != set(examples_by_id):
        raise ValidationArtifactError("validation predictions do not cover the canonical split")
    validation_predictions = tuple(
        ValidationPrediction(
            sample_id=example.sample_id,
            ground_truth=example.ground_truth,
            parsed_prediction=records[example.sample_id]["parsed_prediction"],
            parse_failed=bool(records[example.sample_id]["parse_failed"]),
        )
        for example in examples
    )
    result = adapter.compute_validation_metric(validation_predictions)
    metrics = ValidationMetrics(
        task=identity.task,
        checkpoint_dir=identity.checkpoint_dir,
        epoch=identity.epoch,
        learning_rate=identity.learning_rate,
        seed=identity.seed,
        metric_name=adapter.config.validation.metric.value,
        direction=adapter.config.validation.direction.value,
        primary_metric=float(result.primary_metric),
        parse_failure_rate=float(result.parse_failure_rate),
        details=_jsonable(dict(result.details), "validation details"),
        validation_scope=identity.validation_scope,
    )
    atomic_write_json(output / VALIDATION_METRICS, asdict(metrics))
    marker = ValidationCompleteMarker(
        task=identity.task,
        checkpoint_dir=identity.checkpoint_dir,
        epoch=identity.epoch,
        learning_rate=identity.learning_rate,
        seed=identity.seed,
        validation_scope=identity.validation_scope,
    )
    atomic_write_json(output / VALIDATION_COMPLETE, asdict(marker))
    return ValidationArtifact(output, identity, metrics, marker)


def resume_validation(**kwargs) -> ValidationArtifact:
    return run_validation(**kwargs)


class BackendValidationPredictor:

    def __init__(
        self,
        backend: Any,
        model: Any,
        processor: Any,
        generation: Mapping[str, Any] = GENERATION_CONFIG,
    ) -> None:
        self.backend = backend
        self.model = model
        self.processor = processor
        self.generation = dict(generation)
        if (
            set(self.generation) != {"max_new_tokens", "do_sample"}
            or type(self.generation.get("max_new_tokens")) is not int
            or self.generation["max_new_tokens"] not in {128, 1024, 2048}
            or self.generation.get("do_sample") is not False
        ):
            raise ValueError("validation generation must use configured greedy tokens")

    def __call__(self, example: AdapterExample) -> str:
        import torch

        inputs = self.backend.prepare_generation_inputs(
            self.processor, example, add_generation_prompt=True
        )
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **self.generation)
        finally:
            if was_training:
                self.model.train()
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        decoder = getattr(self.processor, "batch_decode", None)
        if not callable(decoder):
            decoder = self.processor.tokenizer.batch_decode
        return decoder(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class QwenValidationPredictor(BackendValidationPredictor):

    def __init__(
        self,
        model: Any,
        processor: Any,
        generation: Mapping[str, Any] = GENERATION_CONFIG,
    ) -> None:
        super().__init__(get_model_backend("qwen3_vl"), model, processor, generation)
