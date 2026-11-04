"""Prepared task adapter."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from finetune.config.schema import DatasetConfig
from .adapter_contract import (
    AdapterExample,
    AdapterValidationError,
    CanonicalSplit,
    ChatMessage,
    DatasetAdapter,
    ValidationPrediction,
    ValidationResult,
)


IMAGE_PLACEHOLDER = "<image>"


def ensure_image_placeholders(user_prompt: str, image_count: int) -> str:
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise AdapterValidationError("prepared user prompt must be non-empty text")
    if image_count <= 0:
        raise AdapterValidationError("image_count must be positive")
    existing = user_prompt.count(IMAGE_PLACEHOLDER)
    if existing not in {0, image_count}:
        raise AdapterValidationError(
            f"prepared user prompt contains {existing} {IMAGE_PLACEHOLDER} placeholders; "
            f"expected {image_count}"
        )
    if existing == image_count:
        return user_prompt
    return (IMAGE_PLACEHOLDER + "\n") * image_count + user_prompt


RECORD_FILE_BY_SPLIT = {
    CanonicalSplit.TRAIN: "train.jsonl",
    CanonicalSplit.VALIDATION: "validation.jsonl",
    CanonicalSplit.TEST: "test.jsonl",
}
COUNT_KEYS = ("count_before", "no_damage", "minor_damage", "major_damage", "destroyed")
_MODULE_CACHE: dict[Path, Any] = {}


@dataclass(frozen=True)
class AuthoritativeTaskSemantics:

    module: Any
    script: Path
    system_message: str


def load_authoritative_task_semantics(config: DatasetConfig) -> AuthoritativeTaskSemantics:
    script = config.prompt.authoritative_script.expanduser().resolve()
    module = _MODULE_CACHE.get(script)
    if module is None:
        task_dir = script.parent
        utils_dir = task_dir.parent / "utils"

        repository_root = next(
            (
                parent
                for parent in (script.parent, *script.parents)
                if (parent / "evaluation").is_dir()
            ),
            None,
        )
        shared_paths = ()
        if repository_root is not None:
            shared_paths = (
                repository_root / "evaluation" / "tasks",
                repository_root / "evaluation" / "utils",
                repository_root,
            )
        for path in (
            utils_dir,
            task_dir,
            task_dir.parent.parent,
            *shared_paths,
        ):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        module_name = "vleo_authority_" + "_".join(
            part.replace("-", "_")
            for part in (config.task, script.parent.name, script.stem)
        )
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise AdapterValidationError(f"cannot load authoritative script: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        _MODULE_CACHE[script] = module

    system = getattr(module, config.prompt.system_symbol, None)
    if not isinstance(system, str) or not system.strip():
        raise AdapterValidationError(
            f"authoritative system prompt is missing: {config.prompt.system_symbol}"
        )
    return AuthoritativeTaskSemantics(
        module=module,
        script=script,
        system_message=system,
    )


class PreparedTaskAdapter(DatasetAdapter[dict[str, Any], Any]):

    TASK = ""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        records_dir: str | Path | None = None,
        verify_preparation: bool | None = None,
    ) -> None:
        super().__init__(config)
        if config.task != self.TASK:
            raise AdapterValidationError(
                f"{type(self).__name__} requires task={self.TASK!r}, got {config.task!r}"
            )
        self.semantics = load_authoritative_task_semantics(config)
        self.records_dir = (
            Path(records_dir).expanduser().resolve()
            if records_dir is not None
            else (config.paths.prepared_dir / "records").resolve()
        )
        if verify_preparation is None:
            verify_preparation = records_dir is None
        if verify_preparation:
            self._verify_preparation_contract()

    def _verify_preparation_contract(self) -> None:
        manifest_dir = self.config.paths.manifest_dir.resolve()
        marker_path = manifest_dir / "PREPARATION_COMPLETE.json"
        split_path = manifest_dir / "split_manifest.json"
        prompt_path = manifest_dir / "prompt_manifest.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            split = json.loads(split_path.read_text(encoding="utf-8"))
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterValidationError(
                f"prepared task contract is missing or invalid: {manifest_dir}"
            ) from exc
        checks = {
            "task": self.config.task,
        }
        for field, expected in checks.items():
            if marker.get(field) != expected:
                raise AdapterValidationError(f"preparation marker drift for {field}")
        if prompt.get("system_message") != self.semantics.system_message:
            raise AdapterValidationError("authoritative system prompt changed")
        if split.get("split_seed") != self.config.splits.seed:
            raise AdapterValidationError("split manifest seed differs from config")
        if split.get("split_strategy") != self.config.splits.strategy.value:
            raise AdapterValidationError("split manifest strategy differs from config")
        if split.get("test_data_used_for_tuning") is not False:
            raise AdapterValidationError("split manifest permits test-based tuning")

    def iter_raw_records(self, split: CanonicalSplit) -> Iterable[dict[str, Any]]:
        path = self.records_dir / RECORD_FILE_BY_SPLIT[split]
        if not path.is_file():
            raise FileNotFoundError(f"prepared {self.config.task} records are missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AdapterValidationError(
                        f"invalid prepared record at {path}:{line_number}"
                    ) from exc
                if record.get("canonical_split") != split.value:
                    raise AdapterValidationError(
                        f"prepared record split mismatch at {path}:{line_number}"
                    )
                yield record

    def source_id(self, record: dict[str, Any], source_split: str, index: int) -> str:
        del index
        if record.get("raw_split") != source_split:
            raise AdapterValidationError(
                f"raw split mismatch: {record.get('raw_split')!r} != {source_split!r}"
            )
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise AdapterValidationError("prepared record has no source_id")
        return source_id

    def build_example(self, record: dict[str, Any], split: CanonicalSplit) -> AdapterExample:
        if record.get("task") != self.config.task:
            raise AdapterValidationError("prepared record task mismatch")
        image_paths = tuple(Path(path).expanduser().resolve() for path in record["image_paths"])
        user_prompt = ensure_image_placeholders(
            record["user_prompt"], len(image_paths)
        )
        messages = (
            ChatMessage("system", str(record["system_message"])),
            ChatMessage("user", user_prompt),
            ChatMessage("assistant", str(record["assistant_answer"])),
        )
        if messages[0].content != self.semantics.system_message:
            raise AdapterValidationError("prepared system prompt differs from authority")
        self.verify_record_semantics(record)
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "raw_split": record["raw_split"],
                "raw_index": record["raw_index"],
                "source_qualifier": record.get("source_qualifier"),
                "image_order": list(self.config.input.image_order),
            }
        )
        return AdapterExample(
            sample_id=str(record["sample_id"]),
            source_id=str(record["source_id"]),
            canonical_split=split,
            messages=messages,
            image_paths=image_paths,
            ground_truth=record["ground_truth"],
            metadata=metadata,
        )

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        pass

    @staticmethod
    def _failure_rate(predictions: Sequence[ValidationPrediction]) -> float:
        return sum(prediction.parse_failed for prediction in predictions) / len(predictions)


def squared_pearson_r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Squared Pearson correlation, the primary counting metric of the paper.

    Matches ``evaluation/utils/scoring.py::squared_pearson``: it is set to zero
    when undefined (fewer than two samples or constant values).
    """
    if len(actual) != len(predicted):
        raise AdapterValidationError("unpaired counting values")
    if len(actual) < 2:
        return 0.0
    mean_actual = sum(actual) / len(actual)
    mean_predicted = sum(predicted) / len(predicted)
    dx = [float(value) - mean_actual for value in actual]
    dy = [float(value) - mean_predicted for value in predicted]
    denominator = sum(x * x for x in dx) * sum(y * y for y in dy)
    if not math.isfinite(denominator) or denominator <= 0:
        return 0.0
    return min(1.0, sum(x * y for x, y in zip(dx, dy)) ** 2 / denominator)


def count_r2_result(predictions: Sequence[ValidationPrediction]) -> ValidationResult:
    """Primary validation metric for scalar counting tasks: r^2.

    Unparseable or missing counts are replaced by zero, as in the zero-shot
    full-sample counting metrics. MAE is kept as a diagnostic only.
    """
    if not predictions:
        raise AdapterValidationError("validation predictions are empty")
    actual: list[int] = []
    predicted: list[int] = []
    for prediction in predictions:
        actual.append(int(prediction.ground_truth))
        predicted.append(0 if prediction.parse_failed else int(prediction.parsed_prediction))
    errors = [abs(p - a) for a, p in zip(actual, predicted)]
    return ValidationResult(
        primary_metric=squared_pearson_r2(actual, predicted),
        parse_failure_rate=PreparedTaskAdapter._failure_rate(predictions),
        details={
            "metric": "r2",
            "mae": sum(errors) / len(errors),
            "parse_failure_policy": "zero_prediction",
        },
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def require_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value
