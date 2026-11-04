"""Aerial animal adapter."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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


ANIMAL_KEYS = ("elephants", "zebras", "giraffes")
RAW_SPLIT_BY_CANONICAL = {
    CanonicalSplit.TRAIN: "train",
    CanonicalSplit.VALIDATION: "val",
    CanonicalSplit.TEST: "test",
}
RECORD_FILE_BY_CANONICAL = {
    CanonicalSplit.TRAIN: "train.jsonl",
    CanonicalSplit.VALIDATION: "validation.jsonl",
    CanonicalSplit.TEST: "test.jsonl",
}
_AUTHORITATIVE_CONSTANTS = {
    "ANIMAL_RESPONSE_KEYS",
    "SPECIES_TO_RESPONSE_KEY",
    "SYSTEM_MESSAGE",
    "USER_PROMPT",
}
_AUTHORITATIVE_FUNCTIONS = {
    "species_counts_from_objects",
    "format_count_answer",
    "extract_json_candidate",
    "coerce_count_value",
    "parse_count_response",
}


@dataclass(frozen=True)
class AerialAnimalSemantics:
    authoritative_script: Path
    system_message: str
    user_prompt: str
    species_counts_from_objects: Callable[[Any], Optional[Dict[str, int]]]
    format_count_answer: Callable[[Dict[str, int]], str]
    extract_json_candidate: Callable[[str], str]
    parse_count_response: Callable[[str], Tuple[int, Optional[Dict[str, int]]]]


def load_authoritative_semantics(config: DatasetConfig) -> AerialAnimalSemantics:
    script = config.prompt.authoritative_script.expanduser().absolute()
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))

    constants: dict[str, Any] = {}
    function_nodes: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            for name in names:
                if name in _AUTHORITATIVE_CONSTANTS:
                    constants[name] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name in _AUTHORITATIVE_FUNCTIONS:
            function_nodes.append(node)

    required_constants = {
        "ANIMAL_RESPONSE_KEYS",
        "SPECIES_TO_RESPONSE_KEY",
        config.prompt.system_symbol,
        config.prompt.user_symbol,
    }
    missing_constants = sorted(required_constants - set(constants))
    missing_functions = sorted(
        _AUTHORITATIVE_FUNCTIONS - {node.name for node in function_nodes}
    )
    if missing_constants or missing_functions:
        raise AdapterValidationError(
            "authoritative Aerial Animal symbols are missing: "
            f"constants={missing_constants}, functions={missing_functions}"
        )

    namespace: dict[str, Any] = {
        "json": json,
        "math": math,
        "re": re,
        "Optional": Optional,
        "Dict": Dict,
        "Tuple": Tuple,
        **constants,
    }
    function_module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(function_module)
    exec(compile(function_module, str(script), "exec"), namespace)

    response_keys = tuple(namespace["ANIMAL_RESPONSE_KEYS"])
    if response_keys != ANIMAL_KEYS:
        raise AdapterValidationError(
            f"authoritative animal response keys changed: {response_keys}"
        )
    system_message = namespace[config.prompt.system_symbol]
    user_prompt = namespace[config.prompt.user_symbol]
    if not isinstance(system_message, str) or not isinstance(user_prompt, str):
        raise AdapterValidationError("authoritative prompts must be strings")
    return AerialAnimalSemantics(
        authoritative_script=script,
        system_message=system_message,
        user_prompt=user_prompt,
        species_counts_from_objects=namespace["species_counts_from_objects"],
        format_count_answer=namespace["format_count_answer"],
        extract_json_candidate=namespace["extract_json_candidate"],
        parse_count_response=namespace["parse_count_response"],
    )


def snapshot_path(config: DatasetConfig) -> Path:
    slug = "datasets--" + config.dataset.id.replace("/", "--")
    snapshots = config.paths.raw_dataset_root / slug / "snapshots"
    candidates = sorted(path.resolve() for path in snapshots.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one cached snapshot for {config.dataset.id}, found {len(candidates)}"
        )
    return candidates[0]


def parquet_files(config: DatasetConfig, raw_split: str) -> tuple[Path, ...]:
    root = snapshot_path(config)
    files = tuple(sorted((root / "data").glob(f"{raw_split}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no local parquet shards for raw split {raw_split!r}: {root / 'data'}"
        )
    return files


def counts_from_objects(
    semantics: AerialAnimalSemantics, objects: Any
) -> dict[str, int]:
    counts = semantics.species_counts_from_objects(objects)
    if counts is None:
        raise AdapterValidationError("objects.categories is required for Aerial Animal")
    ordered = {key: int(counts.get(key, 0)) for key in ANIMAL_KEYS}
    if any(value < 0 for value in ordered.values()):
        raise AdapterValidationError(f"animal counts must be non-negative: {ordered}")
    if not isinstance(objects, Mapping):
        raise AdapterValidationError("objects must be a mapping")
    categories = objects.get("categories")
    bboxes = objects.get("bbox")
    if categories is None or bboxes is None or len(categories) != len(bboxes):
        raise AdapterValidationError(
            "objects.categories and objects.bbox must exist with equal lengths"
        )
    if sum(ordered.values()) != len(bboxes):
        raise AdapterValidationError(
            "authoritative species counts do not cover every annotated bbox"
        )
    return ordered


class AerialAnimalAdapter(DatasetAdapter[dict[str, Any], dict[str, int]]):

    def __init__(
        self,
        config: DatasetConfig,
        *,
        records_dir: str | Path | None = None,
        verify_preparation: bool | None = None,
    ) -> None:
        super().__init__(config)
        if config.task != "aerial-animal":
            raise AdapterValidationError(
                f"AerialAnimalAdapter requires task='aerial-animal', got {config.task!r}"
            )
        self.semantics = load_authoritative_semantics(config)
        self.records_dir = (
            Path(records_dir).expanduser().resolve()
            if records_dir is not None
            else (config.paths.prepared_dir / "records").resolve()
        )
        self._manifest_entries_by_sample_id: dict[str, dict[str, Any]] = {}
        self._verified_images: set[Path] = set()
        if verify_preparation is None:
            verify_preparation = records_dir is None
        if verify_preparation:
            self._load_preparation_contract()

    def _load_preparation_contract(self) -> None:
        manifest_dir = self.config.paths.manifest_dir.expanduser().resolve()
        marker_path = manifest_dir / "PREPARATION_COMPLETE.json"
        split_manifest_path = manifest_dir / "split_manifest.json"
        prompt_manifest_path = manifest_dir / "prompt_manifest.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterValidationError(
                f"Aerial Animal preparation contract is missing or invalid under {manifest_dir}"
            ) from exc
        if marker.get("task") != self.config.task:
            raise AdapterValidationError("preparation marker task mismatch")
        if marker.get("training_started") is not False or marker.get("test_inference_run") is not False:
            raise AdapterValidationError("preparation marker incorrectly reports training or test inference")

        prompt_checks = {
            "system_message": self.semantics.system_message,
            "user_prompt": self.semantics.user_prompt,
        }
        for field, expected in prompt_checks.items():
            if prompt_manifest.get(field) != expected:
                raise AdapterValidationError(f"prompt manifest drift for {field}")
        split_checks = {
            "task": self.config.task,
            "split_strategy": self.config.splits.strategy.value,
            "split_seed": self.config.splits.seed,
            "test_data_used_for_tuning": False,
        }
        for field, expected in split_checks.items():
            if split_manifest.get(field) != expected:
                raise AdapterValidationError(f"split manifest drift for {field}")
        dataset_identity = split_manifest.get("dataset", {})
        if (
            dataset_identity.get("id") != self.config.dataset.id
            or dataset_identity.get("config") != self.config.dataset.config
        ):
            raise AdapterValidationError("split manifest dataset identity mismatch")

        entries = split_manifest.get("entries")
        if not isinstance(entries, list):
            raise AdapterValidationError("split manifest entries must be a list")
        sample_ids: set[str] = set()
        source_ids: set[str] = set()
        split_names = {"train", "validation", "test"}
        for entry in entries:
            if not isinstance(entry, dict):
                raise AdapterValidationError("split manifest entries must be objects")
            sample_id = entry.get("sample_id")
            source_id = entry.get("source_id")
            canonical_split = entry.get("canonical_split")
            if not isinstance(sample_id, str) or sample_id in sample_ids:
                raise AdapterValidationError(f"duplicate/invalid manifest sample_id: {sample_id!r}")
            if not isinstance(source_id, str) or source_id in source_ids:
                raise AdapterValidationError(f"duplicate/invalid manifest source_id: {source_id!r}")
            if canonical_split not in split_names:
                raise AdapterValidationError(
                    f"invalid manifest canonical split: {canonical_split!r}"
                )
            sample_ids.add(sample_id)
            source_ids.add(source_id)
            self._manifest_entries_by_sample_id[sample_id] = entry

    def _record_path(self, split: CanonicalSplit) -> Path:
        return self.records_dir / RECORD_FILE_BY_CANONICAL[split]

    @property
    def verified_image_count(self) -> int:
        return len(self._verified_images)

    def iter_raw_records(self, split: CanonicalSplit) -> Iterable[dict[str, Any]]:
        path = self._record_path(split)
        if not path.is_file():
            raise FileNotFoundError(
                f"prepared Aerial Animal records are missing: {path}; "
                "submit the required dependencies first"
            )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AdapterValidationError(
                        f"invalid prepared record at {path}:{line_number}"
                    ) from exc
                manifest_record = self._manifest_entries_by_sample_id.get(
                    record.get("sample_id")
                )
                if self._manifest_entries_by_sample_id and manifest_record != record:
                    raise AdapterValidationError(
                        f"prepared record differs from split manifest at {path}:{line_number}"
                    )
                yield record

    def source_id(self, record: dict[str, Any], source_split: str, index: int) -> str:
        if record.get("raw_split") != source_split:
            raise AdapterValidationError(
                f"record raw_split={record.get('raw_split')!r}, expected {source_split!r}"
            )
        if record.get("raw_index") != index:
            raise AdapterValidationError(
                f"native split order changed at {source_split}[{index}]: "
                f"raw_index={record.get('raw_index')!r}"
            )
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise AdapterValidationError(f"missing source_id at {source_split}[{index}]")
        semantic_source_id = f"{self.config.dataset.id}:{source_split}:{index:08d}"
        legacy_suffix = f":{source_split}:{index:08d}"
        if source_id != semantic_source_id and not (
            source_id.startswith(f"{self.config.dataset.id}@")
            and source_id.endswith(legacy_suffix)
        ):
            raise AdapterValidationError(
                f"source_id drift at {source_split}[{index}]: {source_id!r}"
            )
        return source_id

    def build_example(
        self, record: dict[str, Any], split: CanonicalSplit
    ) -> AdapterExample:
        expected_raw_split = RAW_SPLIT_BY_CANONICAL[split]
        if record.get("raw_split") != expected_raw_split:
            raise AdapterValidationError(
                f"{record.get('sample_id')}: expected raw split {expected_raw_split}"
            )
        canonical_name = split.value
        raw_index = record.get("raw_index")
        if type(raw_index) is not int or raw_index < 0:
            raise AdapterValidationError(
                f"{record.get('sample_id')}: raw_index must be non-negative"
            )
        expected_sample_id = f"aerial-animal:{canonical_name}:{raw_index:08d}"
        if record.get("canonical_split") != canonical_name:
            raise AdapterValidationError(
                f"{record.get('sample_id')}: canonical split field drift"
            )
        if record.get("sample_id") != expected_sample_id:
            raise AdapterValidationError(
                f"sample_id drift: {record.get('sample_id')!r} != {expected_sample_id!r}"
            )
        if record.get("dataset_id") != self.config.dataset.id:
            raise AdapterValidationError(f"{expected_sample_id}: dataset ID drift")
        if record.get("dataset_config") != self.config.dataset.config:
            raise AdapterValidationError(f"{expected_sample_id}: dataset config drift")
        counts = record.get("ground_truth")
        if (
            not isinstance(counts, dict)
            or len(counts) != len(ANIMAL_KEYS)
            or set(counts) != set(ANIMAL_KEYS)
        ):
            raise AdapterValidationError(
                f"{record.get('sample_id')}: ground_truth keys must be {ANIMAL_KEYS}"
            )
        if any(type(counts[key]) is not int for key in ANIMAL_KEYS):
            raise AdapterValidationError(
                f"{record.get('sample_id')}: animal counts must be JSON integers"
            )
        normalized_counts = {key: counts[key] for key in ANIMAL_KEYS}
        if any(value < 0 for value in normalized_counts.values()):
            raise AdapterValidationError(
                f"{record.get('sample_id')}: negative animal count"
            )
        image_path = Path(record["image_path"]).expanduser()
        if not image_path.is_absolute():
            raise AdapterValidationError(f"{expected_sample_id}: image path must be absolute")
        prepared_image_root = (self.config.paths.prepared_dir / "images").resolve()
        try:
            image_path.resolve().relative_to(prepared_image_root)
        except ValueError as exc:
            raise AdapterValidationError(
                f"{expected_sample_id}: image path escapes prepared image root"
            ) from exc
        if image_path.parent.name != expected_raw_split or image_path.stem != f"{raw_index:08d}":
            raise AdapterValidationError(
                f"{expected_sample_id}: image path does not preserve native split/index"
            )
        width = record.get("width")
        height = record.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise AdapterValidationError(f"{expected_sample_id}: invalid image dimensions")
        if record.get("pixels") != width * height:
            raise AdapterValidationError(f"{expected_sample_id}: pixel count mismatch")
        if record.get("total_count") != sum(normalized_counts.values()):
            raise AdapterValidationError(f"{expected_sample_id}: total count mismatch")
        if record.get("bbox_count") != record.get("total_count"):
            raise AdapterValidationError(f"{expected_sample_id}: bbox/count mismatch")
        resolved_image = image_path.resolve()
        if resolved_image not in self._verified_images:
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise AdapterValidationError(
                    f"{expected_sample_id}: prepared image is unreadable"
                ) from exc
            self._verified_images.add(resolved_image)
        answer = self.semantics.format_count_answer(normalized_counts)
        return AdapterExample(
            sample_id=str(record["sample_id"]),
            source_id=str(record["source_id"]),
            canonical_split=split,
            messages=(
                ChatMessage("system", self.semantics.system_message),


                ChatMessage("user", f"<image>\n{self.semantics.user_prompt}"),
                ChatMessage("assistant", answer),
            ),
            image_paths=(image_path,),
            ground_truth=normalized_counts,
            metadata={
                "dataset_id": self.config.dataset.id,
                "raw_split": record["raw_split"],
                "raw_index": record["raw_index"],
                "width": record["width"],
                "height": record["height"],
                "total_count": sum(normalized_counts.values()),
            },
        )

    def parse_prediction(self, text: str) -> dict[str, int]:
        parsed_total, parsed_counts = self.semantics.parse_count_response(text)
        if parsed_total == -1 or parsed_counts is None:
            raise ValueError("authoritative Aerial Animal parser rejected the response")
        return {key: int(parsed_counts[key]) for key in ANIMAL_KEYS}

    def compute_validation_metric(
        self, predictions: Sequence[ValidationPrediction]
    ) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError("Aerial Animal validation predictions are empty")
        absolute_errors: list[float] = []
        true_totals: list[int] = []
        predicted_totals: list[int] = []
        parse_failures = 0
        for prediction in predictions:
            ground_truth = prediction.ground_truth
            if not isinstance(ground_truth, Mapping):
                raise AdapterValidationError(
                    f"{prediction.sample_id}: ground truth must be a count mapping"
                )
            true_total = sum(int(ground_truth[key]) for key in ANIMAL_KEYS)
            if prediction.parse_failed:
                predicted_total = 0
                parse_failures += 1
            else:
                parsed = prediction.parsed_prediction
                if not isinstance(parsed, Mapping):
                    raise AdapterValidationError(
                        f"{prediction.sample_id}: parsed prediction must be a mapping"
                    )
                predicted_total = sum(int(parsed[key]) for key in ANIMAL_KEYS)
            absolute_errors.append(abs(true_total - predicted_total))
            true_totals.append(true_total)
            predicted_totals.append(predicted_total)
        prediction_total = len(predictions)
        # Primary metric of the paper: r^2 between predicted and ground-truth
        # total animal counts (MAE kept as a diagnostic only).
        from .prepared_task_adapter import squared_pearson_r2

        return ValidationResult(
            primary_metric=squared_pearson_r2(true_totals, predicted_totals),
            parse_failure_rate=parse_failures / prediction_total,
            details={
                "metric": "total_count_r2",
                "selection_split": "validation",
                "mae": sum(absolute_errors) / prediction_total,
                "sum_absolute_error": sum(absolute_errors),
                "parse_failure_policy": "zero_prediction",
                "authoritative_parser": str(self.semantics.authoritative_script),
            },
        )
