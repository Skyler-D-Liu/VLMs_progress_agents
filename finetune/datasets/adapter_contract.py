"""Adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

from finetune.config.schema import DatasetConfig


RawRecord = TypeVar("RawRecord")
ParsedPrediction = TypeVar("ParsedPrediction")


class CanonicalSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AdapterValidationError(ValueError):
    pass


class SplitLeakageError(AdapterValidationError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise AdapterValidationError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise AdapterValidationError(f"{self.role} message content must be non-empty")


@dataclass(frozen=True)
class AdapterExample:
    sample_id: str
    source_id: str
    canonical_split: CanonicalSplit
    messages: tuple[ChatMessage, ...]
    image_paths: tuple[Path, ...]
    ground_truth: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationPrediction:
    sample_id: str
    ground_truth: Any
    parsed_prediction: Any
    parse_failed: bool


@dataclass(frozen=True)
class ValidationResult:
    primary_metric: float
    parse_failure_rate: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.parse_failure_rate <= 1.0:
            raise AdapterValidationError("parse_failure_rate must be between 0 and 1")


class DatasetAdapter(ABC, Generic[RawRecord, ParsedPrediction]):

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._materialized: dict[CanonicalSplit, tuple[AdapterExample, ...]] = {}

    @abstractmethod
    def iter_raw_records(self, split: CanonicalSplit) -> Iterable[RawRecord]:
        pass

    @abstractmethod
    def source_id(self, record: RawRecord, source_split: str, index: int) -> str:
        pass

    @abstractmethod
    def build_example(self, record: RawRecord, split: CanonicalSplit) -> AdapterExample:
        pass

    @abstractmethod
    def parse_prediction(self, text: str) -> ParsedPrediction:
        pass

    def parse_example_prediction(self, text: str, example: AdapterExample) -> ParsedPrediction:
        return self.parse_prediction(text)

    @abstractmethod
    def compute_validation_metric(
        self, predictions: Sequence[ValidationPrediction]
    ) -> ValidationResult:
        pass

    def _split_config(self, split: CanonicalSplit):
        return getattr(self.config.splits, split.value)

    @property
    def materialized_splits(self) -> frozenset[CanonicalSplit]:
        return frozenset(self._materialized)

    def _validate_example(self, example: AdapterExample, split: CanonicalSplit) -> None:
        if not example.sample_id.strip():
            raise AdapterValidationError("sample_id must be non-empty")
        if not example.source_id.strip():
            raise AdapterValidationError(f"{example.sample_id}: source_id must be non-empty")
        if example.canonical_split is not split:
            raise AdapterValidationError(
                f"{example.sample_id}: emitted split {example.canonical_split.value}, expected {split.value}"
            )
        roles = tuple(message.role for message in example.messages)
        if roles not in {("system", "user", "assistant"), ("user", "assistant")}:
            raise AdapterValidationError(
                f"{example.sample_id}: messages must be [system, user, assistant] or [user, assistant]"
            )
        if not example.messages[-1].content.strip():
            raise AdapterValidationError(f"{example.sample_id}: assistant answer must be non-empty")
        if len(example.image_paths) != self.config.input.image_count:
            raise AdapterValidationError(
                f"{example.sample_id}: expected {self.config.input.image_count} image(s), "
                f"got {len(example.image_paths)}"
            )
        for image_index, image_path in enumerate(example.image_paths):
            if not Path(image_path).is_file():
                raise AdapterValidationError(
                    f"{example.sample_id}: missing image_paths.{image_index}: {Path(image_path).resolve()}"
                )

    def materialize_split(self, split: CanonicalSplit) -> tuple[AdapterExample, ...]:
        if split in self._materialized:
            return self._materialized[split]

        split_config = self._split_config(split)
        examples = []
        sample_ids = set()
        source_ids = set()
        for index, record in enumerate(self.iter_raw_records(split)):
            expected_source_id = self.source_id(record, split_config.source_split, index)
            example = self.build_example(record, split)
            self._validate_example(example, split)
            if example.source_id != expected_source_id:
                raise AdapterValidationError(
                    f"{example.sample_id}: source_id {example.source_id!r} does not match "
                    f"adapter identity {expected_source_id!r}"
                )
            if example.sample_id in sample_ids:
                raise AdapterValidationError(f"duplicate sample_id in {split.value}: {example.sample_id}")
            if example.source_id in source_ids:
                raise AdapterValidationError(f"duplicate source_id in {split.value}: {example.source_id}")
            sample_ids.add(example.sample_id)
            source_ids.add(example.source_id)
            examples.append(example)

        materialized = tuple(examples)
        self._materialized[split] = materialized
        return materialized

    def validate_split_isolation(self) -> dict[CanonicalSplit, tuple[AdapterExample, ...]]:
        materialized = {split: self.materialize_split(split) for split in CanonicalSplit}
        identities = {
            split: {example.source_id for example in examples}
            for split, examples in materialized.items()
        }
        split_pairs = (
            (CanonicalSplit.TRAIN, CanonicalSplit.VALIDATION),
            (CanonicalSplit.TRAIN, CanonicalSplit.TEST),
            (CanonicalSplit.VALIDATION, CanonicalSplit.TEST),
        )
        for left, right in split_pairs:
            overlap = identities[left] & identities[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise SplitLeakageError(
                    f"source_id overlap between {left.value} and {right.value}: {preview}"
                )
        return materialized

    def build_training_records(self) -> list[dict[str, Any]]:
        examples = self.materialize_split(CanonicalSplit.TRAIN)
        records = []
        for example in examples:
            records.append(
                {
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in example.messages
                    ],
                    "images": [str(path) for path in example.image_paths],
                    "meta": {
                        "sample_id": example.sample_id,
                        "source_id": example.source_id,
                        "canonical_split": example.canonical_split.value,
                        **dict(example.metadata),
                    },
                }
            )
        return records

    def evaluate_validation(self, response_text_by_sample_id: Mapping[str, str]) -> ValidationResult:
        examples = self.materialize_split(CanonicalSplit.VALIDATION)
        expected_ids = {example.sample_id for example in examples}
        if set(response_text_by_sample_id) != expected_ids:
            missing = sorted(expected_ids - set(response_text_by_sample_id))
            extra = sorted(set(response_text_by_sample_id) - expected_ids)
            raise AdapterValidationError(
                f"validation responses must match validation sample ids; missing={missing}, extra={extra}"
            )

        predictions = []
        for example in examples:
            text = response_text_by_sample_id[example.sample_id]
            try:
                parsed = self.parse_example_prediction(text, example)
                parse_failed = False
            except (TypeError, ValueError):
                parsed = None
                parse_failed = True
            predictions.append(
                ValidationPrediction(
                    sample_id=example.sample_id,
                    ground_truth=example.ground_truth,
                    parsed_prediction=parsed,
                    parse_failed=parse_failed,
                )
            )
        return self.compute_validation_metric(predictions)
