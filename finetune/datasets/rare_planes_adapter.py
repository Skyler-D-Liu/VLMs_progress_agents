"""Rare planes adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter, count_r2_result, require_nonnegative_int


class RarePlanesAdapter(PreparedTaskAdapter):
    TASK = "rare-planes"
    SOURCE_POOL = "real/train+test"
    RAW_SPLITS = frozenset(("train", "test"))

    def source_id(self, record: dict[str, Any], source_split: str, index: int) -> str:
        del index
        if source_split != self.SOURCE_POOL:
            raise AdapterValidationError(f"RarePlanes source pool differs: {source_split!r}")
        if record.get("raw_split") not in self.RAW_SPLITS:
            raise AdapterValidationError(
                f"RarePlanes official raw split is invalid: {record.get('raw_split')!r}"
            )
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise AdapterValidationError("RarePlanes prepared record has no source_id")
        return source_id

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        count = require_nonnegative_int(record["ground_truth"], "aircraft count")
        if record["user_prompt"] != self.semantics.module.USER_PROMPT:
            raise AdapterValidationError("RarePlanes prompt differs from authority")
        if record["assistant_answer"] != self.semantics.module.format_count_answer(count):
            raise AdapterValidationError("RarePlanes answer differs from authority")
        metadata = record.get("metadata") or {}
        if metadata.get("synthetic_imagery_used") is not False:
            raise AdapterValidationError("RarePlanes prepared record permits synthetic imagery")

    def parse_prediction(self, text: str) -> int:
        value = int(self.semantics.module.parse_count_response(text)[0])
        if value < 0:
            raise ValueError("RarePlanes response is not parseable")
        return value

    def compute_validation_metric(
        self, predictions: Sequence[ValidationPrediction]
    ) -> ValidationResult:
        return count_r2_result(predictions)
