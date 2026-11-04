"""Neon tree adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter, count_r2_result, require_nonnegative_int


class NeonTreeAdapter(PreparedTaskAdapter):
    TASK = "neon-tree"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        count = require_nonnegative_int(record["ground_truth"], "count")
        if record["user_prompt"] != self.semantics.module.USER_PROMPT:
            raise AdapterValidationError("Neon Tree user prompt differs from authority")
        if record["assistant_answer"] != str(count):
            raise AdapterValidationError("Neon Tree assistant answer must be the integer only")

    def parse_prediction(self, text: str) -> int:
        value = int(self.semantics.module.parse_model_response_to_count(text))
        if value < 0:
            raise ValueError("Neon Tree response is not parseable")
        return value

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        return count_r2_result(predictions)
