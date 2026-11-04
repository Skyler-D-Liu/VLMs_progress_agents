"""Cowc adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter, count_r2_result, require_nonnegative_int


class CowcAdapter(PreparedTaskAdapter):
    TASK = "cowc"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        count = require_nonnegative_int(record["ground_truth"], "count")
        expected_prompt = self.semantics.module.get_user_prompt()
        expected_answer = self.semantics.module.format_count_answer(count)
        if record["user_prompt"] != expected_prompt or record["assistant_answer"] != expected_answer:
            raise AdapterValidationError("COWC prompt or answer differs from authority")

    def parse_prediction(self, text: str) -> int:
        value = int(self.semantics.module.parse_model_response(text))
        if value < 0:
            raise ValueError("COWC response is not parseable")
        return value

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        return count_r2_result(predictions)
