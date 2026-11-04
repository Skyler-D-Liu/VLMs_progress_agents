"""Fpcd adapter."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter


class FPCDAdapter(PreparedTaskAdapter):
    TASK = "fpcd"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        answer = str(record["ground_truth"]).upper()
        if answer not in self.semantics.module.OPTION_TO_CLASS:
            raise AdapterValidationError("FPCD ground truth is not A-E")
        if record["user_prompt"] != self.semantics.module.USER_PROMPT:
            raise AdapterValidationError("FPCD prompt differs from authority")
        if record["assistant_answer"] != self.semantics.module.format_answer(answer):
            raise AdapterValidationError("FPCD answer differs from authority")
    def parse_prediction(self, text: str) -> str:
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("FPCD prediction must be a JSON object") from exc
        if not isinstance(value, dict) or set(value) != {"answer"}:
            raise ValueError('FPCD prediction must contain exactly {"answer": ...}')
        answer = value["answer"]
        if not isinstance(answer, str):
            raise ValueError("FPCD answer must be a string")
        normalized = answer.strip().upper()
        if normalized not in self.semantics.module.OPTION_TO_CLASS:
            raise ValueError("FPCD answer must be one option A-E")
        return normalized

    def compute_validation_metric(
        self, predictions: Sequence[ValidationPrediction]
    ) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError("validation predictions are empty")
        correct = sum(
            not prediction.parse_failed
            and str(prediction.parsed_prediction) == str(prediction.ground_truth)
            for prediction in predictions
        )
        return ValidationResult(
            primary_metric=correct / len(predictions),
            parse_failure_rate=self._failure_rate(predictions),
            details={"correct": correct, "denominator": len(predictions)},
        )
