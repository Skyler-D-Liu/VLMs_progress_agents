"""Pattern net adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter


class PatternNetAdapter(PreparedTaskAdapter):
    TASK = "pattern-net"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        category = str(record["ground_truth"])
        expected_prompt = self.semantics.module.get_user_prompt()
        expected_answer = self.semantics.module.format_category_answer(category)
        if category not in self.semantics.module.CATEGORIES:
            raise AdapterValidationError("PatternNet ground truth is outside canonical classes")
        if record["user_prompt"] != expected_prompt or record["assistant_answer"] != expected_answer:
            raise AdapterValidationError("PatternNet prompt or answer differs from authority")

    def parse_prediction(self, text: str) -> str:
        value = str(self.semantics.module.parse_model_response(text))
        if value == "Refused" or value not in self.semantics.module.CATEGORIES:
            raise ValueError("PatternNet response is not a canonical category")
        return value

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
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
            details={"correct": correct},
        )
