"""Dior rsvg recognition adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter


class DiorRsvgRecognitionAdapter(PreparedTaskAdapter):

    TASK = "dior-rsvg-recognition"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        category = str(record["ground_truth"])
        expected_prompt = self.semantics.module.get_user_prompt()
        expected_answer = self.semantics.module.format_category_answer(category)
        categories = getattr(self.semantics.module, "CATEGORIES", ())
        if category not in categories:
            raise AdapterValidationError("DIOR-RSVG recognition category is not canonical")
        if record["user_prompt"] != expected_prompt or record["assistant_answer"] != expected_answer:
            raise AdapterValidationError("DIOR-RSVG recognition prompt or answer differs from authority")

    def parse_prediction(self, text: str) -> str:
        value = self.semantics.module.parse_category_response(text)
        if value == "Refused" or value not in self.semantics.module.CATEGORIES:
            raise ValueError("DIOR-RSVG recognition response is not a canonical category")
        return value

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError("validation predictions are empty")
        correct = sum(
            not item.parse_failed and str(item.parsed_prediction) == str(item.ground_truth)
            for item in predictions
        )
        return ValidationResult(
            primary_metric=correct / len(predictions),
            parse_failure_rate=self._failure_rate(predictions),
            details={"correct": correct},
        )
