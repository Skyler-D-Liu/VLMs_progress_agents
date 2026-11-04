"""Landmarks adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter


class LandmarksAdapter(PreparedTaskAdapter):
    TASK = "landmarks"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        options = list(record["metadata"]["options"])
        expected_prompts = {self.semantics.module.get_user_prompt(options)}
        answer = str(record["ground_truth"]["name"])
        if answer not in options or len(options) != 5:
            raise AdapterValidationError("Landmarks options must contain the gold name and total five")
        if record["user_prompt"] not in expected_prompts or record["assistant_answer"] != answer:
            raise AdapterValidationError("Landmarks prompt or answer differs from authority")

    def parse_prediction(self, text: str) -> str:
        from evaluation.utils.cot_prompting import strip_reasoning
        response = strip_reasoning(text)
        if not response:
            raise ValueError("Empty response")
        return response

    def parse_example_prediction(self, text, example):
        from evaluation.utils.scoring import landmark_answer
        answer = landmark_answer(text, example.metadata['options'])
        if answer is None:
            raise ValueError("Invalid answer format")
        return answer

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError("validation predictions are empty")
        correct = 0
        for prediction in predictions:
            if prediction.parse_failed:
                continue
            from evaluation.utils.scoring import landmark_answer
            truth = prediction.ground_truth
            parsed = landmark_answer(str(prediction.parsed_prediction), list(truth["options"]))
            correct += parsed == str(truth["name"])
        return ValidationResult(
            primary_metric=correct / len(predictions),
            parse_failure_rate=self._failure_rate(predictions),
            details={"correct": correct},
        )
