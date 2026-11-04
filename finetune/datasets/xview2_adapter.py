"""Xview2 adapter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import (
    COUNT_KEYS,
    PreparedTaskAdapter,
    require_nonnegative_int,
    squared_pearson_r2,
)

BUILDING_COUNT_KEY = COUNT_KEYS[0]
DAMAGE_KEYS = COUNT_KEYS[1:]


class XView2Adapter(PreparedTaskAdapter):
    TASK = "xview2"

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        truth = {
            key: require_nonnegative_int(record["ground_truth"][key], key)
            for key in COUNT_KEYS
        }
        expected_answer = self.semantics.module.format_count_answer(truth)
        if record["user_prompt"] != self.semantics.module.USER_PROMPT:
            raise AdapterValidationError("xView2 user prompt differs from authority")
        if record["assistant_answer"] != expected_answer:
            raise AdapterValidationError("xView2 answer differs from authority")

    def parse_prediction(self, text: str) -> dict[str, int]:
        value = self.semantics.module.capture_and_parse_json(text)
        if not isinstance(value, dict) or set(value) != set(COUNT_KEYS):
            raise ValueError("xView2 response must contain all five canonical fields")
        return {key: require_nonnegative_int(value[key], key) for key in COUNT_KEYS}

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError("validation predictions are empty")
        actual = {key: [] for key in COUNT_KEYS}
        predicted_values = {key: [] for key in COUNT_KEYS}
        for prediction in predictions:
            predicted = (
                {key: 0 for key in COUNT_KEYS}
                if prediction.parse_failed
                else prediction.parsed_prediction
            )
            for key in COUNT_KEYS:
                actual[key].append(int(prediction.ground_truth[key]))
                predicted_values[key].append(int(predicted[key]))
        per_field_r2 = {
            key: squared_pearson_r2(actual[key], predicted_values[key]) for key in COUNT_KEYS
        }
        per_field_mae = {
            key: sum(abs(p - a) for a, p in zip(actual[key], predicted_values[key])) / len(actual[key])
            for key in COUNT_KEYS
        }
        # One adapter serves two benchmark tasks: pre-disaster building
        # counting (r^2 of count_before) and change detection (macro r^2 over
        # the four damage categories). The primary validation metric is their
        # equally weighted mean, mirroring equal task weighting in the score.
        building_counting_r2 = per_field_r2[BUILDING_COUNT_KEY]
        change_detection_r2 = sum(per_field_r2[key] for key in DAMAGE_KEYS) / len(DAMAGE_KEYS)
        return ValidationResult(
            primary_metric=(building_counting_r2 + change_detection_r2) / 2,
            parse_failure_rate=self._failure_rate(predictions),
            details={
                "metric": "r2",
                "building_counting_r2": building_counting_r2,
                "change_detection_macro_r2": change_detection_r2,
                "per_field_r2": per_field_r2,
                "per_field_mae": per_field_mae,
            },
        )
