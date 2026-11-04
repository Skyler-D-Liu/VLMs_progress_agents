"""Dior rsvg adapter."""
from __future__ import annotations
import json
from typing import Any, Mapping, Sequence
from .adapter_contract import AdapterValidationError, ValidationPrediction, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter, bbox_iou

class DiorRsvgAdapter(PreparedTaskAdapter):
    TASK = 'dior-rsvg'

    def verify_record_semantics(self, record: Mapping[str, Any]) -> None:
        metadata = record['metadata']
        expected_prompt = self.semantics.module.get_detection_prompt(int(metadata['height']), int(metadata['width']), str(metadata['description']))
        bbox = [float(value) for value in record['ground_truth']]
        normalized = [int(value) if value.is_integer() else value for value in bbox]
        expected_answer = json.dumps({'bbox': normalized}, ensure_ascii=False)
        if record['user_prompt'] != expected_prompt or record['assistant_answer'] != expected_answer:
            raise AdapterValidationError('DIOR-RSVG prompt or answer differs from authority')

    def parse_prediction(self, text: str, *, image_width=None, image_height=None, model=None) -> list[float]:
        boxes = self.semantics.module.parse_detection_response(text, image_height, image_width, model_key=model)
        if len(boxes) != 1 or len(boxes[0]) != 4:
            raise ValueError('DIOR-RSVG response does not contain exactly one bbox')
        return [float(value) for value in boxes[0]]

    def parse_example_prediction(self, text, example):
        return self.parse_prediction(text, image_width=example.metadata['width'],
                                     image_height=example.metadata['height'])

    def compute_validation_metric(self, predictions: Sequence[ValidationPrediction]) -> ValidationResult:
        if not predictions:
            raise AdapterValidationError('validation predictions are empty')
        ious = [0.0 if prediction.parse_failed else bbox_iou(prediction.parsed_prediction, prediction.ground_truth) for prediction in predictions]
        return ValidationResult(primary_metric=sum(ious) / len(ious), parse_failure_rate=self._failure_rate(predictions), details={'iou_sum': sum(ious)})
