"""Geobench refdet adapter."""
from __future__ import annotations
from .adapter_contract import AdapterValidationError, ValidationResult
from .prepared_task_adapter import PreparedTaskAdapter, bbox_iou
from evaluation.utils.geobench_spatial import parse_fair1m_response, parse_bbox_response
from vleo_runtime.localization import polygon_iou


class GeobenchRefdetAdapter(PreparedTaskAdapter):
    TASK = 'geobench-refdet'

    def verify_record_semantics(self, record):
        dataset = (record.get('metadata') or {}).get('source_dataset')
        if dataset not in {'DOTA', 'FAIR1M'}:
            raise AdapterValidationError('Prepare Ref-Det records with source_dataset metadata')
        from vleo_runtime.localization import localization_prompt
        metadata = record['metadata']
        image = metadata['images'][0]
        expected_prompt = localization_prompt(dataset, image['width'], image['height'], metadata['expression'])
        if record.get('user_prompt') != expected_prompt:
            raise AdapterValidationError('Prepare Ref-Det records with the current localization prompt')
        self._datasets = getattr(self, '_datasets', {})
        self._datasets[str(record['sample_id'])] = dataset

    def parse_prediction(self, text, *, source_dataset=None, width=None, height=None):
        if source_dataset not in {'FAIR1M', 'DOTA'}:
            raise ValueError('Ref-Det parsing requires the input source_dataset')
        parsed = (parse_fair1m_response if source_dataset == 'FAIR1M' else parse_bbox_response)(text, width, height)
        if not parsed['format_valid']:
            raise ValueError(parsed.get('parse_error') or 'Invalid Ref-Det box response')
        return parsed['polygons'] if source_dataset == 'FAIR1M' else parsed['bboxes']

    def parse_example_prediction(self, text, example):
        dataset = example.metadata.get('source_dataset')
        image = example.metadata['images'][0]
        return self.parse_prediction(text, source_dataset=dataset, width=image['width'], height=image['height'])

    def compute_validation_metric(self, predictions):
        if not predictions:
            raise AdapterValidationError('validation predictions are empty')
        from scipy.optimize import linear_sum_assignment
        scores = []
        for item in predictions:
            dataset = self._datasets[item.sample_id]
            score = polygon_iou if dataset == 'FAIR1M' else bbox_iou
            predicted = [] if item.parse_failed else item.parsed_prediction
            truth = item.ground_truth
            if predicted and truth:
                ious = [[score(a, b) for b in truth] for a in predicted]
                rows, columns = linear_sum_assignment(ious, maximize=True)
                total = sum(ious[row][column] for row, column in zip(rows, columns))
            else:
                total = 0.0
            scores.append(total / max(len(predicted), len(truth), 1))
        return ValidationResult(primary_metric=sum(scores) / len(scores),
                                parse_failure_rate=self._failure_rate(predictions), details={})
