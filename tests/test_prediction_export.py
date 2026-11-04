import unittest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation/utils'))
from evaluation.utils import reasoning_tasks

from evaluation.analysis import export_predictions
from evaluation.analysis.export_predictions import validate_prediction_protocol


class PredictionProtocolTests(unittest.TestCase):
    def test_baseline_rejects_strategy_drift(self):
        validate_prediction_protocol([{'model': 'm', 'prompt_mode': 'zero-shot',
                                       'few_shot_k': 0, 'reasoning_method': 'baseline'}], 'm', 'baseline')
        validate_prediction_protocol([{'model': 'specialist', 'answer_text': 'x'}], 'label', 'baseline')
        for row in ({'few_shot_k': 4}, {'prompt_mode': 'few-shot'},
                    {'reasoning_method': 'cot'}, {'cot': True}, {'strategy': '4shot'}):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'baseline'):
                validate_prediction_protocol([row], 'm', 'baseline')

    def test_strategy_requires_explicit_provenance(self):
        validate_prediction_protocol([{'strategy': '4shot', 'prompt_mode': 'few-shot', 'few_shot_k': 4,
                                       'reasoning_method': 'baseline'}], 'm', 'four-shot')
        validate_prediction_protocol([{'strategy': 'cot', 'prompt_mode': 'zero-shot', 'few_shot_k': 0,
                                       'reasoning_method': 'cot'}], 'm', 'cot')
        for strategy in ('four-shot', 'cot'):
            with self.subTest(strategy=strategy), self.assertRaises(ValueError):
                validate_prediction_protocol([{}], 'm', strategy)

    def test_mixed_models_and_infrastructure_failures_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mixes models'):
            validate_prediction_protocol([{'model': 'a'}, {'model': 'b'}], 'comparison', 'baseline')
        for status in ('request_failed', 'error', 'failed', 'cancelled', 'environment_failed'):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, 'infrastructure'):
                validate_prediction_protocol([{'status': status}], 'm', 'baseline')
        validate_prediction_protocol([{'status': 'incomplete'}], 'm', 'baseline')

    def test_aerial_specialist_uses_validated_dataset_identity(self):
        source = {'source_id': 'stable-id', 'sample_id': 'display-id', 'index': 3,
                  'label': 2, 'objects': {'bbox': []}}
        prepared = {**source, 'source_dataset': 'animals'}
        observation = {'sample_id': 'display-id', 'predictions': {'model': 0}}
        class Inputs:
            def __init__(self, *args):
                pass
            def prepare(self, row):
                return prepared, []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'predictions.jsonl'
            path.write_text(json.dumps({**source, 'prediction_count': 7}) + '\n')
            with patch.object(export_predictions, 'dataset_rows', return_value=[source]), \
                 patch.object(reasoning_tasks, 'TaskInputs', Inputs), \
                 patch.object(export_predictions, 'observations', return_value=[observation]):
                rows = export_predictions.export('aerial-animal', {'model': path}, folder)
        self.assertEqual(rows[0]['predictions']['model'], 7)


if __name__ == '__main__':
    unittest.main()
