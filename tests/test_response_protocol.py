"""Evaluation protocol checks."""
import unittest

from agents_benchmark.catalog import DATASET_TASKS, Sample
from evaluation.utils.cot_prompting import effective_system_message, parse_reasoning_response
from evaluation.utils.response_protocol import format_valid, parse_answer, query_validated, validated_call
from evaluation.utils.scoring import score_group, squared_pearson


class ParsingTests(unittest.TestCase):
    def test_task_answers(self):
        cases = [
            ('landmarks', '{"answer":"2"}', {'options': ['Alpha', 'Beta']}, 'Beta'),
            ('patternnet', '{"category":"airplane"}', {}, 'Airplane'),
            ('dior_classification', '{"category":"airplane"}', {}, 'Airplane'),
            ('cowc', '{"count":3}', {}, 3),
            ('neon', '**Count: 3**\nExplanation.', {}, 3),
            ('rareplanes', '{"Answer":3}', {}, 3),
            ('fpcd', '{"answer":"D"}', {}, 'D'),
            ('animals', '{"elephants":1,"zebras":0,"giraffes":2}', {},
             {'elephants': 1, 'zebras': 0, 'giraffes': 2}),
            ('dior_localization', '{"bbox":[1,1,3,3]}', {'image_width': 8, 'image_height': 8}, [[1., 1., 3., 3.]]),
            ('dota', '{"bboxes":[[1,1,3,3]]}', {'image_width': 8, 'image_height': 8}, [[1., 1., 3., 3.]]),
            ('fair1m', '{"bboxes":[[[1,1],[3,1],[3,3],[1,3]]]}', {'image_width': 8, 'image_height': 8},
             [[[1., 1.], [3., 1.], [3., 3.], [1., 3.]]]),
        ]
        for task, answer, context, expected in cases:
            with self.subTest(task=task):
                wrapped = '<think>{"count":999}</think>\n```json\n' + answer + '\n```\nExplanation.'
                self.assertEqual(parse_answer(task, wrapped, context), expected)
                self.assertTrue(format_valid(task, wrapped, context))

    def test_negative_and_invalid_counts(self):
        for text in ['-3', '{"count":-3}', '{"count":true}', '{"count":1.5}', '{"count":NaN}', '', 'I cannot answer.']:
            with self.subTest(text=text):
                self.assertIsNone(parse_answer('cowc', text))
        self.assertEqual(parse_answer('cowc', '{"count":0}'), 0)
        self.assertFalse(format_valid('animals', '{"elephants":0}'))

    def test_refusal_with_category_reference(self):
        self.assertIsNone(parse_answer('patternnet', 'I cannot determine whether this is an airplane.'))
        self.assertIsNone(parse_answer('dior_classification', 'Unable to identify the vehicle.'))
        self.assertEqual(parse_answer('patternnet', 'I cannot add details. {"category":"Airplane"}'), 'Airplane')

    def test_structured_precedence(self):
        self.assertEqual(parse_answer('patternnet', '{"category":"Airplane"}\nIt might look like a bridge.'), 'Airplane')
        self.assertIsNone(parse_answer('patternnet', '{"category":"invalid"}\nAirplane'))
        self.assertEqual(parse_answer('cowc', '{"count":2}\n{"count":4}'), 4)

    def test_partial_count_fields(self):
        text = '{"count_before":2,"no_damage":-1,"minor_damage":0,"major_damage":1,"destroyed":0}'
        parsed = parse_answer('xview2', text)
        self.assertEqual(parsed['count_before'], 2)
        self.assertIsNone(parsed['no_damage'])
        self.assertFalse(format_valid('xview2', text))
        self.assertTrue(format_valid('xview2', text.replace(':-1', ':1')))

    def test_geometric_rejection(self):
        for task, key in [('dior_localization', 'bbox'), ('dota', 'bboxes')]:
            for coordinates in ['[-1,0,2,2]', '[0,0,9,2]', '[3,0,2,2]', '[0,0,0,2]', '[0,0,NaN,2]', '[false,0,2,2]']:
                text = '{"' + key + '":' + (coordinates if key == 'bbox' else '[' + coordinates + ']') + '}'
                self.assertIsNone(parse_answer(task, text, {'image_width': 8, 'image_height': 8}))
        self.assertIsNone(parse_answer('fair1m', '{"bboxes":[[[0,0],[2,2],[0,2],[2,0]]]}', {'image_width': 8, 'image_height': 8}))

    def test_labels_do_not_affect_parsing(self):
        for label in ['Alpha', 'Beta']:
            self.assertEqual(parse_answer('landmarks', '2', {'options': ['Alpha', 'Beta'], 'name': label}), 'Beta')


class RetryTests(unittest.TestCase):
    def run_responses(self, replies, method='baseline', **kwargs):
        prompts = []
        source = iter(replies)
        def call(prompt):
            prompts.append(prompt)
            value = next(source)
            if isinstance(value, Exception):
                raise value
            return {'answer_text': value, 'status': 'ok'}
        response, _, parsed = validated_call(call, method, task='cowc', prompt='Original task.', **kwargs)
        return response, parsed, prompts

    def test_first_valid_response(self):
        for method in ['baseline', 'cot']:
            response, parsed, prompts = self.run_responses(['bad', '{"count":2}', '{"count":3}'], method)
            self.assertEqual(len(prompts), 2)
            self.assertEqual(parsed['answer_text'], '{"count":2}')
            self.assertEqual(response['status'], 'ok')
            self.assertTrue(prompts[1].startswith('Original task.'))
            self.assertNotIn('2', prompts[1])
            self.assertIn('non-negative integer', prompts[1])

    def test_no_reasoning_required(self):
        response, _, prompts = self.run_responses(['{"count":0}'], 'cot')
        self.assertEqual(len(prompts), 1)
        self.assertEqual(response['status'], 'ok')
        self.assertIn('JSON-only instruction applies to the final answer', effective_system_message('Only JSON.', 'cot'))

    def test_shared_attempt_cap(self):
        response, _, prompts = self.run_responses(['bad'] * 7, max_attempts=20)
        self.assertEqual(len(prompts), 5)
        self.assertTrue(response['retry_exhausted'])
        self.assertEqual(response['status'], 'parse_failed')

    def test_transport_and_format_share_budget(self):
        response, _, prompts = self.run_responses([ConnectionError('temporary'), 'bad', TimeoutError('temporary'), 'bad', 'bad'])
        self.assertEqual(len(prompts), 5)
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[0], prompts[3])
        self.assertTrue(response['retry_exhausted'])

    def test_persistent_transport_failure(self):
        response, parsed, prompts = self.run_responses([ConnectionError('temporary')] * 5)
        self.assertEqual(len(prompts), 5)
        self.assertEqual(response['status'], 'request_failed')
        self.assertEqual(parsed['answer_text'], '')

    def test_failed_and_incomplete_responses_cannot_be_accepted(self):
        for status in ('request_failed', 'error', 'failed', 'incomplete'):
            with self.subTest(status=status):
                prompts = []
                replies = iter([{'status': status, 'answer_text': '{"count":7}'},
                                {'status': 'ok', 'answer_text': '{"count":3}'}])
                def call(prompt):
                    prompts.append(prompt)
                    return next(replies)
                response, _, parsed = validated_call(call, 'baseline', task='cowc', prompt='Task.')
                self.assertEqual(prompts[0], 'Task.')
                if status == 'incomplete':
                    # A returned but truncated response is followed by a repair prompt.
                    self.assertTrue(prompts[1].startswith('Task.\n\n'))
                    self.assertIn('non-negative integer', prompts[1])
                else:
                    # No model response: the same prompt is retried.
                    self.assertEqual(prompts[1], 'Task.')
                self.assertEqual(response['status'], 'ok')
                self.assertEqual(parsed['answer_text'], '{"count":3}')

    def test_http_status_errors_are_transient(self):
        from evaluation.utils.response_protocol import transient_error
        from evaluation.utils.benchmark_environment import APIStatusError
        self.assertTrue(transient_error(APIStatusError(503, 'unavailable')))
        self.assertTrue(transient_error(APIStatusError(429, 'rate limited')))
        self.assertFalse(transient_error(APIStatusError(400, 'bad request')))

    def test_exhausted_failed_response_is_not_scored(self):
        for status in ('request_failed', 'incomplete'):
            response, _, parsed = validated_call(
                lambda _: {'status': status, 'answer_text': '{"count":7}'},
                'baseline', task='cowc', prompt='Task.', max_attempts=2)
            self.assertEqual(response['status'], status)
            self.assertEqual(parsed['answer_text'], '')
            self.assertTrue(response['retry_exhausted'])

    def test_permanent_errors_propagate(self):
        with self.assertRaises(ValueError):
            self.run_responses([ValueError('configuration')])

    def test_four_shot_examples_preserved(self):
        class Client:
            def __init__(self):
                self.calls = []
            def query(self, system, prompt, images, generation, **kwargs):
                self.calls.append((prompt, kwargs))
                return {'answer_text': 'bad' if len(self.calls) == 1 else '{"count":1}'}
        client = Client()
        examples = [{'answer': str(i)} for i in range(4)]
        prepared = {'system_message': 'Task.', 'prompt': 'Original task.'}
        result = query_validated(client, 'cowc', prepared, [], examples=examples)
        self.assertEqual(result['attempt_count'], 2)
        self.assertTrue(all(call[1]['examples'] == examples for call in client.calls))

    def test_reasoning_removed_before_validation(self):
        response, parsed, prompts = self.run_responses(['<think>{"count":2}</think>\nNo answer.', '{"count":3}\nExplanation.'], 'cot')
        self.assertEqual(len(prompts), 2)
        self.assertEqual(parsed['answer_text'], '{"count":3}')
        self.assertEqual(response['status'], 'ok')


class IntegrationTests(unittest.TestCase):
    def test_transport_is_not_retried_internally(self):
        import sys
        from pathlib import Path
        from unittest.mock import patch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation/utils'))
        from reasoning_effort import RemoteClient, Settings, GPT_MODEL
        calls = []
        def transport(payload):
            calls.append(payload)
            return 503, b'temporary'
        with patch('reasoning_effort.credentials', return_value=('provider', 'unused')):
            client = RemoteClient(Settings(GPT_MODEL, 'default', max_attempts=5),
                                  transport=transport, sleep=lambda _: None)
            result = query_validated(client, 'cowc', {'system_message': 'Task', 'prompt': 'Task'}, [])
        self.assertEqual(len(calls), 5)
        self.assertEqual(result['status'], 'request_failed')

    def test_model_configurations(self):
        from pathlib import Path
        from finetune.config.loader import _read_yaml_mapping
        from finetune.config.schema import BaseLoraConfig
        folder = Path(__file__).resolve().parents[1] / 'finetune/experiment_specs/models'
        for path in folder.glob('*.yaml'):
            config = BaseLoraConfig.model_validate(_read_yaml_mapping(path, 'base_config'))
            policy = config.search.successive_halving
            self.assertEqual(policy.final_training_max_examples,
                             policy.final_training_max_optimizer_steps * config.training.batch.effective_global_batch_size)


class ScoringTests(unittest.TestCase):
    def sample(self, task, row):
        return Sample(task, '', DATASET_TASKS[task], 0, row)

    def test_missing_classification_stays_in_denominator(self):
        sample = self.sample('landmarks', {'name': 'Alpha', 'options': ['Alpha', 'Beta']})
        self.assertEqual(score_group([sample, sample], ['Alpha', None], '')['score'], .5)

    def test_missing_count_is_zero(self):
        samples = [self.sample('cowc', {'label': n}) for n in [0, 2, 4]]
        result = score_group(samples, [None, '-2', '{"count":4}'], '')
        self.assertAlmostEqual(result['score'], squared_pearson([0, 2, 4], [0, 0, 4]))
        self.assertAlmostEqual(result['mae'], 2 / 3)

    def test_constant_predictions(self):
        self.assertEqual(squared_pearson([1, 2, 3], [0, 0, 0]), 0)
        self.assertEqual(squared_pearson([1, 1], [0, 1]), 0)
        self.assertEqual(squared_pearson([], []), 0)

    def test_missing_localization_is_zero(self):
        sample = self.sample('dior_localization', {'object': {'bbox': [1, 1, 3, 3]}, 'image_width': 8, 'image_height': 8})
        result = score_group([sample, sample], ['{"bbox":[1,1,3,3]}', None], '')
        self.assertEqual(result['score'], .5)


if __name__ == '__main__':
    unittest.main()
