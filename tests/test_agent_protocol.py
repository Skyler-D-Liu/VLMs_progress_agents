import base64
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agents_benchmark.catalog import DATASET_TASKS, Sample, load_frozen_catalog, prepare_catalog, stratified_sample
from agents_benchmark.config import COMMON_INSTRUCTIONS, agent_instructions, build_hosted_request
from agents_benchmark.protocol import cohort_digest, run_identity
from agents_benchmark.report import paired_answers
from agents_benchmark.session import SessionInterrupted, SessionRunner, validate_session
from agents_benchmark.storage import atomic_json
from agents_benchmark.cli import direct_answer
from agents_benchmark.images import InputImage


class SamplingTests(unittest.TestCase):
    def test_prepare_freezes_all_tasks_and_input_hashes(self):
        source_rows = {
            'landmarks': {'name': 'place'}, 'pattern-net': {'label': 'scene'},
            'dior-recognition': {'label': 'object'}, 'rare-planes': {'label': 2},
            'cowc': {'label': 2}, 'xview2': {'gt': {'count_before': 2}},
            'neon-trees': {'count': 2}, 'aerial-animal': {'label': 2},
            'fpcd': {'ground_truth': 'A'}, 'dior-refdet': {'object': {'bbox': [0, 0, 1, 1]}},
        }
        def dataset_rows(task, unused):
            if task == 'geobench-refdet':
                return [{'sample_id': name, 'source_dataset': name, 'ground_truth_bboxes': [[0, 0, 1, 1]]}
                        for name in ('DOTA', 'FAIR1M')]
            return [dict(source_rows[task], sample_id=str(i)) for i in range(3)]
        class FakeInputs:
            def __init__(self, task, rows, root):
                pass
            def prepare(self, source):
                return dict(source, prompt='Question', system_message='Task.'), [
                    'data:image/png;base64,' + base64.b64encode(b'image bytes').decode()]
        module = types.ModuleType('evaluation.utils.reasoning_tasks')
        module.TaskInputs = FakeInputs
        with tempfile.TemporaryDirectory() as directory:
            with patch('evaluation.utils.dataset_catalog.dataset_rows', side_effect=dataset_rows), \
                 patch.dict(sys.modules, {'evaluation.utils.reasoning_tasks': module}):
                samples = prepare_catalog(directory, maximum=2, seed=7)
            self.assertEqual(len(samples), 22)
            self.assertEqual(len(load_frozen_catalog(directory)), 22)
            self.assertEqual({sample.task for sample in samples}, set(DATASET_TASKS))
            path = Path(directory) / samples[0].row['image_files'][0]['path']
            self.assertEqual(path.read_bytes(), b'image bytes')
            with self.assertRaises(FileExistsError):
                prepare_catalog(directory, maximum=2)

    def test_proportional_sampling_is_stable_and_seeded(self):
        rows = [{'sample_id': str(i), 'label': 'a' if i < 150 else 'b'} for i in range(200)]
        selected, metadata = stratified_sample('patternnet', rows, 100, 0)
        reordered, _ = stratified_sample('patternnet', list(reversed(rows)), 100, 0)
        alternate, _ = stratified_sample('patternnet', rows, 100, 1)
        self.assertEqual([r['sample_id'] for r in selected], [r['sample_id'] for r in reordered])
        self.assertNotEqual(selected, alternate)
        self.assertEqual(metadata['strata']['a'], {'available': 150, 'selected': 75})
        self.assertEqual(metadata['strata']['b'], {'available': 50, 'selected': 25})

    def test_small_tasks_and_shared_xview2(self):
        rows = [{'sample_id': str(i), 'gt': {'count_before': i % 3}} for i in range(9)]
        selected, metadata = stratified_sample('xview2', rows, 100, 0)
        self.assertEqual(len(selected), 9)
        self.assertEqual(metadata['selected'], 9)
        self.assertEqual({r['sample_id'] for r in selected}, {r['sample_id'] for r in rows})

    def test_equal_remainder_ties_depend_on_seed(self):
        rows = [{'sample_id': str(i), 'label': i} for i in range(30)]
        first, _ = stratified_sample('cowc', rows, 10, 0)
        second, _ = stratified_sample('cowc', rows, 10, 1)
        self.assertEqual(len(first), 10)
        self.assertNotEqual({r['sample_id'] for r in first}, {r['sample_id'] for r in second})

    def test_duplicate_and_missing_annotations_fail(self):
        with self.assertRaises(ValueError):
            stratified_sample('cowc', [{'sample_id': '1', 'label': 2}] * 2, 10)
        with self.assertRaises(KeyError):
            stratified_sample('cowc', [{'sample_id': '1'}], 10)

    def test_cohort_requires_matching_sampling_provenance(self):
        samples = [Sample(task, '1', source, 0, {}) for task, source in DATASET_TASKS.items()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'cohort.json', [sample.identity() for sample in samples])
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_frozen_catalog(root)
            metadata = {'max_per_task': 100,
                        'selected_ids': {sample.task: [sample.sample_id] for sample in samples}}
            atomic_json(root / 'sampling.json', metadata)
            self.assertEqual(len(load_frozen_catalog(root)), len(samples))
            metadata['selected_ids']['cowc'] = ['different']
            atomic_json(root / 'sampling.json', metadata)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_frozen_catalog(root)


class AgentProtocolTests(unittest.TestCase):
    def test_direct_uses_zero_shot_system_message_without_agent_instruction(self):
        sample = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Task system.', 'prompt': 'Question'})
        with tempfile.TemporaryDirectory() as directory, \
             patch('agents_benchmark.cli.read_key', return_value='test-key'), \
             patch('vleo_runtime.api.openai_request', return_value={
                 'model': 'gpt-6-astra', 'choices': [{'finish_reason': 'stop', 'message': {'content': '1'}}]}) as request:
            direct_answer(sample, [InputImage(b'image', '.png', 1, 1)], Path(directory))
        payload = request.call_args.args[0]
        self.assertEqual(payload['messages'][0]['content'], 'Task system.')
        self.assertNotIn(COMMON_INSTRUCTIONS, payload['messages'][0]['content'])
        self.assertEqual(payload['messages'][1]['content'][-1], {'type': 'text', 'text': 'Question'})
        agent = build_hosted_request('', sample.row['system_message'])['agent']['instructions']
        self.assertTrue(agent.startswith('Task system.'))
        self.assertIn(COMMON_INSTRUCTIONS, agent)

    def test_task_system_message_is_in_agent_request_and_identity(self):
        sample = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Task system.'})
        request = build_hosted_request('', sample.row['system_message'])
        self.assertEqual(request['agent']['instructions'], agent_instructions('Task system.'))
        self.assertIn(COMMON_INSTRUCTIONS, request['agent']['instructions'])
        validate_session({'agent': request['agent'], 'environment': request['environment']}, 'Task system.')
        changed = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Different system.'})
        self.assertNotEqual(run_identity(sample, 'autonomous', 'cohort'),
                            run_identity(changed, 'autonomous', 'cohort'))
        direct = run_identity(sample, 'direct', 'cohort')
        self.assertNotEqual(direct, run_identity(changed, 'direct', 'cohort'))

    def test_submitted_session_is_monitored_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {'model': 'gpt-6-astra', 'instructions': agent_instructions('Task.'),
                        'prompt': 'Question', 'input_mode': 'original_image_files',
                        'input_transport': 'inline_chunks', 'network': 'disabled', 'multi_agent': True}
            atomic_json(root / 'settings.json', settings)
            atomic_json(root / 'state.json', {'session_id': 'session', 'input_started': True})
            runner = SessionRunner(None, root, 'Task.')
            with patch.object(runner, '_monitor', return_value={'status': 'completed'}) as monitor:
                self.assertEqual(runner.run('Question', []), {'status': 'completed'})
                monitor.assert_called_once_with()
            with self.assertRaisesRegex(Exception, 'settings changed'):
                runner.run('Different question', [])

    def test_ambiguous_session_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / 'settings.json', {'model': 'gpt-6-astra', 'instructions': agent_instructions('Task.'),
                'prompt': 'Question', 'input_mode': 'original_image_files', 'input_transport': 'inline_chunks',
                'network': 'disabled', 'multi_agent': True})
            atomic_json(root / 'state.json', {'phase': 'creating'})
            with self.assertRaises(SessionInterrupted):
                SessionRunner(None, root, 'Task.').run('Question', [])

    def test_agent_arms_follow_shared_retry_rule(self):
        from agents_benchmark.cli import run_with_retries
        sample = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Task.', 'prompt': 'Question'})
        replies = iter([{'status': 'failed', 'answer_text': ''},
                        {'status': 'completed', 'answer_text': 'no number here'},
                        {'status': 'completed', 'answer_text': '{"count": 4}'}])
        prompts, folders = [], []
        def attempt(folder, prompt):
            folders.append(folder.name)
            prompts.append(prompt)
            return next(replies)
        with tempfile.TemporaryDirectory() as directory:
            outcome = run_with_retries(sample, Path(directory), attempt)
            saved = json.loads((Path(directory) / "outcome.json").read_text())
        self.assertEqual(folders, ['1', '2', '3'])
        self.assertEqual(prompts[0], 'Question')
        self.assertEqual(prompts[1], 'Question')
        self.assertTrue(prompts[2].startswith('Question\n\n'))
        self.assertIn('non-negative integer', prompts[2])
        self.assertEqual(outcome['attempt_count'], 3)
        self.assertFalse(outcome['retry_exhausted'])
        self.assertEqual(saved['answer_text'], '{"count": 4}')

    def test_exhausted_agent_failure_is_scored_as_missing(self):
        sample = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Task.', 'label': 1})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = cohort_digest([sample])
            for arm in ('direct', 'autonomous', 'guided'):
                folder = sample.directory(root / arm)
                atomic_json(folder / 'input_identity.json', run_identity(sample, arm, cohort))
                atomic_json(folder / 'outcome.json', {'status': 'failed', 'answer_text': '', 'retry_exhausted': True}
                            if arm == 'autonomous' else {'status': 'completed', 'answer_text': '1'})
            answers = paired_answers([sample], root)
        self.assertIsNone(answers['autonomous'][0])

    def test_infrastructure_failure_cannot_be_scored(self):
        sample = Sample('cowc', '1', 'cowc', 0, {'system_message': 'Task.'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = cohort_digest([sample])
            for arm in ('direct', 'autonomous', 'guided'):
                folder = sample.directory(root / arm)
                atomic_json(folder / 'input_identity.json', run_identity(sample, arm, cohort))
                atomic_json(folder / 'outcome.json', {'status': 'environment_failed' if arm == 'autonomous' else 'completed',
                                                       'answer_text': '1'})
            with self.assertRaisesRegex(ValueError, 'infrastructure'):
                paired_answers([sample], root)


if __name__ == '__main__':
    unittest.main()
