"""Export predictions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from evaluation.utils.dataset_catalog import dataset_rows, TASK_MODULES
from agents_benchmark.catalog import Sample, DATASET_TASKS, TASKS
from .observations import observations


def validate_prediction_protocol(rows, model, strategy):
    recorded_models = {str(row['model']) for row in rows if row.get('model') is not None}
    if len(recorded_models) > 1:
        raise ValueError(f'{model}: prediction file mixes models: {sorted(recorded_models)}')
    for row in rows:
        mode = row.get('prompt_mode')
        shots = row.get('few_shot_k')
        reasoning = row.get('reasoning_method')
        declared = row.get('strategy')
        if strategy == 'baseline':
            if (mode not in (None, 'zero-shot') or shots not in (None, 0)
                    or reasoning not in (None, 'baseline') or row.get('cot', False)
                    or declared not in (None, 'baseline', 'zero-shot')):
                raise ValueError(f'{model}: prediction is not a zero-shot baseline')
        elif strategy == 'four-shot':
            if (mode != 'few-shot' or shots != 4 or reasoning not in (None, 'baseline')
                    or row.get('cot', False) or declared not in (None, '4shot', 'four-shot', 'few-shot')):
                raise ValueError(f'{model}: prediction is not explicitly identified as four-shot')
        elif strategy == 'cot':
            if (mode not in (None, 'zero-shot') or shots not in (None, 0)
                    or reasoning != 'cot' or declared not in (None, 'cot')):
                raise ValueError(f'{model}: prediction is not explicitly identified as CoT')
        else:
            raise ValueError(f'Unknown strategy: {strategy}')
        if row.get('status') in {'request_failed', 'error', 'failed', 'cancelled',
                                'api_rejected', 'environment_failed', 'protocol_error'} or row.get('error'):
            raise ValueError(f'{model}: infrastructure failure cannot be scored as a model answer')


def export(task, files, data_root, cohort=None, metrics=None, strategy='baseline'):
    sources = dataset_rows(task, data_root)
    from evaluation.utils.reasoning_tasks import TaskInputs
    from evaluation.utils.benchmark_evaluation import identity, response_text
    available = {identity(task, row) for row in sources}
    if cohort is not None:
        manifest = [json.loads(line) for line in Path(cohort).read_text().splitlines() if line.strip()]
        if any(any(key in row for key in ('model','response','answer_text','raw_answer','predictions','prediction')) for row in manifest):
            raise ValueError('A cohort must contain dataset inputs, not model outputs')
        chosen = {identity(task, row) for row in manifest}
        if not chosen or len(chosen) != len(manifest) or not chosen <= available:
            raise ValueError('Invalid dataset cohort identities')
        sources = [row for row in sources if identity(task, row) in chosen]
    if metrics and task == 'geobench-refdet':
        requested = {TASKS.get(metric) for metric in metrics}
        if not requested <= {'dota', 'fair1m'}:
            raise ValueError('Ref-Det metrics must name DOTA or FAIR1M localization')
        sources = [row for row in sources if row['source_dataset'].lower() in requested]
    if not sources:
        raise ValueError('The dataset-defined analysis cohort is empty')
    inputs = TaskInputs(task, sources, data_root)
    expected = {identity(task, row) for row in sources}
    predictions = {}
    for model, path in files.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        indexed = {identity(task, row):row for row in rows}
        if len(indexed) != len(rows) or not expected <= set(indexed) or not set(indexed) <= available:
            raise ValueError(f'{model}: predictions must cover the dataset-defined cohort without foreign or duplicate identities')
        validate_prediction_protocol([indexed[sid] for sid in expected], model, strategy)
        predictions[model] = indexed
    samples, answers = [], {model:[] for model in files}
    aerial_prediction_rows = {model: [] for model in files} if task == 'aerial-animal' else None
    for index, source in enumerate(sources):
        row, _ = inputs.prepare(source)
        slug = row['source_dataset'].lower() if task == 'geobench-refdet' else next(k for k,v in DATASET_TASKS.items() if v==task)
        if task == 'geobench-refdet':
            row['target_boxes'] = row['ground_truth_bboxes']
        samples.append(Sample(slug, row['sample_id'], task, index, row))
        for model, indexed in predictions.items():
            saved = indexed[identity(task, source)]
            if aerial_prediction_rows is not None:
                aerial_prediction_rows[model].append(saved)
            failed = saved.get('status') in {'error','request_failed','incomplete','failed'} or saved.get('error')
            answer = response_text(saved)
            if not answer and task in {'landmarks','fpcd'} and isinstance(saved.get('prediction'), str):
                answer = saved['prediction']
            answers[model].append(None if failed else answer)
    result = observations(samples, answers)
    if task == 'aerial-animal':
        from evaluation.utils.scoring import nonnegative_integer
        for row_index, row in enumerate(result):
            for model, saved_rows in aerial_prediction_rows.items():
                saved = saved_rows[row_index]
                failed = saved.get('error') or saved.get('status') in {'error','request_failed','incomplete','failed'}
                if 'prediction_count' in saved and not failed:
                    value = nonnegative_integer(saved['prediction_count'])
                    if value is None:
                        raise ValueError('Specialist total count must be a nonnegative integer')
                    row['predictions'][model] = value
                elif 'prediction_count' in saved or failed:
                    row['predictions'][model] = 0
    if metrics:
        if not set(metrics) <= {row['task'] for row in result}:
            raise ValueError('Requested metric does not belong to this task')
        result = [row for row in result if row['task'] in metrics]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASK_MODULES, required=True)
    parser.add_argument('--prediction', action='append', required=True, help='MODEL=prediction.jsonl; repeat for every model')
    parser.add_argument('--data-root', type=Path, default=Path('raw_dataset'))
    parser.add_argument('--cohort', type=Path, help='Optional dataset-generated input JSONL, without model outputs')
    parser.add_argument('--metric', action='append', help='Metric task name to include; repeat as needed')
    parser.add_argument('--strategy', choices=('baseline', 'four-shot', 'cot'), default='baseline',
                        help='Require the declared prompting strategy in each prediction')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pairs = [entry.split('=',1) for entry in args.prediction]
    if len({name for name,_ in pairs}) != len(pairs):
        raise ValueError('Duplicate model name')
    rows = export(args.task, dict(pairs), args.data_root, args.cohort, args.metric, args.strategy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row)+'\n' for row in rows))

if __name__ == '__main__':
    main()
