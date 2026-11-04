"""Inference strategy."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from threading import Condition, Event, Lock
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation/utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cot_prompting import validate_cot_run
from inference_strategy_inputs import OUTPUT_ROOT, StrategyInputs, matched_rows
from reasoning_effort import GPT_MODEL, GEMINI_MODEL, ENV_PATH, Settings, RemoteClient, RequestFailed, ProviderUnavailable
from evaluation.experiments.reasoning_comparison import atomic_json, freeze_samples, read_checkpoint, validate_saved, _replace_checkpoint, WorkerInterrupted
from request_budget import RequestBudget
from reasoning_tasks import TASKS, RAW_ROOT, parse_result, read_jsonl, task_metrics
MODELS = (GPT_MODEL, GEMINI_MODEL)
STRATEGIES = ('4shot', 'cot')
REQUEST_LIMIT = 5

class RateGate:

    def __init__(self, workers):
        self.limit, self.active, self.until = (workers, 0, 0.0)
        self.condition = Condition()

    def enter(self):
        with self.condition:
            while self.active >= self.limit or time.monotonic() < self.until:
                self.condition.wait(timeout=1)
            self.active += 1

    def leave(self, status):
        with self.condition:
            self.active -= 1
            if status == 429:
                self.limit = max(1, self.limit // 2)
                self.until = time.monotonic() + 30
                print(f'Rate limited: transport concurrency reduced to {self.limit}', flush=True)
            self.condition.notify_all()

class StrategyClient(RemoteClient):

    def __init__(self, settings, env, workers):
        self.gate = RateGate(workers)
        super().__init__(settings, env)

    def _send(self, payload):
        self.gate.enter()
        status = None
        try:
            status, body = super()._send(payload)
            return (status, body)
        finally:
            self.gate.leave(status)

def query_sample(task, prepared, images, examples, client, budget, stop_requested):
    sid = prepared['sample_id']
    def before_attempt():
        if stop_requested():
            raise WorkerInterrupted('Stopping before request; checkpoints preserved')
        budget.reserve(sid)
    try:
        from evaluation.utils.response_protocol import query_validated
        response = query_validated(client, task, prepared, images, examples=examples,
                                   before_attempt=before_attempt,
                                   max_attempts=budget.limit - budget.used(sid))
    except RequestFailed as exc:
        response = dict(answer_text='', output_text='', status='request_failed', error=str(exc),
                        token_usage={})
    row = {**prepared, 'model': client.settings.model, **response,
           'request_attempts_used': budget.used(sid)}
    row['response'] = {'choices': [{'message': {'content': row['answer_text']}}]}
    return parse_result(task, row)

def strategy_metrics(task, model, strategy, records):
    rows = list(records.values())
    metrics = task_metrics(task, rows)
    for metric in metrics:
        metric.update(model=model, strategy=strategy, reasoning_effort='default')
    return metrics

def write_metrics(folder, task, model, strategy, records):
    metrics = strategy_metrics(task, model, strategy, records)
    atomic_json(folder / 'metrics.json', metrics)
    return metrics

def evaluate(args, stop_requested=lambda: False):
    settings = Settings(args.model, 'default', args.timeout, 5)
    strategy = 'cot' if args.reasoning_method == 'cot' else '4shot'
    all_rows = matched_rows(args.task, Path(args.cache_dir))
    output = Path(args.output_root)
    freeze_samples(output / args.task / 'samples.json', all_rows)
    folder = output / args.task / args.model / strategy
    folder.mkdir(parents=True, exist_ok=True)
    selected = all_rows
    with (folder / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = read_checkpoint(folder / 'predictions.jsonl')
        by_id = {r['sample_id']: r for r in all_rows}
        if set(records) - set(by_id):
            raise ValueError('Checkpoint has unexpected test samples')
        inputs = StrategyInputs(args.task, all_rows, Path(args.cache_dir), strategy)
        for sid, saved in records.items():
            prepared, _, _ = inputs.prepare(by_id[sid])
            validate_saved(saved, prepared, settings)
        budget = RequestBudget(folder / 'request_attempts.json', records, by_id, limit=getattr(args, 'request_limit', REQUEST_LIMIT))
        pending = [r for r in selected if r['sample_id'] not in records or (records[r['sample_id']]['status'] == 'request_failed' and (not budget.exhausted(r['sample_id'])))]
        if pending:
            client = StrategyClient(settings, args.env, args.workers)

            def process(source):
                if stop_requested():
                    raise WorkerInterrupted('Stopping before preparation')
                prepared, images, examples = inputs.prepare(source)
                return query_sample(args.task, prepared, images, examples, client, budget, stop_requested)
            retry_ids = {r['sample_id'] for r in pending}
            records = {sid: row for sid, row in records.items() if sid not in retry_ids}
            _replace_checkpoint(folder / 'predictions.jsonl', records)
            iterator = iter(pending)
            failure = None
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process, next(iterator)) for _ in range(min(args.workers, len(pending)))}
                with (folder / 'predictions.jsonl').open('a') as stream:
                    while futures:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            try:
                                row = future.result()
                            except Exception as exc:
                                failure = failure or exc
                                continue
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                            records[row['sample_id']] = row
                            print(f"{args.task} {args.model} {strategy}: {row['status']}", flush=True)
                            if not failure and (not stop_requested()):
                                source = next(iterator, None)
                                if source is not None:
                                    futures.add(executor.submit(process, source))
            write_metrics(folder, args.task, args.model, strategy, records)
            if failure:
                raise failure
            if stop_requested():
                raise WorkerInterrupted('Saved completed requests')
        write_metrics(folder, args.task, args.model, strategy, records)
        required = {r['sample_id'] for r in selected}
        if not required.issubset(records) or any((records[s]['status'] == 'request_failed' and not records[s].get('retry_exhausted') for s in required)):
            raise RequestFailed('Missing or failed requests remain; resume within the saved budget')
        print(f'Completed {args.task} {args.model} {strategy}', flush=True)

def summarize(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize(output)

def _summarize(output):
    metrics, comparison = ([], [])
    for task in TASKS:
        expected = {r['sample_id'] for r in matched_rows(task)}
        demonstrations = {}
        for model in MODELS:
            zero_path = ROOT / 'results/inference/reasoning_effort' / task / model / 'medium/metrics.json'
            zero = {r['dataset_task']: r for r in json.loads(zero_path.read_text())}
            for strategy in STRATEGIES:
                folder = output / task / model / strategy
                path = folder / 'predictions.jsonl'
                raw = path.read_bytes() if path.exists() else b''
                if raw and (not raw.endswith(b'\n')):
                    raw = raw[:raw.rfind(b'\n') + 1]
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                ids = [r['sample_id'] for r in rows]
                if len(ids) != len(set(ids)) or set(ids) - expected:
                    raise ValueError(f'Duplicate or unexpected samples in {path}')
                for row in rows:
                    if (row.get('model'), row.get('strategy'), row.get('reasoning_effort')) != (model, strategy, 'default'):
                        raise ValueError('Wrong model, strategy or reasoning setting in results')
                    expected_mode = (4, 'baseline') if strategy == '4shot' else (0, 'cot')
                    if (row.get('few_shot_k'), row.get('reasoning_method')) != expected_mode:
                        raise ValueError('Wrong demonstration count or CoT mode in results')
                    if strategy == '4shot':
                        selected = row['demonstrations']
                        if len(selected) != 4:
                            raise ValueError('A result does not contain four demonstrations')
                        sid = row['sample_id']
                        if sid in demonstrations and demonstrations[sid] != selected:
                            raise ValueError('The two models used different demonstrations')
                        demonstrations[sid] = selected
                current = strategy_metrics(task, model, strategy, {r['sample_id']: r for r in rows})
                metrics.extend(current)
                for metric in current:
                    baseline = zero[metric['dataset_task']]
                    comparison.append(dict(dataset_task=metric['dataset_task'], model=model, strategy=strategy, metric=metric['metric'], zero_shot_reasoning='medium', strategy_reasoning='default', zero_shot_score=baseline['score'], strategy_score=metric['score'], complete=metric['complete']))
    for name, rows in [('task_metrics.csv', metrics), ('zero_shot_comparison.csv', comparison)]:
        fields = list(dict.fromkeys((k for row in rows for k in row)))
        temporary = output / (name + '.tmp')
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / name)
    print(f"Summary: {sum((bool(m['complete']) for m in metrics))}/52 complete task metrics", flush=True)

def main(argv=None, stop_requested=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(TASKS))
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--few-shot-k', type=int, choices=(0, 4), default=0)
    parser.add_argument('--reasoning-method', choices=('baseline', 'cot'), default='baseline')
    parser.add_argument('--reasoning-effort', choices=('default',), default='default')
    parser.add_argument('--workers', '--sample-workers', '--sample-parallel', dest='workers', type=int, default=1)
    parser.add_argument('--request-limit', type=int, default=REQUEST_LIMIT, help='Transport request budget')
    parser.add_argument('--output-root', default=str(OUTPUT_ROOT))
    parser.add_argument('--cache-dir', default=str(RAW_ROOT))
    parser.add_argument('--env', default=str(ENV_PATH))
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    if args.summarize:
        return summarize(args.output_root)
    if not args.task or not args.model:
        parser.error('--task and --model are required')
    if (args.few_shot_k, args.reasoning_method) not in ((4, 'baseline'), (0, 'cot')):
        parser.error('Choose exactly 4-shot baseline or zero-shot CoT')
    if args.workers < 1 or not 1 <= args.request_limit <= REQUEST_LIMIT:
        parser.error('Invalid worker count')
    validate_cot_run(reasoning_method=args.reasoning_method, model_key=args.model, few_shot_k=args.few_shot_k)
    event = Event()
    if stop_requested is None:
        signal.signal(signal.SIGUSR1, lambda *_: event.set())
        signal.signal(signal.SIGTERM, lambda *_: event.set())
        end = int('0')
        stop_requested = lambda: event.is_set() or bool(end and time.time() >= end - 3600)
    return evaluate(args, stop_requested)
if __name__ == '__main__':
    try:
        main()
    except WorkerInterrupted as exc:
        print(str(exc), flush=True)
        sys.exit(75)
    except RequestFailed as exc:
        print(str(exc), flush=True)
        sys.exit(76)
    except ProviderUnavailable as exc:
        print(str(exc), flush=True)
        sys.exit(1)
