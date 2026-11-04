"""Reasoning comparison."""
from __future__ import annotations
import argparse
import csv
import fcntl
import http.client
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation/utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reasoning_effort import API_EFFORTS, API_MODELS, DEFAULT_ONLY_MODELS, ENV_PATH, OUTPUT_ROOT, Settings, RemoteClient, ProviderUnavailable, RequestFailed
from reasoning_tasks import LOCALIZATION_TASKS, TASKS, RAW_ROOT, TaskInputs, reference_rows, read_jsonl, parse_result, task_metrics
from request_budget import GPT_REQUEST_LIMIT, RequestBudget, RetryLimitReached

class WorkerInterrupted(RuntimeError):
    pass

class UnfinishedRequests(RuntimeError):
    pass

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def freeze_samples(path, rows):
    identities = [r['sample_id'] for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != identities:
                raise ValueError('Canonical sample IDs differ from the saved task sample list')
        else:
            atomic_json(path, identities)

def pause_provider(path, model, reason):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            atomic_json(path, {'model': model, 'reason': reason})

def read_checkpoint(path):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = {}
    offset = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'Malformed checkpoint at line {i + 1}')
            with path.open('r+b') as stream:
                stream.truncate(offset)
            print('Recovered an interrupted final checkpoint write', flush=True)
            break
        sid = row['sample_id']
        if sid in rows:
            raise ValueError(f'Duplicate checkpoint sample {sid}')
        rows[sid] = row
        offset += len(line)
    return rows

def validate_saved(saved, prepared, settings):
    for key, value in prepared.items():
        if saved.get(key) != value:
            raise ValueError(f"Resume mismatch for sample {prepared['sample_id']}: {key}")
    if saved.get('model') != settings.model or saved.get('reasoning_effort') != settings.effort:
        raise ValueError('Refusing cross-model or cross-effort resume')

def _replace_checkpoint(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        for row in rows.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def _write_metrics(path, task, records, settings):
    metrics = [{'model': settings.model, 'reasoning_effort': settings.effort, **m} for m in task_metrics(task, list(records.values()))]
    atomic_json(path, metrics)
    return metrics

def verify_finished(records, selected):
    required = {r['sample_id'] for r in selected}
    if not required.issubset(records) or any((records[sid]['status'] == 'request_failed' and (not records[sid].get('retry_exhausted', False)) for sid in required)):
        raise UnfinishedRequests('Transport failures remain; resume this same command')
    invalid = sum((records[sid]['status'] == 'parse_failed' for sid in required))

def evaluate(args, stop_requested=None):
    stop_requested = stop_requested or (lambda: False)
    settings = Settings(args.model, args.reasoning_effort, args.timeout, args.max_attempts)
    if args.few_shot_k or args.reasoning_method != 'baseline':
        raise ValueError('This comparison requires the original zero-shot baseline prompts')
    if args.workers < 1:
        raise ValueError('Invalid worker count')
    output = Path(args.output_root)
    provider_pause = output / 'providers' / settings.model / 'paused.json'
    if settings.model in API_MODELS and provider_pause.exists():
        raise ProviderUnavailable(f'Provider is paused; inspect {provider_pause}')
    all_rows = reference_rows(args.task, Path(args.cache_dir))
    freeze_samples(output / args.task / 'samples.json', all_rows)
    selected = all_rows
    inputs = TaskInputs(args.task, all_rows, Path(args.cache_dir), settings.model)
    folder = output / args.task / settings.model / settings.effort
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'predictions.jsonl'
    metrics_path = folder / 'metrics.json'
    lock_path = folder / 'writer.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = read_checkpoint(result_path)
        by_id = {r['sample_id']: r for r in all_rows}
        if set(records) - set(by_id):
            raise ValueError('Checkpoint contains samples outside the fixed task cohort')
        reparsed = False
        for sid, saved in list(records.items()):
            if saved.get('answer_text') and saved.get('status') == 'parse_failed':
                repaired = parse_result(args.task, saved)
                reparsed = reparsed or repaired != saved
                records[sid] = repaired
        for sid, saved in records.items():
            prepared, _ = inputs.prepare(by_id[sid])
            validate_saved(saved, prepared, settings)
        if reparsed:
            _replace_checkpoint(result_path, records)
        budget = RequestBudget(folder / 'request_attempts.json', records, by_id, limit=settings.max_attempts)
        pending = [r for r in selected if r['sample_id'] not in records or (records[r['sample_id']]['status'] == 'request_failed' and (not records[r['sample_id']].get('retry_exhausted', False)))]
        if not pending:
            _write_metrics(metrics_path, args.task, records, settings)
            verify_finished(records, selected)
            print(f'Already complete: {args.task} {settings.model} {settings.effort}', flush=True)
            return
        try:
            client = RemoteClient(settings, args.env)
        except ProviderUnavailable as exc:
            pause_provider(provider_pause, settings.model, str(exc))
            raise
        retry_ids = {r['sample_id'] for r in pending}
        if any((sid in records for sid in retry_ids)):
            records = {sid: r for sid, r in records.items() if sid not in retry_ids}
            _replace_checkpoint(result_path, records)

        def process(source):
            if settings.model in API_MODELS and provider_pause.exists():
                raise ProviderUnavailable('Provider was paused by another task')
            prepared, images = inputs.prepare(source)
            start = time.monotonic()

            def before_attempt():
                if stop_requested():
                    raise WorkerInterrupted('Saved request budget before requeue')
                if provider_pause.exists():
                    raise ProviderUnavailable('Provider was paused by another task')
                if budget:
                    budget.reserve(source['sample_id'])
            try:
                from evaluation.utils.response_protocol import query_validated
                response = query_validated(client, args.task, prepared, images, before_attempt=before_attempt,
                                           max_attempts=budget.limit - budget.used(source['sample_id']))
            except RequestFailed as exc:
                response = {'answer_text': '', 'output_text': '', 'status': 'request_failed', 'error': str(exc), 'elapsed_seconds': time.monotonic() - start, 'token_usage': {}, 'attempt_count': settings.max_attempts}
            except ProviderUnavailable as exc:
                if budget and budget.exhausted(source['sample_id']) and str(exc).endswith(': HTTP 400'):
                    response = {'answer_text': '', 'output_text': '', 'status': 'request_failed', 'error': str(exc), 'elapsed_seconds': time.monotonic() - start, 'token_usage': {}, 'attempt_count': settings.max_attempts}
                else:
                    pause_provider(provider_pause, settings.model, str(exc))
                    raise
            if budget:
                response['request_attempts_used'] = budget.used(source['sample_id'])
                response['retry_exhausted'] = response['status'] == 'request_failed' and budget.exhausted(source['sample_id'])
            row = {**prepared, 'model': settings.model, 'reasoning_effort': settings.effort, **response}
            row['response'] = {'choices': [{'message': {'content': row['answer_text']}}]}
            return parse_result(args.task, row)
        iterator = iter(pending)
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = set()
        fatal_error = None
        try:
            for _ in range(min(args.workers, len(pending))):
                futures.add(executor.submit(process, next(iterator)))
            with result_path.open('a') as stream:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        if future.cancelled():
                            continue
                        try:
                            row = future.result()
                        except Exception as exc:
                            if fatal_error is None:
                                fatal_error = exc
                            continue
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                        records[row['sample_id']] = row
                        print(f"{args.task} {settings.model} {settings.effort}: sample={row['sample_id']} status={row['status']} seconds={row['elapsed_seconds']:.1f}", flush=True)
                    if fatal_error is not None:
                        for future in futures:
                            future.cancel()
                    elif not stop_requested():
                        for _ in range(args.workers - len(futures)):
                            source = next(iterator, None)
                            if source is None:
                                break
                            futures.add(executor.submit(process, source))
                    if len(records) % 10 < args.workers or not futures:
                        _write_metrics(metrics_path, args.task, records, settings)
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            _write_metrics(metrics_path, args.task, records, settings)
        if fatal_error is not None:
            raise fatal_error
        if stop_requested() and (not {r['sample_id'] for r in selected}.issubset(records)):
            raise WorkerInterrupted('Checkpoint saved before the allocation time limit')
        verify_finished(records, selected)

def summarize(output=OUTPUT_ROOT):
    output = Path(output)
    rows = []
    combinations = [(model, effort) for model, efforts in API_EFFORTS.items() for effort in efforts]
    for task, spec in TASKS.items():
        for model, effort in combinations:
            metrics_path = output / task / model / effort / 'metrics.json'
            if metrics_path.exists():
                rows.extend(json.loads(metrics_path.read_text()))
            else:
                for name, _, metric in spec.scores:
                    expected = len(rows)
                    rows.append({'model': model, 'reasoning_effort': effort, 'dataset_task': name, 'metric': metric, 'score': None, 'complete': False, 'elapsed_seconds': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'total_tokens': 0})
    output.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temporary = output / 'summary.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / 'summary.csv')
    text = ['# Reasoning effort evaluation', '', 'Scores on incomplete tasks are provisional.', '', '| Task | Model | Effort | Metric | Score | Complete |', '|---|---|---|---|---:|---|']
    for r in rows:
        score = '—' if r['score'] is None else f"{r['score']:.6f}"
        text.append(f"| {r['dataset_task']} | {r['model']} | {r['reasoning_effort']} | {r['metric']} | {score} | {r['complete']} |")
    temporary = output / 'summary.md.tmp'
    temporary.write_text('\n'.join(text) + '\n')
    temporary.replace(output / 'summary.md')
    print(f'Summary: {len(rows)} metric cells in {output}', flush=True)


def main(argv=None, stop_requested=None):
    supplied = list(sys.argv[1:] if argv is None else argv)

    def option(name, default):
        if name in supplied:
            return supplied[supplied.index(name) + 1]
        return next((arg.split('=', 1)[1] for arg in supplied if arg.startswith(name + '=')), default)
    if option('--few-shot-k', '0') != '0' or option('--reasoning-method', 'baseline') != 'baseline':
        from evaluation.experiments.inference_strategy import main as strategy_main
        return strategy_main(supplied, stop_requested=stop_requested)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(TASKS))
    parser.add_argument('--model', choices=API_MODELS)
    parser.add_argument('--reasoning-effort')
    parser.add_argument('--few-shot-k', type=int, default=0)
    parser.add_argument('--reasoning-method', default='baseline')
    parser.add_argument('--workers', '--sample-workers', '--sample-parallel', dest='workers', type=int, default=1)
    parser.add_argument('--output-root', default=str(OUTPUT_ROOT))
    parser.add_argument('--cache-dir', default=str(RAW_ROOT))
    parser.add_argument('--env', default=str(ENV_PATH))
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--max-attempts', type=int, default=None, help='Maximum attempts per input, including format repair')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    if args.max_attempts is None:
        args.max_attempts = GPT_REQUEST_LIMIT
    if not 1 <= args.max_attempts <= GPT_REQUEST_LIMIT:
        parser.error('--max-attempts must be between 1 and 5')
    if args.summarize:
        summarize(args.output_root)
    else:
        if not args.task or not args.model:
            parser.error('--task and --model are required')
        if args.model in API_MODELS and (not args.reasoning_effort):
            parser.error('API models require explicit --reasoning-effort')
        evaluate(args, stop_requested)
if __name__ == '__main__':
    main()
