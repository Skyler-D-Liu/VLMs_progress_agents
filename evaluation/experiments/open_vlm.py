"""Open vlm."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'utils'))
from open_vlm_registry import EVALUATION_TASKS, MODEL_BY_NAME, OUTPUT_ROOT, ROOT
from open_vlm_runtime import NativeClient, UnsupportedModel
from reasoning_tasks import TASKS, RAW_ROOT, TaskInputs, reference_rows, parse_result, task_metrics

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def read_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return {}
    records, offset = ({}, 0)
    raw = path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(raw):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(raw) - 1:
                raise ValueError(f'Malformed checkpoint row {i + 1}')
            with path.open('r+b') as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
            break
        sid = row['sample_id']
        if sid in records:
            raise ValueError(f'Duplicate checkpoint sample: {sid}')
        records[sid] = row
        offset += len(line)
        if i == len(raw) - 1 and (not line.endswith(b'\n')):
            with path.open('ab') as stream:
                stream.write(b'\n')
                stream.flush()
                os.fsync(stream.fileno())
    return records

@contextmanager
def checkpoint_lock(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)
    with (Path(folder) / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def append_answer(folder, row):
    with checkpoint_lock(folder):
        path = Path(folder) / 'predictions.jsonl'
        records = read_checkpoint(path)
        if row['sample_id'] in records:
            raise ValueError('Refusing to replace an existing answer')
        with path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        records[row['sample_id']] = row
        return records

def retain_completed(path, records):
    completed = {sid: row for sid, row in records.items() if row.get('status') in ('ok', 'parse_failed', 'refused', 'incomplete', 'request_failed')}
    if len(completed) != len(records):
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            for row in completed.values():
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return completed

def save_evaluation_status(folder, tasks, value, partial=False):
    folder = Path(folder)
    with (folder / 'status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task in tasks:
            task_value = dict(value)
            if partial:
                with checkpoint_lock(folder / task):
                    records = read_checkpoint(folder / task / 'predictions.jsonl')
                ids = json.loads((folder.parent / 'samples' / (task + '.json')).read_text())
                if set(records) == set(ids):
                    task_value['state'] = 'complete'
                elif task_value['state'] == 'complete':
                    task_value['state'] = 'resumable'
            atomic_json(folder / task / 'status.json', task_value)
        states = {}
        for task in EVALUATION_TASKS:
            path = folder / task / 'status.json'
            states[task] = json.loads(path.read_text())['state'] if path.exists() else 'pending'
        values = set(states.values())
        if values == {'complete'}:
            state = 'complete'
        else:
            state = next((s for s in ('error', 'unsupported', 'running', 'resumable', 'pending') if s in values), 'resumable')
        atomic_json(folder / 'status.json', {'model': value['model'], 'state': state, 'tasks': states})

def freeze_cohort(task, cache_root):
    return reference_rows(task, Path(cache_root))

def validate_saved(saved, prepared, model):
    if saved.get('model') != model:
        raise ValueError('Refusing cross-model resume')
    for key, value in prepared.items():
        if saved.get(key) != value:
            raise ValueError(f"Resume mismatch: sample={prepared['sample_id']} field={key}")

def require_complete(records, rows):
    expected = {row['sample_id'] for row in rows}
    valid = {sid for sid, row in records.items() if row.get('status') in ('ok', 'parse_failed', 'refused', 'incomplete', 'request_failed')}
    if set(records) != expected or valid != expected:
        raise ValueError(f'Incomplete benchmark coverage: valid={len(valid & expected)}/{len(expected)}')

class TaskDataUnavailable(RuntimeError):
    pass

class Evaluation:

    def __init__(self, args, client_holder=None):
        self.args = args
        self.output = Path(args.output_root)
        self.spec = MODEL_BY_NAME[args.model]
        self.client_holder = client_holder if client_holder is not None else []
        self.client = self.client_holder[0] if self.client_holder else None
        self.generation_settings = {'mode': 'official-default'}
        self.task_inputs = {}
        self.stopping = False
        self.end = int('0')

    def stop(self, *_):
        self.stopping = True

    def must_stop(self):
        return self.stopping or bool(self.end and time.time() >= self.end - getattr(self.args, 'stop_margin', 300))

    def run_task(self, task):
        rows = freeze_cohort(task, self.args.cache_dir)
        folder = self.output / self.spec.name / task
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'predictions.jsonl'
        with ExitStack() as locks:
            lock = locks.enter_context((folder / 'evaluation.lock').open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with checkpoint_lock(folder):
                records = read_checkpoint(path)
            by_id = {r['sample_id']: r for r in rows}
            if set(records) - set(by_id):
                raise ValueError('Checkpoint contains noncanonical sample identities')
            chosen = rows
            setup_start = time.monotonic()
            if task not in self.task_inputs:
                try:
                    self.task_inputs[task] = TaskInputs(task, rows, Path(self.args.cache_dir))
                except FileNotFoundError as error:
                    raise TaskDataUnavailable(f'{task}: {error}') from error
            inputs = self.task_inputs[task]
            print(f'Input setup {task}: seconds={time.monotonic() - setup_start:.1f}', flush=True)

            def prepare(row):
                prepared, images = inputs.prepare(row)
                prepared['generation_settings'] = self.generation_settings
                expected_images = 2 if task in ('xview2', 'fpcd') else 1
                if len(images) != expected_images:
                    raise ValueError(f'{task}: expected {expected_images} separate images')
                return (prepared, images)
            for row in chosen:
                saved = records.get(row['sample_id'])
                if saved is not None:
                    prepared, _ = prepare(row)
                    validate_saved(saved, prepared, self.spec.name)
            with checkpoint_lock(folder):
                records = retain_completed(path, read_checkpoint(path))
            pending = [r for r in chosen if r['sample_id'] not in records]
            print(f'{self.spec.name} {task}: evaluation started', flush=True)
            if pending and self.client is None:
                self.client = NativeClient(self.spec, stop_requested=self.must_stop)
                self.client_holder.append(self.client)
            with ThreadPoolExecutor(max_workers=self.args.prepare_workers) as executor:
                iterator = iter(pending)
                futures = []
                for _ in range(min(len(pending), self.args.prepare_workers * 2)):
                    futures.append(executor.submit(prepare, next(iterator)))
                while futures and (not self.must_stop()):
                    prepare_start = time.monotonic()
                    prepared, images = futures.pop(0).result()
                    print(f"Starting {task} sample={prepared['sample_id']} input_wait_seconds={time.monotonic() - prepare_start:.1f}", flush=True)
                    from evaluation.utils.response_protocol import query_validated
                    response = query_validated(self.client, task, prepared, images)
                    if response.get('status') not in ('ok', 'parse_failed', 'refused', 'incomplete', 'request_failed'):
                        raise RuntimeError('Unfinished response was not committed; sample remains retryable')
                    row = {**prepared, 'model': self.spec.name, 'official_repo': self.spec.repo, 'native_generation': self.client.generation, **response}
                    row['response'] = {'choices': [{'message': {'content': row['answer_text']}}]}
                    row = parse_result(task, row)
                    records = append_answer(folder, row)
                    print(f"{self.spec.name} {task}: sample={row['sample_id']} status={row['status']} seconds={row['elapsed_seconds']:.1f}", flush=True)
                    source = next(iterator, None)
                    if source is not None:
                        futures.append(executor.submit(prepare, source))
                for future in futures:
                    future.cancel()
            with checkpoint_lock(folder):
                records = read_checkpoint(path)
                metrics = task_metrics(task, [records[r['sample_id']] for r in rows if r['sample_id'] in records])
                atomic_json(folder / 'metrics.json', [{'model': self.spec.name, **m} for m in metrics])
            required = {r['sample_id'] for r in chosen}
            missing = required - set(records)
            retryable = {sid for name, sid in getattr(self, 'unfinished', ()) if name == task}
            if missing and (self.must_stop() or not missing <= retryable):
                raise InterruptedError('Checkpoint saved before allocation ends')
            if missing:
                return
            require_complete(records, rows)


    def run(self):
        signal.signal(signal.SIGUSR1, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        folder = self.output / self.spec.name
        folder.mkdir(parents=True, exist_ok=True)

        def save_status(value):
            save_evaluation_status(folder, self.args.tasks, value)
        self.blocked, self.unfinished = ([], [])
        try:
            save_status({'model': self.spec.name, 'state': 'running'})
            for task in self.args.tasks:
                try:
                    self.run_task(task)
                except TaskDataUnavailable as error:
                    self.blocked.append(task)
                    print(f'Blocked {task}: {error}', flush=True)
                except ValueError:
                    if not self.unfinished:
                        raise
                if self.must_stop():
                    raise InterruptedError('Evaluation paused')
            if self.unfinished or self.blocked:
                save_status({'model': self.spec.name, 'state': 'resumable'})
                return 3
            save_status({'model': self.spec.name, 'state': 'complete'})
        except InterruptedError:
            save_status({'model': self.spec.name, 'state': 'resumable'})
            return 75
        except Exception as error:
            save_status({'model': self.spec.name, 'state': 'error', 'reason': str(error)})
            raise
        return 0

def main(argv=None, client_holder=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=tuple(MODEL_BY_NAME))
    parser.add_argument('--tasks', nargs='+', choices=EVALUATION_TASKS, default=list(EVALUATION_TASKS))
    parser.add_argument('--output-root')
    parser.add_argument('--cache-dir', default=str(RAW_ROOT))
    parser.add_argument('--prepare-workers', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=64)
    parser.add_argument('--stop-margin', type=int, default=300)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    args.output_root = args.output_root or str(OUTPUT_ROOT)
    if os.environ.get('VLEO_MODEL_LOCK_HELD') == args.model:
        return Evaluation(args, client_holder).run()
    from runtime_lock import operation_lock
    with operation_lock(args.model):
        return Evaluation(args, client_holder).run()
if __name__ == '__main__':
    sys.exit(main())
