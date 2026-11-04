"""Benchmark evaluation."""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/inference'

@dataclass(frozen=True)
class Task:
    directory: str
    suffix: str
TASKS = {'landmarks': Task('Landmarks', '-landmarks.jsonl'), 'dior-recognition': Task('global_vs_local', '-dior-rsvg-local-classification.jsonl'), 'pattern-net': Task('PatternNet', '-classification.jsonl'), 'rare-planes': Task('aircraft_counting/test_selection', '-counting.jsonl'), 'cowc': Task('cowc', '-combined-counting.jsonl'), 'xview2': Task('xView2', '-counting.jsonl'), 'neon-trees': Task('NeonTreeEvaluation', '-counting.jsonl'), 'aerial-animal': Task('aerial-animal-population-4tu', '-counting.jsonl'), 'dior-refdet': Task('DIOR-RSVG', '-dior-rsvg-detection.jsonl'), 'geobench-refdet': Task('GEOBench-Spatial', '-geobench-refdet.jsonl'), 'fpcd': Task('FPCD', '-fpcd-classification.jsonl')}
API_MODELS = ('gpt-6.1-sol', 'gpt-6-astra', 'gpt-5.6-sol', 'gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.1-pro-preview', 'gemini-3-flash-preview')

def family(model):
    if model.startswith('gemini-'):
        return 'Gemini'
    if model.startswith('Qwen'):
        return 'Qwen'
    if model.startswith('LLaVA'):
        return 'LLaVA'
    return 'GPT'


def result_path(task, model, root=RESULTS):
    spec = TASKS[task]
    candidates = [root / spec.directory / (model + spec.suffix)]
    if task == 'landmarks':
        candidates.insert(0, root / 'Landmarks_qwen32_fixed_20260722' / (model + spec.suffix))
    if task == 'aerial-animal':
        candidates.append(root.parent / 'fewshot' / spec.directory / (model + spec.suffix))
    candidates.extend((root / 'reasoning_effort' / task / model / effort / 'predictions.jsonl' for effort in ('medium', 'native', 'default')))
    return next((p for p in candidates if p.is_file()), candidates[0])

def identity(task, row):
    if task == 'cowc':
        if row.get('city') is None or row.get('index') is None:
            raise ValueError('COWC sample requires city and index')
        return f"{row['city']}:{row['index']}"
    if task == 'geobench-refdet':
        value = row.get('question_id', row.get('source_index'))
        if value is not None:
            return f'refdet:{int(value)}'
    for key in ('source_id', 'sample_id', 'sample_key', 'index'):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError('Sample requires a business identity')

def response_text(row):
    for key in ('answer_text', 'output_text', 'raw_answer'):
        if key in row and row[key] is not None:
            return str(row[key])
    response = row if 'choices' in row or 'candidates' in row else row.get('response', row.get('raw_response', ''))
    if isinstance(response, dict):
        choices = response.get('choices') or []
        if choices:
            return str(choices[0].get('message', {}).get('content') or '')
        candidates = response.get('candidates') or []
        if candidates:
            return ''.join(part.get('text', '') for part in candidates[0].get('content', {}).get('parts', []) if not part.get('thought'))
        return str(response.get('content') or '')
    return str(response or '')

def answered(row):
    if row.get('status') in {'error', 'request_failed', 'incomplete'} or row.get('error'):
        return False
    if response_text(row).strip():
        return True
    response = row.get('response')
    choices = response.get('choices') or [] if isinstance(response, dict) else []
    finish = row.get('finish_reason') or (choices[0].get('finish_reason') if choices else None)
    return bool(row.get('refused') and finish in {'stop', 'STOP'})

def read_checkpoint(path, task, model=None, repair=False):
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    offset = 0
    for i, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError(f'Invalid checkpoint row {i + 1} in {path}') from None
            if repair:
                with path.open('r+b') as stream:
                    stream.truncate(offset)
            break
        if not isinstance(row, dict):
            raise ValueError('Checkpoint row must be an object')
        if task == 'dior-recognition' and (not all((key in row for key in ('row_index', 'bbox', 'label')))):
            offset += len(line)
            continue
        if task == 'geobench-refdet' and (not all((key in row for key in ('question_id', 'ground_truth_bboxes')))):
            offset += len(line)
            continue
        if model is not None and row.get('model') not in (None, model):
            if row.get('model') == family(model).lower():
                offset += len(line)
                continue
            raise ValueError('Checkpoint model differs from selected model')
        if (row.get('few_shot_k') or 0) != 0 or (row.get('reasoning_method') or 'baseline') != 'baseline':
            raise ValueError('Checkpoint must use zero-shot baseline')
        if row.get('prompt_mode') not in (None, 'zero-shot') or row.get('cot', False):
            raise ValueError('Checkpoint must use zero-shot baseline')
        sid = identity(task, row)
        if sid in rows and answered(row) and answered(rows[sid]):
            raise ValueError(f'Duplicate sample {sid}')
        if sid not in rows or answered(row):
            rows[sid] = row
        offset += len(line)
    return rows


class AdaptiveLimiter:

    def __init__(self, maximum, *, initial=None, recovery_successes=100, backoff_errors=False):
        self.maximum = maximum
        self.limit = min(maximum, initial) if initial is not None else maximum
        self.recovery_successes = recovery_successes
        self.backoff_errors = backoff_errors
        self.active = self.stable = 0
        self.condition = threading.Condition()
        self.cooldown = 0.0

    def __enter__(self):
        with self.condition:
            while self.active >= self.limit or time.monotonic() < self.cooldown:
                self.condition.wait(timeout=0.2)
            self.active += 1

    def __exit__(self, *_):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def report(self, status):
        with self.condition:
            if status == 429 or (self.backoff_errors and (status is None or status in {408, 500, 502, 503, 504})):
                self.limit = max(1, self.limit // 2)
                self.stable = 0
                self.cooldown = time.monotonic() + 2
            elif status == 200:
                self.stable += 1
                if self.stable >= self.recovery_successes:
                    self.limit = min(self.maximum, self.limit * 2)
                    self.stable = 0
            else:
                self.stable = 0
            self.condition.notify_all()
