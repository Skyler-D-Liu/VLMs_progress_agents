"""Fpcd."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
for path in (TASKS_DIR, PROJECT_ROOT, UTILS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import xview2 as provider_backend
from cot_prompting import REASONING_METHODS, call_with_reasoning, effective_prompt, effective_system_message, result_path_for_reasoning, validate_cot_run
from finetune.utils.fpcd_data import DATASET_ID, FPCDBundle, FPCDRecord, GEOBENCH_DATASET_ID, MASK_TO_OPTION, OPTION_TO_CLASS, SPLIT_SEED, build_fpcd_bundle, write_frozen_manifest
from benchmark_environment import RESULTS_ROOT, api_host_for_model, api_keys_for_model, key_for_model_index, key_index_for_model, safe_key_display
from qwen_local_runtime import build_lora_model_configs, format_qwen_model_listing, normalize_lora_model_selection, result_model_metadata
from llava_local_runtime import format_llava_model_listing
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'FPCD')
DEFAULT_RANDOM_SEED = 0
DEFAULT_SPLIT_SEED = 0
DEFAULT_FEW_SHOT_SEED = 0
LORA_TASK = 'fpcd'
LORA_MODEL_NAME = 'Qwen3-VL-32B-Instruct-LoRA-fpcd'
_API_REQUEST_INTERVAL_SECONDS = 1.8
_API_MAX_ATTEMPTS = 5
_API_RATE_LOCK = threading.Lock()
_API_NEXT_REQUEST_AT = 0.0
SYSTEM_MESSAGE = 'You are a geospatial image analyst. Compare the pre-image and post-image in the supplied order and return only the requested JSON answer.'
USER_PROMPT = 'Look for changes in the farm ponds between the pre- and post-images. Using the provided options, select the most accurate description of the detected change. Give your answer in Json format.\nA. Farm Pond Demolished\nB. Farm Pond Wetted\nC. Farm Pond Dried\nD. No Change\nE. Farm Pond Constructed'
ANSWER_SCHEMA = '{"answer":"A"}'
PROMPT_PROVENANCE = {'dataset_id': GEOBENCH_DATASET_ID, 'task': 'FPCD', 'source_prompt_index': 5, 'modifications': ['single_prompt', 'json_instruction', 'fixed_options']}
PROMPT_PROTOCOL = {'system_message': SYSTEM_MESSAGE, 'user_prompt': USER_PROMPT, 'answer_schema': ANSWER_SCHEMA, 'valid_answers': list('ABCDE'), 'mask_to_option': {str(key): value for key, value in MASK_TO_OPTION.items()}, 'image_order': ['T0/pre-image', 'T1/post-image'], 'provenance': PROMPT_PROVENANCE}
_FENCE_RE = re.compile('^\\s*```(?:json)?\\s*(.*?)\\s*```\\s*$', re.IGNORECASE | re.DOTALL)
_REFUSAL_RE = re.compile("\\b(?:cannot|can't|unable|won't|refus(?:e|ed|al)|not able)\\b", re.IGNORECASE)
_OPTION_DESCRIPTIONS = {'A': ('farm pond demolished', 'demolished'), 'B': ('farm pond wetted', 'wetted'), 'C': ('farm pond dried', 'dried'), 'D': ('no change',), 'E': ('farm pond constructed', 'constructed')}
_ANSWER_KEYS = {'answer', 'change', 'change_type', 'option', 'description'}

def format_answer(option: str) -> str:
    normalized = str(option).strip().upper()
    if normalized not in OPTION_TO_CLASS:
        raise ValueError(f'FPCD answer must be A-E, got {option!r}')
    return json.dumps({'answer': normalized}, separators=(',', ':'), ensure_ascii=True)

def _parse_option_text(candidate: str) -> str:
    options: set[str] = set()
    leading = re.match('^\\s*([A-E])(?:\\s*[.)\\-:]|\\s*$)', candidate, re.IGNORECASE)
    if leading:
        options.add(leading.group(1).upper())
    labeled = re.findall('(?<![a-z0-9_])[\\"\']?(?:answer|change(?:[_ ]type)?|option|description)[\\"\']?\\s*(?:is\\s*)?(?:[:=]\\s*)?[\\"\']?([A-E])(?:\\b|[.)])', candidate, flags=re.IGNORECASE)
    options.update((option.upper() for option in labeled))
    alternatives = re.findall('(?:\\b(?:or|and)\\b|[,/])\\s*([A-E])(?:\\b|[.)])', candidate, flags=re.IGNORECASE)
    options.update((option.upper() for option in alternatives))
    normalized_text = re.sub('[^a-z0-9]+', ' ', candidate.casefold()).strip()
    for option, descriptions in _OPTION_DESCRIPTIONS.items():
        if any((re.search(f'\\b{re.escape(description)}\\b', normalized_text) for description in descriptions)):
            options.add(option)
    if len(options) != 1:
        raise ValueError('FPCD response does not identify exactly one option')
    return next(iter(options))

def _parse_structured_option(value: Any) -> str:
    if isinstance(value, str):
        return _parse_option_text(value.strip())
    if isinstance(value, dict):
        normalized = {str(key).strip().casefold().replace('-', '_').replace(' ', '_'): item for key, item in value.items()}
        values = [item for key, item in normalized.items() if key in _ANSWER_KEYS]
        if not values:
            if len(value) != 1:
                raise ValueError('FPCD JSON answer is ambiguous')
            values = [next(iter(value.values()))]
    elif isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            raise ValueError('FPCD JSON answer list is empty')
    else:
        raise ValueError('FPCD JSON answer has an unsupported value type')
    parsed_options = {_parse_structured_option(item) for item in values}
    if len(parsed_options) != 1:
        raise ValueError('FPCD JSON answer fields disagree')
    return next(iter(parsed_options))

def parse_answer(text: str) -> str:
    from evaluation.utils.response_protocol import parse_answer as shared_parse
    value = shared_parse('fpcd', text)
    if value is None:
        raise ValueError('Invalid answer format')
    return value

def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(str(text)))

def _base_model_configs() -> dict[str, dict[str, Any]]:
    return {name: dict(config) for name, config in provider_backend.MODEL_CONFIGS.items() if config.get('model_kind') != 'lora'}
MODEL_CONFIGS = _base_model_configs()
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))

def _sync_provider_backend() -> None:
    provider_backend.MODEL_CONFIGS = MODEL_CONFIGS
    provider_backend.LORA_TASK = LORA_TASK
    provider_backend.SYSTEM_MESSAGE = SYSTEM_MESSAGE
    provider_backend.USER_PROMPT = USER_PROMPT

def configure_cache_environment_from_args(cache_dir: str | None=None) -> str:
    return provider_backend.configure_cache_environment_from_args(cache_dir)

def pin_canonical_fpcd_source() -> None:
    override = os.environ.pop('FPCD_ZENODO_RECORD_ID', None)
    if override:
        print('Ignoring FPCD_ZENODO_RECORD_ID={!r}: model evaluation is pinned to {}.'.format(override, DATASET_ID), flush=True)

def expand_model_selection(model_arg: str | None) -> list[str]:
    model_arg = normalize_lora_model_selection(model_arg, LORA_TASK)
    if model_arg is None:
        return [name for name, config in MODEL_CONFIGS.items() if config.get('family') not in {'Qwen', 'LLaVA'}]
    family_aliases = {'GPT-all': 'GPT', 'Gemini-all': 'Gemini', 'Qwen-all': 'Qwen', 'LLaVA-all': 'LLaVA'}
    if model_arg in family_aliases:
        family = family_aliases[model_arg]
        return [name for name, config in MODEL_CONFIGS.items() if config.get('family') == family and config.get('model_kind') != 'lora']
    if model_arg not in MODEL_CONFIGS:
        raise ValueError(f'unknown model: {model_arg}')
    return [model_arg]

def _is_local(model_key: str) -> bool:
    return MODEL_CONFIGS[model_key].get('api_type') == 'hf_local'

def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, 'item', None)
    if callable(item):
        return item()
    raise TypeError(f'cannot JSON-encode {type(value).__name__}')

def _strip_generated_tracking_metadata(response: Any) -> None:
    if not isinstance(response, dict):
        return
    original_response = response.get('original_response')
    if isinstance(original_response, dict):
        original_response.pop('transformers_version', None)

def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=True, default=_json_default) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid JSONL at {path}:{line_number}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'non-object JSONL row at {path}:{line_number}')
            yield value

def clean_result_jsonl_for_resume(path: str | Path) -> set[str]:
    result_path = Path(path)
    if not result_path.exists():
        return set()
    valid_lines: list[str] = []
    completed: set[str] = set()
    damaged = False
    with result_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                sample_id = str(value['sample_id'])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                damaged = True
                break
            if sample_id in completed:
                raise ValueError(f'duplicate sample_id in result file: {sample_id}')
            completed.add(sample_id)
            valid_lines.append(json.dumps(value, ensure_ascii=True, default=_json_default) + '\n')
    if damaged:
        temporary = result_path.with_name(f'.{result_path.name}.resume-{os.getpid()}')
        temporary.write_text(''.join(valid_lines), encoding='utf-8')
        os.replace(temporary, result_path)
    return completed

def _run_metadata(model_key: str, few_shot_k: int, few_shot_source_ids: Sequence[str]=(), evaluation_source_ids: Sequence[str]=(), reasoning_method: str='baseline') -> dict[str, Any]:
    metadata = {'model': model_key, 'dataset_id': DATASET_ID, 'canonical_split': 'test', 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'prompt_mode': 'few-shot' if few_shot_k else 'zero-shot', 'few_shot_k': few_shot_k, 'few_shot_seed': DEFAULT_FEW_SHOT_SEED if few_shot_k else None, 'few_shot_source_ids': list(few_shot_source_ids), 'few_shot_selection': 'seeded_shuffle_prefix_nested_2_4_8' if few_shot_k else None, 'label_protocol': 'original_fpcd_multi_class_mask_single_nonzero_class'}
    if reasoning_method == 'cot':
        metadata['reasoning_method'] = reasoning_method
    return metadata

def validate_resume_config(path: str | Path, expected: Mapping[str, Any], evaluation_source_ids: Sequence[str]=()) -> None:
    result_path = Path(path)
    if not result_path.exists():
        return
    for record in iter_jsonl(result_path):
        for key in expected:
            if record.get(key) != expected[key]:
                raise ValueError(f'resume configuration mismatch for {key}: {record.get(key)!r} != {expected[key]!r} in {result_path}')

def validate_resume_records(path: str | Path, expected_records: Sequence[FPCDRecord]) -> None:
    result_path = Path(path)
    if not result_path.exists():
        return
    expected_by_id = {record.source_id: record for record in expected_records}
    for row in iter_jsonl(result_path):
        sample_id = str(row.get('sample_id') or '')
        record = expected_by_id.get(sample_id)
        if record is None:
            raise ValueError(f'existing result sample {sample_id!r} is outside the current evaluation')
        expected_fields = {'source_id': record.source_id, 'raw_split': record.raw_split, 'raw_index': record.raw_index, 'canonical_split': 'test', 'ground_truth': record.answer, 'ground_truth_class': record.class_name, 'mask_classes': list(record.mask_classes), 'image_order': ['T0/pre-image', 'T1/post-image'], 'prompt': USER_PROMPT, 'answer_schema': ANSWER_SCHEMA}
        for key, expected_value in expected_fields.items():
            if row.get(key) != expected_value:
                raise ValueError(f'resume record mismatch for {sample_id} field {key}: {row.get(key)!r} != {expected_value!r}')

def select_few_shot_records(train_records: Sequence[FPCDRecord], k: int, seed: int=DEFAULT_FEW_SHOT_SEED) -> tuple[FPCDRecord, ...]:
    if k < 0:
        raise ValueError('few-shot k must be non-negative')
    candidates = sorted(train_records, key=lambda record: record.source_id)
    random.Random(seed).shuffle(candidates)
    if k > len(candidates):
        raise ValueError(f'requested {k} FPCD examples from only {len(candidates)} train rows')
    selected = tuple(candidates[:k])
    if any((record.canonical_split != 'train' for record in selected)):
        raise RuntimeError('few-shot examples must come only from canonical train')
    return selected

def build_few_shot_examples(bundle: FPCDBundle, records: Sequence[FPCDRecord]) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        t0, t1 = bundle.read_image_bytes(record)
        examples.append({'source_id': record.source_id, 'image_urls': provider_backend.image_data_uris_from_raw_bytes_list([t0, t1]), 'user_prompt': USER_PROMPT, 'answer': format_answer(record.answer)})
    return examples

def configure_api_request_policy(*, interval_seconds: float, max_attempts: int) -> None:
    global _API_REQUEST_INTERVAL_SECONDS, _API_MAX_ATTEMPTS
    if interval_seconds < 0:
        raise ValueError('API request interval must be non-negative')
    if max_attempts < 1:
        raise ValueError('API max attempts must be positive')
    _API_REQUEST_INTERVAL_SECONDS = float(interval_seconds)
    _API_MAX_ATTEMPTS = int(max_attempts)

def _wait_for_api_slot() -> None:
    global _API_NEXT_REQUEST_AT
    with _API_RATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _API_NEXT_REQUEST_AT - now)
        if delay:
            time.sleep(delay)
        _API_NEXT_REQUEST_AT = time.monotonic() + _API_REQUEST_INTERVAL_SECONDS

def _provider_error_kind(exc: BaseException) -> str:
    message = str(exc).casefold()
    if any((token in message for token in ('invalid token', 'invalid api key'))):
        return 'invalid_key'
    if any((token in message for token in ('rate limit', 'too many requests'))):
        return 'rate_limit'
    if 'model_not_found' in message:
        return 'unavailable_model'
    return 'transient'

def _undecorated(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    return getattr(function, '__wrapped__', function)

def _query_function(model_key: str, api_key: str | None) -> Callable[..., dict[str, Any]]:
    _sync_provider_backend()
    if _is_local(model_key):
        return provider_backend.get_query_function(model_key, api_key=api_key)
    configured_keys = api_keys_for_model(model_key, MODEL_CONFIGS)
    keys = list(dict.fromkeys(configured_keys or ([api_key] if api_key else [])))
    if not keys:
        raise ValueError(f'No API key configured for {model_key}')
    if api_key in keys:
        offset = keys.index(api_key)
        keys = keys[offset:] + keys[:offset]
    family = MODEL_CONFIGS[model_key].get('family')
    provider_call = _undecorated(provider_backend.query_gemini if family == 'Gemini' else provider_backend.query_openai)
    key_cursor = 0
    cursor_lock = threading.Lock()
    unavailable_error: BaseException | None = None

    def query(*, image_urls: Sequence[str], system: str, user: str, few_shot_examples=None, sample_index: int | None=None) -> dict[str, Any]:
        nonlocal key_cursor, unavailable_error
        with cursor_lock:
            if unavailable_error is not None:
                raise unavailable_error
        last_error: BaseException | None = None
        for attempt in range(1):
            with cursor_lock:
                selected_key = keys[key_cursor % len(keys)]
                key_cursor += 1
            _wait_for_api_slot()
            try:
                return provider_call(image_urls=list(image_urls), system=system, user=user, model_key=model_key, api_key=selected_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
            except Exception as exc:
                last_error = exc
                kind = _provider_error_kind(exc)
                if kind == 'unavailable_model':
                    with cursor_lock:
                        unavailable_error = last_error
                    break
        assert last_error is not None
        raise last_error
    return query

def _result_path(output_dir: str | Path, model_key: str, few_shot_k: int, reasoning_method: str='baseline') -> Path:
    suffix = f'-{few_shot_k}shot' if few_shot_k else ''
    path = Path(output_dir) / f'{model_key}-fpcd-classification{suffix}.jsonl'
    return result_path_for_reasoning(path, reasoning_method)

def query_model_on_dataset(*, model_key: str, bundle: FPCDBundle, test_records: Sequence[FPCDRecord], output_dir: str | Path=DEFAULT_RESULT_DIR, result_path: str | Path | None=None, api_key: str | None=None, few_shot_k: int=0, sample_parallel: int=4, reasoning_method: str='baseline') -> Path:
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if model_key == LORA_MODEL_NAME and few_shot_k:
        raise ValueError('the FPCD LoRA model is evaluated zero-shot only')
    if any((record.canonical_split != 'test' for record in test_records)):
        raise RuntimeError('FPCD inference scope must contain canonical test records only')
    selected_examples = select_few_shot_records(bundle.records('train'), few_shot_k)
    few_shot_source_ids = [record.source_id for record in selected_examples]
    evaluation_source_ids = [record.source_id for record in test_records]
    metadata = _run_metadata(model_key, few_shot_k, few_shot_source_ids, evaluation_source_ids, reasoning_method)
    path = Path(result_path) if result_path else _result_path(output_dir, model_key, few_shot_k, reasoning_method)
    path = result_path_for_reasoning(path, reasoning_method)
    completed = clean_result_jsonl_for_resume(path)
    validate_resume_config(path, metadata, evaluation_source_ids)
    validate_resume_records(path, test_records)
    unexpected = completed - set(evaluation_source_ids)
    if unexpected:
        raise ValueError(f'result file contains sample IDs outside this evaluation selection: {sorted(unexpected)[:3]}')
    pending = [record for record in test_records if record.source_id not in completed]
    workers = 1 if _is_local(model_key) else max(1, int(sample_parallel))
    print(f"[{model_key}] FPCD {metadata['prompt_mode']}: workers={workers}", flush=True)
    if not pending:
        return path
    query_fn = _query_function(model_key, api_key)
    few_shot_examples = build_few_shot_examples(bundle, selected_examples)

    def process(record: FPCDRecord) -> tuple[dict[str, Any], str]:
        raw_images = list(bundle.read_image_bytes(record))
        task_prompt = USER_PROMPT
        query_prompt = effective_prompt(task_prompt, reasoning_method)
        query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

        def call_model(user_prompt: str):
            return provider_backend.query_images(query_fn=query_fn, raw_image_bytes_list=raw_images, model_key=model_key, sample_index=record.raw_index, system=query_system, user=user_prompt, few_shot_examples=few_shot_examples)
        response, content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='fpcd', context={}, prompt=query_prompt, max_attempts=_API_MAX_ATTEMPTS)
        _strip_generated_tracking_metadata(response)
        answer_text = reasoning_response['answer_text']
        refused = is_refusal(answer_text)
        try:
            parsed = parse_answer(answer_text)
            parse_failed = False
        except ValueError:
            parsed = None
            parse_failed = True
        row = {'sample_id': record.source_id, 'source_id': record.source_id, 'raw_split': record.raw_split, 'raw_index': record.raw_index, **metadata, **result_model_metadata(model_key, MODEL_CONFIGS), 'image_order': ['T0/pre-image', 'T1/post-image'], 'prompt': task_prompt, 'answer_schema': ANSWER_SCHEMA, 'prompt_provenance': PROMPT_PROVENANCE, 'mask_classes': list(record.mask_classes), 'ground_truth': record.answer, 'ground_truth_class': record.class_name, 'response': response, 'parsed_answer': parsed, 'parse_failed': parse_failed, 'refused': refused}
        if reasoning_method == 'cot':
            row['answer_text'] = answer_text
            row.update({'reasoning_method': 'cot', 'reasoning': reasoning_response.get('reasoning')})
        return (row, content)
    if workers == 1:
        for record in pending:
            row, content = process(record)
            _append_jsonl(path, row)
            print(f'[{model_key}] {record.source_id}: {content[:80]!r} (GT={record.answer})', flush=True)
    else:
        failures: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, record): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    row, content = future.result()
                except Exception as exc:
                    failures.append((record.source_id, exc))
                    print(f'[{model_key}] {record.source_id} failed and remains resumable: {exc}', flush=True)
                    continue
                _append_jsonl(path, row)
                print(f"[{model_key}] {row['sample_id']}: {content[:80]!r} (GT={row['ground_truth']})", flush=True)
        if failures:
            source_id, error = failures[0]
            raise RuntimeError(f'{model_key} failed on {len(failures)} FPCD sample(s); first failure {source_id}: {error}. Completed rows remain resumable in {path}') from error
    return path

def _empty_confusion() -> dict[str, dict[str, int]]:
    return {truth: {prediction: 0 for prediction in (*OPTION_TO_CLASS, 'INVALID')} for truth in OPTION_TO_CLASS}

def evaluation(result_path: str | Path, ax: Any=None) -> dict[str, Any]:
    del ax
    records = list(iter_jsonl(result_path))
    if not records:
        raise ValueError(f'FPCD result has no predictions: {result_path}')
    seen: set[str] = set()
    confusion = _empty_confusion()
    correct = parse_failures = refusals = 0
    true_positive = Counter()
    predicted = Counter()
    observed_truth = Counter()
    metadata: dict[str, Any] = {}
    invariant_keys = ('model', 'dataset_id', 'canonical_split', 'canonical_split_seed', 'prompt_mode', 'few_shot_k', 'few_shot_seed', 'few_shot_source_ids', 'few_shot_selection', 'label_protocol', 'reasoning_method', 'image_order', 'prompt', 'answer_schema')
    first_invariants = {key: records[0].get(key) for key in invariant_keys}
    for row in records:
        sample_id = str(row.get('sample_id') or '')
        if not sample_id or sample_id in seen:
            raise ValueError(f'missing or duplicate sample_id in {result_path}: {sample_id!r}')
        seen.add(sample_id)
        truth = str(row.get('ground_truth') or '').upper()
        if truth not in OPTION_TO_CLASS:
            raise ValueError(f'invalid FPCD ground truth in {result_path}: {truth!r}')
        for key, expected_value in first_invariants.items():
            if row.get(key) != expected_value:
                raise ValueError(f'mixed FPCD run metadata for {key} in {result_path}: {row.get(key)!r} != {expected_value!r}')
        content = row.get('answer_text')
        if not isinstance(content, str):
            content = provider_backend.extract_response_content(row.get('response'))
        refused = is_refusal(content)
        try:
            answer = parse_answer(content)
        except ValueError:
            answer = None
        parse_failed = answer is None
        parse_failures += int(parse_failed)
        refusals += int(refused)
        observed_truth[truth] += 1
        predicted[answer or 'INVALID'] += 1
        confusion[truth][answer or 'INVALID'] += 1
        if answer == truth:
            correct += 1
            true_positive[truth] += 1
        if not metadata:
            metadata = {key: row.get(key) for key in ('model', 'model_kind', 'adapter_name', 'prompt_mode', 'few_shot_k', 'few_shot_seed', 'dataset_id', 'canonical_split', 'reasoning_method')}
    requested_count = len(records)
    recall = {}
    f1 = {}
    for option in OPTION_TO_CLASS:
        support = observed_truth[option]
        tp = true_positive[option]
        recall[option] = tp / support if support else 0.0
        fp = predicted[option] - tp
        fn = support - tp
        denominator = 2 * tp + fp + fn
        f1[option] = 2 * tp / denominator if denominator else 0.0
    metrics = {**metadata, 'result_file': str(Path(result_path).resolve()), 'primary_metric': 'accuracy', 'accuracy': correct / requested_count, 'parse_failure_rate': parse_failures / requested_count, 'refusal_rate': refusals / requested_count, 'per_class_recall': recall, 'macro_f1': sum(f1.values()) / len(f1), 'per_class_f1': f1}
    metrics_path = Path(result_path).with_suffix('.metrics.json')
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=True) + '\n', encoding='utf-8')
    print(f"{Path(result_path).name}: accuracy={metrics['accuracy']:.4f}, parse_failure={metrics['parse_failure_rate']:.4f}", flush=True)
    return metrics

def _result_files(result_dir: str | Path) -> list[Path]:
    return sorted((path for path in Path(result_dir).glob('*-fpcd-classification*.jsonl') if not path.name.endswith('-CoT.jsonl')))

def plot_model_comparison(result_dir: str | Path=DEFAULT_RESULT_DIR) -> list[dict[str, Any]]:
    output_dir = Path(result_dir)
    metrics = [evaluation(path) for path in _result_files(output_dir)]
    for value in metrics:
        model = str(value.get('model') or '')
        config = MODEL_CONFIGS.get(model, {})
        value['release_date'] = config.get('release_date')
        value['family'] = config.get('family')
    summary_path = output_dir / 'model_metrics_summary_all_shots.json'
    summary_path.write_text(json.dumps({'dataset': 'FPCD', 'runs': metrics}, indent=2, sort_keys=True, ensure_ascii=True) + '\n', encoding='utf-8')
    zero_shot = [value for value in metrics if int(value.get('few_shot_k') or 0) == 0]
    csv_path = output_dir / 'model_metrics_zero_shot.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        columns = ('model', 'family', 'release_date', 'accuracy', 'parse_failure_rate', 'refusal_rate', 'macro_f1')
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for value in zero_shot:
            writer.writerow({column: value.get(column) for column in columns})
    if not zero_shot:
        return metrics
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except ImportError:
        return metrics
    colors = provider_backend.FAMILY_COLORS
    for metric_name in ('accuracy', 'parse_failure_rate'):
        figure, axis = plt.subplots(figsize=(12, 6))
        for family in provider_backend.PLOT_FAMILY_ORDER:
            family_rows = sorted((value for value in zero_shot if value.get('family') == family and value.get('release_date')), key=lambda value: value['release_date'])
            if not family_rows:
                continue
            dates = [datetime.fromisoformat(value['release_date']) for value in family_rows]
            values = [float(value[metric_name]) for value in family_rows]
            axis.plot(dates, values, marker='o', label=family, color=colors[family])
        axis.set_xlabel('Release date')
        axis.set_ylabel(metric_name.replace('_', ' ').title())
        axis.set_title(f"FPCD zero-shot {metric_name.replace('_', ' ')}")
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output_dir / f'fpcd_zero_shot_{metric_name}_over_time.pdf')
        plt.close(figure)
    return metrics

def _list_models(qwen_finetuned_root: str | None, llava_model_root: str | None) -> None:
    for family in ('GPT', 'Gemini', 'Qwen', 'LLaVA'):
        print(f'\n{family} Models:')
        for name, config in MODEL_CONFIGS.items():
            if config.get('family') != family:
                continue
            if family == 'Qwen':
                print(format_qwen_model_listing(name, config, qwen_finetuned_root))
            elif family == 'LLaVA':
                print(format_llava_model_listing(name, config, llava_model_root))
            else:
                print(f"  {name} (released: {config['release_date']})")

def _build_run_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(model, args.few_shot_k) for model in expand_model_selection(args.model)]

def select_test_records(records: Sequence[FPCDRecord], seed: int=DEFAULT_RANDOM_SEED) -> tuple[FPCDRecord, ...]:
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return tuple(selected)

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Model name, GPT-all, Gemini-all, or Qwen-all')
    parser.add_argument('--few-shot', action='store_true', help='Use 4-shot for a normal model run')
    parser.add_argument('--few-shot-k', type=int, default=0)
    parser.add_argument('--reasoning-method', choices=REASONING_METHODS, default='baseline', help='Prompting method (default: baseline)')
    parser.add_argument('--output-dir', default=DEFAULT_RESULT_DIR)
    parser.add_argument('--output', help='Custom JSONL path for one model/shot run')
    parser.add_argument('--cache-dir')
    parser.add_argument('--sample-parallel', type=int, default=4)
    parser.add_argument('--api-request-interval', type=float, default=1.8)
    parser.add_argument('--api-max-attempts', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--api-key-offset', type=int, default=0)
    parser.add_argument('--qwen-model-path')
    parser.add_argument('--llava-model-path')
    parser.add_argument('--qwen-finetuned-root')
    parser.add_argument('--evaluate-only')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args(argv)
    if args.few_shot and args.few_shot_k == 0:
        args.few_shot_k = 4
    if args.few_shot_k < 0:
        parser.error('--few-shot-k must be non-negative')
    if args.sample_parallel < 1:
        parser.error('--sample-parallel must be at least 1')
    if args.api_request_interval < 0:
        parser.error('--api-request-interval must be non-negative')
    if args.api_max_attempts < 1:
        parser.error('--api-max-attempts must be positive')
    if args.api_key_offset < 0:
        parser.error('--api-key-offset must be non-negative')
    return args

def main(argv: Sequence[str] | None=None) -> int:
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('fpcd'):
        return
    args = parse_args(argv)
    configure_api_request_policy(interval_seconds=args.api_request_interval, max_attempts=args.api_max_attempts)
    cache_root = configure_cache_environment_from_args(args.cache_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.list_models:
        _list_models(args.qwen_finetuned_root, args.llava_model_path)
        return 0
    if args.evaluate_only:
        evaluation(args.evaluate_only)
        return 0
    if args.plot_only:
        plot_model_comparison(output_dir)
        return 0
    run_tasks = _build_run_tasks(args)
    for model_key, few_shot_k in run_tasks:
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if args.output and len(run_tasks) != 1:
        raise ValueError('--output is valid only for one model/shot run')
    local_models = list(dict.fromkeys((model for model, _ in run_tasks if _is_local(model))))
    _sync_provider_backend()
    provider_backend._QWEN_MODEL_PATHS = provider_backend.validate_qwen_model_paths(local_models, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    pin_canonical_fpcd_source()
    if SPLIT_SEED != DEFAULT_SPLIT_SEED:
        raise RuntimeError('FPCD split seed differs from the xView2 seed-0 protocol')
    bundle = build_fpcd_bundle(cache_root, strict=True)
    write_frozen_manifest(bundle, output_dir / 'fpcd_split_manifest.json', prompt=USER_PROMPT, answer_schema=ANSWER_SCHEMA, prompt_protocol=PROMPT_PROTOCOL)
    test_records = select_test_records(bundle.records('test'))
    result_paths = []
    for task_index, (model_key, few_shot_k) in enumerate(run_tasks):
        api_key = None
        if not _is_local(model_key):
            key_position = args.api_key_offset + task_index
            api_key = key_for_model_index(model_key, key_position, MODEL_CONFIGS)
            shown = safe_key_display(api_key) if api_key else 'missing'
            host = api_host_for_model(model_key, MODEL_CONFIGS)
            print(f'{model_key} -> {host}, API key #{key_index_for_model(model_key, key_position, MODEL_CONFIGS)} ({shown})', flush=True)
        path = query_model_on_dataset(model_key=model_key, bundle=bundle, test_records=test_records, output_dir=output_dir, result_path=args.output, api_key=api_key, few_shot_k=few_shot_k, sample_parallel=args.sample_parallel, reasoning_method=args.reasoning_method)
        result_paths.append(path)
        evaluation(path)
    if len(result_paths) > 1:
        plot_model_comparison(output_dir)
    return 0
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    raise SystemExit(main())
