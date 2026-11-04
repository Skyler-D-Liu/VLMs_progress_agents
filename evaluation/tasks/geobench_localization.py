"""Geobench localization."""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(UTILS_DIR))
import dior_rsvg_localization as dior
import geobench_spatial as geo
from cot_prompting import REASONING_METHODS, call_with_reasoning, effective_prompt, effective_system_message, validate_cot_run
QWEN_MODEL_KEYS = tuple(dior.build_qwen_model_configs())
TARGET_MODELS = ('gpt-5.5', 'gemini-3.5-flash', *QWEN_MODEL_KEYS, *tuple(dior.build_llava_model_configs()))
LORA_TASK = 'geobench-refdet'
LORA_MODELS = tuple(dior.build_lora_model_configs(LORA_TASK))
dior.MODEL_CONFIGS.update(dior.build_lora_model_configs(LORA_TASK))
LLAVA_MODEL_KEYS = tuple(dior.build_llava_model_configs())
DEFAULT_DATA_DIR = './raw_dataset/geobench_spatial'
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'results/inference/GEOBench-Spatial')
SYSTEM_MESSAGE = 'You are evaluating aerial and satellite imagery. Follow the requested output schema exactly and do not provide an explanation.'

def _load_evaluation_cohort(path: Optional[str | Path]) -> Optional[Dict[str, set[str]]]:
    if path is None:
        return None
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid GEOBench cohort manifest: {manifest_path}') from exc
    splits = payload.get('splits') if isinstance(payload, dict) else None
    if not isinstance(splits, dict):
        raise ValueError(f'Cohort manifest has no splits object: {manifest_path}')
    result: Dict[str, set[str]] = {}
    for name in ('train', 'validation', 'test'):
        values = splits.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f'Cohort manifest has an invalid {name} split: {manifest_path}')
        result[name] = {str(value) for value in values}
    union = set().union(*result.values())
    if len(union) != sum((len(values) for values in result.values())):
        raise ValueError(f'Cohort manifest has split leakage: {manifest_path}')
    return result

def _image_uri(raw_bytes: bytes) -> str:
    return dior.raw_bytes_to_base64_uri(raw_bytes)

def _image_size(raw_bytes: bytes) -> Tuple[int, int]:
    height, width = dior.get_image_size(raw_bytes)
    return (int(width), int(height))

def _response_text(response: Mapping[str, Any]) -> str:
    try:
        value = response['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        return ''
    return value if isinstance(value, str) else str(value or '')

def _response_for_result(response: Mapping[str, Any]) -> Dict[str, Any]:
    return {'choices': [{'message': {'content': _response_text(response)}}]}

def _usage(response: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if isinstance(response.get('usage'), Mapping):
        return response['usage']
    original = response.get('original_response')
    if isinstance(original, Mapping):
        for key in ('usageMetadata', 'usage'):
            if isinstance(original.get(key), Mapping):
                return original[key]
    return None

def _result_path(output_dir: str, model: str, task: str, few_shot_k: int=0, reasoning_method: str='baseline') -> Path:
    suffix = ''
    if few_shot_k:
        suffix += f'-{few_shot_k}shot'
    if reasoning_method == 'cot':
        suffix += '-CoT'
    return Path(output_dir) / f'{model}-geobench-{task}{suffix}.jsonl'

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: ignoring malformed row {line_number} in {path}')
                continue
            if isinstance(row, dict):
                yield row

def _prepare_result_file(path: Path, resume: bool, expected: Mapping[str, Any]) -> set[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not resume):
        raise FileExistsError(f'Result exists: {path}; pass --resume or choose another output directory')
    return geo.clean_resume_jsonl(path, expected)

def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    existed = path.exists()
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    if not existed:
        _fsync_directory(path.parent)

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _bbox_prompt(description: str, width: int, height: int, dataset: str) -> str:
    return geo.localization_prompt(dataset, width, height, description)

def _refdet_items(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for row in rows:
        item = dict(row)
        item['sample_key'] = row['sample_id']
        items.append(item)
    return items

def build_items(manifest: Mapping[str, Any], task: str, support_rows: Sequence[Mapping[str, Any]]=(), cohort_ids: Optional[set[str]]=None) -> List[Dict[str, Any]]:
    rows = [row for row in manifest['tasks']['refdet'] if geo.source_dataset(row) in {'DOTA', 'FAIR1M'}]
    items = _refdet_items(rows)
    if cohort_ids is not None:
        items = [item for item in items if str(item['sample_id']) in cohort_ids]
    if support_rows and len(geo.exclude_few_shot_support_images(items, support_rows)) != len(items):
        # Support samples come from the image-disjoint training partition, so no
        # evaluated query may share an image with a demonstration.
        raise ValueError('Few-shot support images overlap the evaluated Ref-Det samples')
    return items

def _model_metadata(model: str) -> Dict[str, Any]:
    metadata = {'model': model}
    config = dior.MODEL_CONFIGS[model]
    metadata.update({'requested_model': model, 'configured_model_name': config.get('model_name'), 'model_family': config.get('family')})
    if dior.is_local_model(model, dior.MODEL_CONFIGS):
        metadata['decoding'] = 'greedy-do_sample-false'
    elif dior.MODEL_CONFIGS[model].get('family') == 'Gemini':
        metadata['decoding'] = 'temperature-0-candidate-count-1'
    else:
        metadata['decoding'] = 'provider-default-no-stable-seed-contract'
    return metadata


def _few_shot_image_uri(raw_bytes: bytes) -> str:
    from PIL import Image
    with Image.open(io.BytesIO(raw_bytes)) as source:
        image = source.convert('RGB')
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90, optimize=False, progressive=False)
    return _image_uri(output.getvalue())

def _build_few_shot_examples(data_root: Path, support_rows: Sequence[Mapping[str, Any]], task: str, few_shot_k: int, prompt_index: int=0) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in support_rows[:few_shot_k]:
        raw_bytes = (data_root / str(row['image_path'])).read_bytes()
        width, height = (int(row['image_width']), int(row['image_height']))
        source = geo.source_metadata(row)
        dataset = geo.source_dataset(row)
        if dataset is None:
            raise ValueError('Localization input requires source_dataset')
        prompt = _bbox_prompt(str(row['prompt']), width, height, dataset)
        if dataset == 'FAIR1M':
            answer_boxes = geo.ground_truth_polygons(row)
            answer_value = {'bboxes': answer_boxes}
        else:
            answer_boxes = row['ground_truth_bboxes']
            answer_value = {'bboxes': answer_boxes}
        answer = json.dumps(answer_value, ensure_ascii=False, separators=(',', ':'))
        examples.append({'sample_id': row['sample_id'], 'image_url': _few_shot_image_uri(raw_bytes), 'prompt': prompt, 'answer': answer})
    if len(examples) != few_shot_k:
        raise ValueError(f'Expected {few_shot_k} few-shot examples for {task}, found {len(examples)}')
    return examples

def run_model(model: str, task: str, items: Sequence[Dict[str, Any]], data_root: Path, output_dir: str, resume: bool, sample_workers: int, few_shot_k: int=0, reasoning_method: str='baseline', support_rows: Sequence[Mapping[str, Any]]=(), api_key: Optional[str]=None) -> Path:
    result_path = _result_path(output_dir, model, task, few_shot_k, reasoning_method)
    run_metadata = {'dataset': geo.HF_REPO_ID, 'task': task, 'zero_shot': few_shot_k == 0, 'cot': reasoning_method == 'cot'}
    resume_metadata = dict(run_metadata)
    if few_shot_k or reasoning_method == 'cot':
        run_metadata.update({'prompt_mode': 'few-shot' if few_shot_k else 'zero-shot', 'few_shot_k': few_shot_k, 'reasoning_method': reasoning_method})
        if few_shot_k:
            run_metadata.update({'few_shot_support_sample_ids': [str(row['sample_id']) for row in support_rows[:few_shot_k]], 'few_shot_support_partition': 'train'})
        resume_metadata = dict(run_metadata)
    if resume and result_path.exists():
        for row in _iter_jsonl(result_path):
            try:
                prompt = _bbox_prompt(row['official_prompt'], row['image_width'], row['image_height'], row['source_dataset'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Existing localization result {result_path} is incompatible; choose another output directory') from exc
            if row.get('prompt') != prompt:
                raise ValueError(f'Existing localization result {result_path} uses an outdated prompt; choose another output directory')
    completed = _prepare_result_file(result_path, resume, resume_metadata)
    target_keys = {item['sample_key'] for item in items}
    pending = [item for item in items if item['sample_key'] not in completed]
    if len(target_keys) != len(items):
        raise ValueError('Selected items contain duplicate sample keys')
    if not pending:
        print('Evaluation state updated.', flush=True)
        return result_path
    local_query = None
    if dior.is_local_model(model, dior.MODEL_CONFIGS):
        model_path = dior._QWEN_MODEL_PATHS.get(model)
        if not model_path:
            raise ValueError(f'No local checkpoint resolved for {model}')
        local_query = dior.make_qwen_query_callable(model, model_path)
        if os.environ.get('GEOBENCH_ALLOW_LOCAL_SAMPLE_WORKERS') != '1':
            sample_workers = 1
    sample_workers = max(1, sample_workers)
    write_lock = threading.Lock()
    failures: List[Tuple[str, str]] = []
    few_shot_by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    if few_shot_k:
        prompt_indices = [0]
        few_shot_by_prompt = {index: _build_few_shot_examples(data_root, support_rows, task, few_shot_k, prompt_index=index) for index in prompt_indices}
    print('Evaluation state updated.', flush=True)

    def process(item: Dict[str, Any]) -> None:
        key = item['sample_key']
        try:
            relative_image = item['image_path']
            image_path = data_root / relative_image
            raw_bytes = image_path.read_bytes()
            width, height = _image_size(raw_bytes)
            image_uri = _image_uri(raw_bytes)
            source = geo.source_metadata(item)
            dataset = geo.source_dataset(item)
            if task == 'refdet' and dataset is None:
                raise ValueError('Localization input requires source_dataset')
            oriented = dataset == 'FAIR1M'
            base_prompt = _bbox_prompt(item['prompt'], width, height, dataset)
            examples = few_shot_by_prompt.get(int(item.get('prompt_index') or 0), [])
            query_prompt = effective_prompt(base_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)
            if local_query is not None:
                query = local_query
            else:
                request_key = dior.random_api_key_for_model(model, dior.MODEL_CONFIGS, fallback=api_key or '')
                query = dior.get_query_function(model, api_key=request_key, few_shot_examples=examples, deterministic=True)

            def query_once(user_prompt: str) -> Mapping[str, Any]:
                if local_query is not None:
                    return query(image_url=image_uri, system=query_system, user=user_prompt, few_shot_examples=examples)
                return query(image_url=image_uri, system=query_system, user=user_prompt)

            started = time.perf_counter()
            response, raw_response, reasoning_response = call_with_reasoning(
                query_once, reasoning_method, task=dataset.lower(),
                context={'image_width': width, 'image_height': height}, prompt=query_prompt)
            elapsed = time.perf_counter() - started
            answer_text = str(reasoning_response['answer_text'])
            parsed = (geo.parse_fair1m_response if oriented else geo.parse_bbox_response)(answer_text, width, height)
            row: Dict[str, Any] = {'sample_key': key, 'sample_id': item['sample_id'], 'question_id': item.get('question_id'), 'image_path': item['image_path'], 'input_image_path': relative_image, 'image_width': width, 'image_height': height, 'prompt': base_prompt, 'official_prompt': item['prompt'], 'raw_response': raw_response, 'response': _response_for_result(response), 'parsed': parsed, 'format_valid': parsed['format_valid'], 'refused': bool(parsed['refused']), 'elapsed_seconds': elapsed, 'token_usage': _usage(response), **_model_metadata(model), **run_metadata}
            if reasoning_method == 'cot':
                row.update({'answer_text': answer_text, 'reasoning_method': 'cot', 'reasoning': reasoning_response.get('reasoning')})
            row.update({'ground_truth_bboxes': item['ground_truth_bboxes'], 'predicted_bboxes': parsed['bboxes'], 'source': item['source'], 'has_location_cue': item['has_location_cue'], 'source_dataset': dataset})
            if oriented:
                target_polygons = geo.ground_truth_polygons(item)
                row['ground_truth_polygons'] = target_polygons
                row['predicted_polygons'] = parsed['polygons']
                row.update(geo.evaluate_bbox_query(parsed['bboxes'], item['ground_truth_bboxes'], width, height, predicted_polygons=parsed['polygons'], target_polygons=target_polygons))
            else:
                row.update(geo.evaluate_bbox_query(parsed['bboxes'], item['ground_truth_bboxes'], width, height))
            with write_lock:
                _append_jsonl(result_path, row)
                completed.add(key)
                print('Evaluation state updated.', flush=True)
        except Exception as exc:
            with write_lock:
                failures.append((key, repr(exc)))
            print(f'[{model}] failed {key}: {exc}', file=sys.stderr)
            if local_query is not None:
                raise
    with ThreadPoolExecutor(max_workers=sample_workers) as executor:
        futures = [executor.submit(process, item) for item in pending]
        for future in as_completed(futures):
            future.result()
    missing = target_keys - completed
    if missing:
        raise RuntimeError(f'Incomplete {model}/{task}; errors={failures[:3]}. Re-run with --resume.')
    return result_path

def parse_args(argv: Optional[Sequence[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('refdet',))
    parser.add_argument('--model', choices=(*TARGET_MODELS, *LORA_MODELS), help='API model, local model, or task LoRA adapter.')
    parser.add_argument('--few-shot', action='store_true', help='Compatibility alias for 4-shot')
    parser.add_argument('--few-shot-k', type=int, choices=(0, *geo.FEW_SHOT_LEVELS), default=0)
    parser.add_argument('--reasoning-method', choices=REASONING_METHODS, default='baseline')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-data', action='store_true')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--sample-workers', type=int, default=1)
    parser.add_argument('--qwen-model-path')
    parser.add_argument('--qwen-finetuned-root')
    parser.add_argument('--llava-model-path')
    parser.add_argument('--cohort-manifest', help='Optional explicit train/validation/test manifest. For Ref-Det, only the listed test IDs are evaluated; this is required for the fixed Overall Score cohort.')
    args = parser.parse_args(argv)
    if args.few_shot:
        if args.few_shot_k not in (0, 4):
            parser.error('--few-shot cannot be combined with a non-4 --few-shot-k')
        args.few_shot_k = 4
    return args

def main() -> None:
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('geobench-refdet'):
        return
    args = parse_args()
    if args.sample_workers < 1:
        raise ValueError('--sample-workers must be positive')
    dior.configure_cache_environment_from_args(args.cache_dir)
    if args.prepare_data:
        manifest_path = geo.prepare_dataset(args.data_dir, cache_dir=os.environ.get('HF_HUB_CACHE'))
        print(f'Prepared pinned GEOBench-VLM data: {manifest_path}')
        if args.task is None and args.model is None:
            return
    if not args.task or not args.model:
        raise ValueError('--task and --model are required unless --prepare-data is used alone')
    validate_cot_run(reasoning_method=args.reasoning_method, model_key=args.model, few_shot_k=args.few_shot_k, model_kind=dior.MODEL_CONFIGS[args.model].get('model_kind'))
    data_root, manifest = geo.load_manifest(args.data_dir)
    cohort = _load_evaluation_cohort(args.cohort_manifest)
    if cohort is None:
        from evaluation.tasks.create_geobench_refdet_manifest import partition_rows
        cohort = {name: set(ids) for name, ids in partition_rows(build_items(manifest, args.task)).items()}
    support_rows: List[Dict[str, Any]] = []
    if args.few_shot_k:
        # One fixed set of four training-partition samples with distinct images,
        # drawn with a seed-0 shuffle, for every test query and every model.
        train_rows = [row for row in build_items(manifest, args.task) if str(row['sample_id']) in cohort['train']]
        support_rows = geo.few_shot_support_rows(train_rows, args.task)
    items = build_items(manifest, args.task, support_rows, cohort['test'] if cohort is not None and args.task == 'refdet' else None)
    if not items:
        raise ValueError('No samples selected')
    resolved = dior.validate_qwen_model_paths([args.model], dior.MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    dior._QWEN_MODEL_PATHS = resolved
    if not dior.is_local_model(args.model, dior.MODEL_CONFIGS):
        if not dior.api_keys_for_model(args.model, dior.MODEL_CONFIGS):
            raise ValueError(f'No API key configured for {args.model}')
        dior.print_provider_summary()
    path = run_model(model=args.model, task=args.task, items=items, data_root=data_root, output_dir=args.output_dir, resume=args.resume, sample_workers=args.sample_workers, few_shot_k=args.few_shot_k, reasoning_method=args.reasoning_method, support_rows=support_rows)
    print(f'Complete: {path}')
from reasoning_effort import register_reasoning_models
register_reasoning_models(dior.MODEL_CONFIGS)
if __name__ == '__main__':
    main()
