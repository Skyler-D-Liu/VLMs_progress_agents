"""Dior rsvg recognition."""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import random
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UTILS_DIR))
import dior_rsvg_localization as dior
TASK_NAME = 'dior-local-classification'
RESULT_TASK_STEM = 'dior-rsvg-local-classification'
DEFAULT_RESULT_DIR = str(dior.RESULTS_ROOT / 'global_vs_local')
DEFAULT_RANDOM_SEED = dior.DEFAULT_RANDOM_SEED
DEFAULT_FEW_SHOT_K = dior.DEFAULT_FEW_SHOT_K
DEFAULT_FEW_SHOT_SEED = dior.DEFAULT_FEW_SHOT_SEED
DEFAULT_SAMPLE_WORKERS = dior.DEFAULT_SAMPLE_WORKERS
QWEN_MODEL_KEYS = tuple(dior.build_qwen_model_configs())
LLAVA_MODEL_KEYS = tuple(dior.build_llava_model_configs())
TARGET_MODELS = ('gpt-5.5', 'gemini-3.5-flash', 'gpt-6-astra', *QWEN_MODEL_KEYS, *LLAVA_MODEL_KEYS)
DEFAULT_MODELS = TARGET_MODELS[:2]
LORA_TASK = 'dior-rsvg-recognition'
dior.MODEL_CONFIGS.update(dior.build_lora_model_configs(LORA_TASK))
MARKER_COLOR = (255, 32, 32)
MARKER_HALO_COLOR = (255, 255, 255)
MARKER_PADDING_PIXELS = 2
ANNOTATED_JPEG_QUALITY = 95
CATEGORIES = tuple(sorted(set(dior.CATEGORY_NAME.values())))
SYSTEM_MESSAGE = 'You are a helpful image analyst who specializes in recognizing objects and facilities in aerial and satellite images. Classify only the object inside the marked red box.'

def get_user_prompt() -> str:
    options = '\n'.join((f'{index}. {category}' for index, category in enumerate(CATEGORIES, 1)))
    return f'What object or facility is shown inside the marked red box? Classify the marked target into exactly one of these options:\n{options}\nReturn only a JSON object in this exact format: {{"category": "<exact category name>"}}'
USER_PROMPT = get_user_prompt()

def _label_key(value: Any) -> str:
    return re.sub('[^a-z0-9]+', '', str(value or '').casefold())
CATEGORY_BY_KEY: Dict[str, str] = {}
for raw_name, display_name in dior.CATEGORY_NAME.items():
    CATEGORY_BY_KEY[_label_key(raw_name)] = display_name
    CATEGORY_BY_KEY[_label_key(display_name)] = display_name

def normalize_category(value: Any) -> Optional[str]:
    return CATEGORY_BY_KEY.get(_label_key(value))

def format_category_answer(category: str) -> str:
    return json.dumps({'category': category}, ensure_ascii=False)


def parse_category_response(response: Any) -> str:
    from evaluation.utils.response_protocol import parse_answer
    return parse_answer('dior_classification', response) or 'Refused' 


def response_text_from_record(record: Dict[str, Any]) -> str:
    answer_text = record.get('answer_text')
    if isinstance(answer_text, str):
        return answer_text
    response = record.get('response')
    if isinstance(response, dict):
        try:
            content = response['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            content = None
        if isinstance(content, str):
            return content
    parsed_answer = record.get('parsed_answer')
    return '' if parsed_answer is None else str(parsed_answer)

def _validated_bbox(bbox: Any, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    values = dior._coerce_bbox(bbox)
    if not values:
        raise ValueError(f'Invalid DIOR bounding box: {bbox!r}')
    xmin, ymin, xmax, ymax = values
    xmin = max(0, min(image_width - 1, int(round(xmin))))
    ymin = max(0, min(image_height - 1, int(round(ymin))))
    xmax = max(0, min(image_width - 1, int(round(xmax))))
    ymax = max(0, min(image_height - 1, int(round(ymax))))
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f'Bounding box {bbox!r} is empty after clipping to {image_width}x{image_height}')
    return (xmin, ymin, xmax, ymax)

def annotate_target_box(raw_bytes: bytes, bbox: Any) -> Tuple[str, int, int]:
    from PIL import Image, ImageDraw
    with Image.open(io.BytesIO(raw_bytes)) as source:
        image = source.convert('RGB')
    width, height = image.size
    xmin, ymin, xmax, ymax = _validated_bbox(bbox, width, height)
    line_width = max(2, min(5, int(round(min(width, height) / 256))))

    def expanded_box(offset: int) -> Tuple[int, int, int, int]:
        return (max(0, xmin - offset), max(0, ymin - offset), min(width - 1, xmax + offset), min(height - 1, ymax + offset))
    halo_width = line_width + 2
    halo_bbox = expanded_box(MARKER_PADDING_PIXELS + halo_width)
    marker_bbox = expanded_box(MARKER_PADDING_PIXELS + line_width)
    draw = ImageDraw.Draw(image)
    draw.rectangle(halo_bbox, outline=MARKER_HALO_COLOR, width=halo_width)
    draw.rectangle(marker_bbox, outline=MARKER_COLOR, width=line_width)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=ANNOTATED_JPEG_QUALITY, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return (f'data:image/jpeg;base64,{encoded}', height, width)

def _object_values(objects: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(objects, list):
        for value in objects:
            if isinstance(value, dict):
                yield value
        return
    if not isinstance(objects, dict):
        return
    fields = ('bbox', 'captions', 'categories', 'categories_normalized')
    lengths = [len(objects.get(field) or []) for field in fields]
    object_count = max(lengths, default=0)
    for object_index in range(object_count):
        value = {}
        for field in fields:
            values = objects.get(field) or []
            value[field] = values[object_index] if object_index < len(values) else None
        yield value

def build_object_items(hf_dataset, row_indices: Sequence[int], raw_split: str, canonical_split: str) -> List[Dict[str, Any]]:
    metadata = hf_dataset.select_columns([column for column in hf_dataset.column_names if column != 'image'])
    items: List[Dict[str, Any]] = []
    skipped = 0
    for row_index in row_indices:
        row = metadata[row_index]
        for object_index, obj in enumerate(_object_values(row.get('objects'))):
            label = normalize_category(obj.get('categories_normalized') or obj.get('categories'))
            bbox = dior._coerce_bbox(obj.get('bbox'))
            if not label or not bbox:
                skipped += 1
                continue
            sample_id = f'{raw_split}:{row_index}:{object_index}'
            items.append({'index': sample_id, 'sample_id': sample_id, 'row_index': row_index, 'object_index': object_index, 'raw_split': raw_split, 'canonical_split': canonical_split, 'path': row.get('path'), 'bbox': bbox, 'label': label, 'raw_category': obj.get('categories'), 'caption': obj.get('captions')})
    if skipped:
        print(f'Skipped {skipped} DIOR objects with an invalid category or bounding box')
    return items

def _select_object_items(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = list(candidates)
    random.Random(DEFAULT_RANDOM_SEED).shuffle(selected)
    for rank, item in enumerate(selected):
        item['selected_index_rank'] = rank
    return selected

def build_few_shot_source(hf_dataset, candidates: Sequence[Dict[str, Any]], k: int, raw_split: str, canonical_split: str) -> Dict[str, Any]:
    if len(candidates) < k:
        raise ValueError(f'Could not find {k} DIOR local-classification few-shot candidates in split {raw_split!r}; found {len(candidates)}.')
    return {'hf_dataset': hf_dataset, 'candidates': list(candidates), 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'task': TASK_NAME}

def sample_few_shot_examples(source: Dict[str, Any], seed: int, current_item: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    candidates = list(source['candidates'])
    if current_item is not None:
        candidates = [item for item in candidates if item['sample_id'] != current_item['sample_id']]
    if len(candidates) < source['k']:
        raise ValueError(f"Need {source['k']} non-overlapping few-shot samples, found {len(candidates)}.")
    examples = []
    for item in random.Random(seed).sample(candidates, source['k']):
        raw_bytes = dior.get_raw_image_bytes(source['hf_dataset'][item['row_index']]['image'])
        image_url, _, _ = annotate_target_box(raw_bytes, item['bbox'])
        examples.append({'index': item['sample_id'], 'sample_id': item['sample_id'], 'split': item['raw_split'], 'raw_split': item['raw_split'], 'canonical_split': item['canonical_split'], 'image_url': image_url, 'prompt': USER_PROMPT, 'user_prompt': USER_PROMPT, 'answer': format_category_answer(item['label'])})
    return examples

def resolve_canonical_pools(few_shot_k: int) -> Dict[str, Any]:
    raw_split_names = dior.discover_dior_raw_split_names()
    split_map = dior.discover_available_splits_from_names(raw_split_names)
    strategy = dior.resolve_split_strategy(split_map)
    datasets_by_split = {}
    split_lengths = {}
    for raw_split in sorted(set(split_map.values())):
        dataset = dior.load_dior_dataset(raw_split)
        datasets_by_split[raw_split] = dataset
        split_lengths[raw_split] = len(dataset)
    assignments = dior.build_split_assignments(split_map=split_map, split_lengths=split_lengths, seed=dior.DEFAULT_SPLIT_SEED)
    eval_assignment = assignments['test']
    train_assignment = assignments['train']
    evaluation_candidates = build_object_items(datasets_by_split[eval_assignment['raw_split']], eval_assignment['indices'], raw_split=eval_assignment['raw_split'], canonical_split='test')
    evaluation_items = _select_object_items(evaluation_candidates)
    few_shot_source = None
    if few_shot_k > 0:
        few_shot_candidates = build_object_items(datasets_by_split[train_assignment['raw_split']], train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(datasets_by_split[train_assignment['raw_split']], few_shot_candidates, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train')
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': dior.DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print(f'Detected raw splits: {split_map}')
    print(f"Canonical strategy mode: {strategy['mode']}")
    return {'evaluation_dataset': datasets_by_split[eval_assignment['raw_split']], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def prompt_metadata(few_shot_source: Optional[Dict[str, Any]], split_context: Dict[str, Any], few_shot_seed: Optional[int], reasoning_method: str) -> Dict[str, Any]:
    source = few_shot_source or {}
    k = int(source.get('k', 0) or 0)
    metadata = {'canonical_strategy_mode': split_context['canonical_strategy_mode'], 'canonical_split_seed': split_context['canonical_split_seed'], 'eval_raw_split': split_context['eval_raw_split'], 'few_shot_source_raw_split': split_context['few_shot_source_raw_split'], 'eval_canonical_split': split_context['eval_canonical_split'], 'few_shot_source_canonical_split': split_context['few_shot_source_canonical_split'], 'prompt_mode': 'few-shot' if k else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k else None, 'few_shot_seed': few_shot_seed if k else None, 'task': TASK_NAME, 'input_scope': 'full-image-with-marked-target-box', 'target_caption_exposed': False, 'marker_padding_pixels': MARKER_PADDING_PIXELS, 'category_count': len(CATEGORIES)}
    if reasoning_method == 'cot':
        metadata.update({'reasoning_method': reasoning_method})
    return metadata

def few_shot_sample_metadata(examples: Optional[Sequence[Dict[str, Any]]], sample_seed: Optional[int]) -> Dict[str, Any]:
    selected = list(examples or [])
    return {'few_shot_sample_seed': sample_seed, 'few_shot_indices': [example['sample_id'] for example in selected], 'few_shot_raw_splits': [example.get('raw_split') for example in selected], 'few_shot_canonical_splits': [example.get('canonical_split') for example in selected], 'few_shot_answers': [example['answer'] for example in selected]}

def build_few_shot_seeds(items: Sequence[Dict[str, Any]], few_shot_seed: Optional[int]) -> Dict[str, int]:
    if few_shot_seed is None:
        return {}
    rng = random.Random(few_shot_seed)
    return {item['sample_id']: rng.randrange(2 ** 32) for item in items}

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: ignoring malformed JSONL row {line_number} in {path}')
                continue
            if isinstance(value, dict):
                yield value

def clean_result_jsonl_for_resume(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    latest: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    malformed_or_unkeyed = 0
    duplicate_count = 0
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                malformed_or_unkeyed += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_or_unkeyed += 1
                continue
            if not isinstance(record, dict):
                malformed_or_unkeyed += 1
                continue
            sample_id = record.get('sample_id') or record.get('index')
            if sample_id is None:
                malformed_or_unkeyed += 1
                continue
            sample_id = str(sample_id)
            if sample_id in latest:
                duplicate_count += 1
            latest[sample_id] = (line_number, record)
    if malformed_or_unkeyed or duplicate_count:
        backup_path = f'{path}.before-resume-cleanup'
        if not os.path.exists(backup_path):
            shutil.copyfile(path, backup_path)
        temporary_path = f'{path}.cleaning'
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            for _, record in sorted(latest.values(), key=lambda item: item[0]):
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(temporary_path, path)
        print(f'Cleaned malformed or duplicate rows in {path}')
    return set(latest)

def append_jsonl(record: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')

def validate_resume_config(path: str, expected: Dict[str, Any]) -> None:
    keys = ('model', 'canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'task', 'input_scope', 'target_caption_exposed', 'marker_padding_pixels', 'category_count', 'reasoning_method')
    for record in iter_jsonl(path):
        for key in keys:
            existing = record.get(key)
            expected_value = expected.get(key)
            if existing is None and expected_value is not None:
                continue
            if existing != expected_value:
                raise ValueError(f'Existing result {path} has incompatible {key}: {existing!r} != {expected_value!r}. Use another output directory or keep the same run configuration.')

def validate_resume_samples(path: str, dataset_items: Sequence[Dict[str, Any]]) -> None:
    expected_by_id = {str(item['sample_id']): item for item in dataset_items}
    for record in iter_jsonl(path):
        sample_id = record.get('sample_id') or record.get('index')
        expected = expected_by_id.get(str(sample_id))
        if expected is None:
            continue
        for key, saved_key in (('row_index', 'row_index'), ('object_index', 'object_index'), ('path', 'path'), ('bbox', 'bbox'), ('label', 'label')):
            saved_value = record.get(saved_key)
            expected_value = expected.get(key)
            if saved_value is not None and expected_value is not None and (saved_value != expected_value):
                raise ValueError(f'Existing result {path} maps sample {sample_id!r} to a different DIOR object ({saved_key} mismatch).')

def get_models_to_test(model_arg: Optional[str]) -> List[str]:
    if model_arg is None:
        return list(DEFAULT_MODELS)
    model_arg = dior.normalize_lora_model_selection(model_arg, LORA_TASK)
    aliases = {'target-all': TARGET_MODELS, 'all': TARGET_MODELS, 'GPT-all': ('gpt-5.5', 'gpt-6-astra'), 'Gemini-all': ('gemini-3.5-flash',), 'Qwen-all': QWEN_MODEL_KEYS, 'LLaVA-all': LLAVA_MODEL_KEYS}
    if model_arg in aliases:
        return list(aliases[model_arg])
    if model_arg in TARGET_MODELS or model_arg in dior.build_lora_model_configs(LORA_TASK):
        return [model_arg]
    raise ValueError(f'Unknown model {model_arg!r}. This experiment is limited to: ' + ', '.join(TARGET_MODELS))

def query_model_on_dataset(model_key: str, dataset_items: Sequence[Dict[str, Any]], hf_dataset, split_context: Dict[str, Any], output_dir: str=DEFAULT_RESULT_DIR, api_key: Optional[str]=None, few_shot_source: Optional[Dict[str, Any]]=None, few_shot_seed: Optional[int]=None, sample_workers: int=DEFAULT_SAMPLE_WORKERS, reasoning_method: str='baseline', result_path: Optional[str]=None) -> str:
    metadata = prompt_metadata(few_shot_source, split_context, few_shot_seed, reasoning_method)
    metadata['model'] = model_key
    dior.validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=metadata['few_shot_k'], model_kind=dior.MODEL_CONFIGS[model_key].get('model_kind'))
    if result_path is None:
        suffix = f"-{metadata['few_shot_k']}shot" if metadata['few_shot_k'] else ''
        result_path = os.path.join(output_dir, f'{model_key}-{RESULT_TASK_STEM}{suffix}.jsonl')
    result_path = dior.result_path_for_reasoning(result_path, reasoning_method)
    os.makedirs(os.path.dirname(result_path) or '.', exist_ok=True)
    completed = clean_result_jsonl_for_resume(result_path)
    validate_resume_config(result_path, metadata)
    validate_resume_samples(result_path, dataset_items)
    pending = [item for item in dataset_items if item['sample_id'] not in completed]
    target_ids = {item['sample_id'] for item in dataset_items}
    few_shot_seeds = build_few_shot_seeds(dataset_items, few_shot_seed if few_shot_source else None)
    sample_workers = max(1, int(sample_workers))
    if dior.is_local_model(model_key, dior.MODEL_CONFIGS):
        sample_workers = 1
    print('Evaluation state updated.', flush=True)
    if not pending:
        print(f'[{model_key}] Result file is complete; no model will be loaded or queried.')
        return result_path
    local_query = None
    if dior.is_local_model(model_key, dior.MODEL_CONFIGS):
        model_path = dior._QWEN_MODEL_PATHS.get(model_key)
        if not model_path:
            raise ValueError(f'Model {model_key} requires --qwen-model-path or --llava-model-path.')
        local_query = dior.make_qwen_query_callable(model_key, model_path)
    write_lock = threading.Lock()

    def process_item(item: Dict[str, Any]) -> None:
        sample_id = item['sample_id']
        try:
            raw_bytes = dior.get_raw_image_bytes(hf_dataset[item['row_index']]['image'])
            image_url, image_height, image_width = annotate_target_box(raw_bytes, item['bbox'])
            sample_seed = few_shot_seeds.get(sample_id)
            examples = sample_few_shot_examples(few_shot_source, sample_seed, current_item=item) if few_shot_source and sample_seed is not None else []
            query_prompt = dior.effective_prompt(USER_PROMPT, reasoning_method)
            query_system = dior.effective_system_message(SYSTEM_MESSAGE, reasoning_method)
            request_query = None
            if local_query is None:
                request_key = dior.random_api_key_for_model(model_key, dior.MODEL_CONFIGS, fallback=api_key or '')
                request_query = dior.get_query_function(model_key, api_key=request_key, few_shot_examples=examples)

            def call_model(prompt: str):
                if local_query is not None:
                    return local_query(image_url=image_url, system=query_system, user=prompt, few_shot_examples=examples)
                return request_query(image_url=image_url, system=query_system, user=prompt)
            response, content, reasoning_response = dior.call_with_reasoning(call_model, reasoning_method, task='dior_classification', context={}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer = parse_category_response(answer_text)
            record = {'index': sample_id, 'sample_id': sample_id, 'row_index': item['row_index'], 'object_index': item['object_index'], 'selected_index_rank': item.get('selected_index_rank'), 'raw_split': item['raw_split'], 'canonical_split': item['canonical_split'], 'path': item.get('path'), 'bbox': item['bbox'], 'label': item['label'], 'raw_category': item.get('raw_category'), 'caption': item.get('caption'), 'image_height': image_height, 'image_width': image_width, 'prompt': USER_PROMPT, 'response': response, 'parsed_answer': parsed_answer, **dior.result_model_metadata(model_key, dior.MODEL_CONFIGS), 'random_seed': DEFAULT_RANDOM_SEED, **metadata, **few_shot_sample_metadata(examples, sample_seed)}
            if reasoning_method == 'cot':
                record['answer_text'] = answer_text
                record.update(dior.cot_row_metadata(base_prompt=USER_PROMPT, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            with write_lock:
                append_jsonl(record, result_path)
                completed.add(sample_id)
                print('Evaluation state updated.', flush=True)
        except Exception as exc:
            print(f'[{model_key}] Error processing {sample_id}: {exc}')
            import traceback
            traceback.print_exc()
            if dior.is_local_model(model_key, dior.MODEL_CONFIGS):
                raise
    with ThreadPoolExecutor(max_workers=sample_workers) as executor:
        futures = [executor.submit(process_item, item) for item in pending]
        for future in as_completed(futures):
            future.result()
    completed_target_ids = completed & target_ids
    missing_target_ids = target_ids - completed_target_ids
    if missing_target_ids:
        preview = ', '.join(sorted(missing_target_ids)[:10])
        raise RuntimeError(f'[{model_key}] Incomplete run. Missing sample IDs include: {preview}. Re-run the same command to resume.')
    print(f'[{model_key}] Done. Results in {result_path}')
    return result_path

def _model_key_from_result_name(filename: str) -> Optional[str]:
    for model_key in sorted(TARGET_MODELS, key=len, reverse=True):
        if filename.startswith(f'{model_key}-{RESULT_TASK_STEM}'):
            return model_key
    return None

def evaluation(result_path: str) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
    records = list(iter_jsonl(result_path))
    if not records:
        raise ValueError(f'No valid result rows found in {result_path}')
    results = pd.DataFrame(records)
    results['model_response'] = results.apply(lambda row: response_text_from_record(row.to_dict()), axis=1)
    results['label'] = results['label'].apply(lambda value: normalize_category(value) or str(value))
    results['model_answer'] = results['model_response'].apply(parse_category_response)
    results['is_refusal'] = results['model_answer'] == 'Refused'
    refusal_pattern = re.compile("\\b(sorry|cannot|can't|unable|difficult)\\b", re.IGNORECASE)
    results['is_explicit_refusal'] = results['model_response'].apply(lambda value: bool(refusal_pattern.search(str(value))))
    results['is_correct'] = results['model_answer'] == results['label']
    valid = results[~results['is_refusal']].copy()
    accuracy = float(results['is_correct'].mean())
    accuracy_no_refusal = float(accuracy_score(valid['label'], valid['model_answer'])) if len(valid) else 0.0
    refusal_rate = float(results['is_refusal'].mean())
    explicit_refusal_rate = float(results['is_explicit_refusal'].mean())
    y_true = results['label']
    y_pred = results['model_answer']
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    macro = report.get('macro avg', {})
    precision_macro = float(macro.get('precision', 0.0))
    recall_macro = float(macro.get('recall', 0.0))
    f1_macro = float(macro.get('f1-score', 0.0))
    result_stem = Path(result_path).stem
    model_key = _model_key_from_result_name(Path(result_path).name) or result_stem
    result_dir = os.path.dirname(result_path) or DEFAULT_RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    artifact_stem = os.path.join(result_dir, result_stem)
    pd.DataFrame(report).transpose().round(4).to_csv(f'{artifact_stem}-classification-report.csv')
    export = results.drop(columns=['response'], errors='ignore')
    export.to_csv(f'{artifact_stem}-all.csv', index=False)
    export[~export['is_correct']].to_csv(f'{artifact_stem}-incorrect.csv', index=False)
    export[export['is_refusal']].to_csv(f'{artifact_stem}-refusal.csv', index=False)
    try:
        labels = list(CATEGORIES)
        if 'Refused' in set(y_pred):
            labels.append('Refused')
        fig, axis = plt.subplots(figsize=(14, 14))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, xticks_rotation=90, colorbar=False, ax=axis)
        axis.set_title('DIOR-RSVG Local Classification')
        fig.tight_layout()
        fig.savefig(f'{artifact_stem}-confusion-matrix.pdf', bbox_inches='tight', dpi=300)
        fig.savefig(f'{artifact_stem}-confusion-matrix.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    except Exception as exc:
        print(f'Warning: could not generate confusion matrix: {exc}')
    print(f'\nResults for {model_key}:')
    print('Evaluation state updated.', flush=True)
    print('Evaluation state updated.', flush=True)
    print(f'  Accuracy (all): {accuracy:.4f}')
    print(f'  Accuracy (excl. refusals): {accuracy_no_refusal:.4f}')
    print(f'  Precision (macro): {precision_macro:.4f}')
    print(f'  Recall (macro): {recall_macro:.4f}')
    print(f'  F1 (macro): {f1_macro:.4f}')
    print(f'  Refusal/no-match rate: {refusal_rate:.2%}')
    return {'model': model_key, 'accuracy': accuracy, 'correct_rate': accuracy, 'accuracy_no_refusal': accuracy_no_refusal, 'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro, 'refusal_rate': refusal_rate, 'explicit_refusal_rate': explicit_refusal_rate}

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR):
    import matplotlib.pyplot as plt
    import pandas as pd
    if not os.path.isdir(result_dir):
        print(f'No result directory found: {result_dir}')
        return None
    result_files = sorted((filename for filename in os.listdir(result_dir) if filename.endswith('.jsonl') and f'-{RESULT_TASK_STEM}' in filename and (not filename.endswith('-CoT.jsonl'))))
    if not result_files:
        print(f'No DIOR local-classification result files found in {result_dir}')
        return None
    all_metrics = []
    for filename in result_files:
        metrics = evaluation(os.path.join(result_dir, filename))
        model_key = _model_key_from_result_name(filename)
        if model_key is None:
            print(f'Warning: cannot infer configured model from {filename}')
            continue
        metrics.update({'model_key': model_key, 'family': dior.MODEL_CONFIGS[model_key]['family'], 'release_date': dior.MODEL_CONFIGS[model_key]['release_date'], 'result_file': filename})
        metrics.update(dior.shot_from_metrics(metrics, filename))
        all_metrics.append(metrics)
    if not all_metrics:
        return None
    dior.write_shot_summary_json(all_metrics, os.path.join(result_dir, 'dior_local_model_metrics_all_shots.json'), dataset_name='DIOR-RSVG local classification')
    zero_shot = [metrics for metrics in all_metrics if dior.is_zero_shot_metrics(metrics, metrics.get('result_file'))]
    if not zero_shot:
        print('No zero-shot metrics found; skipping comparison plot.')
        return pd.DataFrame(all_metrics)
    frame = pd.DataFrame(zero_shot)
    frame.to_csv(os.path.join(result_dir, 'dior_local_model_metrics_summary.csv'), index=False)
    metrics_to_plot = (('accuracy', 'Accuracy'), ('accuracy_no_refusal', 'Accuracy (excl. refusals)'), ('refusal_rate', 'Refusal rate'), ('precision_macro', 'Precision (macro)'), ('recall_macro', 'Recall (macro)'), ('f1_macro', 'F1 (macro)'))
    colors = [dior.FAMILY_COLORS[family] for family in frame['family']]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for axis, (metric, title) in zip(axes.flat, metrics_to_plot):
        axis.bar(frame['model_key'], frame[metric], color=colors)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.tick_params(axis='x', rotation=35)
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('DIOR-RSVG Local Classification')
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, 'dior_local_model_comparison.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(result_dir, 'dior_local_model_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return frame
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_split_context = None
_shared_output_dir = DEFAULT_RESULT_DIR
_shared_few_shot_source = None
_shared_few_shot_seed = None
_shared_sample_workers = DEFAULT_SAMPLE_WORKERS
_shared_reasoning_method = 'baseline'

def run_single_model(task: Tuple[str, Optional[str]]) -> Optional[str]:
    model_key, api_key = task
    try:
        return query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, split_context=_shared_split_context, output_dir=_shared_output_dir, api_key=api_key, few_shot_source=_shared_few_shot_source, few_shot_seed=_shared_few_shot_seed, sample_workers=_shared_sample_workers, reasoning_method=_shared_reasoning_method)
    except Exception as exc:
        print(f'[{model_key}] Failed: {exc}')
        import traceback
        traceback.print_exc()
        raise

def print_models() -> None:
    print('Models in the global-vs-local experiment:')
    for model_key in TARGET_MODELS:
        config = dior.MODEL_CONFIGS[model_key]
        runtime = 'local checkpoint' if config['api_type'] == 'hf_local' else 'API'
        print(f"  {model_key:<28} {config['family']:<7} {runtime}")
    print('\nAliases: target-all, GPT-all, Gemini-all, Qwen-all, LLaVA-all')
    print('Default: gpt-5.5 and gemini-3.5-flash')

def main() -> None:
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('dior-recognition'):
        return
    parser = argparse.ArgumentParser(description='Evaluate marked-box local object classification on DIOR-RSVG', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=f'\nExamples:\n  python {Path(__file__).name}\n  python {Path(__file__).name} --model Qwen3-VL-32B-Instruct --qwen-model-path /path/to/models\n  python {Path(__file__).name} --model LLaVA-OV-2-8B --llava-model-path /path/to/models\n  python {Path(__file__).name} --evaluate-only RESULT.jsonl\n')
    parser.add_argument('--model', default=None, help='One target model or family alias; default runs the two API models')
    parser.add_argument('--output-dir', default=DEFAULT_RESULT_DIR)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--evaluate-only', default=None)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--download-all-splits', action='store_true')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--parallel', type=int, default=None)
    parser.add_argument('--sample-workers', type=int, default=DEFAULT_SAMPLE_WORKERS)
    parser.add_argument('--qwen-model-path', default=None)
    parser.add_argument('--qwen-finetuned-root', default=None)
    parser.add_argument('--llava-model-path', default=None)
    parser.add_argument('--few-shot', action='store_true')
    parser.add_argument('--few-shot-k', type=int, default=0)
    parser.add_argument('--few-shot-seed', type=int, default=DEFAULT_FEW_SHOT_SEED)
    parser.add_argument('--reasoning-method', choices=dior.REASONING_METHODS, default='baseline')
    args = parser.parse_args()
    dior.configure_cache_environment_from_args(args.cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.download_all_splits and (not args.download_only):
        raise ValueError('--download-all-splits requires --download-only')
    if args.sample_workers < 1:
        raise ValueError('--sample-workers must be positive')
    if args.list_models:
        print_models()
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir)
        return
    if args.evaluate_only:
        dior.evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative')
    if args.download_all_splits:
        dior.ensure_dior_parquet_splits_downloaded(list(dior.DIOR_RSVG_ALL_SPLITS))
        print('DIOR-RSVG split cache is complete; no model calls were made.')
        return
    canonical_pools = resolve_canonical_pools(few_shot_k)
    if args.download_only:
        print('DIOR-RSVG local-classification data is ready; no model calls were made.')
        return
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        dior.validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=dior.MODEL_CONFIGS[model_key].get('model_kind'))
    resolved_paths = dior.validate_qwen_model_paths(models_to_test, dior.MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    dior._QWEN_MODEL_PATHS = resolved_paths
    if dior.has_api_models(models_to_test, dior.MODEL_CONFIGS):
        dior.print_provider_summary()
    global _shared_dataset_items, _shared_hf_dataset, _shared_split_context
    global _shared_output_dir, _shared_few_shot_source, _shared_few_shot_seed
    global _shared_sample_workers, _shared_reasoning_method
    _shared_dataset_items = canonical_pools['evaluation_items']
    _shared_hf_dataset = canonical_pools['evaluation_dataset']
    _shared_split_context = canonical_pools['split_context']
    _shared_output_dir = args.output_dir
    _shared_few_shot_source = canonical_pools['few_shot_source']
    _shared_few_shot_seed = args.few_shot_seed if _shared_few_shot_source else None
    _shared_sample_workers = args.sample_workers
    _shared_reasoning_method = args.reasoning_method
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Prompt mode: {('few-shot' if _shared_few_shot_source else 'zero-shot')}")
    print(f'Results: {args.output_dir}')
    tasks: List[Tuple[str, Optional[str]]] = []
    for model_key in models_to_test:
        if dior.is_local_model(model_key, dior.MODEL_CONFIGS):
            print(f'  {model_key} -> local checkpoint: {resolved_paths[model_key]}')
        else:
            keys = dior.api_keys_for_model(model_key, dior.MODEL_CONFIGS)
            if not keys:
                raise ValueError(f'No API key configured for {model_key}')
            print(f'  {model_key} -> {len(keys)} configured API key(s)')
        tasks.append((model_key, None))
    if args.parallel is None:
        parallel_models = 1 if dior.has_local_models(models_to_test, dior.MODEL_CONFIGS) else len(tasks)
    else:
        parallel_models = max(1, min(args.parallel, len(tasks)))
    result_paths: List[str] = []
    if parallel_models > 1:
        with ThreadPoolExecutor(max_workers=parallel_models) as executor:
            futures = {executor.submit(run_single_model, task): task[0] for task in tasks}
            for future in as_completed(futures):
                result_path = future.result()
                if result_path:
                    result_paths.append(result_path)
    else:
        for task in tasks:
            result_path = run_single_model(task)
            if result_path:
                result_paths.append(result_path)
    if len(result_paths) != len(tasks):
        raise RuntimeError(f'Only {len(result_paths)} of {len(tasks)} model run(s) completed. Inspect the errors above and re-run the same command to resume.')
    for result_path in result_paths:
        dior.evaluate_and_write_cot_metrics(evaluation, result_path)
    if len(result_paths) > 1:
        plot_model_comparison(args.output_dir)
    print(f'Completed {len(result_paths)} model run(s). Results saved to {args.output_dir}')
from reasoning_effort import register_reasoning_models
register_reasoning_models(dior.MODEL_CONFIGS)
if __name__ == '__main__':
    try:
        main()
    except dior.DIORDatasetAccessError as exc:
        print(f'\n{exc}', file=sys.stderr)
        raise SystemExit(1)
