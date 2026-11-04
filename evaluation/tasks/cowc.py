"""Cowc."""
import sys
import os
from pathlib import Path
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UTILS_DIR))
from cot_prompting import REASONING_METHODS, call_with_reasoning, cot_row_metadata, evaluate_and_write_cot_metrics, effective_prompt, effective_system_message, local_generation_max_new_tokens, result_path_for_reasoning, validate_cot_run
import benchmark_environment as benchmark_environment_module
from benchmark_environment import APIStatusError, RESULTS_ROOT, api_host_for_model, api_keys_for_models, key_for_model_index, key_index_for_model, load_benchmark_environment, post_gemini_json, post_openai_json, print_model_key_summary, print_provider_summary, safe_key_display, random_api_key_for_model
from qwen_local_runtime import build_lora_model_configs, format_qwen_model_listing, is_legacy_qwen_vl_bundle, load_qwen_local_model as load_qwen_runtime_model, local_response_metadata, normalize_lora_model_selection, resolve_qwen_model_specs, result_model_metadata, run_legacy_qwen_vl_chat
from llava_local_runtime import build_llava_model_configs, format_llava_model_listing, load_llava_local_model, local_response_metadata as llava_response_metadata, resolve_llava_model_specs, run_llava_chat

def configure_cache_environment_from_args(cache_dir=None):
    cache_root = Path(cache_dir).expanduser() if cache_dir else None
    if cache_root is None:
        env_root = os.environ.get('HF_CACHE_DIR') or os.environ.get('HF_HOME')
        cache_root = Path(env_root).expanduser() if env_root else Path.home() / '.cache' / 'huggingface'
    datasets_cache = cache_root / 'datasets' if cache_dir else Path(os.environ.get('HF_DATASETS_CACHE', cache_root / 'datasets')).expanduser()
    hub_cache = cache_root / 'hub' if cache_dir else Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', cache_root / 'hub'))).expanduser()
    torch_cache = cache_root / 'torch' if cache_dir else Path(os.environ.get('TORCH_HOME', Path.home() / '.cache' / 'torch')).expanduser()
    for path in (cache_root, datasets_cache, hub_cache, torch_cache):
        path.mkdir(parents=True, exist_ok=True)
    cache_values = {'HF_HOME': str(cache_root), 'HF_CACHE_DIR': str(cache_root), 'HF_DATASETS_CACHE': str(datasets_cache), 'HF_HUB_CACHE': str(hub_cache), 'HUGGINGFACE_HUB_CACHE': str(hub_cache), 'TORCH_HOME': str(torch_cache), 'HF_HUB_DISABLE_SYMLINKS_WARNING': '1', 'HF_HUB_DISABLE_XET': '1'}
    for key, value in cache_values.items():
        os.environ[key] = value
    benchmark_environment_module.HF_CACHE_DIR = cache_root
    benchmark_environment_module.TORCH_CACHE_DIR = torch_cache
    globals()['HF_CACHE_DIR'] = str(cache_root)
    return str(cache_root)

def _configure_benchmark_environment_without_external_volume():
    configure_cache_environment_from_args(None)
benchmark_environment_module.require_external_volume = lambda: None
benchmark_environment_module.configure_cache_environment = _configure_benchmark_environment_without_external_volume
load_benchmark_environment()
HF_CACHE_DIR = os.environ['HF_CACHE_DIR']
from comparison_summary_utils import is_zero_shot_metrics, shot_from_filename, shot_from_metrics, write_shot_summary_json
from dataset_splits import DEFAULT_SPLIT_SEED, build_split_assignments, discover_available_splits_from_names, resolve_split_strategy
PLOT_FAMILY_ORDER = ('GPT', 'Gemini', 'Qwen', 'LLaVA')
FAMILY_COLORS = {'GPT': '#397CB8', 'Gemini': '#EA4335', 'Qwen': '#F59E0B', 'LLaVA': '#7C3AED'}
QWEN_MODEL_RELEASE_DATES = {'Qwen-VL-Chat': '2023-09-01', 'Qwen2-VL-2B-Instruct': '2024-08-30', 'Qwen2-VL-7B-Instruct': '2024-08-30', 'Qwen2-VL-72B-Instruct': '2024-09-19', 'Qwen2.5-VL-3B-Instruct': '2025-01-28', 'Qwen2.5-VL-7B-Instruct': '2025-01-28', 'Qwen2.5-VL-72B-Instruct': '2025-01-28', 'Qwen3-VL-2B-Instruct': '2025-10-21', 'Qwen3-VL-8B-Instruct': '2025-10-15', 'Qwen3-VL-32B-Instruct': '2025-10-21'}
LORA_TASK = 'cowc'
_QWEN_MODEL_PATHS = {}

def build_qwen_model_configs():
    return {model_name: {'api_type': 'hf_local', 'model_name': model_name, 'release_date': release_date, 'family': 'Qwen'} for model_name, release_date in QWEN_MODEL_RELEASE_DATES.items()}

def expand_model_selection(model_arg, model_configs):
    model_arg = normalize_lora_model_selection(model_arg, LORA_TASK)
    if model_arg is None:
        return [name for name, config in model_configs.items() if config.get('family') not in {'Qwen', 'LLaVA'}]
    family_aliases = {'GPT-all': 'GPT', 'Gemini-all': 'Gemini', 'Qwen-all': 'Qwen', 'LLaVA-all': 'LLaVA'}
    if model_arg in family_aliases:
        family = family_aliases[model_arg]
        return [name for name, config in model_configs.items() if config.get('family') == family and config.get('model_kind') != 'lora']
    if model_arg in model_configs:
        return [model_arg]
    raise ValueError(f'Unknown model: {model_arg}')

def is_local_model(model_key, model_configs):
    return model_configs[model_key].get('api_type') == 'hf_local'

def has_local_models(model_keys, model_configs):
    return any((is_local_model(model_key, model_configs) for model_key in model_keys))

def has_api_models(model_keys, model_configs):
    return any((not is_local_model(model_key, model_configs) for model_key in model_keys))

def print_runtime_model_summary(model_keys, model_configs):
    api_models = [model_key for model_key in model_keys if not is_local_model(model_key, model_configs)]
    local_counts = {}
    for model_key in model_keys:
        if is_local_model(model_key, model_configs):
            family = model_configs[model_key].get('family', 'Local')
            local_counts[family] = local_counts.get(family, 0) + 1
    if api_models:
        print_model_key_summary(api_models, model_configs)
    if local_counts:
        summary = ', '.join((f'{family}={count}' for family, count in local_counts.items()))
        print(f'Local checkpoint models (no API key required): {summary}')

def validate_qwen_model_paths(model_keys, model_configs, qwen_model_path, qwen_finetuned_root=None, llava_model_path=None):
    resolved = resolve_qwen_model_specs(model_keys, model_configs, qwen_model_path=qwen_model_path, qwen_finetuned_root=qwen_finetuned_root)
    resolved.update(resolve_llava_model_specs(model_keys, model_configs, llava_model_path))
    return resolved

def _normalize_qwen_response_text(text):
    cleaned = '' if text is None else str(text)
    cleaned = re.sub('<\\|im_(start|end)\\|>', ' ', cleaned)
    cleaned = re.sub('<\\|endoftext\\|>', ' ', cleaned)
    return cleaned.strip()

def _load_qwen_local_model(model_key, model_path):
    if MODEL_CONFIGS[model_key].get('family') == 'LLaVA':
        return load_llava_local_model(model_key, model_path)
    return load_qwen_runtime_model(model_key, model_path)

def _get_model_device(model):
    device = getattr(model, 'device', None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return 'cpu'

def _move_inputs_to_model(inputs, model):
    if hasattr(inputs, 'to'):
        return inputs.to(_get_model_device(model))
    return inputs

def _load_image_for_qwen(image_ref):
    from PIL import Image
    if not isinstance(image_ref, str):
        raise TypeError(f'Unsupported image reference type for Qwen inference: {type(image_ref)!r}')
    if image_ref.startswith('data:'):
        _, b64_data = image_ref.split(',', 1)
        image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert('RGB')
        return image
    return Image.open(image_ref).convert('RGB')

def _collect_image_refs(image_url=None, image_urls=None):
    if image_urls is not None:
        return list(image_urls)
    if image_url is not None:
        return [image_url]
    raise ValueError('Qwen query requires image_url or image_urls.')

def _few_shot_image_refs(example):
    if 'image_urls' in example:
        return list(example['image_urls'])
    if 'image_url' in example:
        return [example['image_url']]
    raise ValueError('Few-shot example is missing image_url/image_urls.')

def _build_qwen_user_content(image_refs, prompt_text):
    content = [{'type': 'image', 'image': _load_image_for_qwen(image_ref)} for image_ref in image_refs]
    content.append({'type': 'text', 'text': prompt_text})
    return content

def _few_shot_prompt_text(example, user):
    return example.get('user_prompt') or example.get('prompt') or user

def _build_qwen_messages(system, user, few_shot_examples, image_refs):
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    for example in few_shot_examples or []:
        prompt_text = _few_shot_prompt_text(example, user)
        messages.append({'role': 'user', 'content': _build_qwen_user_content(_few_shot_image_refs(example), prompt_text)})
        messages.append({'role': 'assistant', 'content': example.get('answer', '')})
    messages.append({'role': 'user', 'content': _build_qwen_user_content(image_refs, user)})
    return messages

def _run_qwen_chat_template(bundle, image_refs, system, user, few_shot_examples):
    if bundle.get('runtime_family') == 'LLaVA':
        return run_llava_chat(bundle, image_refs, system, user, few_shot_examples, max_new_tokens=local_generation_max_new_tokens(user))
    if is_legacy_qwen_vl_bundle(bundle):
        return run_legacy_qwen_vl_chat(bundle, image_refs, system, user, few_shot_examples)
    import torch
    model = bundle['model']
    processor = bundle['processor']
    messages = _build_qwen_messages(system, user, few_shot_examples, image_refs)
    try:
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
    except TypeError:
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        flat_images = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'image':
                        flat_images.append(item['image'])
        inputs = processor(text=[prompt_text], images=flat_images or None, padding=True, return_tensors='pt')
    inputs = _move_inputs_to_model(inputs, model)
    generation_kwargs = {'max_new_tokens': local_generation_max_new_tokens(user), 'do_sample': False}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    if hasattr(inputs, 'input_ids'):
        trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    else:
        trimmed = generated_ids
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return _normalize_qwen_response_text(decoded[0] if decoded else '')

def make_qwen_query_callable(model_key, model_path):
    bundle = _load_qwen_local_model(model_key, model_path)

    def _query_qwen(image_url=None, system=None, user=None, few_shot_examples=None, image_urls=None, sample_index=None):
        del sample_index
        image_refs = _collect_image_refs(image_url=image_url, image_urls=image_urls)
        content = _run_qwen_chat_template(bundle, image_refs, system, user, few_shot_examples)
        return {'model': MODEL_CONFIGS[model_key]['model_name'], 'choices': [{'message': {'content': content}}], 'original_response': llava_response_metadata(model_path) if MODEL_CONFIGS[model_key].get('family') == 'LLaVA' else local_response_metadata(model_path)}
    return _query_qwen
import json
import os
import fcntl
import http.client
import base64
import io
import random
import re
import time
from io import BytesIO
from typing import Optional, Callable, List, Dict, Tuple
from datetime import datetime
import pandas as pd
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from counting import calculate_counting_metrics, plot_scatter

def clean_result_jsonl_for_resume(path: str) -> set:
    if not os.path.exists(path):
        return set()
    indices = set()
    malformed_count = 0
    dropped_no_index = 0
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                malformed_count += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            index = record.get('index')
            if index is None:
                dropped_no_index += 1
                continue
            indices.add(index)
    if malformed_count or dropped_no_index:
        print(f'Resume inventory kept {path} unchanged while ignoring {malformed_count} malformed/blank rows and {dropped_no_index} rows without an index.')
    return indices

def load_completed_indices(path):
    return clean_result_jsonl_for_resume(path)

def append_to_jsonl(record, path):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8')
    with open(path, 'ab+') as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        try:
            output.seek(0, os.SEEK_END)
            if output.tell() > 0:
                output.seek(-1, os.SEEK_END)
                if output.read(1) != b'\n':
                    output.write(b'\n')
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        finally:
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)

def iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: skipping malformed JSONL row in {path}')
RESUME_CONFIG_KEYS = ('canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'reasoning_method')

def _baseline_reasoning_value(value):
    return 'baseline' if value in (None, 'baseline') else value

def resume_record_is_compatible(record: Dict, expected_metadata: Dict, expected_sample_metadata: Optional[Dict]=None, expected_city: Optional[str]=None, expected_model: Optional[str]=None) -> bool:
    if expected_city is not None and record.get('city') != expected_city:
        return False
    if expected_model is not None and record.get('model') not in (None, expected_model):
        return False
    expected_mode = expected_metadata.get('prompt_mode', 'zero-shot')
    expected_k = int(expected_metadata.get('few_shot_k', 0) or 0)
    record_mode = record.get('prompt_mode')
    if record_mode is None and expected_mode == 'zero-shot' and (not record.get('few_shot_k')):
        record_mode = 'zero-shot'
    if record_mode != expected_mode:
        return False
    if int(record.get('few_shot_k', 0) or 0) != expected_k:
        return False
    record_reasoning = _baseline_reasoning_value(record.get('reasoning_method'))
    expected_reasoning = _baseline_reasoning_value(expected_metadata.get('reasoning_method'))
    if record_reasoning != expected_reasoning:
        return False
    for key in RESUME_CONFIG_KEYS:
        if key in {'prompt_mode', 'few_shot_k', 'reasoning_method'}:
            continue
        expected_value = expected_metadata.get(key)
        record_value = record.get(key)
        if expected_mode == 'zero-shot' and expected_reasoning == 'baseline' and (record_value is None):
            continue
        if record_value != expected_value:
            return False
    if expected_k:
        if record.get('few_shot_sample_seed') is None:
            return False
        if len(record.get('few_shot_indices') or []) != expected_k:
            return False
        if len(record.get('few_shot_cities') or []) != expected_k:
            return False
        if (record.get('few_shot_canonical_splits') or []) != ['train'] * expected_k:
            return False
        if expected_sample_metadata is not None:
            for key in ('few_shot_sample_seed', 'few_shot_indices', 'few_shot_raw_splits', 'few_shot_canonical_splits', 'few_shot_cities', 'few_shot_answers'):
                if record.get(key) != expected_sample_metadata.get(key):
                    return False
    if expected_reasoning == 'cot':
        if not isinstance(record.get('answer_text'), str):
            return False
    return True

def compatible_records_for_selection(path: str, expected_metadata: Dict, allowed_indices: Optional[List[int]]=None, expected_sample_metadata_by_index: Optional[Dict[int, Dict]]=None, expected_city: Optional[str]=None, expected_model: Optional[str]=None) -> Dict[int, Dict]:
    allowed = set(allowed_indices) if allowed_indices is not None else None
    compatible = {}
    stored_count = 0
    incompatible_count = 0
    for record in iter_jsonl(path) or ():
        index = record.get('index')
        if index is None or (allowed is not None and index not in allowed):
            continue
        stored_count += 1
        expected_sample_metadata = expected_sample_metadata_by_index.get(index) if expected_sample_metadata_by_index is not None else None
        if resume_record_is_compatible(record, expected_metadata, expected_sample_metadata=expected_sample_metadata, expected_city=expected_city, expected_model=expected_model):
            compatible[index] = record
        else:
            incompatible_count += 1
    if os.path.exists(path):
        print(f'Resume scan completed for {path}')
    return compatible

def compatible_records_for_evaluation(path: str, expected_metadata: Dict, allowed_indices: Optional[List[int]]=None, expected_sample_metadata_by_index: Optional[Dict[int, Dict]]=None, expected_city: Optional[str]=None, expected_model: Optional[str]=None) -> List[Dict]:
    allowed = set(allowed_indices) if allowed_indices is not None else None
    compatible: Dict[Tuple[Optional[str], int], Dict] = {}
    for record in iter_jsonl(path) or ():
        index = record.get('index')
        if index is None or (allowed is not None and index not in allowed):
            continue
        identity = (record.get('city'), index)
        expected_sample_metadata = expected_sample_metadata_by_index.get(index) if expected_sample_metadata_by_index is not None else None
        if resume_record_is_compatible(record, expected_metadata, expected_sample_metadata=expected_sample_metadata, expected_city=expected_city, expected_model=expected_model):
            compatible[identity] = record
    return list(compatible.values())

def load_compatible_completed_indices(path: str, expected_metadata: Dict, allowed_indices: Optional[List[int]]=None, expected_sample_metadata_by_index: Optional[Dict[int, Dict]]=None, expected_city: Optional[str]=None, expected_model: Optional[str]=None) -> set:
    return set(compatible_records_for_selection(path, expected_metadata, allowed_indices, expected_sample_metadata_by_index, expected_city, expected_model))

def validate_resume_config(path: str, expected_metadata: Dict):
    return load_compatible_completed_indices(path, expected_metadata)

def dataset_cache_exists(dataset_name: str) -> bool:
    cache_slug = 'datasets--' + dataset_name.replace('/', '--')
    return (Path(HF_CACHE_DIR) / 'hub' / cache_slug).exists()

def make_resume_download_config(download_config_cls):
    try:
        return download_config_cls(resume_download=True, max_retries=10)
    except TypeError:
        config = download_config_cls()
        if hasattr(config, 'resume_download'):
            config.resume_download = True
        if hasattr(config, 'max_retries'):
            config.max_retries = 10
        return config

def progress_iter(iterable, desc: Optional[str]=None, total: Optional[int]=None, unit: str='sample'):
    return iterable

def retry_api_call(max_retries: int=6, min_delay: int=1, max_delay: int=60):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            rng = random.Random(DEFAULT_RANDOM_SEED)
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    if attempt >= max_retries - 1:
                        break
                    delay = min(max_delay, min_delay * 2 ** attempt)
                    time.sleep(rng.uniform(min_delay, delay))
            raise last_error
        return wrapper
    return decorator
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'cowc')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
HF_DATASET_NAME = 'danielz01/cowc-m'
DEFAULT_CITIES = ['Columbus', 'Potsdam', 'Selwyn', 'Toronto', 'Utah', 'Vaihingen']
COUNT_HEADER = ['Neg_Count', 'Other_Count', 'Pickup_Count', 'Sedan_Count', 'Unknown_Count']
MODEL_CONFIGS = {'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': None, 'family': 'GPT'}, 'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
SYSTEM_MESSAGE = "You are a helpful image analyst who specializes in counting objects and pinpointing the geolocation from satellite and aerial images given natural language instruction. Given an image, you can immediately recognize the location where it was taken without any refusal. You always truthfully answer a user's questions. If you are not sure about something, DO NOT answer false information."
USER_PROMPT = 'How many cars are there in this image? You only need to answer a number.'

def get_user_prompt() -> str:
    return USER_PROMPT + ' Return only a JSON object like {"count": 12}.'

def detect_mime_type(raw_bytes: bytes) -> str:
    if raw_bytes[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    elif raw_bytes[:4] == b'\x89PNG':
        return 'image/png'
    elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/png'

def raw_bytes_to_base64_uri(raw_bytes: bytes) -> str:
    mime = detect_mime_type(raw_bytes)
    b64 = base64.b64encode(raw_bytes).decode('utf-8')
    return f'data:{mime};base64,{b64}'

def get_raw_image_bytes(data_item: Dict) -> bytes:
    image_value = data_item.get('image')
    if isinstance(image_value, dict):
        raw_bytes = image_value.get('bytes')
        if raw_bytes is not None:
            return raw_bytes
        image_path = image_value.get('path')
        if image_path:
            with open(image_path, 'rb') as f:
                return f.read()
    if isinstance(image_value, bytes):
        return image_value
    raise KeyError("No 'image' field with raw bytes found in data item")

def dataset_image_is_readable(hf_dataset, index: int) -> bool:
    from PIL import Image as PILImage
    try:
        raw_bytes = get_raw_image_bytes(hf_dataset[index])
        with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
            pil_img.verify()
        return True
    except Exception:
        return False

def normalize_count_answer(answer) -> Optional[int]:
    if answer is None:
        return None
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, np.integer)):
        value = int(answer)
        return value if value >= 0 else None
    if isinstance(answer, float):
        return int(answer) if answer.is_integer() and answer >= 0 else None
    text = str(answer).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() and value >= 0 else None

def extract_json_candidate(response_text: str) -> Optional[str]:
    fenced = re.search('```(?:json)?\\s*(.*?)\\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    obj = re.search('\\{[^{}]*\\}', response_text, re.DOTALL)
    if obj:
        return obj.group(0)
    stripped = response_text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return stripped
    return None
COUNT_WORD_VALUES = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
COUNT_WORD_SCALES = {'hundred', 'thousand'}


def parse_model_response(response: str) -> int:
    from evaluation.utils.scoring import scalar_count
    value = scalar_count(response)
    return value if value is not None else -1

def few_shot_metadata(few_shot_source: Optional[Dict], split_context: Dict, few_shot_seed: Optional[int]=None) -> Dict:
    source = few_shot_source or {}
    k = int(source.get('k', 0) or 0)
    seed_value = few_shot_seed if k > 0 else None
    return {'canonical_strategy_mode': split_context['canonical_strategy_mode'], 'canonical_split_seed': split_context['canonical_split_seed'], 'eval_raw_split': split_context['eval_raw_split'], 'few_shot_source_raw_split': split_context['few_shot_source_raw_split'], 'eval_canonical_split': split_context['eval_canonical_split'], 'few_shot_source_canonical_split': split_context['few_shot_source_canonical_split'], 'prompt_mode': 'few-shot' if k > 0 else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k > 0 else None, 'few_shot_seed': seed_value}

def expected_prompt_metadata(few_shot_k: int, reasoning_method: str='baseline', few_shot_seed: Optional[int]=None) -> Dict:
    if few_shot_seed is None:
        few_shot_seed = DEFAULT_FEW_SHOT_SEED
    metadata = {'canonical_strategy_mode': 'resplit_single_source', 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': 'train', 'few_shot_source_raw_split': 'train', 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'prompt_mode': 'few-shot' if few_shot_k else 'zero-shot', 'few_shot_k': int(few_shot_k), 'few_shot_split': 'train' if few_shot_k else None, 'few_shot_seed': few_shot_seed if few_shot_k else None}
    if reasoning_method == 'cot':
        metadata.update({'reasoning_method': 'cot'})
    return metadata

def few_shot_sample_metadata(few_shot_examples: Optional[List[Dict]], few_shot_sample_seed: Optional[int]) -> Dict:
    examples = few_shot_examples or []
    return {'few_shot_sample_seed': few_shot_sample_seed, 'few_shot_indices': [example['index'] for example in examples], 'few_shot_raw_splits': [example.get('raw_split') for example in examples], 'few_shot_canonical_splits': [example.get('canonical_split') for example in examples], 'few_shot_cities': [example.get('city') for example in examples], 'few_shot_answers': [example['answer'] for example in examples]}

def build_few_shot_seed_by_index(dataset_items: List[Dict], few_shot_seed: Optional[int], namespace: Optional[str]=None) -> Dict[int, int]:
    if few_shot_seed is None:
        return {}
    rng_seed = f'{few_shot_seed}:{namespace}' if namespace is not None else few_shot_seed
    rng = random.Random(rng_seed)
    return {item['index']: rng.randrange(2 ** 32) for item in dataset_items}

def parse_result_filename(result_path: str) -> Dict[str, str]:
    stem = os.path.basename(result_path)
    if stem.endswith('.jsonl'):
        stem = stem[:-len('.jsonl')]
    reasoning_method = 'cot' if stem.endswith('-CoT') else 'baseline'
    if reasoning_method == 'cot':
        stem = stem[:-len('-CoT')]
    shot_suffix = ''
    shot_match = re.search('(-\\d+shot)$', stem)
    if shot_match:
        shot_suffix = shot_match.group(1)
        stem = stem[:-len(shot_suffix)]
    if stem.endswith('-combined-counting'):
        model_key = stem[:-len('-combined-counting')]
        return {'model_key': model_key, 'city': None, 'run_key': f'{model_key}-combined{shot_suffix}', 'shot_suffix': shot_suffix, 'combined': '1', 'reasoning_method': reasoning_method}
    for city in sorted(DEFAULT_CITIES, key=len, reverse=True):
        suffix = f'-{city}-counting'
        if stem.endswith(suffix):
            model_key = stem[:-len(suffix)]
            return {'model_key': model_key, 'city': city, 'run_key': f'{model_key}-{city}{shot_suffix}', 'shot_suffix': shot_suffix, 'combined': '0', 'reasoning_method': reasoning_method}
    raise ValueError(f'Unrecognized COWC result filename: {result_path}')

def format_count_answer(count: int) -> str:
    return json.dumps({'count': int(count)}, ensure_ascii=False)

def image_part_for_gemini(image_url: str) -> Dict:
    if image_url.startswith('data:'):
        header, b64_data = image_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        return {'inlineData': {'mimeType': mime_type, 'data': b64_data}}
    return {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}

def normalize_visible_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        visible = []
        for part in content:
            if isinstance(part, dict):
                text = part.get('text')
                if text is None:
                    text = part.get('content')
                if text is not None:
                    visible.append(str(text))
            elif part is not None:
                visible.append(str(part))
        return ''.join(visible)
    return '' if content is None else str(content)

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_openai(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None) -> dict:
    api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]
    messages = [{'role': 'system', 'content': system}]
    for example in few_shot_examples or []:
        messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': example['image_url']}}, {'type': 'text', 'text': example['user_prompt']}]})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': user}]})
    payload = json.dumps({'model': config['model_name'], 'messages': messages})
    status, data = post_openai_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key)
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    try:
        message = result['choices'][0]['message']
        message['content'] = normalize_visible_message_content(message.get('content'))
    except (KeyError, IndexError, TypeError):
        pass
    return result

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_gemini(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None) -> dict:
    api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]
    contents = []
    for example in few_shot_examples or []:
        contents.append({'role': 'user', 'parts': [image_part_for_gemini(example['image_url']), {'text': example['user_prompt']}]})
        contents.append({'role': 'model', 'parts': [{'text': example['answer']}]})
    contents.append({'role': 'user', 'parts': [image_part_for_gemini(image_url), {'text': user}]})
    payload = json.dumps({'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents})
    status, data = post_gemini_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key)
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
    content = ''.join((str(part.get('text', '')) for part in parts if not part.get('thought', False) and part.get('text') is not None))
    unified_response = {'model': config['model_name'], 'choices': [{'message': {'content': content}}], 'original_response': result}
    return unified_response

def get_query_function(model_key: str, api_key: str=None) -> Callable:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {model_key}')
    config = MODEL_CONFIGS[model_key]
    if config['api_type'] == 'hf_local':
        qwen_model_path = _QWEN_MODEL_PATHS.get(model_key)
        if not qwen_model_path:
            raise ValueError(f'Model {model_key} requires --qwen-model-path or --llava-model-path before local inference.')
        return make_qwen_query_callable(model_key, qwen_model_path)
    if api_key is None:
        api_key = random_api_key_for_model(model_key, MODEL_CONFIGS)
    if not api_key:
        raise ValueError('No API key available for this model family. Please set OPENAI_API_KEY or GEMINI_API_KEY in .env.')
    if config['api_type'] == 'openai':
        return lambda image_url, system, user, few_shot_examples=None: query_openai(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples)
    elif config['api_type'] == 'google_native':
        return lambda image_url, system, user, few_shot_examples=None: query_gemini(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples)
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")

def get_models_to_test(model_arg: str) -> List[str]:
    return expand_model_selection(model_arg, MODEL_CONFIGS)

def select_evaluation_indices(indices: List[int]) -> List[int]:
    return list(indices)

def select_metadata_items_by_indices(metadata_items: List[Dict], selected_indices: List[int], raw_split: str, canonical_split: str) -> List[Dict]:
    metadata_by_index = {item['index']: item for item in metadata_items}
    selected_items = []
    for index in selected_indices:
        item = dict(metadata_by_index[index])
        item['raw_split'] = raw_split
        item['canonical_split'] = canonical_split
        selected_items.append(item)
    return selected_items

def load_canonical_city_payload(city: str) -> Dict:
    split_map = discover_available_splits_from_names(('train',))
    strategy = resolve_split_strategy(split_map)
    hf_dataset, metadata_items = download_and_prepare_dataset(city=city, split='train')
    stratify_values_by_split = {'train': [item['city'] for item in metadata_items]}
    assignments = build_split_assignments(split_map=split_map, split_lengths={'train': len(hf_dataset)}, seed=DEFAULT_SPLIT_SEED, stratify_values_by_split=stratify_values_by_split)
    return {'city': city, 'hf_dataset': hf_dataset, 'metadata_items': metadata_items, 'assignments': assignments, 'strategy_mode': strategy['mode']}

def resolve_cowc_split_context() -> Dict:
    split_map = discover_available_splits_from_names(('train',))
    strategy = resolve_split_strategy(split_map)
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw train | Eval: canonical test <- raw train | split seed={}'.format(DEFAULT_SPLIT_SEED))
    return {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': 'train', 'few_shot_source_raw_split': 'train', 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}

def download_and_prepare_dataset(city: str, split: str='train'):
    from datasets import load_dataset, Image, DownloadConfig
    token = os.environ.get('HF_TOKEN')
    cache_dir = HF_CACHE_DIR
    download_config = make_resume_download_config(DownloadConfig)
    print(f'\nLoading dataset: {HF_DATASET_NAME} (city={city}, split={split})')
    print(f'Hugging Face cache directory: {cache_dir}')
    print('Hugging Face download resume: enabled (max_retries=10)')
    if dataset_cache_exists(HF_DATASET_NAME):
        print(f'Found local dataset cache for {HF_DATASET_NAME}; reusing it when possible.')
    else:
        print(f'Local dataset cache for {HF_DATASET_NAME} not found; Hugging Face will download/cache it.')
    dataset = load_dataset(HF_DATASET_NAME, city, split=split, token=token, cache_dir=cache_dir, download_mode='reuse_dataset_if_exists', download_config=download_config)
    dataset = dataset.cast_column('image', Image(decode=False))
    print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c != 'image'])
    metadata_items = []
    for i in progress_iter(range(len(metadata_ds)), desc=f'Scanning metadata for {city}'):
        row = metadata_ds[i]
        label = 0
        if 'objects' in row:
            obj = row['objects']
            if isinstance(obj, dict) and 'bbox' in obj and (obj['bbox'] is not None):
                label = len(obj['bbox'])
        if label <= 0:
            label = int(sum((row.get(h, 0) for h in COUNT_HEADER)))
        metadata_items.append({'index': i, 'label': label, 'city': city})
    selected_indices = select_evaluation_indices([item['index'] for item in metadata_items])
    metadata_by_index = {item['index']: item for item in metadata_items}
    items = [metadata_by_index[index] for index in selected_indices]
    return (dataset, items)

def result_suffix_from_prompt_metadata(prompt_metadata: Dict) -> str:
    if prompt_metadata['prompt_mode'] == 'few-shot':
        return f"-{prompt_metadata['few_shot_k']}shot"
    return ''

def result_path_for_model_city(output_dir: str, model_key: str, city: str, prompt_metadata: Dict) -> str:
    suffix = result_suffix_from_prompt_metadata(prompt_metadata)
    path = os.path.join(output_dir, f'{model_key}-{city}-counting{suffix}.jsonl')
    return result_path_for_reasoning(path, prompt_metadata.get('reasoning_method', 'baseline'))

def build_few_shot_source(city_datasets: Dict[str, Dict], k: int, raw_split: str, canonical_split: str):
    valid_candidates = []
    for city, payload in city_datasets.items():
        hf_dataset = payload['hf_dataset']
        metadata_items = payload['metadata_items']
        for item in metadata_items:
            count = normalize_count_answer(item.get('label'))
            if count is None:
                continue
            if not dataset_image_is_readable(hf_dataset, item['index']):
                print(f"Skipping invalid few-shot candidate #{city}/{item['index']}: image is not readable")
                continue
            valid_candidates.append({'index': item['index'], 'city': city, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'hf_dataset': hf_dataset, 'user_prompt': get_user_prompt(), 'answer': format_count_answer(count)})
    if len(valid_candidates) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}'. Found only {len(valid_candidates)} examples with readable images and non-negative counts.")
    return {'candidates': valid_candidates, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split}

def build_candidate_identity(item: Dict) -> Tuple[str, str, int]:
    return (item.get('city'), item.get('raw_split'), item['index'])

def select_few_shot_candidates(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidate_pool = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidate_pool = [candidate for candidate in candidate_pool if build_candidate_identity(candidate) != current_identity]
    if len(candidate_pool) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidate_pool)))
    return random.Random(seed).sample(candidate_pool, few_shot_source['k'])

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    sampled_candidates = select_few_shot_candidates(few_shot_source, seed, current_item=current_item)
    examples = []
    for candidate in sampled_candidates:
        raw_bytes = get_raw_image_bytes(candidate['hf_dataset'][candidate['index']])
        examples.append({'index': candidate['index'], 'city': candidate['city'], 'split': candidate['split'], 'raw_split': candidate['raw_split'], 'canonical_split': candidate['canonical_split'], 'image_url': raw_bytes_to_base64_uri(raw_bytes), 'user_prompt': candidate['user_prompt'], 'answer': candidate['answer']})
    return examples

def expected_few_shot_sample_metadata_by_index(dataset_items: List[Dict], few_shot_source: Optional[Dict], few_shot_seed: Optional[int], namespace: Optional[str]=None) -> Dict[int, Dict]:
    if not few_shot_source or few_shot_seed is None:
        return {}
    seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed, namespace=namespace)
    planned = {}
    for item in dataset_items:
        sample_seed = seed_by_index[item['index']]
        candidates = select_few_shot_candidates(few_shot_source, sample_seed, current_item=item)
        planned[item['index']] = few_shot_sample_metadata(candidates, sample_seed)
    return planned

def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, city: str, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, sample_parallel: int=4, reasoning_method: str='baseline'):
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': reasoning_method})
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None, namespace=city)
    expected_sample_metadata_by_index = expected_few_shot_sample_metadata_by_index(dataset_items, few_shot_source, few_shot_seed, namespace=city)
    if result_path is None:
        os.makedirs(output_dir, exist_ok=True)
        result_path = result_path_for_model_city(output_dir, model_key, city, prompt_metadata)
    else:
        result_path = result_path_for_reasoning(result_path, reasoning_method)
    print(f'[{model_key}] Starting city={city}, results -> {result_path}')
    selected_indices = [item['index'] for item in dataset_items]
    completed_indices = load_compatible_completed_indices(result_path, prompt_metadata, allowed_indices=selected_indices, expected_sample_metadata_by_index=expected_sample_metadata_by_index, expected_city=city, expected_model=model_key)
    pending_items = [item for item in dataset_items if item['index'] not in completed_indices]
    sample_parallel = max(1, int(sample_parallel or 1))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_parallel = 1
    print(f'[{model_key}] Processing {city} with {sample_parallel} sample threads')
    if not pending_items:
        print('Evaluation state updated.', flush=True)
        return result_path
    query_fn = get_query_function(model_key, api_key=api_key)
    failed_indices = []

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[i])
            image_data_uri = raw_bytes_to_base64_uri(raw_bytes)
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            task_prompt = get_user_prompt()
            query_prompt = effective_prompt(task_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

            def call_model(user_prompt: str):
                return query_fn(image_url=image_data_uri, system=query_system, user=user_prompt, few_shot_examples=sample_few_shot_examples_list)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='cowc', context={}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer = parse_model_response(answer_text)
            result_record = {'index': i, 'label': item['label'], 'city': item['city'], 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), **result_model_metadata(model_key, MODEL_CONFIGS), 'response': response, 'prompt': task_prompt, 'parsed_answer': parsed_answer}
            result_record.update(prompt_metadata)
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            if reasoning_method == 'cot':
                result_record['answer_text'] = answer_text
                result_record.update(cot_row_metadata(base_prompt=task_prompt, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            return (result_record, response_content)
        except Exception as e:
            print(f'[{model_key}] Error processing {city} sample {i}: {e}')
            import traceback
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS):
                raise
            failed_indices.append(i)
            return None
    if sample_parallel == 1:
        for item in progress_iter(pending_items, desc=f'{model_key} {city}', unit='sample'):
            result = _process_single_sample(item)
            if result is None:
                continue
            result_record, response_content = result
            append_to_jsonl(result_record, result_path)
            completed_indices.add(result_record['index'])
            print(f"[{model_key}] {city} #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['label']}")
    else:
        with ThreadPoolExecutor(max_workers=sample_parallel) as executor:
            futures = [executor.submit(_process_single_sample, item) for item in pending_items]
            for future in progress_iter(as_completed(futures), desc=f'{model_key} {city}', unit='sample'):
                result = future.result()
                if result is None:
                    continue
                result_record, response_content = result
                append_to_jsonl(result_record, result_path)
                completed_indices.add(result_record['index'])
                print(f"[{model_key}] {city} #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['label']}")
    if failed_indices:
        failed_preview = ', '.join((str(index) for index in failed_indices[:10]))
        raise RuntimeError(f'{model_key}/{city} left {len(failed_indices)} API samples incomplete after preserving all successful rows (indices: {failed_preview}). Resubmit the same command to continue only those items.')
    print(f'[{model_key}] Done for {city}. Results: {result_path}')
    return result_path

def evaluation(result_path: str, ax=None, allowed_indices: Optional[List[int]]=None, expected_metadata: Optional[Dict]=None, expected_sample_metadata_by_index: Optional[Dict[int, Dict]]=None) -> Dict:
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {result_path}')
    print(f"{'=' * 60}")
    parsed_name = parse_result_filename(result_path)
    if expected_metadata is None:
        parsed_shot = int(parsed_name['shot_suffix'][1:-4]) if parsed_name['shot_suffix'] else 0
        expected_metadata = expected_prompt_metadata(parsed_shot, parsed_name['reasoning_method'])
    selected_records = compatible_records_for_evaluation(result_path, expected_metadata, allowed_indices=allowed_indices, expected_sample_metadata_by_index=expected_sample_metadata_by_index, expected_city=parsed_name['city'] if parsed_name['combined'] == '0' else None, expected_model=parsed_name['model_key'])
    result_json = pd.DataFrame(selected_records)
    if result_json.empty:
        print(f'No compatible current-method rows found in {result_path}')
        return None
    result_json['model_response'] = result_json.apply(lambda row: row['answer_text'] if isinstance(row.get('answer_text'), str) else row['response']['choices'][0]['message']['content'] if isinstance(row['response'], dict) and 'choices' in row['response'] else row['response'], axis=1)

    def _parse_scored_response(row):
        return parse_model_response(row['model_response'])
    result_json['parsed_response'] = result_json.apply(_parse_scored_response, axis=1)
    result_json['count'] = result_json['label']
    result_json_no_refusal = result_json[result_json['parsed_response'] != -1].copy()
    _, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_json, result_json_no_refusal)
    rr = float((result_json['parsed_response'] == -1).mean())
    print(f'\nResults for {os.path.basename(result_path)}:')
    print('Evaluation state updated.', flush=True)
    print('Evaluation state updated.', flush=True)
    print(f'  Refusal Rate: {rr:.2%}')
    print(f'\nMetrics (failed answers counted as zero):')
    print(f'  MAE: {mae:.3f}')
    print(f'  MAPE: {mape:.3f}')
    print(f'  R²: {r2:.3f}')
    print(f'\nLaTeX format:')
    print(f'MAE & MAPE & R² & Refusal Rate')
    print(f'{mae:.3f} & {mape:.3f} & {r2:.3f} & {rr:.2f}')
    try:
        model_name = parsed_name['run_key']
        plot_scatter(result_json, ax=ax)
        if ax:
            ax.set_title(model_name)
    except Exception as e:
        print(f'Warning: Could not generate scatter plot: {e}')
    return {'model': model_name, 'mae': mae, 'mape': mape, 'r2': r2, 'refusal_rate': rr}

def combine_cities(result_dir: str=DEFAULT_RESULT_DIR, models: List[str]=None, cities: Optional[List[str]]=None, allowed_indices_by_city: Optional[Dict[str, List[int]]]=None, reasoning_method: Optional[str]=None, few_shot_k: Optional[int]=None, few_shot_seed: Optional[int]=None, expected_sample_metadata_by_city: Optional[Dict[str, Dict[int, Dict]]]=None):
    if models is None:
        model_set = set()
        for f in os.listdir(result_dir):
            if not f.endswith('.jsonl'):
                continue
            try:
                parsed = parse_result_filename(os.path.join(result_dir, f))
            except ValueError:
                continue
            if parsed['combined'] == '1':
                continue
            model_set.add(parsed['model_key'])
        models = sorted(model_set)
    if not models:
        print('No model results found to combine')
        return []
    combined_paths = []
    for model_key in models:
        grouped: Dict[str, List[str]] = {}
        for f in os.listdir(result_dir):
            if not f.endswith('.jsonl'):
                continue
            file_path = os.path.join(result_dir, f)
            try:
                parsed = parse_result_filename(file_path)
            except ValueError:
                continue
            if parsed['combined'] == '1' or parsed['model_key'] != model_key:
                continue
            if reasoning_method is not None and parsed['reasoning_method'] != reasoning_method:
                continue
            parsed_shot = int(parsed['shot_suffix'][1:-4]) if parsed['shot_suffix'] else 0
            if few_shot_k is not None and parsed_shot != few_shot_k:
                continue
            if cities is not None and parsed['city'] not in cities:
                continue
            suffix = parsed['shot_suffix'] or ''
            if parsed['reasoning_method'] == 'cot':
                suffix += '-CoT'
            grouped.setdefault(suffix, []).append(os.path.join(result_dir, f))
        for shot_suffix, paths in sorted(grouped.items(), key=lambda item: item[0]):
            dfs = []
            for file_path in sorted(paths):
                print(f'Loading: {file_path}')
                parsed = parse_result_filename(file_path)
                parsed_shot = int(parsed['shot_suffix'][1:-4]) if parsed['shot_suffix'] else 0
                expected = expected_prompt_metadata(parsed_shot, parsed['reasoning_method'], few_shot_seed=few_shot_seed)
                allowed_indices = allowed_indices_by_city.get(parsed['city'], []) if allowed_indices_by_city is not None else None
                selected_records = list(compatible_records_for_selection(file_path, expected, allowed_indices=allowed_indices, expected_sample_metadata_by_index=expected_sample_metadata_by_city.get(parsed['city']) if expected_sample_metadata_by_city is not None else None, expected_city=parsed['city'], expected_model=model_key).values())
                df = pd.DataFrame(selected_records)
                if df.empty:
                    continue
                dfs.append(df)
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['city', 'index'], keep='last')
                expected_count = sum((len(indices) for indices in allowed_indices_by_city.values())) if allowed_indices_by_city is not None else None
                if expected_count is not None and len(combined_df) != expected_count:
                    print(f'Not replacing the incomplete combined artifact for {model_key}{shot_suffix}.')
                    continue
                combined_path = os.path.join(result_dir, f'{model_key}-combined-counting{shot_suffix}.jsonl')
                if os.path.exists(combined_path) and allowed_indices_by_city is None:
                    try:
                        existing_count = sum((1 for _ in iter_jsonl(combined_path)))
                    except OSError:
                        existing_count = 0
                    if len(combined_df) < existing_count:
                        print(f'Not replacing {combined_path}: the rebuilt artifact is incomplete.')
                        continue
                temporary_path = f'{combined_path}.tmp'
                combined_df.to_json(temporary_path, orient='records', lines=True)
                os.replace(temporary_path, combined_path)
                print(f'Combined results saved to: {combined_path} ({len(combined_df)} samples)')
                combined_paths.append(combined_path)
    return combined_paths

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR, few_shot_seed: int=DEFAULT_FEW_SHOT_SEED):
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    result_files = []
    for f in os.listdir(result_dir):
        if not f.endswith('.jsonl'):
            continue
        file_path = os.path.join(result_dir, f)
        try:
            parsed = parse_result_filename(file_path)
        except ValueError:
            continue
        if parsed['combined'] != '1':
            continue
        if parsed['reasoning_method'] != 'baseline':
            continue
        result_files.append(f)
    if not result_files:
        print(f'No combined result files found in {result_dir}')
        print('Run --combine first to merge city results')
        return
    print(f'Found {len(result_files)} combined result files')
    all_metrics = []
    for result_file in progress_iter(result_files, desc='Evaluating models', total=len(result_files), unit='file'):
        result_path = os.path.join(result_dir, result_file)
        try:
            parsed = parse_result_filename(result_path)
            parsed_shot = int(parsed['shot_suffix'][1:-4]) if parsed['shot_suffix'] else 0
            metrics = evaluation(result_path, ax=None, expected_metadata=expected_prompt_metadata(parsed_shot, parsed['reasoning_method'], few_shot_seed=few_shot_seed))
            if metrics is not None:
                parsed = parse_result_filename(result_path)
                base_model_key = parsed['model_key']
                if base_model_key in MODEL_CONFIGS:
                    metrics['release_date'] = MODEL_CONFIGS[base_model_key]['release_date']
                    metrics['family'] = MODEL_CONFIGS[base_model_key]['family']
                    metrics['base_model_key'] = base_model_key
                    metrics['model_key'] = parsed['run_key']
                    metrics['result_file'] = result_file
                    metrics.update(shot_from_metrics(metrics, result_file))
                    all_metrics.append(metrics)
                else:
                    print(f'Warning: Model {base_model_key} not found in MODEL_CONFIGS')
        except Exception as e:
            print(f'Error evaluating {result_file}: {e}')
            continue
    if not all_metrics:
        print('No valid metrics collected, cannot generate plots')
        return
    write_shot_summary_json(all_metrics, os.path.join(result_dir, 'model_metrics_summary_all_shots.json'), dataset_name='COWC')
    zero_shot_metrics = [metrics for metrics in all_metrics if is_zero_shot_metrics(metrics, metrics.get('result_file'))]
    if not zero_shot_metrics:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return pd.DataFrame(all_metrics)
    df_metrics = pd.DataFrame(zero_shot_metrics)
    df_metrics['release_date'] = pd.to_datetime(df_metrics['release_date'])
    df_metrics = df_metrics.sort_values('release_date')
    print(f'\nCollected metrics for {len(df_metrics)} models')
    print(df_metrics[['model_key', 'release_date', 'mae', 'r2', 'refusal_rate']])
    import re
    from adjustText import adjust_text
    sns.set(font_scale=1.2, style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = FAMILY_COLORS
    ax1 = axes[0, 0]
    texts1 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax1.plot(df_family['release_date'], df_family['mae'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts1.append(ax1.text(row['release_date'], row['mae'], label, fontsize=12, color=colors[family]))
    ax1.set_xlabel('Release Date')
    ax1.set_ylabel('MAE (Mean Absolute Error)')
    ax1.set_title('MAE over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if texts1:
        adjust_text(texts1, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax2 = axes[0, 1]
    texts2 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax2.plot(df_family['release_date'], df_family['r2'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts2.append(ax2.text(row['release_date'], row['r2'], label, fontsize=12, color=colors[family]))
    ax2.set_xlabel('Release Date')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if texts2:
        adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax3 = axes[1, 0]
    texts3 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax3.plot(df_family['release_date'], df_family['mape'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts3.append(ax3.text(row['release_date'], row['mape'], label, fontsize=12, color=colors[family]))
    ax3.set_xlabel('Release Date')
    ax3.set_ylabel('MAPE (Mean Absolute Percentage Error)')
    ax3.set_title('MAPE over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if texts3:
        adjust_text(texts3, ax=ax3, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax4 = axes[1, 1]
    texts4 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax4.plot(df_family['release_date'], df_family['refusal_rate'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts4.append(ax4.text(row['release_date'], row['refusal_rate'], label, fontsize=12, color=colors[family]))
    ax4.set_xlabel('Release Date')
    ax4.set_ylabel('Refusal Rate')
    ax4.set_title('Refusal Rate over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    if texts4:
        adjust_text(texts4, ax=ax4, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle('COWC: Car Counting', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path_pdf = os.path.join(result_dir, 'model_comparison_over_time.pdf')
    output_path_png = os.path.join(result_dir, 'model_comparison_over_time.png')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 60}")
    print(f'Plots saved to:')
    print(f'  {output_path_pdf}')
    print(f'  {output_path_png}')
    print(f"{'=' * 60}\n")
    csv_path = os.path.join(result_dir, 'model_metrics_summary.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f'Metrics summary saved to: {csv_path}\n')
    return df_metrics

def main():
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('cowc'):
        return
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate vision models on COWC (Cars Overhead With Context) dataset', epilog='\nAvailable models:\n  Individual: gpt-5.2-2025-12-11, gemini-3-pro-preview, Qwen2.5-VL-7B-Instruct, etc.\n  Families:   GPT-all, Gemini-all, Qwen-all\n  Default:    Test all GPT/Gemini models\n\nExamples:\n  # Test single model on single city\n  python cowc.py --model gpt-5.2-2025-12-11 --city Utah\n\n  # Test all GPT models on all default cities\n  python cowc.py --model GPT-all\n\n  # Test all GPT/Gemini models in parallel on all cities\n  python cowc.py --parallel 8\n\n  # Force sequential execution\n  python cowc.py --parallel 1\n\n  # List models\n  python cowc.py --list-models\n\n  # Evaluate existing results\n  python cowc.py --evaluate-only results/inference/cowc/gpt-5.2-2025-12-11-Utah-counting.jsonl\n\n  # Combine city results and generate plots\n  python cowc.py --combine\n  python cowc.py --plot-only\n        ', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None, help='Model to test (model name, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--city', type=str, nargs='+', default=None, help='Cities to evaluate (default: all default cities)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--plot-only', action='store_true', help='Only generate comparison plots from existing results')
    parser.add_argument('--combine', action='store_true', help='Combine per-city results into combined files')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel threads (default: auto-detect based on models and API keys)')
    parser.add_argument('--sample-parallel', type=int, default=4, help='Concurrent sample/API calls per model (default: 4; use 1 to disable sample-level threading)')
    parser.add_argument('--qwen-model-path', type=str, default=None, help='Local Qwen checkpoint directory, or a root containing per-model subdirectories')
    parser.add_argument('--llava-model-path', type=str, default=None, help='Local LLaVA checkpoint directory, or a root containing per-model subdirectories')
    parser.add_argument('--qwen-finetuned-root', type=str, default=None, help='Root containing task-specific Qwen LoRA adapter directories')
    parser.add_argument('--few-shot', action='store_true', help=f'Enable few-shot prompting with {DEFAULT_FEW_SHOT_K} examples by default')
    parser.add_argument('--few-shot-k', type=int, default=0, help=f'Number of few-shot examples to use (default when --few-shot is set: {DEFAULT_FEW_SHOT_K})')
    parser.add_argument('--few-shot-seed', type=int, default=DEFAULT_FEW_SHOT_SEED, help=f'Random seed for few-shot example sampling (default: {DEFAULT_FEW_SHOT_SEED})')
    parser.add_argument('--reasoning-method', choices=REASONING_METHODS, default='baseline', help='Prompting method (default: baseline)')
    args = parser.parse_args()
    configure_cache_environment_from_args(args.cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.list_models:
        print('\nAvailable models:')
        print('\nGPT Models:')
        for name, config in MODEL_CONFIGS.items():
            if config['family'] == 'GPT':
                print(f"  {name} (released: {config['release_date']})")
        print('\nGemini Models:')
        for name, config in MODEL_CONFIGS.items():
            if config['family'] == 'Gemini':
                print(f"  {name} (released: {config['release_date']})")
        print('\nQwen Models:')
        for name, config in MODEL_CONFIGS.items():
            if config['family'] == 'Qwen':
                print(format_qwen_model_listing(name, config, args.qwen_finetuned_root))
        print('\nLLaVA Models:')
        for name, config in MODEL_CONFIGS.items():
            if config['family'] == 'LLaVA':
                print(format_llava_model_listing(name, config, args.llava_model_path))
        print('\nModel families:')
        print('  GPT-all    - Test all GPT models')
        print('  Gemini-all - Test all Gemini models')
        print('  Qwen-all   - Test all base Qwen models (excludes LoRA)')
        print('  LLaVA-all  - Test all LLaVA models')
        print('  (default)  - Test all GPT/Gemini models')
        print_runtime_model_summary(list(MODEL_CONFIGS.keys()), MODEL_CONFIGS)
        print(f"\nDefault cities: {', '.join(DEFAULT_CITIES)}")
        return
    if args.combine:
        combine_cities(args.output_dir, few_shot_seed=args.few_shot_seed)
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir, few_shot_seed=args.few_shot_seed)
        return
    if args.evaluate_only:
        parsed_result = parse_result_filename(args.evaluate_only)
        parsed_shot = int(parsed_result['shot_suffix'][1:-4]) if parsed_result['shot_suffix'] else 0
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only, expected_metadata=expected_prompt_metadata(parsed_shot, parsed_result['reasoning_method'], few_shot_seed=args.few_shot_seed))
        return
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative.')
    prompt_metadata = None
    cities = args.city or DEFAULT_CITIES
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        print_provider_summary()
    split_context = resolve_cowc_split_context()
    needed_payload_cities = set(cities)
    if few_shot_k > 0:
        needed_payload_cities.update(DEFAULT_CITIES)
    city_payload_cache = {}
    for city_name in sorted(needed_payload_cities):
        city_payload_cache[city_name] = load_canonical_city_payload(city_name)
    print(f"\n{'=' * 60}")
    print(f'Models to test: {len(models_to_test)}')
    print(f"Cities to test: {', '.join(cities)}")
    print_runtime_model_summary(models_to_test, MODEL_CONFIGS)
    print(f"{'=' * 60}\n")
    few_shot_source = None
    if few_shot_k > 0:
        print(f'\nLoading few-shot source candidates from canonical train pool across default cities: k={few_shot_k}')
        few_shot_city_payloads = {}
        for city in DEFAULT_CITIES:
            city_payload = city_payload_cache[city]
            train_assignment = city_payload['assignments']['train']
            few_shot_items = select_metadata_items_by_indices(city_payload['metadata_items'], train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
            few_shot_city_payloads[city] = {'hf_dataset': city_payload['hf_dataset'], 'metadata_items': few_shot_items}
        few_shot_source = build_few_shot_source(few_shot_city_payloads, k=few_shot_k, raw_split='train', canonical_split='train')
        print('Evaluation state updated.', flush=True)
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=args.few_shot_seed)
    if args.reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': args.reasoning_method})
    if args.parallel is None:
        if has_local_models(models_to_test, MODEL_CONFIGS):
            num_parallel = 1
        else:
            num_parallel = min(len(models_to_test), len(api_keys_for_models(models_to_test, MODEL_CONFIGS)))
    else:
        num_parallel = min(args.parallel, len(models_to_test))
    tasks = []
    for idx, model_key in enumerate(models_to_test):
        if is_local_model(model_key, MODEL_CONFIGS):
            api_key = None
            print(f'  {model_key:<35} -> local checkpoint: {_QWEN_MODEL_PATHS[model_key]}')
        else:
            api_key = key_for_model_index(model_key, idx, MODEL_CONFIGS)
            key_display = safe_key_display(api_key)
            print(f'  {model_key:<35} -> API Key #{key_index_for_model(model_key, idx, MODEL_CONFIGS)} ({key_display})')
        tasks.append((model_key, api_key, args.output_dir, args.sample_parallel))
    global _shared_dataset_items, _shared_hf_dataset, _shared_city, _shared_split_context
    all_result_paths = []
    evaluation_indices_by_city = {}
    expected_sample_metadata_by_city = {}
    run_failures = []
    for city in cities:
        print(f"\n{'=' * 60}")
        print(f'Processing city: {city}')
        print(f"{'=' * 60}\n")
        city_payload = city_payload_cache[city]
        eval_assignment = city_payload['assignments']['test']
        eval_indices = select_evaluation_indices(eval_assignment['indices'])
        evaluation_indices_by_city[city] = eval_indices
        _shared_hf_dataset = city_payload['hf_dataset']
        _shared_dataset_items = select_metadata_items_by_indices(city_payload['metadata_items'], eval_indices, raw_split=eval_assignment['raw_split'], canonical_split='test')
        _shared_city = city
        _shared_split_context = split_context
        expected_sample_metadata_by_city[city] = expected_few_shot_sample_metadata_by_index(_shared_dataset_items, few_shot_source, args.few_shot_seed if few_shot_source else None, namespace=city)
        pending_tasks = tasks
        if num_parallel > 1 and len(pending_tasks) > 1:
            city_parallel = min(num_parallel, len(pending_tasks))
            print(f'Starting {city_parallel} parallel threads for {city}...')
            with ThreadPoolExecutor(max_workers=city_parallel) as executor:
                futures = {executor.submit(run_single_model, (model_key, api_key, output_dir, sample_parallel, few_shot_source, args.few_shot_seed if few_shot_source else None, args.reasoning_method)): model_key for model_key, api_key, output_dir, sample_parallel in pending_tasks}
                for future in as_completed(futures):
                    model_key = futures[future]
                    try:
                        result_path = future.result()
                        if result_path is not None:
                            all_result_paths.append(result_path)
                            print(f'Completed: {model_key} on {city}')
                    except Exception as e:
                        print(f'Failed: {model_key} on {city}: {e}')
                        run_failures.append((model_key, city, str(e)))
                        if is_local_model(model_key, MODEL_CONFIGS):
                            raise
        else:
            print(f'Running sequentially for {city}...')
            for model_key, api_key, output_dir, sample_parallel in progress_iter(pending_tasks, desc=f'Testing models on {city}', total=len(pending_tasks), unit='model'):
                try:
                    result_path = run_single_model((model_key, api_key, output_dir, sample_parallel, few_shot_source, args.few_shot_seed if few_shot_source else None, args.reasoning_method))
                    if result_path is not None:
                        all_result_paths.append(result_path)
                except Exception as e:
                    print(f'Failed: {model_key} on {city}: {e}')
                    run_failures.append((model_key, city, str(e)))
                    if is_local_model(model_key, MODEL_CONFIGS):
                        raise
                    continue
    if len(cities) > 1:
        print(f'\nCombining city results...')
        combined_paths = combine_cities(args.output_dir, models=[task[0] for task in tasks], cities=list(evaluation_indices_by_city), allowed_indices_by_city=evaluation_indices_by_city, reasoning_method=args.reasoning_method, few_shot_k=few_shot_k, few_shot_seed=args.few_shot_seed, expected_sample_metadata_by_city=expected_sample_metadata_by_city)
    else:
        combined_paths = []
    print(f"\n{'=' * 60}")
    print(f'Evaluating {len(all_result_paths)} result files')
    print(f"{'=' * 60}\n")
    for result_path in all_result_paths:
        try:
            parsed_result = parse_result_filename(result_path)
            evaluate_and_write_cot_metrics(evaluation, result_path, allowed_indices=evaluation_indices_by_city.get(parsed_result['city']), expected_metadata=prompt_metadata, expected_sample_metadata_by_index=expected_sample_metadata_by_city.get(parsed_result['city']))
        except Exception as e:
            print(f'Error evaluating {result_path}: {e}')
            continue
    for combined_path in combined_paths:
        try:
            evaluate_and_write_cot_metrics(evaluation, combined_path, expected_metadata=prompt_metadata)
        except Exception as e:
            print(f'Error evaluating {combined_path}: {e}')
            continue
    if len(combined_paths) > 1:
        print(f'\nGenerating comparison plots...')
        plot_model_comparison(args.output_dir, few_shot_seed=args.few_shot_seed)
    print(f"\n{'=' * 60}")
    print(f'All tasks completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")
    if run_failures:
        summary = '; '.join((f'{model}/{city}: {message}' for model, city, message in run_failures[:10]))
        raise RuntimeError(f'COWC run finished with {len(run_failures)} incomplete model/city groups. Successful items were preserved for resume. {summary}')
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_city = None
_shared_split_context = None

def run_single_model(args_tuple):
    model_key, api_key, output_dir, sample_parallel, few_shot_source, few_shot_seed, reasoning_method = args_tuple
    return query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, city=_shared_city, split_context=_shared_split_context, result_path=None, output_dir=output_dir, api_key=api_key, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, sample_parallel=sample_parallel, reasoning_method=reasoning_method)
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
