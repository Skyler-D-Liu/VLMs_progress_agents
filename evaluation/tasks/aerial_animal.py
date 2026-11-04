"""Aerial animal."""
import sys
import os
import threading
import time
from pathlib import Path
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UTILS_DIR))
from cot_prompting import REASONING_METHODS, call_with_reasoning, cot_row_metadata, evaluate_and_write_cot_metrics, effective_prompt, effective_system_message, local_generation_max_new_tokens, result_path_for_reasoning, validate_cot_run
import benchmark_environment as benchmark_environment_module
from benchmark_environment import APIStatusError, RESULTS_ROOT, api_host_for_model, api_keys_for_models, is_fatal_provider_error, key_for_model_index, key_index_for_model, load_benchmark_environment, post_gemini_json, post_openai_json, print_model_key_summary, print_provider_summary, safe_key_display, random_api_key_for_model
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
from comparison_summary_utils import shot_from_metrics, write_shot_summary_json
from dataset_splits import DEFAULT_SPLIT_SEED, build_split_assignments, discover_available_splits_from_names, resolve_split_strategy
PLOT_FAMILY_ORDER = ('GPT', 'Gemini', 'Qwen', 'LLaVA')
FAMILY_COLORS = {'GPT': '#397CB8', 'Gemini': '#EA4335', 'Qwen': '#F59E0B', 'LLaVA': '#7C3AED'}
QWEN_MODEL_RELEASE_DATES = {'Qwen-VL-Chat': '2023-09-01', 'Qwen2-VL-2B-Instruct': '2024-08-30', 'Qwen2-VL-7B-Instruct': '2024-08-30', 'Qwen2-VL-72B-Instruct': '2024-09-19', 'Qwen2.5-VL-3B-Instruct': '2025-01-28', 'Qwen2.5-VL-7B-Instruct': '2025-01-28', 'Qwen2.5-VL-72B-Instruct': '2025-01-28', 'Qwen3-VL-2B-Instruct': '2025-10-21', 'Qwen3-VL-8B-Instruct': '2025-10-15', 'Qwen3-VL-32B-Instruct': '2025-10-21'}
LORA_TASK = 'aerial-animal'
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

def _run_qwen_chat_template(bundle, image_refs, system, user, few_shot_examples, *, allow_repeated_output=False):
    if bundle.get('runtime_family') == 'LLaVA':
        return run_llava_chat(bundle, image_refs, system, user, few_shot_examples, max_new_tokens=local_generation_max_new_tokens(user))
    if is_legacy_qwen_vl_bundle(bundle):
        return run_legacy_qwen_vl_chat(bundle, image_refs, system, user, few_shot_examples, allow_repeated_output=allow_repeated_output)
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

def make_qwen_query_callable(model_key, model_path, *, allow_repeated_output=False):
    bundle = _load_qwen_local_model(model_key, model_path)

    def _query_qwen(image_url=None, system=None, user=None, few_shot_examples=None, image_urls=None, sample_index=None):
        del sample_index
        image_refs = _collect_image_refs(image_url=image_url, image_urls=image_urls)
        content = _run_qwen_chat_template(bundle, image_refs, system, user, few_shot_examples, allow_repeated_output=allow_repeated_output)
        return {'model': MODEL_CONFIGS[model_key]['model_name'], 'choices': [{'message': {'content': content}}], 'original_response': llava_response_metadata(model_path) if MODEL_CONFIGS[model_key].get('family') == 'LLaVA' else local_response_metadata(model_path)}
    return _query_qwen
import json
import os
import http.client
import base64
import io
import random
import re
import math
import csv
from datetime import datetime
from typing import Optional, Callable, List, Dict, Tuple
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential
except ImportError:

    def retry(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]) and (not decorator_kwargs):
            return decorator_args[0]

        def _decorator(func):
            return func
        return _decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None

    def retry_if_exception(*args, **kwargs):
        return None

def progress_iter(iterable, desc: Optional[str]=None, total: Optional[int]=None, unit: str='sample'):
    return iterable

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _squared_pearson_corr(y_true: List[float], y_pred: List[float]) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    mean_true = _mean(y_true)
    mean_pred = _mean(y_pred)
    num = sum(((a - mean_true) * (b - mean_pred) for a, b in zip(y_true, y_pred)))
    den_true = math.sqrt(sum(((a - mean_true) ** 2 for a in y_true)))
    den_pred = math.sqrt(sum(((b - mean_pred) ** 2 for b in y_pred)))
    if den_true == 0 or den_pred == 0:
        return 0.0
    corr = num / (den_true * den_pred)
    return corr * corr

def normalize_result_index(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch('\\+?\\d+', text):
            return int(text)
    return None

def calculate_counting_metrics(result_records: List[Dict], result_records_no_refusal: List[Dict]):
    if not result_records:
        raise ValueError('No results available for metric calculation.')
    y_true = [float(row['count']) for row in result_records]
    y_pred = [0.0 if row['parsed_response'] == -1 else float(row['parsed_response']) for row in result_records]
    y_true_nr = [float(row['count']) for row in result_records_no_refusal]
    y_pred_nr = [float(row['parsed_response']) for row in result_records_no_refusal]

    def _mape(true_values: List[float], pred_values: List[float]) -> float:
        if not true_values:
            return 0.0
        eps = sys.float_info.epsilon
        return _mean([abs((t - p) / max(abs(t), eps)) for t, p in zip(true_values, pred_values)])

    def _mae(true_values: List[float], pred_values: List[float]) -> float:
        if not true_values:
            return 0.0
        return _mean([abs(t - p) for t, p in zip(true_values, pred_values)])
    rr = 1.0 - len(result_records_no_refusal) / len(result_records)
    mape = _mape(y_true, y_pred)
    mape_no_refusal = _mape(y_true_nr, y_pred_nr) if y_true_nr else None
    mae = _mae(y_true, y_pred)
    mae_no_refusal = _mae(y_true_nr, y_pred_nr) if y_true_nr else None
    r2 = _squared_pearson_corr(y_true, y_pred)
    r2_no_refusal = _squared_pearson_corr(y_true_nr, y_pred_nr) if y_true_nr else None
    return (rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal))

def plot_scatter(records: List[Dict], ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: plotting libraries not installed; skipping scatter plot.')
        return ax
    if not records:
        return ax
    df = records
    pred = [row['parsed_response'] for row in df]
    true = [row['count'] for row in df]
    max_limit = max(max(pred), max(true)) if pred and true else 0
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)
    xs = pred
    ys = true
    ax.scatter(xs, ys, marker='x', color='k')
    if len(xs) > 1:
        x_mean = _mean([float(x) for x in xs])
        y_mean = _mean([float(y) for y in ys])
        denom = sum(((x - x_mean) ** 2 for x in xs))
        slope = 0.0 if denom == 0 else sum(((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))) / denom
        intercept = y_mean - slope * x_mean
        line_x = [0, max_limit]
        line_y = [intercept, slope * max_limit + intercept]
        ax.plot(line_x, line_y, linestyle='--', color='red')
    r2 = _squared_pearson_corr([float(v) for v in true], [float(v) for v in pred])
    ax.text(0.95, 0.95, f'$R^2 = {r2:.2f}$', verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=24)
    return ax

def load_jsonl_records(path: str) -> List[Dict]:
    latest_by_index = {}
    for position, record in enumerate(iter_jsonl(path)):
        if not isinstance(record, dict):
            continue
        index = normalize_result_index(record.get('index'))
        if index is None:
            continue
        latest_by_index[index] = (position, record)
    ordered_records = list(latest_by_index.values())
    return [record for _, record in sorted(ordered_records, key=lambda item: item[0])]

def clean_result_jsonl_for_resume(path: str) -> set:
    if not os.path.exists(path):
        return set()
    return {index for record in iter_jsonl(path) if isinstance(record, dict) for index in (normalize_result_index(record.get('index')),) if index is not None}

def completed_dataset_indices(path: str, dataset_items: List[Dict]) -> set:
    del dataset_items
    return clean_result_jsonl_for_resume(path)

def validate_resume_samples(path: str, dataset_items: List[Dict], model_key: str) -> None:
    expected_model = str(model_key)
    filename_method = classify_result_filename(path)
    filename_model = model_key_from_result_filename(path) if filename_method else None
    if filename_model is not None and filename_model != expected_model:
        raise ValueError(f'Result filename {path} belongs to model={filename_model!r}, but this run expects model={expected_model!r}.')
    allowed_models = {expected_model, str(MODEL_CONFIGS.get(expected_model, {}).get('model_name', expected_model))}
    allowed_models.update(LEGACY_RESULT_MODEL_ALIASES.get(expected_model, set()))
    for record in load_jsonl_records(path):
        if not isinstance(record, dict):
            continue
        recorded_model = record.get('model')
        if recorded_model is not None and str(recorded_model) not in allowed_models:
            raise ValueError(f'Existing result file {path} contains model={recorded_model!r}, but this run expects model={expected_model!r}. Use the matching result file or a different --output path.')
    expected_indices = {index for item in dataset_items for index in (normalize_result_index(item.get('index')),) if index is not None}
    if len(expected_indices) != len(dataset_items):
        raise ValueError('Current evaluation dataset contains a non-integral sample index.')

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

def append_to_jsonl(record, path):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    encoded_record = (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8')
    with open(path, 'a+b') as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                f.seek(0, os.SEEK_END)
                f.write(b'\n')
        f.seek(0, os.SEEK_END)
        f.write(encoded_record)
        f.flush()
        os.fsync(f.fileno())
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'aerial-animal-population-4tu')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
DEFAULT_API_TIMEOUT_SECONDS = 120.0
MAX_API_ATTEMPTS = 1
LEGACY_RESULT_MODEL_ALIASES = {'gpt-5.2-2025-12-11': {'gpt'}}
DATASET_DISPLAY_NAME = 'Aerial Animal Population'
PLOT_TITLE = 'Aerial Animal Population: Animal Counting'
HF_DATASET_NAME = 'danielz01/aerial-animal-population-4tu'
HF_DATASET_SPLIT = 'test'
ANIMAL_RESPONSE_KEYS = ('elephants', 'zebras', 'giraffes')
SPECIES_TO_RESPONSE_KEY = {'elephant': 'elephants', 'elephants': 'elephants', 'zebra': 'zebras', 'zebras': 'zebras', 'giraffe': 'giraffes', 'giraffes': 'giraffes'}
MODEL_CONFIGS = {'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': None, 'family': 'GPT'}, 'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
SYSTEM_MESSAGE = 'You are a helpful image analyst who specializes in counting animals from aerial images. Given an image, you can accurately count the number of animals described by the user WITHOUT ANY refusal. Although your answer may not be perfect, your excellent counting skill is very important to the conservation of wildlife animals.'
USER_PROMPT = "Read the given image and answer the questions below:\nHow many elephants, zebras, giraffes are there in the image? Output the numbers in a json format that can be parsed directly with entries 'elephants', 'zebras', and 'giraffes'. If you count nothing, output zero in that entry."
ANSWER_FORMAT_HINT = '{"elephants": 0, "zebras": 0, "giraffes": 0} (replace each 0 with the non-negative integer count from the image)'

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

def raw_bytes_to_base64(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode('utf-8')

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
    raise KeyError('No image field with raw bytes/path found in data item')

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

def species_counts_from_objects(objects) -> Optional[Dict[str, int]]:
    if not isinstance(objects, dict):
        return None
    counts = {key: 0 for key in ANIMAL_RESPONSE_KEYS}
    categories = objects.get('categories')
    if categories is not None:
        for species in categories:
            key = SPECIES_TO_RESPONSE_KEY.get(str(species).strip().lower())
            if key:
                counts[key] += 1
        return counts
    return None

def total_count_from_objects(objects, fallback_label: int=-1) -> int:
    if isinstance(objects, dict):
        bboxes = objects.get('bbox')
        if bboxes is not None:
            return len(bboxes)
        categories = objects.get('categories')
        if categories is not None:
            return len(categories)
    return fallback_label

def format_count_answer(counts: Dict[str, int]) -> str:
    ordered = {key: int(counts.get(key, 0)) for key in ANIMAL_RESPONSE_KEYS}
    return json.dumps(ordered, ensure_ascii=False)

def extract_response_content(response) -> str:
    if isinstance(response, dict):
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            if isinstance(choice, dict):
                message = choice.get('message', {})
                if isinstance(message, dict):
                    return message.get('content') or ''
        if 'content' in response:
            return response.get('content') or ''
    return '' if response is None else str(response)

def extract_json_candidate(text: str) -> str:
    text = '' if text is None else str(text).strip()
    fenced = re.search('```(?:json)?\\s*(.*?)\\s*```', text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and (end > start):
        return text[start:end + 1].strip()
    return text.replace('json\n', '').replace('```', '').replace('\n', '')

def coerce_count_value(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search('-?\\d+(?:\\.\\d+)?', str(value))
        if not match:
            return None
        number = float(match.group(0))
    if not math.isfinite(number) or number < 0 or (not number.is_integer()):
        return None
    return int(number)

def parse_count_response(response_text: str) -> Tuple[int, Optional[Dict[str, int]]]:
    from evaluation.utils.response_protocol import parse_answer
    counts = parse_answer('animals', response_text)
    if counts is None or any(value is None for value in counts.values()):
        return (-1, None)
    return sum(counts.values()), counts

def build_candidate_identity(item: Dict) -> Tuple[str, int]:
    return (item.get('raw_split'), item['index'])

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str) -> Dict:
    annotations = hf_dataset.select_columns(['objects'])
    valid_examples = []
    for item in metadata_items:
        original_idx = item['index']
        data_item = annotations[original_idx]
        counts = species_counts_from_objects(data_item.get('objects'))
        if counts is None:
            continue
        valid_examples.append({'index': original_idx, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'counts': counts, 'answer': format_count_answer(counts)})
    if len(valid_examples) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}'. Found only {len(valid_examples)} examples with readable images and count annotations.")

    def materialize_candidate(candidate):
        raw_bytes = get_raw_image_bytes(hf_dataset[candidate['index']])
        return {**candidate, 'base64_image': raw_bytes_to_base64(raw_bytes), 'mime_type': detect_mime_type(raw_bytes)}
    return {'candidates': valid_examples, 'materialize_candidate': materialize_candidate, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split}

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidates = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidates = [candidate for candidate in candidates if build_candidate_identity(candidate) != current_identity]
    if len(candidates) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidates)))
    sampled_candidates = random.Random(seed).sample(candidates, few_shot_source['k'])
    examples = []
    for candidate in sampled_candidates:
        materializer = few_shot_source.get('materialize_candidate')
        if callable(materializer):
            candidate = materializer(candidate)
        example = dict(candidate)
        example['image_url'] = f"data:{example['mime_type']};base64,{example['base64_image']}"
        example['user_prompt'] = USER_PROMPT
        examples.append(example)
    return examples

def few_shot_metadata(few_shot_source: Optional[Dict], split_context: Dict, few_shot_seed: Optional[int]=None) -> Dict:
    source = few_shot_source or {}
    k = int(source.get('k', 0) or 0)
    seed_value = few_shot_seed if k > 0 else None
    return {'canonical_strategy_mode': split_context['canonical_strategy_mode'], 'canonical_split_seed': split_context['canonical_split_seed'], 'eval_raw_split': split_context['eval_raw_split'], 'few_shot_source_raw_split': split_context['few_shot_source_raw_split'], 'eval_canonical_split': split_context['eval_canonical_split'], 'few_shot_source_canonical_split': split_context['few_shot_source_canonical_split'], 'prompt_mode': 'few-shot' if k > 0 else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k > 0 else None, 'few_shot_seed': seed_value}

def few_shot_sample_metadata(few_shot_examples: Optional[List[Dict]], few_shot_sample_seed: Optional[int]) -> Dict:
    examples = few_shot_examples or []
    return {'few_shot_sample_seed': few_shot_sample_seed, 'few_shot_indices': [example['index'] for example in examples], 'few_shot_raw_splits': [example.get('raw_split') for example in examples], 'few_shot_canonical_splits': [example.get('canonical_split') for example in examples], 'few_shot_answers': [example['answer'] for example in examples]}

def build_few_shot_seed_by_index(dataset_items: List[Dict], few_shot_seed: Optional[int]) -> Dict[int, int]:
    if few_shot_seed is None:
        return {}
    rng = random.Random(few_shot_seed)
    return {item['index']: rng.randrange(2 ** 32) for item in dataset_items}

def classify_result_filename(filename: str) -> Optional[str]:
    basename = os.path.basename(filename)
    if re.fullmatch('.+-counting-CoT\\.jsonl', basename):
        return 'cot'
    if re.fullmatch('.+-counting-\\d+shot\\.jsonl', basename):
        return 'few-shot'
    if re.fullmatch('.+-counting\\.jsonl', basename):
        return 'zero-shot'
    return None

def result_filename_metadata(path: str) -> Dict:
    basename = os.path.basename(path)
    method = classify_result_filename(basename)
    if method == 'cot':
        return {'prompt_mode': 'zero-shot', 'few_shot_k': 0, 'reasoning_method': 'cot'}
    if method == 'few-shot':
        match = re.fullmatch('.+-counting-(\\d+)shot\\.jsonl', basename)
        return {'prompt_mode': 'few-shot', 'few_shot_k': int(match.group(1)), 'reasoning_method': 'baseline'}
    if method == 'zero-shot':
        return {'prompt_mode': 'zero-shot', 'few_shot_k': 0, 'reasoning_method': 'baseline'}
    return {}

def validate_resume_config(path: str, expected_metadata: Dict):
    if not os.path.exists(path):
        return
    comparable_keys = ['canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'reasoning_method']
    expected = dict(expected_metadata)
    expected.setdefault('reasoning_method', 'baseline')
    inferred = result_filename_metadata(path)
    for key, inferred_value in inferred.items():
        if expected.get(key) != inferred_value:
            raise ValueError(f'Result filename {path} implies {key}={inferred_value!r}, but this run expects {expected.get(key)!r}.')
    for record in load_jsonl_records(path):
        if not isinstance(record, dict):
            continue
        for key in comparable_keys:
            record_value = record.get(key)
            if record_value is None:
                record_value = inferred.get(key)
            if record_value is None:
                continue
            if record_value != expected.get(key):
                raise ValueError(f'Existing result file {path} has {key}={record_value!r}, but this run expects {expected.get(key)!r}. Use a different --output path or remove the old result file.')

def model_key_from_result_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    match = re.match('(.+)-counting(?:-\\d+shot)?(?:-CoT)?\\.jsonl$', basename)
    if match:
        return match.group(1)
    return basename.replace('.jsonl', '')
_API_RATE_LOCK = threading.Lock()
_API_NEXT_REQUEST_AT: Dict[str, float] = {}

def api_request_timeout_seconds() -> float:
    raw_timeout = os.environ.get('VLEO_API_TIMEOUT_SECONDS', str(DEFAULT_API_TIMEOUT_SECONDS))
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError('VLEO_API_TIMEOUT_SECONDS must be a positive number') from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('VLEO_API_TIMEOUT_SECONDS must be a positive number')
    return timeout

def extract_gemini_answer_text(response: Dict) -> str:
    if not isinstance(response, dict):
        return ''
    candidates = response.get('candidates') or []
    if not candidates or not isinstance(candidates[0], dict):
        return ''
    content = candidates[0].get('content')
    if not isinstance(content, dict):
        return ''
    parts = content.get('parts') or []
    if not isinstance(parts, list):
        return ''
    return ''.join((part['text'] for part in parts if isinstance(part, dict) and (not part.get('thought', False)) and isinstance(part.get('text'), str)))

def _wait_for_api_rate_slot(model_key: str) -> None:
    raw_interval = os.environ.get('VLEO_API_MIN_REQUEST_INTERVAL_SECONDS', '0')
    try:
        interval = float(raw_interval)
    except ValueError as exc:
        raise ValueError('VLEO_API_MIN_REQUEST_INTERVAL_SECONDS must be a non-negative number') from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError('VLEO_API_MIN_REQUEST_INTERVAL_SECONDS must be a non-negative number')
    if interval == 0:
        return
    family = str(MODEL_CONFIGS[model_key].get('family', model_key))
    with _API_RATE_LOCK:
        now = time.monotonic()
        request_at = max(now, _API_NEXT_REQUEST_AT.get(family, now))
        delay = request_at - now
        if delay > 0:
            time.sleep(delay)
        _API_NEXT_REQUEST_AT[family] = time.monotonic() + interval

def _retry_transient_provider_error(exc: BaseException) -> bool:
    return not is_fatal_provider_error(str(exc))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(MAX_API_ATTEMPTS), retry=retry_if_exception(_retry_transient_provider_error))
def query_openai(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, rotate_api_key: bool=True) -> dict:
    if rotate_api_key:
        api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    _wait_for_api_rate_slot(model_key)
    config = MODEL_CONFIGS[model_key]
    messages = [{'role': 'system', 'content': system}]
    for example in few_shot_examples or []:
        messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': example['image_url'], 'detail': 'high'}}, {'type': 'text', 'text': example['user_prompt']}]})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}}, {'type': 'text', 'text': user}]})
    payload = json.dumps({'model': config['model_name'], 'messages': messages})
    status, data = post_openai_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=api_request_timeout_seconds())
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    return json.loads(data.decode('utf-8'))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(MAX_API_ATTEMPTS), retry=retry_if_exception(_retry_transient_provider_error))
def query_gemini(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, rotate_api_key: bool=True) -> dict:
    if rotate_api_key:
        api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    _wait_for_api_rate_slot(model_key)
    config = MODEL_CONFIGS[model_key]
    if image_url.startswith('data:'):
        header, b64_data = image_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        image_part = {'inlineData': {'mimeType': mime_type, 'data': b64_data}}
    else:
        image_part = {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}
    contents = []
    for example in few_shot_examples or []:
        contents.append({'role': 'user', 'parts': [{'inlineData': {'mimeType': example['mime_type'], 'data': example['base64_image']}}, {'text': example['user_prompt']}]})
        contents.append({'role': 'model', 'parts': [{'text': example['answer']}]})
    contents.append({'role': 'user', 'parts': [image_part, {'text': user}]})
    payload = json.dumps({'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents})
    status, data = post_gemini_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=api_request_timeout_seconds())
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    content = extract_gemini_answer_text(result)
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
    if config['api_type'] == 'openai':
        return lambda image_url, system, user, few_shot_examples=None: query_openai(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples)
    elif config['api_type'] == 'google_native':
        return lambda image_url, system, user, few_shot_examples=None: query_gemini(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples)
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")

def get_models_to_test(model_arg: str) -> List[str]:
    return expand_model_selection(model_arg, MODEL_CONFIGS)

def select_evaluation_indices(total_size: int) -> List[int]:
    return list(range(total_size))

def select_evaluation_indices_from_pool(candidate_indices: List[int]) -> List[int]:
    return list(candidate_indices)

def select_metadata_items_by_indices(metadata_items: List[Dict], selected_indices: List[int], raw_split: str, canonical_split: str) -> List[Dict]:
    metadata_by_index = {item['index']: item for item in metadata_items}
    selected_items = []
    for index in selected_indices:
        item = dict(metadata_by_index[index])
        item['raw_split'] = raw_split
        item['canonical_split'] = canonical_split
        selected_items.append(item)
    return selected_items

def resolve_canonical_pools(few_shot_k: int) -> Dict:
    split_map = discover_available_splits_from_names(('train', 'val', 'test'))
    strategy = resolve_split_strategy(split_map)
    datasets_by_raw_split = {}
    metadata_by_raw_split = {}
    split_lengths = {}
    for raw_split in sorted(set(split_map.values())):
        hf_dataset, metadata_items = download_and_prepare_dataset(split=raw_split)
        datasets_by_raw_split[raw_split] = hf_dataset
        metadata_by_raw_split[raw_split] = metadata_items
        split_lengths[raw_split] = len(hf_dataset)
    assignments = build_split_assignments(split_map=split_map, split_lengths=split_lengths, seed=DEFAULT_SPLIT_SEED)
    eval_assignment = assignments['test']
    train_assignment = assignments['train']
    eval_selected_indices = select_evaluation_indices_from_pool(eval_assignment['indices'])
    evaluation_items = select_metadata_items_by_indices(metadata_by_raw_split[eval_assignment['raw_split']], eval_selected_indices, raw_split=eval_assignment['raw_split'], canonical_split='test')
    few_shot_source = None
    if few_shot_k > 0:
        few_shot_source_items = select_metadata_items_by_indices(metadata_by_raw_split[train_assignment['raw_split']], train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(datasets_by_raw_split[train_assignment['raw_split']], few_shot_source_items, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train')
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw {} | Eval: canonical test <- raw {} | split seed={}'.format(train_assignment['raw_split'], eval_assignment['raw_split'], DEFAULT_SPLIT_SEED))
    return {'evaluation_dataset': datasets_by_raw_split[eval_assignment['raw_split']], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def download_and_prepare_dataset(split: str='test'):
    from datasets import load_dataset, Image, DownloadConfig
    token = os.environ.get('HF_TOKEN')
    cache_dir = HF_CACHE_DIR
    download_config = make_resume_download_config(DownloadConfig)
    print(f'\nLoading dataset: {HF_DATASET_NAME} (split={split})')
    print(f'Hugging Face cache directory: {cache_dir}')
    print('Hugging Face download resume: enabled (max_retries=10)')
    if dataset_cache_exists(HF_DATASET_NAME):
        print(f'Found local dataset cache for {HF_DATASET_NAME}; reusing it when possible.')
    else:
        print(f'Local dataset cache for {HF_DATASET_NAME} not found; Hugging Face will download/cache it.')
    dataset = load_dataset(HF_DATASET_NAME, split=split, token=token, cache_dir=cache_dir, download_mode='reuse_dataset_if_exists', download_config=download_config)
    dataset = dataset.cast_column('image', Image(decode=False))
    print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c != 'image'])
    selected_indices = select_evaluation_indices(len(metadata_ds))
    print('Evaluation state updated.', flush=True)
    items = []
    for i in progress_iter(selected_indices, desc='Building metadata index'):
        row = metadata_ds[i]
        label = -1
        if 'objects' in row:
            label = total_count_from_objects(row['objects'], fallback_label=-1)
        if label == -1:
            label = row.get('label', -1)
        items.append({'index': i, 'label': label, 'objects': row.get('objects')})
    print('Evaluation state updated.', flush=True)
    return (dataset, items)

def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, sample_parallel: int=4, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, reasoning_method: str='baseline'):
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': reasoning_method})
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None)
    if result_path is None:
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"-{prompt_metadata['few_shot_k']}shot" if prompt_metadata['prompt_mode'] == 'few-shot' else ''
        result_path = os.path.join(output_dir, f'{model_key}-counting{suffix}.jsonl')
    else:
        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
    result_path = result_path_for_reasoning(result_path, reasoning_method)
    print(f'[{model_key}] Starting, results -> {result_path}')
    completed_indices = completed_dataset_indices(result_path, dataset_items)
    validate_resume_config(result_path, prompt_metadata)
    validate_resume_samples(result_path, dataset_items, model_key)
    pending_items = [item for item in dataset_items if normalize_result_index(item.get('index')) not in completed_indices]
    sample_parallel = max(1, int(sample_parallel or 1))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_parallel = 1
    print('Evaluation state updated.', flush=True)
    if not pending_items:
        print(f'[{model_key}] Result is already complete for the requested samples; no model/API call is needed.')
        return result_path
    query_fn = get_query_function(model_key, api_key=api_key)

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[i])
            image_data_uri = raw_bytes_to_base64_uri(raw_bytes)
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            task_prompt = USER_PROMPT
            query_prompt = effective_prompt(task_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

            def call_model(user_prompt: str):
                return query_fn(image_url=image_data_uri, system=query_system, user=user_prompt, few_shot_examples=sample_few_shot_examples_list)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='rareplanes' if LORA_TASK == 'rare-planes' else 'animals', context={}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer, _ = parse_count_response(answer_text)
            result_record = {'index': i, 'label': item['label'], 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'objects': item.get('objects'), **result_model_metadata(model_key, MODEL_CONFIGS), 'response': response, 'prompt': task_prompt, 'parsed_answer': parsed_answer, **prompt_metadata}
            for provenance_key in ('source_id', 'official_split', 'file_name', 'source_scene_id', 'dataset_revision'):
                if provenance_key in item:
                    result_record[provenance_key] = item[provenance_key]
            if reasoning_method == 'cot':
                result_record['answer_text'] = answer_text
                result_record.update(cot_row_metadata(base_prompt=task_prompt, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            return (result_record, response_content)
        except Exception as e:
            print(f'[{model_key}] Error processing sample {i}: {e}')
            import traceback
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS) or is_fatal_provider_error(str(e)):
                raise
            return None
    if sample_parallel == 1:
        for item in progress_iter(pending_items, desc=f'{model_key} samples', unit='sample'):
            result = _process_single_sample(item)
            if result is None:
                continue
            result_record, response_content = result
            append_to_jsonl(result_record, result_path)
            completed_indices.add(normalize_result_index(result_record['index']))
            print(f"[{model_key}] #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['label']}")
    else:
        executor = ThreadPoolExecutor(max_workers=sample_parallel)
        futures = {executor.submit(_process_single_sample, item) for item in pending_items}
        try:
            for future in progress_iter(as_completed(futures), desc=f'{model_key} samples', unit='sample'):
                result = future.result()
                if result is None:
                    continue
                result_record, response_content = result
                append_to_jsonl(result_record, result_path)
                completed_indices.add(normalize_result_index(result_record['index']))
                print(f"[{model_key}] #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['label']}")
        except BaseException:
            for future in futures:
                cancel = getattr(future, 'cancel', None)
                if callable(cancel):
                    cancel()
            shutdown = getattr(executor, 'shutdown', None)
            if callable(shutdown):
                try:
                    shutdown(wait=True, cancel_futures=True)
                except TypeError:
                    shutdown(wait=True)
            raise
        else:
            shutdown = getattr(executor, 'shutdown', None)
            if callable(shutdown):
                shutdown(wait=True)
    missing_indices = [item['index'] for item in dataset_items if normalize_result_index(item.get('index')) not in completed_indices]
    if missing_indices:
        preview = ', '.join((str(index) for index in missing_indices[:10]))
        raise RuntimeError(f'[{model_key}] Run finished with {len(missing_indices)} samples still pending (first indices: {preview}). Completed rows were checkpointed to {result_path}; rerun the same command to resume only the missing samples.')
    print('Evaluation state updated.', flush=True)
    return result_path

def format_metric(value: Optional[float]) -> str:
    return 'N/A' if value is None else f'{value:.3f}'

def evaluation(result_path: str, ax=None) -> Dict:
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {result_path}')
    print(f"{'=' * 60}")
    result_json = load_jsonl_records(result_path)
    if not result_json:
        print(f'No valid JSONL rows found in {result_path}')
        return None
    for row in result_json:
        row['response'] = row['answer_text'] if isinstance(row.get('answer_text'), str) else extract_response_content(row.get('response'))
        row['model_response'] = extract_json_candidate(row['response'])
        parsed_total, parsed_counts = parse_count_response(row['response'])
        row['parsed_response'] = parsed_total
        row['parsed_counts'] = parsed_counts
        row['count'] = total_count_from_objects(row.get('objects'), fallback_label=row.get('label', -1))
    result_json_no_refusal = [row for row in result_json if row['parsed_response'] != -1]
    rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_json, result_json_no_refusal)
    scored_result_json = [{**row, 'parsed_response': 0 if row['parsed_response'] == -1 else row['parsed_response']} for row in result_json]
    print(f'\nResults for {os.path.basename(result_path)}:')
    print('Evaluation state updated.', flush=True)
    print('Evaluation state updated.', flush=True)
    print(f'  Refusal Rate: {rr:.2%}')
    print(f'\nMetrics (all samples):')
    print(f'  MAE: {mae:.3f}')
    print(f'  MAPE: {mape:.3f}')
    print(f'  R²: {r2:.3f}')
    print(f'\nMetrics (excluding refusals):')
    print(f'  MAE: {format_metric(mae_no_refusal)}')
    print(f'  MAPE: {format_metric(mape_no_refusal)}')
    print(f'  R²: {format_metric(r2_no_refusal)}')
    print(f'\nLaTeX format:')
    print('MAE & MAPE & R² & Refusal Rate')
    print(f'{mae:.3f} & {mape:.3f} & {r2:.3f} & {rr:.2f}')
    model_name = model_key_from_result_filename(result_path)
    try:
        scatter_ax = plot_scatter(scored_result_json, ax=ax)
        if ax:
            ax.set_title(model_name)
        elif scatter_ax is not None:
            import matplotlib.pyplot as plt
            plt.close(scatter_ax.figure)
    except Exception as e:
        print(f'Warning: Could not generate scatter plot: {e}')
    return {'model': model_name, 'mae_all': mae, 'mape_all': mape, 'r2_all': r2, 'mae': mae, 'mape': mape, 'r2': r2, 'mae_no_refusal': mae_no_refusal, 'mape_no_refusal': mape_no_refusal, 'r2_no_refusal': r2_no_refusal, 'refusal_rate': rr}

def discover_result_files(result_dir: str, methods: Tuple[str, ...]) -> List[str]:
    return sorted((filename for filename in os.listdir(result_dir) if classify_result_filename(filename) in methods and os.path.isfile(os.path.join(result_dir, filename))))

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR, evaluate_all_methods: bool=False):
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    methods = ('zero-shot', 'few-shot', 'cot') if evaluate_all_methods else ('zero-shot',)
    result_files = discover_result_files(result_dir, methods)
    if not result_files:
        print(f'No result files found in {result_dir}')
        return
    print(f'Found {len(result_files)} result files')
    all_metrics = []
    for result_file in progress_iter(result_files, desc='Evaluating models'):
        result_path = os.path.join(result_dir, result_file)
        result_method = classify_result_filename(result_file)
        try:
            metrics = evaluate_and_write_cot_metrics(evaluation, result_path) if result_method == 'cot' else evaluation(result_path, ax=None)
            if metrics is not None:
                model_key = model_key_from_result_filename(result_file)
                if model_key in MODEL_CONFIGS:
                    metrics['release_date'] = MODEL_CONFIGS[model_key]['release_date']
                    metrics['family'] = MODEL_CONFIGS[model_key]['family']
                    metrics['model_key'] = model_key
                    metrics['result_file'] = result_file
                    metrics.update(shot_from_metrics(metrics, result_file))
                    metrics['reasoning_method'] = 'cot' if result_method == 'cot' else 'baseline'
                    all_metrics.append(metrics)
                else:
                    print(f'Warning: Model {model_key} not found in MODEL_CONFIGS')
        except Exception as e:
            print(f'Error evaluating {result_file}: {e}')
            continue
    if not all_metrics:
        print('No valid metrics collected, cannot generate plots')
        return
    all_metrics = sorted(all_metrics, key=lambda item: datetime.fromisoformat(item['release_date']) if item.get('release_date') else datetime.min)
    if evaluate_all_methods:
        write_shot_summary_json(all_metrics, os.path.join(result_dir, 'model_metrics_summary_all_shots.json'), dataset_name=DATASET_DISPLAY_NAME)
    zero_shot_metrics = [metrics for metrics in all_metrics if classify_result_filename(metrics.get('result_file', '')) == 'zero-shot']
    if not zero_shot_metrics:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return all_metrics
    evaluated_metrics = all_metrics
    all_metrics = zero_shot_metrics
    print(f'\nCollected metrics for {len(all_metrics)} models')
    for row in all_metrics:
        print(f"{row['model_key']:<35} {row['release_date']} MAE={row['mae']:.3f} R2={row['r2']:.3f} RR={row['refusal_rate']:.3f}")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print('Warning: matplotlib is not installed; skipping comparison plots.')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = FAMILY_COLORS

        def format_date_axis(ax):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        def plot_metric(ax, metric_key: str, ylabel: str, title: str):
            for family in PLOT_FAMILY_ORDER:
                rows = [row for row in all_metrics if row['family'] == family and row.get('release_date')]
                if not rows:
                    continue
                dates = [datetime.fromisoformat(row['release_date']) for row in rows]
                values = [row[metric_key] for row in rows]
                ax.plot(dates, values, marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
                for date_value, metric_value, row in zip(dates, values, rows):
                    label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                    ax.text(date_value, metric_value, label, fontsize=10, color=colors[family])
            ax.set_xlabel('Release Date')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            format_date_axis(ax)
        plot_metric(axes[0, 0], 'mae', 'MAE (Mean Absolute Error)', 'MAE over Time')
        plot_metric(axes[0, 1], 'r2', 'R2 Score', 'R2 Score over Time')
        plot_metric(axes[1, 0], 'mape', 'MAPE (Mean Absolute Percentage Error)', 'MAPE over Time')
        plot_metric(axes[1, 1], 'refusal_rate', 'Refusal Rate', 'Refusal Rate over Time')
        fig.suptitle(PLOT_TITLE, fontsize=16, fontweight='bold')
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
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = sorted({key for row in all_metrics for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f'Metrics summary saved to: {csv_path}\n')
    return evaluated_metrics if evaluate_all_methods else all_metrics

def main():
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('aerial-animal'):
        return
    import argparse
    global DEFAULT_RANDOM_SEED
    parser = argparse.ArgumentParser(description='Evaluate vision models on Aerial Animal Population dataset', epilog='\nAvailable models:\nIndividual: gpt-5.2-2025-12-11, gemini-3-pro-preview, Qwen2.5-VL-7B-Instruct, etc.\nFamilies:   GPT-all, Gemini-all, Qwen-all\nDefault:    Test all GPT/Gemini models in parallel\n\nExamples:\n# Test all GPT/Gemini models in parallel\npython aerial_animal.py\n\n# Test GPT models in parallel (auto: 9 threads)\npython aerial_animal.py --model GPT-all\n\n# Manually specify 8 parallel threads\npython aerial_animal.py --parallel 8\n\n# Force sequential execution\npython aerial_animal.py --parallel 1\n\n# Test single model\npython aerial_animal.py --model gpt-5.2-2025-12-11\n\n# List models and API keys\npython aerial_animal.py --list-models\n\n# Evaluate all existing results and generate plots\npython aerial_animal.py --evaluate-all\n\n# Generate plots only\npython aerial_animal.py --plot-only\n        ', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None, help='Model to test (model name, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file path (only allowed with a single model)')
    parser.add_argument('--dataset-seed', type=int, default=DEFAULT_RANDOM_SEED, help=f'Random seed for deterministic dataset sample selection (default: {DEFAULT_RANDOM_SEED})')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--evaluate-all', action='store_true', help='Evaluate all existing results in the output directory and generate comparison plots')
    parser.add_argument('--plot-only', action='store_true', help='Only generate comparison plots from existing results')
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
    if args.few_shot_k < 0:
        parser.error('--few-shot-k must be non-negative')
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k == 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    configure_cache_environment_from_args(args.cache_dir)
    DEFAULT_RANDOM_SEED = int(args.dataset_seed)
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
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir)
        return
    if args.evaluate_all:
        plot_model_comparison(args.output_dir, evaluate_all_methods=True)
        return
    if args.evaluate_only:
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        api_request_timeout_seconds()
        print_provider_summary()
    if args.output and len(models_to_test) != 1:
        raise ValueError('--output can only be used when testing a single model.')
    canonical_pools = resolve_canonical_pools(few_shot_k)
    global _shared_dataset_items, _shared_hf_dataset, _shared_few_shot_source, _shared_few_shot_seed, _shared_split_context
    _shared_hf_dataset = canonical_pools['evaluation_dataset']
    _shared_dataset_items = canonical_pools['evaluation_items']
    _shared_few_shot_source = canonical_pools['few_shot_source']
    _shared_few_shot_seed = args.few_shot_seed if _shared_few_shot_source else None
    _shared_split_context = canonical_pools['split_context']
    if _shared_few_shot_source:
        print('Evaluation state updated.', flush=True)
    print(f"\n{'=' * 60}")
    print(f'Models to test: {len(models_to_test)}')
    print(f"Prompt mode: {('few-shot' if _shared_few_shot_source else 'zero-shot')}")
    if _shared_few_shot_source:
        print(f'Few-shot seed: {_shared_few_shot_seed}')
    print_runtime_model_summary(models_to_test, MODEL_CONFIGS)
    print(f"{'=' * 60}\n")
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
        output_path = args.output if len(models_to_test) == 1 else None
        tasks.append((model_key, api_key, args.output_dir, args.sample_parallel, output_path, args.reasoning_method))
    failed_models = {}
    if num_parallel > 1 and len(models_to_test) > 1:
        print(f"\n{'=' * 60}")
        print(f'Starting {num_parallel} parallel threads...')
        print(f"{'=' * 60}\n")
        result_paths = []
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(run_single_model, task): task[0] for task in tasks}
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    result_path = future.result()
                    if result_path is not None:
                        result_paths.append(result_path)
                        print(f'Completed: {model_key}')
                except Exception as e:
                    print(f'Failed: {model_key}: {e}')
                    failed_models[model_key] = str(e)
    else:
        print(f"\n{'=' * 60}")
        print(f'Running sequentially (use --parallel N for parallel execution)...')
        print(f"{'=' * 60}\n")
        result_paths = []
        for task in progress_iter(tasks, desc='Testing models'):
            result_path = run_single_model(task)
            if result_path is not None:
                result_paths.append(result_path)
    print(f"\n{'=' * 60}")
    print(f'Evaluating {len(result_paths)} result files')
    print(f"{'=' * 60}\n")
    for result_path in progress_iter(result_paths, desc='Final evaluation'):
        try:
            evaluate_and_write_cot_metrics(evaluation, result_path)
        except Exception as e:
            print(f'Error evaluating {result_path}: {e}')
            continue
    if len(result_paths) > 1:
        print(f'\nGenerating comparison plots...')
        plot_model_comparison(args.output_dir)
    if failed_models:
        summary = '; '.join((f'{model}: {message}' for model, message in failed_models.items()))
        raise RuntimeError(f'One or more model evaluations did not complete: {summary}')
    print(f"\n{'=' * 60}")
    print(f'All tasks completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_few_shot_source = None
_shared_few_shot_seed = None
_shared_split_context = None

def run_single_model(args_tuple):
    model_key, api_key, output_dir, sample_parallel, output_path, reasoning_method = args_tuple
    try:
        result_path = query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, split_context=_shared_split_context, result_path=output_path, output_dir=output_dir, api_key=api_key, sample_parallel=sample_parallel, few_shot_source=_shared_few_shot_source, few_shot_seed=_shared_few_shot_seed, reasoning_method=reasoning_method)
        return result_path
    except Exception as e:
        print(f'\n[{model_key}] Error: {e}')
        import traceback
        traceback.print_exc()
        raise
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
