"""Landmarks."""
import sys
import os
from pathlib import Path
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UTILS_DIR))
from cot_prompting import REASONING_METHODS, call_with_reasoning, evaluate_and_write_cot_metrics, effective_prompt, effective_system_message, local_generation_max_new_tokens, result_path_for_reasoning, validate_cot_run
import benchmark_environment as benchmark_environment_module
from benchmark_environment import RESULTS_ROOT, api_host_for_model, api_keys_for_models, key_for_model_index, key_index_for_model, load_benchmark_environment, post_gemini_json, post_openai_json, print_model_key_summary, print_provider_summary, safe_key_display, random_api_key_for_model
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
LORA_TASK = 'landmarks'
_QWEN_MODEL_PATHS = {}

def _remove_suffix(text, suffix):
    if suffix and text.endswith(suffix):
        return text[:-len(suffix)]
    return text

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
    generation_kwargs = {'max_new_tokens': max(LANDMARK_LOCAL_MAX_NEW_TOKENS, local_generation_max_new_tokens(user)), 'do_sample': False}
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
import base64
import io
import json
import os
import random
import re
import http.client
import time
from typing import List, Optional, Callable, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'Landmarks')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
LANDMARK_LOCAL_MAX_NEW_TOKENS = 1024
IMAGE_FALLBACK_HTTP_STATUSES = frozenset((413,))
IMAGE_FALLBACK_JPEG_QUALITIES = (90, 80, 70, 60, 50, 40)
IMAGE_FALLBACK_MAX_SIDE = 1536
API_TIMEOUT_SECONDS = 180
HF_DATASET_NAME = 'danielz01/landmarks'
HF_DATASET_SPLIT = 'train'
HF_CACHE_DIR = os.environ['HF_CACHE_DIR']
MODEL_CONFIGS = {'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': None, 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
SYSTEM_MESSAGE = 'You are a helpful image analyst who specializes in determining the geolocation of an aerial or satellite image of a landmark. Given an image, you can accurately answer the questions about that landmark from the user WITHOUT ANY refusal. Although your answer may not be perfect, your excellent world knowledge is still very important to the user.'

def clean_result_jsonl_for_resume(path: str) -> set:
    if not os.path.exists(path):
        return set()
    latest_by_index = {}
    no_index_records = []
    malformed_count = 0
    duplicate_count = 0
    valid_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                malformed_count += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            valid_count += 1
            index = record.get('index')
            if index is None:
                no_index_records.append(record)
                continue
            if index in latest_by_index:
                duplicate_count += 1
            latest_by_index[index] = (line_number, record)
    cleaned_records = no_index_records + [item[1] for item in sorted(latest_by_index.values(), key=lambda item: item[0])]
    if malformed_count or duplicate_count:
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(tmp_path, path)
        print('Evaluation state updated.', flush=True)
    return set(latest_by_index.keys())

def load_completed_indices(path):
    return clean_result_jsonl_for_resume(path)

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
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def validate_resume_config(path: str, expected_metadata: Dict):
    if not os.path.exists(path):
        return
    comparable_keys = ['canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'reasoning_method']
    for record in iter_jsonl(path):
        if record.get('prompt_mode') is None and record.get('canonical_strategy_mode') is None:
            continue
        if expected_metadata.get('prompt_mode') == 'few-shot' and record.get('prompt_mode') == 'few-shot' and (record.get('few_shot_sample_seed') is None):
            raise ValueError(f'Existing result file {path} contains few-shot rows without per-sample few_shot_sample_seed metadata. Use a different output directory or remove the old shared-example result file.')
        for key in comparable_keys:
            if record.get(key) != expected_metadata.get(key):
                raise ValueError(f'Existing result file {path} has {key}={record.get(key)!r}, but this run expects {expected_metadata.get(key)!r}. Use a different output directory or remove the old result file.')

def dataset_cache_exists(dataset_name: str) -> bool:
    cache_slug = 'datasets--' + dataset_name.replace('/', '--')
    return (Path(HF_CACHE_DIR) / 'hub' / cache_slug).exists()

def cached_snapshot_dir(dataset_name: str) -> Optional[Path]:
    cache_slug = 'datasets--' + dataset_name.replace('/', '--')
    repo_cache = Path(HF_CACHE_DIR) / 'hub' / cache_slug
    snapshots_dir = repo_cache / 'snapshots'
    if not snapshots_dir.exists():
        return None
    ref_path = repo_cache / 'refs' / 'main'
    if ref_path.exists():
        revision = ref_path.read_text(encoding='utf-8').strip()
        snapshot = snapshots_dir / revision
        if snapshot.exists():
            return snapshot
    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda p: p.stat().st_mtime)

def cached_parquet_files(dataset_name: str, split: Optional[str]=None) -> List[Path]:
    snapshot = cached_snapshot_dir(dataset_name)
    if snapshot is None:
        return []
    pattern = f'{split}-*.parquet' if split else '*.parquet'
    return sorted((p for p in snapshot.rglob(pattern) if p.is_file()))

def cached_arrow_files(dataset_name: str, split: Optional[str]=None) -> List[Path]:
    dataset_slug = dataset_name.replace('/', '___')
    dataset_root = Path(HF_CACHE_DIR) / dataset_slug
    if not dataset_root.exists():
        return []
    pattern = f'*{split}-*.arrow' if split else '*.arrow'
    return sorted((p for p in dataset_root.rglob(pattern) if p.is_file()))

def expected_parquet_shard_count(parquet_files: List[Path]) -> Optional[int]:
    counts = []
    for parquet_path in parquet_files:
        match = re.search('-of-(\\d+)-', parquet_path.name)
        if match:
            counts.append(int(match.group(1)))
    return max(counts) if counts else None

def load_cached_parquet_dataset(dataset_name: str, split: str, image_columns: List[str]):
    arrow_files = cached_arrow_files(dataset_name, split)
    if arrow_files:
        from datasets import Dataset, Image, concatenate_datasets
        expected_shards = expected_parquet_shard_count(arrow_files)
        split_complete = expected_shards is None or len(arrow_files) >= expected_shards
        if split_complete:
            print(f'Using {len(arrow_files)} local Arrow shard(s) for {dataset_name} (split={split}); skipping Hugging Face builder/download.')
            datasets = [Dataset.from_file(str(path)) for path in arrow_files]
            dataset = concatenate_datasets(datasets)
            for col in image_columns:
                if col in dataset.column_names:
                    dataset = dataset.cast_column(col, Image(decode=False))
                    break
            return dataset
        else:
            print(f'Local Arrow split is incomplete ({len(arrow_files)}/{expected_shards} shard(s)); checking the local parquet snapshot.')
    parquet_files = cached_parquet_files(dataset_name, split)
    if not parquet_files:
        return None
    from datasets import Image, load_dataset
    print(f'Using {len(parquet_files)} local parquet shard(s) for {dataset_name} (split={split}); skipping Hugging Face builder/download.')
    expected_shards = expected_parquet_shard_count(parquet_files)
    split_complete = expected_shards is None or len(parquet_files) >= expected_shards
    if not split_complete:
        print(f'Local parquet split is incomplete ({len(parquet_files)}/{expected_shards} shard(s)); falling back to Hugging Face download.')
        return None
    dataset = load_dataset('parquet', data_files=[str(path) for path in parquet_files], split='train', cache_dir=HF_CACHE_DIR, download_mode='reuse_dataset_if_exists')
    for col in image_columns:
        if col in dataset.column_names:
            dataset = dataset.cast_column(col, Image(decode=False))
            break
    return dataset

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

def download_full_dataset_cache(dataset_name: str, split: Optional[str]=None, max_workers: int=2) -> str:
    from huggingface_hub import snapshot_download
    token = os.environ.get('HF_TOKEN')
    allow_patterns = ['*.parquet']
    if split:
        allow_patterns = [f'**/{split}-*.parquet']
    hub_cache_dir = Path(HF_CACHE_DIR) / 'hub'
    print(f'\nDownloading full Hugging Face dataset cache: {dataset_name}')
    print(f'Hugging Face hub cache directory: {hub_cache_dir}')
    print(f'Allow patterns: {allow_patterns}')
    snapshot_path = snapshot_download(repo_id=dataset_name, repo_type='dataset', cache_dir=hub_cache_dir, token=token, allow_patterns=allow_patterns, max_workers=max_workers, endpoint='https://huggingface.co')
    print(f'Snapshot cache ready: {snapshot_path}')
    cached = cached_parquet_files(dataset_name, split) if split else sorted((p for p in Path(snapshot_path).rglob('*.parquet') if p.is_file()))
    print(f'Cached parquet files: {len(cached)}')
    return snapshot_path

def progress_iter(iterable, desc: Optional[str]=None, total: Optional[int]=None, unit: str='sample'):
    return iterable

def is_non_retryable_api_error(exc: Exception) -> bool:
    normalized = str(exc).casefold()
    return any((marker in normalized for marker in ('insufficient_quota', 'model_not_found', 'model not found')))

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
                    if is_image_fallback_error(exc):
                        raise
                    if is_non_retryable_api_error(exc):
                        raise
                    last_error = exc
                    if attempt >= max_retries - 1:
                        break
                    delay = min(max_delay, min_delay * 2 ** attempt)
                    time.sleep(rng.uniform(min_delay, delay))
            raise last_error
        return wrapper
    return decorator

class APIStatusError(IOError):

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code

def image_fallback_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, 'status_code', None)
    if status_code in IMAGE_FALLBACK_HTTP_STATUSES:
        return status_code
    match = re.search('\\b(413)\\b', str(exc))
    if match:
        parsed_status = int(match.group(1))
        if parsed_status in IMAGE_FALLBACK_HTTP_STATUSES:
            return parsed_status
    return None

def is_image_fallback_error(exc: Exception) -> bool:
    return image_fallback_status_code(exc) is not None

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

def pil_to_base64_uri(pil_image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'

def pil_to_rgb_for_jpeg(pil_image):
    from PIL import Image as PILImage
    if pil_image.mode in ('RGBA', 'LA'):
        background = pil_image.convert('RGBA')
        white = PILImage.new('RGBA', background.size, (255, 255, 255, 255))
        white.alpha_composite(background)
        return white.convert('RGB')
    if pil_image.mode == 'P' and 'transparency' in pil_image.info:
        background = pil_image.convert('RGBA')
        white = PILImage.new('RGBA', background.size, (255, 255, 255, 255))
        white.alpha_composite(background)
        return white.convert('RGB')
    if pil_image.mode != 'RGB':
        return pil_image.convert('RGB')
    return pil_image.copy()

def pil_to_jpeg_base64_uri(pil_image, quality: int) -> str:
    image = pil_to_rgb_for_jpeg(pil_image)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'

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
    raise KeyError("No 'image' field with raw bytes found in data item")

def image_data_uri_from_raw_bytes(raw_bytes: bytes, model_key: str=None, jpeg_quality: int=None) -> str:
    from PIL import Image as PILImage
    with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
        pil_img.load()
        if jpeg_quality is not None:
            return pil_to_jpeg_base64_uri(pil_img, jpeg_quality)
        return pil_to_base64_uri(pil_img)

def image_data_uri_from_dataset(hf_dataset, index: int, model_key: str=None, jpeg_quality: int=None) -> str:
    raw_bytes = get_raw_image_bytes(hf_dataset[index])
    return image_data_uri_from_raw_bytes(raw_bytes, model_key=model_key, jpeg_quality=jpeg_quality)


def query_with_image_fallback(query_fn: Callable, raw_bytes: bytes, model_key: str, sample_index: int, system: str, user: str, few_shot_examples: Optional[List[Dict]]=None) -> dict:
    return query_fn(image_url=image_data_uri_from_raw_bytes(raw_bytes, model_key=model_key),
                    system=system, user=user, few_shot_examples=few_shot_examples,
                    sample_index=sample_index)

def dataset_image_is_readable(hf_dataset, index: int) -> bool:
    from PIL import Image as PILImage
    try:
        raw_bytes = get_raw_image_bytes(hf_dataset[index])
        with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
            pil_img.verify()
        return True
    except Exception:
        return False

def image_part_for_gemini(image_url: str) -> Dict:
    if image_url.startswith('data:'):
        header, b64_data = image_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        return {'inlineData': {'mimeType': mime_type, 'data': b64_data}}
    return {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_openai(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, sample_index: Optional[int]=None) -> dict:
    api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]
    messages = [{'role': 'system', 'content': system}]
    for example in few_shot_examples or []:
        messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': example['image_url']}}, {'type': 'text', 'text': example['user_prompt']}]})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': user}]})
    payload = json.dumps({'model': config['model_name'], 'messages': messages})
    status, data = post_openai_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    if status != 200:
        error_msg = data.decode('utf-8')
        sample_prefix = f'[{model_key}] #{sample_index} ' if sample_index is not None else f'{model_key} '
        print(f'{sample_prefix}API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    return json.loads(data.decode('utf-8'))

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_gemini(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, sample_index: Optional[int]=None) -> dict:
    api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]
    contents = []
    for example in few_shot_examples or []:
        contents.append({'role': 'user', 'parts': [image_part_for_gemini(example['image_url']), {'text': example['user_prompt']}]})
        contents.append({'role': 'model', 'parts': [{'text': example['answer']}]})
    contents.append({'role': 'user', 'parts': [image_part_for_gemini(image_url), {'text': user}]})
    payload = json.dumps({'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents})
    status, data = post_gemini_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=API_TIMEOUT_SECONDS)
    if status != 200:
        error_msg = data.decode('utf-8')
        sample_prefix = f'[{model_key}] #{sample_index} ' if sample_index is not None else f'{model_key} '
        print(f'{sample_prefix}API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
    content = ''
    for part in parts:
        if part.get('thought', False):
            continue
        content = part.get('text', '')
        break
    if not content and parts:
        content = parts[-1].get('text', '')
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
        return lambda image_url, system, user, few_shot_examples=None, sample_index=None: query_openai(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
    elif config['api_type'] == 'google_native':
        return lambda image_url, system, user, few_shot_examples=None, sample_index=None: query_gemini(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")

def get_models_to_test(model_arg: str) -> List[str]:
    return expand_model_selection(model_arg, MODEL_CONFIGS)
landmark_categories = {'Natural Parks and Reserves': ['Alabama state park', 'Arizona state park', 'California state beach', 'California state park', 'Delaware state park', 'Georgia state park', 'Idaho state park', 'Illinois state park', 'Maryland state park', 'Massachusetts state park', 'Mississippi state park', 'National Conservation Area', 'National Park System unit', 'National Recreation Area', 'National Reserve', 'National Wildlife Refuge', 'Nevada state park', 'New Hampshire state park', 'New Jersey State Forest', 'New Jersey state park', 'North Carolina state park', 'Oregon state park', 'Pennsylvania state park', 'South Carolina state park', 'Tennessee state park', 'United States National Forest', 'Utah state park', 'Vermont state park', 'Virginia state park', 'Washington state park', 'West Virginia state park', 'US Wilderness Area', 'bay', 'beach', 'botanical garden', 'forest', 'glacial lake', 'glacier', 'island', 'lagoon', 'lake', 'landform', 'mountain park', 'mountain range', 'nature area', 'nature center', 'nature reserve', 'park', 'protected area', 'recreation area', 'reservoir', 'river', 'valley', 'wildlife refuge', 'provincial park of Canada', 'national park of Canada', 'dark-sky preserve', 'Connecticut state park', 'United States National Marine Sanctuary', 'waterfall', 'state forest of Wisconsin'], 'Places of Worship': ['African-American museum', 'Baptists', 'Catholic cathedral', 'Eastern Orthodox cathedral', 'Methodist church building', 'Southern Baptist Convention Church', 'Spanish missions in California', 'church building', 'mission church', 'mission station', 'protestant church building', 'Southern Baptist Convention Church', 'chapel', 'synagogue', 'Anglican or episcopal cathedral'], 'Sports and Entertainment Venues': ['American football stadium', 'arena', 'art museum', 'arts centre', 'assembly plant', 'astronomical observatory', 'aviation museum', 'baseball venue', 'circus museum', "children's museum", 'entertainment district', 'fair ground', 'history museum', 'house museum', 'movie theater', 'museum', 'museum building', 'music venue', 'performing arts center', 'public aquarium', 'public garden', 'radio quiet zone', 'railway museum', 'science center', 'science museum', 'sculpture garden', 'shopping center', 'shopping mall', 'show cave', 'skyscraper', 'sports venue', 'stadium', 'theatre', 'tourist attraction', 'velodrome', 'venue', 'zoo', 'amusement park', 'transport museum'], 'Historical and Cultural Sites': ['archaeological site', 'battlefield', 'column', 'cultural institution', 'estate', 'fixed construction', 'fort', 'fountain', 'geographic region', 'group of sculptures', 'heritage site', 'historic district', 'historic site', 'lighthouse', 'mansion', 'military cemetery', 'military museum', 'missile launch facility', 'monument', 'national monument', 'neighborhood', 'neighborhood of Pittsburgh', 'neighborhood of Washington, D.C.', 'statistical territorial entity', 'strait', 'streetcar suburb', 'war memorial', 'university art museum', 'National Historic Site', 'National Historical Park', 'List of California State Historic Parks', 'National Memorial of the United States', 'National Monument of the United States', 'National Battlefield', 'Spanish missions in California', 'Jewish cemetery', 'historic building', 'national military park'], 'Government and Public Buildings': ['academic library', 'airport', 'archive building', 'capitol building', 'casino', 'cemetery', 'chancery', 'courthouse', 'dam', 'embankment dam', 'embassy', 'government building', 'hotel', 'house', 'library', 'library building', 'lodge', 'office building', 'organization', 'parterre', 'pier', 'plantation', 'public housing', 'rathaus', 'reclaimed land', 'reservoir', 'ruins', 'town hall', 'official residence'], 'Infrastructure and Urban Features': ['apartment building', 'architectural structure', 'building', 'channel', 'convention center', 'cove', 'disaster remains', 'downtown', 'embassy', 'entertainment district', 'estate', 'fair ground', 'fixed construction', 'forest', 'fountain', 'geographic region', 'glacial lake', 'glacier', 'group of sculptures', 'heritage site', 'hiking trail', 'historic district', 'historic site', 'history museum', 'hotel', 'house', 'house museum', 'housing estate', 'hydroelectric power station', 'island', 'lagoon', 'lake', 'landform', 'library', 'library building', 'lighthouse', 'lodge', 'mansion', 'military cemetery', 'military museum', 'missile launch facility', 'mission church', 'mission station', 'monument', 'mountain park', 'mountain range', 'movie theater', 'museum', 'museum building', 'music venue', 'national monument', 'nature area', 'nature center', 'nature reserve', 'neighborhood', 'neighborhood of Pittsburgh', 'neighborhood of Washington, D.C.', 'nonprofit corporation', 'nonprofit organization', 'office building', 'organization', 'park', 'parterre', 'performing arts center', 'pier', 'plantation', 'protected area', 'protestant church building', 'public aquarium', 'public garden', 'public housing', 'radio quiet zone', 'railway museum', 'rathaus', 'reclaimed land', 'recreation area', 'reservoir', 'river', 'ruins', 'rural cemetery', 'science center', 'science museum', 'sculpture', 'sculpture garden', 'shopping center', 'shopping mall', 'show cave', 'single-family detached home', 'skyscraper', 'sports venue', 'stadium', 'state forest', 'state historic site in the United States', 'state park', 'state recreation area', 'statistical territorial entity', 'strait', 'streetcar suburb', 'theatre', 'thermal solar power station', 'tied island', 'tourist attraction', 'town hall', 'urban park', 'valley', 'velodrome', 'venue', 'war memorial', 'watercourse', 'wildlife refuge', 'thermal solar power station', 'television tower', 'public library', 'community center', ''], 'Miscellaneous': ['Indian reservation of the United States', 'Passport to Your National Parks cancellation location', 'census-designated place in the United States', 'tied island']}

def find_category(instance_types):
    for instance_type in instance_types:
        for category, instances in landmark_categories.items():
            if instance_type.lower() in list(map(str.lower, instances)):
                return category
            elif 'park' in instance_type.lower():
                return 'Natural Parks and Reserves'
    return 'Miscellaneous'

def get_user_prompt(options: List[str]):
    prompt = "Make an educated guess about the name of the landmark shown in the image. Choose one option below. The option order is randomized independently for each image. Return exactly one line in the form '<number>. <option>'. Do not include analysis or explanation.\n"
    prompt += '\n'.join([f'{i + 1}. {option}' for i, option in enumerate(options)])
    return prompt


def parse_landmark_answer(answer_text: str, options: List[str]) -> Optional[str]:
    from evaluation.utils.scoring import landmark_answer
    return landmark_answer(answer_text, options)


def build_landmark_options(item: Dict, num_distractors: int=4, seed: int=DEFAULT_RANDOM_SEED) -> Optional[List[str]]:
    name = str(item.get('name') or '').strip()
    if not name:
        return None
    distractors = set(item.get('distractorLabels') or [])
    distractors.discard(name)
    distractors = sorted((label for label in distractors if label and (not str(label).startswith('Q'))))
    if not distractors:
        return None
    identity = '{}|{}|{}|{}'.format(seed, item.get('raw_split', ''), item.get('index', ''), name)
    rng = random.Random(identity)
    selected_distractors = rng.sample(distractors, k=min(len(distractors), num_distractors))
    target_position = rng.randrange(len(selected_distractors) + 1)
    selected_distractors.insert(target_position, name)
    return selected_distractors

def format_few_shot_answer(name: str, options: List[str]) -> str:
    canonical_name = str(name).strip()
    try:
        answer_index = options.index(canonical_name)
    except ValueError as exc:
        raise ValueError('Few-shot landmark answer is missing from its options') from exc
    return f'{answer_index + 1}. {canonical_name}'

def few_shot_metadata(few_shot_source: Optional[Dict], split_context: Dict, few_shot_seed: Optional[int]=None) -> Dict:
    source = few_shot_source or {}
    k = int(source.get('k', 0) or 0)
    return {'canonical_strategy_mode': split_context['canonical_strategy_mode'], 'canonical_split_seed': split_context['canonical_split_seed'], 'eval_raw_split': split_context['eval_raw_split'], 'few_shot_source_raw_split': split_context['few_shot_source_raw_split'], 'eval_canonical_split': split_context['eval_canonical_split'], 'few_shot_source_canonical_split': split_context['few_shot_source_canonical_split'], 'prompt_mode': 'few-shot' if k > 0 else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k > 0 else None, 'few_shot_seed': few_shot_seed if k > 0 else None}

def few_shot_sample_metadata(few_shot_examples: Optional[List[Dict]], few_shot_sample_seed: Optional[int]) -> Dict:
    examples = few_shot_examples or []
    return {'few_shot_sample_seed': few_shot_sample_seed, 'few_shot_indices': [example['index'] for example in examples], 'few_shot_raw_splits': [example.get('raw_split') for example in examples], 'few_shot_canonical_splits': [example.get('canonical_split') for example in examples], 'few_shot_answers': [example['answer'] for example in examples]}

def build_few_shot_seed_by_index(dataset_items: List[Dict], few_shot_seed: Optional[int]) -> Dict[int, int]:
    if few_shot_seed is None:
        return {}
    rng = random.Random(few_shot_seed)
    return {item['index']: rng.randrange(2 ** 32) for item in dataset_items}

def build_candidate_identity(item: Dict) -> Tuple[str, int]:
    return (item.get('raw_split'), item['index'])

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str, num_distractors: int=4, support_checkpoint: Optional[Path]=None):
    valid_candidates = []
    eligible = [item for item in metadata_items if build_landmark_options(item, num_distractors=num_distractors) is not None]
    contract = json.loads(json.dumps(dict(raw_split=raw_split, canonical_split=canonical_split, num_distractors=num_distractors, candidates=eligible)))
    checked = {}
    if support_checkpoint is not None and Path(support_checkpoint).exists():
        saved = json.loads(Path(support_checkpoint).read_text())
        if saved.get('contract') != contract:
            raise ValueError('Landmarks support checkpoint differs from candidate pool')
        checked = saved['readable']
        expected = {str(item['index']) for item in eligible}
        if set(checked) - expected or any((type(value) is not bool for value in checked.values())):
            raise ValueError('Invalid Landmarks support checkpoint')
    image_dataset = hf_dataset.select_columns(['image']) if hasattr(hf_dataset, 'select_columns') else hf_dataset

    def check(item):
        return dataset_image_is_readable(image_dataset, item['index'])

    def candidate(item):
        return {'index': item['index'], 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'name': str(item['name']).strip(), 'distractorLabels': list(item.get('distractorLabels') or [])}
    print(f'Landmarks support image checks: resuming {len(checked)}/{len(eligible)}', flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check, item): item for item in eligible if str(item['index']) not in checked}
        for future in as_completed(futures):
            item = futures[future]
            checked[str(item['index'])] = future.result()
            if support_checkpoint is not None:
                path = Path(support_checkpoint)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(contract=contract, readable=checked)))
                temporary.replace(path)
            if len(checked) % 25 == 0 or len(checked) == len(eligible):
                print(f'Landmarks support image checks: {len(checked)}/{len(eligible)}', flush=True)
    for item in eligible:
        if checked[str(item['index'])]:
            valid_candidates.append(candidate(item))
        else:
            print(f"Skipping invalid few-shot candidate #{item['index']}: image is not readable")
    if len(valid_candidates) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}'. Found only {len(valid_candidates)} examples with readable images, non-empty names, and distractors.")
    return {'hf_dataset': hf_dataset, 'candidates': valid_candidates, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split}

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidate_pool = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidate_pool = [candidate for candidate in candidate_pool if build_candidate_identity(candidate) != current_identity]
    if len(candidate_pool) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidate_pool)))
    sampled_candidates = random.Random(seed).sample(candidate_pool, few_shot_source['k'])
    examples = []
    for candidate in sampled_candidates:
        options = build_landmark_options(candidate)
        if options is None:
            raise ValueError(f"Previously validated few-shot candidate #{candidate['index']} no longer has valid Landmarks options")
        examples.append({'index': candidate['index'], 'split': candidate['split'], 'raw_split': candidate['raw_split'], 'canonical_split': candidate['canonical_split'], 'image_url': image_data_uri_from_dataset(few_shot_source['hf_dataset'], candidate['index']), 'options': options, 'user_prompt': get_user_prompt(options), 'answer': format_few_shot_answer(candidate['name'], options)})
    return examples

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

def resolve_canonical_pools(few_shot_k: int, support_checkpoint: Optional[Path]=None) -> Dict:
    split_map = discover_available_splits_from_names(('train',))
    strategy = resolve_split_strategy(split_map)
    hf_dataset, metadata_items = download_and_prepare_dataset(split='train')
    split_lengths = {'train': len(hf_dataset)}
    assignments = build_split_assignments(split_map=split_map, split_lengths=split_lengths, seed=DEFAULT_SPLIT_SEED)
    eval_assignment = assignments['test']
    train_assignment = assignments['train']
    eval_selected_indices = select_evaluation_indices_from_pool(eval_assignment['indices'])
    evaluation_items = select_metadata_items_by_indices(metadata_items, eval_selected_indices, raw_split=eval_assignment['raw_split'], canonical_split='test')
    few_shot_source = None
    if few_shot_k > 0:
        few_shot_items = select_metadata_items_by_indices(metadata_items, train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(hf_dataset, few_shot_items, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train', support_checkpoint=support_checkpoint)
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw {} | Eval: canonical test <- raw {} | split seed={}'.format(train_assignment['raw_split'], eval_assignment['raw_split'], DEFAULT_SPLIT_SEED))
    return {'evaluation_dataset': hf_dataset, 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def download_and_prepare_dataset(split: str='train'):
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
    dataset = load_cached_parquet_dataset(HF_DATASET_NAME, split=split, image_columns=['image'])
    if dataset is None:
        dataset = load_dataset(HF_DATASET_NAME, split=split, token=token, cache_dir=cache_dir, download_mode='reuse_dataset_if_exists', download_config=download_config)
        dataset = dataset.cast_column('image', Image(decode=False))
        print('Evaluation state updated.', flush=True)
    else:
        print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c != 'image'])
    selected_indices = select_evaluation_indices(len(metadata_ds))
    print('Evaluation state updated.', flush=True)
    items = []
    for i in progress_iter(selected_indices, desc='Building metadata index'):
        row = metadata_ds[i]
        items.append({'index': i, 'name': row.get('name', ''), 'instanceOfLabels': row.get('instanceOfLabels', []), 'distractorLabels': row.get('distractorLabels', []), 'lat': row.get('lat', 0), 'lon': row.get('lon', 0)})
    print('Evaluation state updated.', flush=True)
    return (dataset, items)

def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, num_distractors: int=4, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, sample_parallel: int=4, reasoning_method: str='baseline'):
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if reasoning_method == 'cot':
        prompt_metadata['reasoning_method'] = reasoning_method
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None)
    if result_path is None:
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"-{prompt_metadata['few_shot_k']}shot" if prompt_metadata['prompt_mode'] == 'few-shot' else ''
        result_path = os.path.join(output_dir, f'{model_key}-landmarks{suffix}.jsonl')
    result_path = result_path_for_reasoning(result_path, reasoning_method)
    completed_indices = clean_result_jsonl_for_resume(result_path)
    validate_resume_config(result_path, prompt_metadata)
    print(f'[{model_key}] Starting, results -> {result_path}')
    pending_items = [item for item in dataset_items if item['index'] not in completed_indices]
    sample_parallel = max(1, int(sample_parallel or 1))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_parallel = 1
    print('Evaluation state updated.', flush=True)
    if not pending_items:
        print('Evaluation state updated.', flush=True)
        return result_path
    query_fn = get_query_function(model_key, api_key=api_key)

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[i])
            options = build_landmark_options(item, num_distractors=num_distractors)
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            if options is None:
                result_record = {'index': i, 'options': None, 'name': item['name'], 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'instanceOfLabels': item['instanceOfLabels'], 'lat': item['lat'], 'lon': item['lon'], **result_model_metadata(model_key, MODEL_CONFIGS), 'response': None}
                result_record.update(prompt_metadata)
                result_record.update(few_shot_sample_metadata([], few_shot_sample_seed))
                return (result_record, None)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            task_prompt = get_user_prompt(options)
            query_prompt = effective_prompt(task_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

            def call_model(prompt_text):
                return query_with_image_fallback(query_fn=query_fn, raw_bytes=raw_bytes, model_key=model_key, sample_index=i, system=query_system, user=prompt_text, few_shot_examples=sample_few_shot_examples_list)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='landmarks', context={'options': options}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer = parse_landmark_answer(answer_text, options)
            result_record = {'index': i, 'options': options, 'name': item['name'], 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'instanceOfLabels': item['instanceOfLabels'], 'lat': item['lat'], 'lon': item['lon'], **result_model_metadata(model_key, MODEL_CONFIGS), 'response': response, 'prompt': task_prompt, 'answer_text': answer_text, 'parsed_answer': parsed_answer}
            result_record.update(prompt_metadata)
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            if reasoning_method == 'cot':
                result_record.update({'reasoning_method': 'cot', 'reasoning': reasoning_response.get('reasoning')})
            return (result_record, response_content)
        except Exception as e:
            print(f'[{model_key}] Error processing sample {i}: {e}')
            import traceback
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS) or is_non_retryable_api_error(e):
                raise
            return None
    if sample_parallel == 1:
        for item in progress_iter(pending_items, desc=f'{model_key} samples', unit='sample'):
            result = _process_single_sample(item)
            if result is None:
                continue
            result_record, response_content = result
            append_to_jsonl(result_record, result_path)
            completed_indices.add(result_record['index'])
            done = len(completed_indices)
            if response_content is None:
                print('Evaluation state updated.', flush=True)
            else:
                print('Evaluation state updated.', flush=True)
    else:
        buffered_results = {}
        next_write_pos = 0
        with ThreadPoolExecutor(max_workers=sample_parallel) as pool:
            futures = {pool.submit(_process_single_sample, item): pos for pos, item in enumerate(pending_items)}
            for future in progress_iter(as_completed(futures), desc=f'{model_key} samples', unit='sample'):
                pos = futures[future]
                buffered_results[pos] = future.result()
                while next_write_pos in buffered_results:
                    result = buffered_results.pop(next_write_pos)
                    if result is not None:
                        result_record, response_content = result
                        append_to_jsonl(result_record, result_path)
                        completed_indices.add(result_record['index'])
                        done = len(completed_indices)
                        if response_content is None:
                            print('Evaluation state updated.', flush=True)
                        else:
                            print('Evaluation state updated.', flush=True)
                    next_write_pos += 1
    print('Evaluation state updated.', flush=True)
    return result_path

def response_text_from_record(record: Dict) -> str:
    response = record.get('response')
    if isinstance(response, dict):
        choices = response.get('choices')
        if isinstance(choices, list) and choices:
            message = choices[0].get('message')
            if isinstance(message, dict):
                return str(message.get('content') or '')
    return '' if response is None else str(response)

def scoreable_landmark_answer(record: Dict) -> Optional[str]:
    options = record.get('options')
    if not isinstance(options, list) or not options:
        return None
    answer_text = record.get('answer_text')
    if isinstance(answer_text, str):
        return parse_landmark_answer(answer_text, options)
    parsed = record.get('parsed_answer')
    if isinstance(parsed, str) and parsed.strip():
        canonical = parse_landmark_answer(parsed, options)
        if canonical is not None:
            return canonical
    return parse_landmark_answer(response_text_from_record(record), options)

def canonical_landmarks_records(records: List[Dict]) -> List[Dict]:
    if not records:
        return []
    indices = [record.get('index') for record in records]
    if not all((isinstance(index, int) and index >= 0 for index in indices)):
        raise ValueError('Landmarks results cannot be scoped because some rows lack a non-negative integer index.')
    latest_by_index = {}
    for record in records:
        latest_by_index[record['index']] = record
    unique_records = list(latest_by_index.values())
    if all((record.get('canonical_split') is not None for record in unique_records)):
        return [record for record in unique_records if record.get('canonical_split') == 'test']
    raise ValueError('Landmarks results require canonical_split metadata')

def evaluation(result_path):
    import pandas as pd
    model_name = _remove_suffix(os.path.basename(result_path), '.jsonl')
    result_dir = os.path.dirname(result_path) or DEFAULT_RESULT_DIR
    print(f'---------------- {model_name} ----------------')
    records = canonical_landmarks_records(list(iter_jsonl(result_path)))
    if not records:
        raise ValueError(f'No valid Landmarks result rows found in {result_path}')
    result_json = pd.DataFrame(records)
    result_json['model_response'] = result_json.apply(lambda row: response_text_from_record(row), axis=1)
    refusal_keywords = ['sorry', 'difficult']
    result_json['model_answer'] = result_json.apply(scoreable_landmark_answer, axis=1)
    result_json['is_refusal'] = result_json['model_response'].apply(lambda text: any((keyword in text.casefold() for keyword in refusal_keywords)))
    result_json['parse_failed'] = result_json['model_answer'].isna()
    result_json['is_correct'] = result_json.apply(lambda row: isinstance(row['model_answer'], str) and row['model_answer'].casefold() == str(row['name']).casefold(), axis=1)
    result_json['category'] = result_json['instanceOfLabels'].apply(lambda x: find_category(x))
    correct_rate = result_json['is_correct'].mean()
    refusal_rate = result_json['is_refusal'].mean()
    parse_failure_rate = result_json['parse_failed'].mean()
    print(correct_rate, refusal_rate)
    correct_results = result_json[result_json['is_correct']]
    incorrect_results = result_json[~result_json['is_correct']]
    category_counts = result_json['category'].value_counts()
    print(category_counts)
    print((correct_results['category'].value_counts() / result_json['category'].value_counts()).loc[category_counts.index] * 100)
    export_columns = ['index', 'name', 'category', 'model_response']
    os.makedirs(result_dir, exist_ok=True)
    result_json.to_csv(os.path.join(result_dir, f'{model_name}.csv'), index=False)
    correct_results[export_columns].to_csv(os.path.join(result_dir, f'{model_name}-correct.csv'), index=False)
    incorrect_results[export_columns].to_csv(os.path.join(result_dir, f'{model_name}-incorrect.csv'), index=False)
    try:
        import geopandas
        from matplotlib import pyplot as plt
        shp = geopandas.GeoDataFrame(result_json, crs='EPSG:4269', geometry=geopandas.points_from_xy(result_json.lon, result_json.lat))
        usa = geopandas.read_file('./data/Landmarks/cb_2018_us_state_500k.zip')
        shp = shp.sjoin(usa, how='left', predicate='within')
        state_accuracy = shp.groupby('NAME').is_correct.mean().reset_index()
        heatmap_data = usa.merge(state_accuracy, on='NAME', how='left')
        heatmap_data['is_correct'] *= 100
        print(f'\nTop 10 worst performing states:')
        print(heatmap_data[['NAME', 'is_correct']].dropna().sort_values(by='is_correct').iloc[:10])
        heatmap_data.plot(column='is_correct', cmap='jet_r', legend=True, legend_kwds={'orientation': 'horizontal'})
        plt.title('Zero-shot Landmark Recognition Accuracy (%) by State')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'landmarks-{model_name}-accuracy-state.pdf'), dpi=500)
        plt.close(plt.gcf())
    except Exception as e:
        print(f'Warning: Could not generate state map: {e}')
    return {'model': model_name, 'correct_rate': correct_rate, 'refusal_rate': refusal_rate, 'parse_failure_rate': parse_failure_rate}

def plot():
    import geoplot as gplt
    import geoplot.crs as gcrs
    import geopandas
    from matplotlib import pyplot as plt
    contiguous_usa = geopandas.read_file(gplt.datasets.get_path('contiguous_usa'))
    shp = geopandas.read_file('./data/Landmarks/all_unions_polygons_convexhulls_metadata_wikidata_us.gpkg')
    shp['area'] = shp.to_crs('+proj=cea').area / 1000000.0
    shp['centroids'] = shp['geometry'].to_crs('+proj=cea').centroid.to_crs(4326)
    shp.set_geometry('centroids', inplace=True)
    shp['instanceOfLabels'] = shp['instanceOfLabels'].apply(json.loads)
    shp['categories'] = shp['instanceOfLabels'].apply(find_category)
    print(shp['categories'])
    print(shp[['categories', 'area']].groupby('categories').median()['area'])
    print(shp['area'].median())
    print(shp[shp['categories'] == 'Miscellaneous'][['name', 'instanceOfLabels']])
    ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea(), edgecolor='white', facecolor='lightgray', figsize=(12, 8))
    gplt.pointplot(shp, ax=ax, hue='area', cmap='Blues', scheme='quantiles', scale='area', limits=(1, 10), legend=True, legend_var='scale')
    ax.set_title('Aerial Landmarks Coverage', fontsize=16)
    plt.show()

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR):
    import re
    import matplotlib.dates as mdates
    import pandas as pd
    import seaborn as sns
    from adjustText import adjust_text
    from matplotlib import pyplot as plt
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    result_files = [f for f in os.listdir(result_dir) if re.search('-landmarks(?:-\\d+shot)?\\.jsonl$', f)]
    if not result_files:
        print(f'No result files found in {result_dir}')
        return
    print(f'Found {len(result_files)} result files')
    all_metrics = []
    for result_file in progress_iter(result_files, desc='Evaluating models'):
        result_path = os.path.join(result_dir, result_file)
        try:
            metrics = evaluation(result_path)
            if metrics is not None:
                model_key = re.sub('-landmarks(?:-\\d+shot)?\\.jsonl$', '', result_file)
                if model_key in MODEL_CONFIGS:
                    metrics['release_date'] = MODEL_CONFIGS[model_key]['release_date']
                    metrics['family'] = MODEL_CONFIGS[model_key]['family']
                    metrics['model_key'] = _remove_suffix(result_file, '.jsonl')
                    metrics['base_model_key'] = model_key
                    metrics['result_file'] = result_file
                    metrics.update(shot_from_metrics(metrics, result_file))
                    all_metrics.append(metrics)
                else:
                    print(f'Warning: Model {model_key} not found in MODEL_CONFIGS')
        except Exception as e:
            print(f'Error evaluating {result_file}: {e}')
            continue
    if not all_metrics:
        print('No valid metrics collected, cannot generate plots')
        return
    write_shot_summary_json(all_metrics, os.path.join(result_dir, 'model_metrics_summary_all_shots.json'), dataset_name='Landmarks')
    zero_shot_metrics = [metrics for metrics in all_metrics if is_zero_shot_metrics(metrics, metrics.get('result_file'))]
    if not zero_shot_metrics:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return pd.DataFrame(all_metrics)
    df_metrics = pd.DataFrame(zero_shot_metrics)
    df_metrics['release_date'] = pd.to_datetime(df_metrics['release_date'])
    df_metrics = df_metrics.sort_values('release_date')
    print(f'\nCollected metrics for {len(df_metrics)} models')
    print(df_metrics[['model_key', 'release_date', 'correct_rate', 'refusal_rate']])
    sns.set(font_scale=1.2, style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = FAMILY_COLORS
    ax1 = axes[0]
    texts1 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax1.plot(df_family['release_date'], df_family['correct_rate'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts1.append(ax1.text(row['release_date'], row['correct_rate'], label, fontsize=12, color=colors[family]))
    ax1.set_xlabel('Release Date')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if texts1:
        adjust_text(texts1, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax2 = axes[1]
    texts2 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family]
        if len(df_family) > 0:
            ax2.plot(df_family['release_date'], df_family['refusal_rate'], marker='o', label=family, color=colors[family], linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts2.append(ax2.text(row['release_date'], row['refusal_rate'], label, fontsize=12, color=colors[family]))
    ax2.set_xlabel('Release Date')
    ax2.set_ylabel('Refusal Rate')
    ax2.set_title('Refusal Rate over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if texts2:
        adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle('Landmarks: Landmark Recognition', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
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
    if run_reasoning_task_cli('landmarks'):
        return
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate vision models on Landmarks dataset')
    parser.add_argument('--model', type=str, default=None, help='Model to test (model name, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--plot-only', action='store_true', help='Generate model comparison plots from existing results')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel threads (default: auto-detect based on models and API keys)')
    parser.add_argument('--sample-parallel', type=int, default=4, help='Concurrent sample/API calls per model (default: 4; use 1 to disable sample-level threading)')
    parser.add_argument('--qwen-model-path', type=str, default=None, help='Local Qwen checkpoint directory, or a root containing per-model subdirectories')
    parser.add_argument('--llava-model-path', type=str, default=None, help='Local LLaVA checkpoint directory, or a root containing per-model subdirectories')
    parser.add_argument('--qwen-finetuned-root', type=str, default=None, help='Root containing task-specific Qwen LoRA adapter directories')
    parser.add_argument('--download-full-dataset', action='store_true', help='Download all local parquet shards for the requested dataset/split before evaluation')
    parser.add_argument('--download-only', action='store_true', help='Only download/cache the dataset, then exit')
    parser.add_argument('--download-workers', type=int, default=2, help='Parallel workers for Hugging Face snapshot download (default: 2)')
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
        print_runtime_model_summary(list(MODEL_CONFIGS.keys()), MODEL_CONFIGS)
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir)
        return
    if args.evaluate_only:
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    if args.download_only:
        args.download_full_dataset = True
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative.')
    if args.download_full_dataset:
        download_full_dataset_cache(HF_DATASET_NAME, split=None, max_workers=args.download_workers)
        if args.download_only:
            return
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        print_provider_summary()
    canonical_pools = resolve_canonical_pools(few_shot_k)
    global _shared_dataset_items, _shared_hf_dataset, _shared_split_context
    _shared_hf_dataset = canonical_pools['evaluation_dataset']
    _shared_dataset_items = canonical_pools['evaluation_items']
    _shared_split_context = canonical_pools['split_context']
    few_shot_source = canonical_pools['few_shot_source']
    if few_shot_source:
        print('Evaluation state updated.', flush=True)
    print(f"\n{'=' * 60}")
    print(f'Models to test: {len(models_to_test)}')
    print_runtime_model_summary(models_to_test, MODEL_CONFIGS)
    print(f"Prompt mode: {('few-shot' if few_shot_source else 'zero-shot')}")
    print(f"{'=' * 60}\n")
    if args.parallel is None:
        if has_local_models(models_to_test, MODEL_CONFIGS):
            num_parallel = 1
        else:
            available_keys = api_keys_for_models(models_to_test, MODEL_CONFIGS)
            num_parallel = min(len(models_to_test), len(available_keys)) if available_keys else 1
    else:
        num_parallel = max(1, min(args.parallel, len(models_to_test)))
    tasks = []
    for idx, model_key in enumerate(models_to_test):
        if is_local_model(model_key, MODEL_CONFIGS):
            api_key = None
            print(f'  {model_key:<35} -> local checkpoint: {_QWEN_MODEL_PATHS[model_key]}')
        else:
            api_key = key_for_model_index(model_key, idx, MODEL_CONFIGS)
            key_display = safe_key_display(api_key)
            print(f'  {model_key:<35} -> API Key #{key_index_for_model(model_key, idx, MODEL_CONFIGS)} ({key_display})')
        tasks.append((model_key, api_key, args.output_dir, few_shot_source, args.few_shot_seed if few_shot_source else None, args.sample_parallel, args.reasoning_method))
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
                    if is_local_model(model_key, MODEL_CONFIGS):
                        raise
    else:
        print(f"\n{'=' * 60}")
        print(f'Running sequentially...')
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
    print(f"\n{'=' * 60}")
    print(f'All tasks completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_split_context = None

def run_single_model(args_tuple):
    model_key, api_key, output_dir, few_shot_source, few_shot_seed, sample_parallel, reasoning_method = args_tuple
    try:
        result_path = query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, split_context=_shared_split_context, result_path=None, output_dir=output_dir, api_key=api_key, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, sample_parallel=sample_parallel, reasoning_method=reasoning_method)
        return result_path
    except Exception as e:
        print(f'\n[{model_key}] Error: {e}')
        import traceback
        traceback.print_exc()
        if is_local_model(model_key, MODEL_CONFIGS):
            raise
        return None
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
