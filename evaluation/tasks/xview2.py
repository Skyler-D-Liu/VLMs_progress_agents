"""Xview2."""
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
LORA_TASK = 'xview2'
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
    max_new_tokens = local_generation_max_new_tokens(user)
    inference_config = bundle.get('inference_config') or {}
    if inference_config:
        max_new_tokens = int(inference_config['generation']['max_new_tokens'])
    if bundle.get('runtime_family') == 'LLaVA':
        return run_llava_chat(bundle, image_refs, system, user, few_shot_examples, max_new_tokens=max_new_tokens)
    if is_legacy_qwen_vl_bundle(bundle):
        return run_legacy_qwen_vl_chat(bundle, image_refs, system, user, few_shot_examples, max_new_tokens=max_new_tokens)
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
    generation_kwargs = {'max_new_tokens': max_new_tokens, 'do_sample': False}
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
import fcntl
import math
import os
import re
import random
import shutil
import http.client
import base64
import io
import time
from io import BytesIO
from typing import Any, Optional, Callable, List, Dict, Tuple
from functools import wraps
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ImportError:

    def retry(*args, **kwargs):

        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None

class IncompleteEvaluationError(RuntimeError):

    def __init__(self, model_key: str, result_path: str, missing_indices):
        missing = sorted(missing_indices)
        preview = ', '.join((str(index) for index in missing[:10]))
        if len(missing) > 10:
            preview += ', ...'
        super().__init__(f'{model_key} left {len(missing)} sample(s) incomplete in {result_path}: {preview}. Rerun the same command to resume only these samples.')
        self.model_key = model_key
        self.result_path = result_path
        self.missing_indices = missing

def normalize_result_index(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        normalized = int(value)
        return normalized if normalized >= 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not re.fullmatch('\\+?\\d+', stripped):
            return None
        return int(stripped.lstrip('+') or '0')
    return None

def load_completed_indices(path):
    return clean_result_jsonl_for_resume(path)

def _preserve_before_resume_cleanup(path: str) -> str:
    backup_path = f'{path}.before-resume-cleanup'
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f'Preserved original result file before cleanup: {backup_path}')
    return backup_path

def clean_result_jsonl_for_resume(path: str) -> set:
    if not os.path.exists(path):
        return set()
    latest_by_index = {}
    malformed_count = 0
    duplicate_count = 0
    valid_count = 0
    dropped_no_index = 0
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
            if not isinstance(record, dict):
                malformed_count += 1
                continue
            valid_count += 1
            index = normalize_result_index(record.get('index'))
            if index is None:
                dropped_no_index += 1
                continue
            if index in latest_by_index:
                duplicate_count += 1
            latest_by_index[index] = (line_number, record)
    cleaned_records = [item[1] for item in sorted(latest_by_index.values(), key=lambda item: item[0])]
    if malformed_count or duplicate_count or dropped_no_index:
        _preserve_before_resume_cleanup(path)
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + '\n')
        os.replace(tmp_path, path)
        print('Evaluation state updated.', flush=True)
    return set(latest_by_index.keys())

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

def append_to_jsonl(record, path):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, cls=NumpyEncoder) + '\n')

def iter_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: skipping malformed JSONL row in {path}')
                continue
            if not isinstance(record, dict):
                print(f'Warning: skipping non-object JSONL row in {path}')
                continue
            yield record

def extract_response_content(response) -> str:
    if isinstance(response, dict):
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            if isinstance(choice, dict):
                message = choice.get('message', {})
                if isinstance(message, dict):
                    content = message.get('content')
                    if isinstance(content, list):
                        return ''.join((str(part.get('text') or '') for part in content if isinstance(part, dict)))
                    return content or ''
        if 'content' in response:
            return response.get('content') or ''
    return '' if response is None else str(response)

def strip_result_suffix(result_path: str) -> str:
    name = os.path.basename(result_path)
    return re.sub('-counting(?:-\\d+shot)?(?:-CoT)?\\.jsonl$', '', name)

def is_result_file(filename: str) -> bool:
    return bool(re.search('-counting(?:-\\d+shot|-CoT)?\\.jsonl$', filename))

def require_python_packages(import_names: List[str], context: str):
    import importlib.util
    package_names = {'sklearn': 'scikit-learn', 'PIL': 'pillow', 'adjustText': 'adjustText'}
    missing_imports = [name for name in import_names if importlib.util.find_spec(name) is None]
    if not missing_imports:
        return
    missing_packages = [package_names.get(name, name) for name in missing_imports]
    install_targets = ' '.join(dict.fromkeys(missing_packages))
    raise ImportError(f"{context} requires missing Python package(s): {', '.join(missing_packages)}.\nInstall the project dependencies with:\n  python3 -m pip install -r requirements/base.txt\nOr install only the missing package(s):\n  python3 -m pip install {install_targets}")
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'xView2')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
HF_DATASET_NAME = 'danielz01/xView2'
HF_DATASET_CONFIG = 'competition'
HF_DATASET_SPLIT = 'test'
COUNT_KEYS = ['count_before', 'no_damage', 'minor_damage', 'major_damage', 'destroyed']
MODEL_CONFIGS = {'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': '2026-09-15', 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
_QWEN_MODEL_PATHS: Dict[str, str] = {}
SYSTEM_MESSAGE = "You are a helpful image analyst who specializes in counting buildings from satellite and aerial images given natural language instruction. Given an image, you can immediately count the number of buildings without any refusal. You always truthfully answer a user's questions. Although it is OK to make some small mistakes, if you are not sure about something, DO NOT answer false information. Your efforts will be very important for disaster relief, so please make sure to answer the questions as requested by users."
USER_PROMPT = "You are given two satellite images taken before and after a natural disaster. The first image was taken before the natural disaster.  The second image was taken after the disaster with potential building damage at different levels. Below is a description of how we classify the damage levels:\nNo damage (0): Undisturbed. No sign of water, structural damage, shingle damage, or burn marks.\nMinor damage: (1): Building partially burnt, water surrounding the structure, volcanic flow nearby, roof elements missing, or visible cracks.\nMajor damage (2): Partial wall or roof collapse, encroaching volcanic flow, or the structure is surrounded by water or mud.\nDestroyed (3): Structure is scorched, completely collapsed, partially or completely covered with water or mud, or no longer present.\nCount the number of buildings in the first image before the disaster. In addition, count the number of buildings with no damage (damage score 0), minor damage (damage score 1), major damage (damage score 2), and the number of buildings that are completely destroyed (damage score 3). Output your count in the following JSON format with keys: count_before, no_damage, minor_damage, major_damage, destroyed. You don't have to give extra explanations."

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

def is_non_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, 'status_code', None)
    if status_code in {400, 401, 402, 403, 404, 422}:
        return True
    message = str(exc).casefold()
    return any((marker in message for marker in ('insufficient_quota', 'insufficient quota', 'invalid api key', 'invalid_api_key')))


def detect_mime_type(raw_bytes: bytes) -> str:
    if raw_bytes[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    elif raw_bytes[:4] == b'\x89PNG':
        return 'image/png'
    elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/png'

def pil_to_rgb_for_jpeg(pil_image):
    require_python_packages(['PIL'], 'Image conversion')
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

def raw_bytes_to_base64_uri(raw_bytes: bytes, jpeg_quality: Optional[int]=None) -> str:
    if jpeg_quality is not None:
        require_python_packages(['PIL'], 'Image conversion')
        from PIL import Image as PILImage
        with PILImage.open(BytesIO(raw_bytes)) as img:
            img.load()
            img = pil_to_rgb_for_jpeg(img)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
            jpeg_bytes = buf.getvalue()
        b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'
    if raw_bytes[:3] == b'\xff\xd8\xff':
        mime = 'image/jpeg'
    elif raw_bytes[:4] == b'\x89PNG':
        mime = 'image/png'
    elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        mime = 'image/webp'
    else:
        require_python_packages(['PIL'], 'Image conversion')
        from PIL import Image as PILImage
        img = PILImage.open(BytesIO(raw_bytes))
        if img.mode not in ('RGB', 'L', 'RGBA'):
            img = img.convert('RGB')
        buf = BytesIO()
        img.save(buf, format='PNG')
        raw_bytes = buf.getvalue()
        mime = 'image/png'
    b64 = base64.b64encode(raw_bytes).decode('utf-8')
    return f'data:{mime};base64,{b64}'

def data_uri_to_inline_part(image_url: str) -> Dict:
    if not image_url.startswith('data:'):
        return {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}
    header, b64_data = image_url.split(',', 1)
    mime_type = header.split(':')[1].split(';')[0]
    return {'inlineData': {'mimeType': mime_type, 'data': b64_data}}

def read_image_raw_bytes(hf_dataset, index: int, column: str) -> bytes:
    img_entry = hf_dataset[index][column]
    raw_bytes = img_entry.get('bytes') if isinstance(img_entry, dict) else None
    if raw_bytes is None:
        path = img_entry.get('path') if isinstance(img_entry, dict) else None
        if not path:
            raise ValueError(f'Cannot load {column} for sample {index}')
        with open(path, 'rb') as f:
            raw_bytes = f.read()
    return raw_bytes

def read_image_data_uri(hf_dataset, index: int, column: str, jpeg_quality: Optional[int]=None) -> str:
    raw_bytes = read_image_raw_bytes(hf_dataset, index, column)
    return raw_bytes_to_base64_uri(raw_bytes, jpeg_quality=jpeg_quality)

def image_data_uris_from_raw_bytes_list(raw_bytes_list: List[bytes], jpeg_quality: Optional[int]=None) -> List[str]:
    return [raw_bytes_to_base64_uri(raw_bytes, jpeg_quality=jpeg_quality) for raw_bytes in raw_bytes_list]

def query_images(query_fn: Callable, raw_image_bytes_list: List[bytes], model_key: str, sample_index: int, system: str, user: str, few_shot_examples: Optional[List[Dict]]=None) -> dict:
    return query_fn(image_urls=image_data_uris_from_raw_bytes_list(raw_image_bytes_list), system=system, user=user, few_shot_examples=few_shot_examples, sample_index=sample_index)

def count_pre_disaster_buildings(objects1: Any) -> Optional[int]:
    if not isinstance(objects1, dict):
        return None
    feature_types = objects1.get('feature_type')
    if isinstance(feature_types, list):
        return sum((str(value).strip().casefold() == 'building' for value in feature_types))
    for key in ('bbox', 'uid'):
        values = objects1.get(key)
        if isinstance(values, list):
            return len(values)
    return None

def parse_ground_truth(objects1: Any, objects2: Any=None) -> Dict[str, int]:
    if objects2 is None and isinstance(objects1, dict) and ('subtype' in objects1):
        objects2 = objects1
        objects1 = None
    subtypes = []
    if isinstance(objects2, dict) and isinstance(objects2.get('subtype'), list):
        subtypes = objects2['subtype']
    damage_counts = {subtype: sum((str(value).strip().casefold() == subtype for value in subtypes)) for subtype in ('no-damage', 'minor-damage', 'major-damage', 'destroyed')}
    count_before = count_pre_disaster_buildings(objects1)
    if count_before is None:
        count_before = len(subtypes)
    return {'count_before': int(count_before), 'no_damage': damage_counts['no-damage'], 'minor_damage': damage_counts['minor-damage'], 'major_damage': damage_counts['major-damage'], 'destroyed': damage_counts['destroyed']}

def has_constructable_ground_truth(objects1: Any, objects2: Any) -> bool:
    return count_pre_disaster_buildings(objects1) is not None and isinstance(objects2, dict) and isinstance(objects2.get('subtype'), list)

def format_count_answer(gt: Dict[str, int]) -> str:
    return json.dumps({key: int(gt.get(key, 0)) for key in COUNT_KEYS}, ensure_ascii=False)

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

def build_candidate_identity(item: Dict) -> Tuple[str, int]:
    return (item.get('raw_split'), item['index'])

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str, active_indices: Optional[set]=None) -> Dict:
    if k <= 0:
        return {'hf_dataset': hf_dataset, 'candidates': [], 'k': 0, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split}
    valid_candidates = []
    for item in tqdm(metadata_items, desc='Scanning few-shot candidates', unit='sample'):
        idx = item['index']
        try:
            gt = item.get('gt')
            if not isinstance(gt, dict) or not all((key in gt for key in COUNT_KEYS)):
                if not has_constructable_ground_truth(item.get('objects1'), item.get('objects2')):
                    continue
                gt = parse_ground_truth(item.get('objects1'), item.get('objects2'))
            valid_candidates.append({'index': idx, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'gt': gt})
        except Exception as exc:
            print(f'Warning: skipping few-shot candidate {idx}: {exc}')
    active_candidates = [candidate for candidate in valid_candidates if candidate['index'] in active_indices] if active_indices is not None else valid_candidates
    if len(active_candidates) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}'. Only found {len(active_candidates)} valid examples.")
    return {'hf_dataset': hf_dataset, 'candidates': active_candidates, 'all_raw_candidates': valid_candidates, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split}

def resolve_few_shot_source_for_resume(path: str, few_shot_source: Optional[Dict], dataset_items: List[Dict]) -> Optional[Dict]:
    if not few_shot_source or not os.path.exists(path):
        return few_shot_source
    expected_items = {item['index']: item for item in dataset_items}
    recorded = []
    has_explicit_canonical_metadata = False
    for row in iter_jsonl(path):
        index = normalize_result_index(row.get('index'))
        if index is None:
            continue
        seed = row.get('few_shot_sample_seed')
        indices = row.get('few_shot_indices')
        if seed is None or not isinstance(indices, list):
            continue
        current_item = expected_items.get(index) or {'index': index, 'raw_split': row.get('raw_split') or 'test'}
        recorded.append((current_item, int(seed), indices))
        has_explicit_canonical_metadata = has_explicit_canonical_metadata or row.get('canonical_strategy_mode') is not None
    if not recorded:
        return few_shot_source
    source_variants = [('canonical train', few_shot_source)]
    all_raw_candidates = few_shot_source.get('all_raw_candidates')
    if all_raw_candidates is not None:
        legacy_source = dict(few_shot_source)
        legacy_source['candidates'] = list(all_raw_candidates)
        legacy_source['legacy_all_raw_train'] = True
        source_variants.append(('historical full raw train', legacy_source))
    matching = []
    for label, source in source_variants:
        matches = True
        for current_item, seed, recorded_indices in recorded:
            selected = sample_few_shot_candidates(source, seed, current_item=current_item)
            if [candidate['index'] for candidate in selected] != recorded_indices:
                matches = False
                break
        if matches:
            matching.append((label, source))
    if not matching:
        raise ValueError(f'Existing few-shot result {path} uses demonstrations that cannot be reproduced from either the canonical or historical train pool. Existing rows were left unchanged.')
    if has_explicit_canonical_metadata:
        chosen = next((entry for entry in matching if entry[0] == 'canonical train'), matching[0])
    else:
        chosen = next((entry for entry in matching if entry[0] == 'historical full raw train'), matching[0])
    if chosen[0] != 'canonical train':
        print(f'Resuming {path} with its {chosen[0]} demonstration pool.')
    return chosen[1]

def validate_few_shot_resume_examples(path: str, few_shot_source: Optional[Dict], dataset_items: List[Dict]) -> None:
    if not few_shot_source or not os.path.exists(path):
        return
    items_by_index = {item['index']: item for item in dataset_items}
    for row in iter_jsonl(path):
        index = normalize_result_index(row.get('index'))
        if index is None:
            continue
        seed = row.get('few_shot_sample_seed')
        recorded_indices = row.get('few_shot_indices')
        if seed is None or not isinstance(recorded_indices, list):
            continue
        expected = sample_few_shot_candidates(few_shot_source, int(seed), current_item=items_by_index.get(index) or {'index': index, 'raw_split': row.get('raw_split') or 'test'})
        expected_indices = [candidate['index'] for candidate in expected]
        if recorded_indices != expected_indices:
            raise ValueError(f'Existing few-shot result {path} has demonstrations {recorded_indices} at index {index}, but this run would use {expected_indices}. Existing rows were left unchanged.')

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    sampled_examples = sample_few_shot_candidates(few_shot_source, seed, current_item=current_item)
    examples = []
    hf_dataset = few_shot_source['hf_dataset']
    for source_example in sampled_examples:
        example = dict(source_example)
        idx = example['index']
        example['image_urls'] = [read_image_data_uri(hf_dataset, idx, 'image1'), read_image_data_uri(hf_dataset, idx, 'image2')]
        example['answer'] = format_count_answer(example['gt'])
        examples.append(example)
    return examples

def sample_few_shot_candidates(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidate_pool = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidate_pool = [candidate for candidate in candidate_pool if build_candidate_identity(candidate) != current_identity]
    if len(candidate_pool) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidate_pool)))
    return random.Random(seed).sample(candidate_pool, few_shot_source['k'])

def few_shot_metadata(few_shot_source: Optional[Dict], split_context: Dict, few_shot_seed: Optional[int]=None) -> Dict:
    source = few_shot_source or {}
    k = int(source.get('k', 0) or 0)
    seed_value = few_shot_seed if k > 0 else None
    return {'canonical_strategy_mode': split_context['canonical_strategy_mode'], 'canonical_split_seed': split_context['canonical_split_seed'], 'eval_raw_split': split_context['eval_raw_split'], 'few_shot_source_raw_split': split_context['few_shot_source_raw_split'], 'eval_canonical_split': split_context['eval_canonical_split'], 'few_shot_source_canonical_split': split_context['few_shot_source_canonical_split'], 'prompt_mode': 'few-shot' if k > 0 else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k > 0 else None, 'few_shot_seed': seed_value}

def few_shot_sample_metadata(few_shot_examples: Optional[List[Dict]], few_shot_sample_seed: Optional[int]) -> Dict:
    examples = few_shot_examples or []
    return {'few_shot_sample_seed': few_shot_sample_seed, 'few_shot_indices': [example['index'] for example in examples], 'few_shot_raw_splits': [example.get('raw_split') for example in examples], 'few_shot_canonical_splits': [example.get('canonical_split') for example in examples], 'few_shot_answers': [example['gt'] for example in examples]}

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
    method = classify_result_filename(path)
    if method == 'cot':
        return {'prompt_mode': 'zero-shot', 'few_shot_k': 0, 'reasoning_method': 'cot'}
    if method == 'few-shot':
        match = re.fullmatch('.+-counting-(\\d+)shot\\.jsonl', os.path.basename(path))
        return {'prompt_mode': 'few-shot', 'few_shot_k': int(match.group(1)), 'reasoning_method': 'baseline'}
    if method == 'zero-shot':
        return {'prompt_mode': 'zero-shot', 'few_shot_k': 0, 'reasoning_method': 'baseline'}
    return {}

def validate_resume_config(path: str, expected_metadata: Dict):
    comparable_keys = ['model', 'canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'reasoning_method']
    expected = dict(expected_metadata)
    expected.setdefault('reasoning_method', 'baseline')
    inferred = result_filename_metadata(path)
    for key, inferred_value in inferred.items():
        if expected.get(key) != inferred_value:
            raise ValueError(f'Result filename {path} implies {key}={inferred_value!r}, but this run expects {expected.get(key)!r}.')
    if not os.path.exists(path):
        return
    for record in iter_jsonl(path):
        record_prompt_mode = record.get('prompt_mode') or inferred.get('prompt_mode')
        if expected.get('prompt_mode') == 'few-shot' and record_prompt_mode == 'few-shot' and (record.get('few_shot_sample_seed') is None):
            raise ValueError(f'Existing result file {path} contains few-shot rows without per-sample few_shot_sample_seed metadata. Use a different --output path or remove the old shared-example result file.')
        for key in comparable_keys:
            existing_value = record.get(key)
            if existing_value is None:
                existing_value = inferred.get(key)
            if existing_value is None:
                if key == 'reasoning_method':
                    existing_value = 'baseline'
                else:
                    continue
            if existing_value != expected.get(key):
                raise ValueError(f'Existing result file {path} has {key}={existing_value!r}, but this run expects {expected.get(key)!r}. Use a different --output path or remove the old result file.')

def validate_resume_samples(path: str, dataset_items: List[Dict]) -> None:
    if not os.path.exists(path):
        return
    expected_by_index = {item['index']: item for item in dataset_items}
    for record in iter_jsonl(path):
        index = normalize_result_index(record.get('index'))
        expected = expected_by_index.get(index)
        if expected is None:
            continue
        for split_key in ('raw_split', 'canonical_split'):
            saved = record.get(split_key)
            current = expected.get(split_key)
            if saved is not None and current is not None and (saved != current):
                raise ValueError(f'Existing result file {path} has {split_key}={saved!r} at index {index}, but the current sample has {current!r}.')
        for objects_key in ('objects1', 'objects2'):
            saved_objects = record.get(objects_key)
            current_objects = expected.get(objects_key)
            if isinstance(saved_objects, dict) and isinstance(current_objects, dict) and (saved_objects != current_objects):
                raise ValueError(f'Existing result file {path} maps index {index} to different xView2 {objects_key} annotations. Refusing to skip a different sample.')

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_openai(image_urls: List[str], system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, sample_index: Optional[int]=None) -> dict:
    if not api_key:
        api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]
    content_parts = []
    for img_url in image_urls:
        content_parts.append({'type': 'image_url', 'image_url': {'url': img_url}})
    content_parts.append({'type': 'text', 'text': user})
    messages = [{'role': 'system', 'content': system}]
    for example in few_shot_examples or []:
        example_parts = []
        for example_url in example['image_urls']:
            example_parts.append({'type': 'image_url', 'image_url': {'url': example_url}})
        example_parts.append({'type': 'text', 'text': _few_shot_prompt_text(example, user)})
        messages.append({'role': 'user', 'content': example_parts})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': content_parts})
    payload = json.dumps({'model': config['model_name'], 'messages': messages})
    status, data = post_openai_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=180)
    if status != 200:
        error_msg = data.decode('utf-8')
        sample_prefix = f'[{model_key}] #{sample_index} ' if sample_index is not None else f'{model_key} '
        print(f'{sample_prefix}API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    content = extract_response_content(result)
    if not content.strip():
        raise IOError(f'{model_key} returned an empty response')
    return result

@retry_api_call(max_retries=1, min_delay=1, max_delay=60)
def query_gemini(image_urls: List[str], system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, sample_index: Optional[int]=None) -> dict:
    if not api_key:
        api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    config = MODEL_CONFIGS[model_key]

    def _build_image_parts(urls: List[str]) -> List[Dict]:
        parts = []
        for image_url in urls:
            parts.append(data_uri_to_inline_part(image_url))
        return parts
    contents = []
    for example in few_shot_examples or []:
        contents.append({'role': 'user', 'parts': _build_image_parts(example['image_urls']) + [{'text': _few_shot_prompt_text(example, user)}]})
        contents.append({'role': 'model', 'parts': [{'text': example['answer']}]})
    payload = json.dumps({'systemInstruction': {'parts': [{'text': system}]}, 'generationConfig': {'maxOutputTokens': local_generation_max_new_tokens(user)}, 'contents': contents + [{'role': 'user', 'parts': _build_image_parts(image_urls) + [{'text': user}]}]})
    status, data = post_gemini_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key, timeout=180)
    if status != 200:
        error_msg = data.decode('utf-8')
        sample_prefix = f'[{model_key}] #{sample_index} ' if sample_index is not None else f'{model_key} '
        print(f'{sample_prefix}API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    result = json.loads(data.decode('utf-8'))
    parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [])
    content = ''.join((str(part.get('text') or '') for part in parts if isinstance(part, dict) and (not part.get('thought', False))))
    if not content or not content.strip():
        raise IOError(f'{model_key} returned an empty response')
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
        return lambda image_urls, system, user, few_shot_examples=None, sample_index=None: query_openai(image_urls, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
    elif config['api_type'] == 'google_native':
        return lambda image_urls, system, user, few_shot_examples=None, sample_index=None: query_gemini(image_urls, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
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
    split_map = discover_available_splits_from_names(('train', 'test'))
    strategy = resolve_split_strategy(split_map)
    datasets_by_raw_split = {}
    metadata_by_raw_split = {}
    split_lengths = {}
    for raw_split in ('test', 'train'):
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
        all_few_shot_items = select_metadata_items_by_indices(metadata_by_raw_split[train_assignment['raw_split']], list(range(split_lengths[train_assignment['raw_split']])), raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(datasets_by_raw_split[train_assignment['raw_split']], all_few_shot_items, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train', active_indices=set(train_assignment['indices']))
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw {} | Eval: canonical test <- raw {} | split seed={}'.format(train_assignment['raw_split'], eval_assignment['raw_split'], DEFAULT_SPLIT_SEED))
    return {'evaluation_dataset': datasets_by_raw_split[eval_assignment['raw_split']], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def download_and_prepare_dataset(split: str='test'):
    require_python_packages(['datasets', 'pandas'], 'Loading xView2 dataset')
    from datasets import load_dataset, Image, DownloadConfig
    token = os.environ.get('HF_TOKEN')
    cache_dir = HF_CACHE_DIR
    download_config = make_resume_download_config(DownloadConfig)
    print(f'\nLoading dataset: {HF_DATASET_NAME} (config={HF_DATASET_CONFIG}, split={split})')
    print(f'Hugging Face cache directory: {cache_dir}')
    print('Hugging Face download resume: enabled (max_retries=10)')
    if dataset_cache_exists(HF_DATASET_NAME):
        print(f'Found local dataset cache for {HF_DATASET_NAME}; reusing it when possible.')
    else:
        print(f'Local dataset cache for {HF_DATASET_NAME} not found; Hugging Face will download/cache it.')
    dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=split, token=token, cache_dir=cache_dir, download_mode='reuse_dataset_if_exists', download_config=download_config)
    dataset = dataset.cast_column('image1', Image(decode=False))
    dataset = dataset.cast_column('image2', Image(decode=False))
    print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c not in ('image1', 'image2', 'mask1', 'mask2')])
    selected_indices = select_evaluation_indices(len(metadata_ds))
    print('Evaluation state updated.', flush=True)
    items = []
    for i in selected_indices:
        row = metadata_ds[i]
        gt = parse_ground_truth(row.get('objects1'), row.get('objects2'))
        items.append({'index': i, 'gt': gt, 'objects1': row.get('objects1'), 'objects2': row.get('objects2')})
    print('Evaluation state updated.', flush=True)
    return (dataset, items)

def select_dataset_items(metadata_items: List[Dict]) -> List[Dict]:
    return list(metadata_items)

def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, sample_parallel: int=4, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, reasoning_method: str='baseline'):
    few_shot_k = int((few_shot_source or {}).get('k', 0) or 0)
    if result_path is None:
        suffix = f'-{few_shot_k}shot' if few_shot_k > 0 else ''
        result_path = os.path.join(output_dir, f'{model_key}-counting{suffix}.jsonl')
    final_result_path = str(result_path_for_reasoning(result_path, reasoning_method))
    os.makedirs(os.path.dirname(final_result_path) or '.', exist_ok=True)
    writer_lock = open(f'{final_result_path}.lock', 'a+', encoding='utf-8')
    try:
        try:
            fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is already writing {final_result_path}. Wait for it to finish, then rerun to resume its completed rows.') from exc
        return _query_model_on_dataset_locked(model_key=model_key, dataset_items=dataset_items, hf_dataset=hf_dataset, split_context=split_context, result_path=final_result_path, output_dir=output_dir, api_key=api_key, sample_parallel=sample_parallel, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, reasoning_method=reasoning_method)
    finally:
        writer_lock.close()

def _query_model_on_dataset_locked(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, sample_parallel: int=4, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, reasoning_method: str='baseline'):
    selected_items = select_dataset_items(dataset_items)
    few_shot_source = resolve_few_shot_source_for_resume(result_path, few_shot_source, dataset_items)
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    if MODEL_CONFIGS[model_key].get('model_kind') == 'lora' and prompt_metadata['few_shot_k'] > 0:
        raise ValueError('The promoted xView2 LoRA adapter is zero-shot only; --few-shot-k must be 0.')
    if few_shot_source and few_shot_source.get('legacy_all_raw_train'):
        for key in ('canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split'):
            prompt_metadata[key] = None
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': reasoning_method})
    print(f'[{model_key}] Starting, results -> {result_path}')
    completed_indices = clean_result_jsonl_for_resume(result_path)
    expected_resume_metadata = {**prompt_metadata, **result_model_metadata(model_key, MODEL_CONFIGS)}
    validate_resume_config(result_path, expected_resume_metadata)
    validate_resume_samples(result_path, dataset_items)
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None)
    validate_few_shot_resume_examples(result_path, few_shot_source, dataset_items)
    completed_target_indices = completed_indices & {item['index'] for item in selected_items}
    print('Evaluation state updated.', flush=True)
    pending_items = [item for item in selected_items if item['index'] not in completed_indices]
    total_pending = len(pending_items)
    sample_parallel = max(1, int(sample_parallel or 1))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_parallel = 1
    print('Evaluation state updated.', flush=True)
    if total_pending == 0:
        print(f'[{model_key}] Complete; no model checkpoint or API connection was opened.')
        return result_path
    query_fn = get_query_function(model_key, api_key=api_key)

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_image_bytes_list = [read_image_raw_bytes(hf_dataset, i, 'image1'), read_image_raw_bytes(hf_dataset, i, 'image2')]
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            task_prompt = USER_PROMPT
            query_prompt = effective_prompt(task_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

            def call_model(user_prompt: str):
                return query_images(query_fn=query_fn, raw_image_bytes_list=raw_image_bytes_list, model_key=model_key, sample_index=i, system=query_system, user=user_prompt, few_shot_examples=sample_few_shot_examples_list)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='xview2', context={}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer = capture_and_parse_json(answer_text)
            result_record = {'index': i, 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'gt': item['gt'], 'objects1': item.get('objects1'), 'objects2': item.get('objects2'), **result_model_metadata(model_key, MODEL_CONFIGS), 'response': response, 'prompt': task_prompt, 'parsed_answer': parsed_answer, **prompt_metadata}
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            if reasoning_method == 'cot':
                result_record['answer_text'] = answer_text
                result_record.update(cot_row_metadata(base_prompt=task_prompt, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            return (result_record, response_content)
        except Exception as e:
            print(f'[{model_key}] Error processing sample {i}: {e}')
            import traceback
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS) or is_non_retryable_api_error(e):
                raise
            return None
    if sample_parallel == 1:
        for item in pending_items:
            result = _process_single_sample(item)
            if result is None:
                continue
            result_record, response_content = result
            append_to_jsonl(result_record, result_path)
            completed_indices.add(result_record['index'])
            print(f"[{model_key}] #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['gt']}")
    else:
        executor = ThreadPoolExecutor(max_workers=sample_parallel)
        item_iterator = iter(pending_items)
        futures = {}

        def _submit_next_item() -> bool:
            try:
                next_item = next(item_iterator)
            except StopIteration:
                return False
            future = executor.submit(_process_single_sample, next_item)
            futures[future] = next_item['index']
            return True
        for _ in range(min(sample_parallel, total_pending)):
            _submit_next_item()
        try:
            while futures:
                future = next(as_completed(futures))
                futures.pop(future)
                result = future.result()
                if result is not None:
                    result_record, response_content = result
                    append_to_jsonl(result_record, result_path)
                    completed_indices.add(result_record['index'])
                    print(f"[{model_key}] #{result_record['index']}  Response: {response_content[:80]}...  GT: {result_record['gt']}")
                _submit_next_item()
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    target_indices = {item['index'] for item in selected_items}
    missing_indices = target_indices - completed_indices
    if missing_indices:
        raise IncompleteEvaluationError(model_key, result_path, missing_indices)
    print(f'[{model_key}] Done. Results saved to: {result_path}')
    return result_path

def capture_and_parse_count_fields(json_string):
    from evaluation.utils.scoring import count_fields
    return count_fields(json_string)


def capture_and_parse_json(json_string):
    response_text = str(json_string)
    candidates = [match.group(0) for match in re.finditer('\\{[^{}]*\\}', response_text, re.DOTALL)]
    for candidate in reversed(candidates):
        parsed = capture_and_parse_count_fields(candidate)
        if isinstance(parsed, dict) and set(parsed) == set(COUNT_KEYS) and all(value is not None for value in parsed.values()):
            return {key: parsed[key] for key in COUNT_KEYS}
    return -1


def parse_ground_truth_from_record(record: Dict) -> Dict[str, int]:
    objects1 = record.get('objects1')
    objects2 = record.get('objects2')
    saved_gt = record.get('gt') if isinstance(record.get('gt'), dict) else {}
    reconstructed = parse_ground_truth(objects1, objects2)
    count_before = count_pre_disaster_buildings(objects1)
    if count_before is None:
        try:
            count_before = int(saved_gt['count_before'])
        except (KeyError, TypeError, ValueError, OverflowError):
            count_before = reconstructed['count_before']
    truth = {'count_before': int(count_before)}
    can_rebuild_damage = isinstance(objects2, dict) and isinstance(objects2.get('subtype'), list)
    for key in COUNT_KEYS[1:]:
        if can_rebuild_damage:
            truth[key] = reconstructed[key]
        else:
            try:
                truth[key] = int(saved_gt.get(key, 0))
            except (TypeError, ValueError, OverflowError):
                truth[key] = 0
    return truth

def parse_answer_from_result_record(record: Dict) -> Any:
    answer_text = record.get('answer_text')
    if not isinstance(answer_text, str):
        answer_text = extract_response_content(record.get('response'))
    return capture_and_parse_count_fields(answer_text)

def parsed_count_for_key(model_answer: Any, key: str) -> int:
    value = model_answer.get(key) if isinstance(model_answer, dict) else None
    return value if value is not None else -1

def evaluation(result_path: str, ax=None) -> Dict:
    require_python_packages(['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'], 'Evaluation')
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from counting import calculate_counting_metrics, plot_scatter
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {result_path}')
    print(f"{'=' * 60}")
    result_json = pd.read_json(result_path, lines=True)
    result_json['model_answer'] = result_json.apply(lambda row: parse_answer_from_result_record(row.to_dict()), axis=1)
    result_json['gt'] = result_json.apply(lambda row: parse_ground_truth_from_record(row.to_dict()), axis=1)
    model_name = strip_result_suffix(result_path)
    if ax is not None:
        result_tmp = result_json.copy()
        result_tmp['parsed_response'] = result_tmp['model_answer'].apply(lambda x: parsed_count_for_key(x, 'count_before'))
        result_tmp['count'] = result_tmp['gt'].apply(lambda x: x['count_before'])
        result_tmp_no_refusal = result_tmp[result_tmp['parsed_response'] != -1].copy()
        plot_scatter(result_tmp, ax=ax)
        ax.set_title(model_name)
        return None
    sns.set(font_scale=2)
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
    all_metrics = {}
    for i, key in enumerate(['count_before', 'no_damage', 'minor_damage', 'major_damage', 'destroyed']):
        result_tmp = result_json.copy()
        result_tmp['parsed_response'] = result_tmp['model_answer'].apply(lambda x, k=key: parsed_count_for_key(x, k))
        result_tmp['count'] = result_tmp['gt'].apply(lambda x, k=key: x[k])
        result_tmp_no_refusal = result_tmp[result_tmp['parsed_response'] != -1].copy()
        rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_tmp, result_tmp_no_refusal)
        print(f'\n  {key}:')
        print(f'    MAE & MAE (No Refusal) & MAPE & MAPE (No Refusal) & R2 & R2 (No Refusal) & Refusal Rate')
        print(f'    {mae:.3f} & {mae_no_refusal:.3f} & {mape:.3f} & {mape_no_refusal:.3f} & {r2:.3f} & {r2_no_refusal:.3f} & {rr:.3f}')
        all_metrics[key] = {'mae': mae, 'mape': mape, 'r2': r2, 'refusal_rate': float(rr)}
        plot_scatter(result_tmp, axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(key, fontsize='x-large')
    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    output_pdf = result_path.replace('.jsonl', '.pdf')
    plt.savefig(output_pdf)
    print(f'\n  Plot saved to: {output_pdf}')
    count_before_metrics = all_metrics.get('count_before', {})
    primary_predictions = result_json['model_answer'].apply(lambda x: parsed_count_for_key(x, 'count_before'))
    rr_total = (primary_predictions == -1).mean()
    return {'model': model_name, 'mae': count_before_metrics.get('mae', float('nan')), 'mape': count_before_metrics.get('mape', float('nan')), 'r2': count_before_metrics.get('r2', float('nan')), 'refusal_rate': rr_total, 'all_metrics': all_metrics}

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR):
    require_python_packages(['pandas', 'matplotlib', 'seaborn', 'adjustText'], 'Model comparison plotting')
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    result_files = [f for f in os.listdir(result_dir) if is_result_file(f)]
    if not result_files:
        print(f'No result files found in {result_dir}')
        return
    print(f'Found {len(result_files)} result files')
    all_metrics = []
    for result_file in tqdm(result_files, desc='Evaluating models'):
        result_path = os.path.join(result_dir, result_file)
        try:
            metrics = evaluation(result_path, ax=None)
            if metrics is not None:
                model_key = strip_result_suffix(result_file)
                if model_key in MODEL_CONFIGS:
                    metrics['release_date'] = MODEL_CONFIGS[model_key]['release_date']
                    metrics['family'] = MODEL_CONFIGS[model_key]['family']
                    metrics['model_key'] = model_key
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
    write_shot_summary_json(all_metrics, os.path.join(result_dir, 'model_metrics_summary_all_shots.json'), dataset_name='xView2')
    zero_shot_metrics = [metrics for metrics in all_metrics if is_zero_shot_metrics(metrics, metrics.get('result_file'))]
    if not zero_shot_metrics:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return pd.DataFrame(all_metrics)
    df_metrics = pd.DataFrame(zero_shot_metrics)
    df_metrics['release_date'] = pd.to_datetime(df_metrics['release_date'])
    df_metrics = df_metrics.sort_values('release_date')
    print(f'\nCollected metrics for {len(df_metrics)} models')
    print(df_metrics[['model_key', 'release_date', 'mae', 'r2', 'refusal_rate']])
    from adjustText import adjust_text
    sns.set(font_scale=1.2, style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = dict(FAMILY_COLORS)
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
    ax1.set_title('MAE over Time (count_before)')
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
    ax2.set_title('R² Score over Time (count_before)')
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
    ax3.set_title('MAPE over Time (count_before)')
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
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle('xView2: Building Damage Assessment', fontsize=16, fontweight='bold')
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
    df_metrics.drop(columns=['all_metrics'], errors='ignore').to_csv(csv_path, index=False)
    print(f'Metrics summary saved to: {csv_path}\n')
    return df_metrics

def plot_damage_comparison(result_dir: str=DEFAULT_RESULT_DIR):
    require_python_packages(['pandas', 'numpy', 'matplotlib', 'seaborn', 'adjustText'], 'Damage comparison plotting')
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    print(f"\n{'=' * 60}")
    print(f'Generating damage category comparison plots')
    print(f"{'=' * 60}\n")
    result_files = [f for f in os.listdir(result_dir) if is_result_file(f)]
    if not result_files:
        print(f'No result files found in {result_dir}')
        return
    print(f'Found {len(result_files)} result files')
    all_results = []
    for result_file in tqdm(result_files, desc='Evaluating models'):
        result_path = os.path.join(result_dir, result_file)
        try:
            metrics = evaluation(result_path, ax=None)
            if metrics is not None and 'all_metrics' in metrics:
                model_key = strip_result_suffix(result_file)
                if model_key in MODEL_CONFIGS:
                    result = {'model_key': model_key, 'release_date': MODEL_CONFIGS[model_key]['release_date'], 'family': MODEL_CONFIGS[model_key]['family'], 'result_file': result_file, 'all_metrics': metrics['all_metrics']}
                    result.update(shot_from_metrics(metrics, result_file))
                    all_results.append(result)
                else:
                    print(f'Warning: Model {model_key} not found in MODEL_CONFIGS')
        except Exception as e:
            print(f'Error evaluating {result_file}: {e}')
            continue
    if not all_results:
        print('No valid metrics collected, cannot generate plots')
        return
    write_shot_summary_json(all_results, os.path.join(result_dir, 'damage_metrics_summary_all_shots.json'), dataset_name='xView2', summary_type='damage_category_comparison')
    zero_shot_results = [result for result in all_results if is_zero_shot_metrics(result, result.get('result_file'))]
    if not zero_shot_results:
        print('No zero-shot damage metrics collected; skipping damage comparison plots and zero-shot CSV.')
        return pd.DataFrame(all_results)
    all_results = zero_shot_results
    print(f'\nCollected metrics for {len(all_results)} models')
    damage_categories = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']
    damage_labels = {'no_damage': 'No Damage (0)', 'minor_damage': 'Minor Damage (1)', 'major_damage': 'Major Damage (2)', 'destroyed': 'Destroyed (3)'}
    metric_types = ['mae', 'r2', 'mape', 'refusal_rate']
    metric_labels = {'mae': 'MAE', 'r2': 'R² Score', 'mape': 'MAPE', 'refusal_rate': 'Refusal Rate'}
    from adjustText import adjust_text
    sns.set(font_scale=1.0, style='whitegrid')
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    colors = dict(FAMILY_COLORS)
    for row_idx, category in enumerate(damage_categories):
        for col_idx, metric in enumerate(metric_types):
            ax = axes[row_idx, col_idx]
            texts = []
            for family in PLOT_FAMILY_ORDER:
                dates = []
                values = []
                labels = []
                for result in all_results:
                    if result['family'] != family:
                        continue
                    cat_metrics = result['all_metrics'].get(category, {})
                    val = cat_metrics.get(metric)
                    if val is not None and (not (isinstance(val, float) and np.isnan(val))):
                        dates.append(pd.to_datetime(result['release_date']))
                        values.append(val)
                        labels.append(re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', result['model_key']))
                if dates:
                    sorted_data = sorted(zip(dates, values, labels))
                    dates_s, values_s, labels_s = zip(*sorted_data)
                    ax.plot(dates_s, values_s, marker='o', label=family, color=colors[family], linewidth=2, markersize=6)
                    for d, v, l in zip(dates_s, values_s, labels_s):
                        texts.append(ax.text(d, v, l, fontsize=8, color=colors[family]))
            ax.set_xlabel('Release Date')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f'{damage_labels[category]} - {metric_labels[metric]}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if texts:
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    fig.suptitle('xView2: Damage Category Assessment Performance over Time', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_pdf = os.path.join(result_dir, 'damage_comparison_over_time.pdf')
    output_png = os.path.join(result_dir, 'damage_comparison_over_time.png')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 60}")
    print(f'Damage comparison plots saved to:')
    print(f'  {output_pdf}')
    print(f'  {output_png}')
    print(f"{'=' * 60}\n")
    rows = []
    for result in all_results:
        for category in damage_categories:
            cat_metrics = result['all_metrics'].get(category, {})
            rows.append({'model_key': result['model_key'], 'release_date': result['release_date'], 'family': result['family'], 'category': category, **cat_metrics})
    df_damage = pd.DataFrame(rows)
    csv_path = os.path.join(result_dir, 'damage_metrics_summary.csv')
    df_damage.to_csv(csv_path, index=False)
    print(f'Damage metrics summary saved to: {csv_path}\n')
    return df_damage

def main():
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('xview2'):
        return
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate vision models on xView2 Building Damage Assessment dataset', epilog='\nAvailable models:\n  Individual: gpt-5.5, gemini-3.5-flash, Qwen2.5-VL-7B-Instruct, etc.\n  Families:   GPT-all, Gemini-all, Qwen-all\n  Default:    Test all GPT/Gemini models\n\nExamples:\n  python xview2.py --model gpt-5.5\n  python xview2.py --model GPT-all\n  python xview2.py --model gemini-3.5-flash --few-shot\n  python xview2.py --evaluate-only results/inference/xView2/gpt-5.5-counting.jsonl\n        ', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None, help='Model to test (model name, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL file path. Only valid for single-model runs.')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--plot-only', action='store_true', help='Only generate comparison plots from existing results (count_before)')
    parser.add_argument('--plot-damage', action='store_true', help='Generate damage category comparison plots from existing results')
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
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir)
        return
    if args.plot_damage:
        plot_damage_comparison(args.output_dir)
        return
    if args.evaluate_only:
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    models_to_test = get_models_to_test(args.model)
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        print_provider_summary()
    if args.output and len(models_to_test) != 1:
        raise ValueError('--output can only be used when testing a single model.')
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative.')
    for model_key in models_to_test:
        if MODEL_CONFIGS[model_key].get('model_kind') == 'lora' and few_shot_k > 0:
            raise ValueError('The promoted xView2 LoRA adapter is zero-shot only; --few-shot-k must be 0.')
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
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
            if api_key:
                key_display = safe_key_display(api_key)
                print(f'  {model_key:<35} -> API Key #{key_index_for_model(model_key, idx, MODEL_CONFIGS)} ({key_display})')
            else:
                print(f'  {model_key:<35} -> No API key')
        output_path = args.output if args.output and len(models_to_test) == 1 else None
        tasks.append((model_key, api_key, output_path))
    failed_models = []
    if num_parallel > 1 and len(models_to_test) > 1:
        print(f"\n{'=' * 60}")
        print(f'Starting {num_parallel} parallel threads...')
        print(f"{'=' * 60}\n")
        result_paths = []
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(run_single_model, model_key, api_key, output_path, args.output_dir, args.sample_parallel, few_shot_source, args.few_shot_seed if few_shot_source else None, args.reasoning_method): model_key for model_key, api_key, output_path in tasks}
            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    result_path = future.result()
                    if result_path is not None:
                        result_paths.append(result_path)
                        print(f'Completed: {model_key}')
                except Exception as e:
                    print(f'Failed: {model_key}: {e}')
                    failed_models.append((model_key, str(e)))
    else:
        print(f"\n{'=' * 60}")
        print(f'Running sequentially (use --parallel N for parallel execution)...')
        print(f"{'=' * 60}\n")
        result_paths = []
        for model_key, api_key, output_path in tqdm(tasks, desc='Testing models'):
            try:
                result_path = run_single_model(model_key, api_key, output_path, args.output_dir, args.sample_parallel, few_shot_source, args.few_shot_seed if few_shot_source else None, args.reasoning_method)
                result_paths.append(result_path)
            except Exception as e:
                print(f'Failed: {model_key}: {e}')
                failed_models.append((model_key, str(e)))
    print(f"\n{'=' * 60}")
    print(f'Evaluating {len(result_paths)} result files')
    print(f"{'=' * 60}\n")
    evaluation_failures = []
    for result_path in tqdm(result_paths, desc='Final evaluation'):
        try:
            evaluate_and_write_cot_metrics(evaluation, result_path)
        except Exception as e:
            print(f'Error evaluating {result_path}: {e}')
            evaluation_failures.append((result_path, str(e)))
    if len(result_paths) > 1:
        print(f'\nGenerating comparison plots...')
        plot_model_comparison(args.output_dir)
    if failed_models:
        summary = '; '.join((f'{model}: {message}' for model, message in failed_models))
        raise RuntimeError(f'One or more model evaluations are incomplete. Successful rows were checkpointed and the same command will resume them. Failures: {summary}')
    if evaluation_failures:
        summary = '; '.join((f'{path}: {message}' for path, message in evaluation_failures))
        raise RuntimeError(f'Inference completed, but result evaluation failed: {summary}')
    print(f"\n{'=' * 60}")
    print('All tasks completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_split_context = None

def run_single_model(model_key: str, api_key: str, result_path: Optional[str], output_dir: str, sample_parallel: int, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, reasoning_method: str='baseline'):
    return query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, split_context=_shared_split_context, result_path=result_path, output_dir=output_dir, api_key=api_key, sample_parallel=sample_parallel, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, reasoning_method=reasoning_method)
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
