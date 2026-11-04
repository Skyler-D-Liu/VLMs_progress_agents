"""Neon trees."""
import base64
import fcntl
import http.client
import io
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
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
LORA_TASK = 'neon-tree'
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
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from PIL import Image as PILImage

def parse_digit_response(x: str):
    replace_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    x = str(x).strip().lower()
    for old, new in replace_dict.items():
        x = re.sub(f'\\b{re.escape(old)}\\b', str(new), x)
    number = re.search('(?<![\\d,])(?<!\\d\\.)(?P<sign>-?)\\s*(?P<value>\\d{1,3}(?:,\\d{3})+|\\d+)(?![\\d.,])', x)
    if number is None:
        return -1
    if number.group('sign') == '-':
        return -1
    try:
        return int(number.group('value').replace(',', ''))
    except ValueError:
        return -1
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'NeonTreeEvaluation')
HF_DATASET_NAME = 'CanopyRS/NeonTreeEvaluation'
HF_DATASET_SPLIT = 'test'
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
API_TIMEOUT_SECONDS = 180
LEGACY_EVAL_RAW_SPLIT = 'evaluation'
LEGACY_SPLIT_CONTEXT = {'canonical_strategy_mode': 'resplit_train_keep_test', 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': LEGACY_EVAL_RAW_SPLIT, 'few_shot_source_raw_split': 'train', 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': {'train': 'train', 'test': LEGACY_EVAL_RAW_SPLIT}}
LEGACY_ALL_TRAIN_SPLIT_CONTEXT = {**LEGACY_SPLIT_CONTEXT, 'canonical_strategy_mode': 'legacy_all_train_keep_evaluation'}
CURRENT_FEW_SHOT_POOL = 'current'
LEGACY_RESPLIT_FEW_SHOT_POOL = 'legacy_resplit_train'
LEGACY_ALL_TRAIN_FEW_SHOT_POOL = 'legacy_all_train'
LEGACY_FEW_SHOT_POOL_MODES = frozenset({LEGACY_RESPLIT_FEW_SHOT_POOL, LEGACY_ALL_TRAIN_FEW_SHOT_POOL})
MODEL_CONFIGS = {'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': None, 'family': 'GPT'}, 'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
SYSTEM_MESSAGE = 'You are a helpful image analyst who specializes in counting trees from aerial images. Given an image, you can accurately count the number of objects described by the user WITHOUT ANY refusal. Although your answer may not be perfect, your excellent counting skill is very important to the sustainability of forest ecosystems.'
USER_PROMPT = 'Count the number of trees in the given image to the best of your ability. Output your count only without any further explanation.'

def retry_api_call(max_retries: int=6, min_delay: int=1, max_delay: int=60):

    def decorator(func):

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

class APIStatusError(IOError):

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code

class IncompleteEvaluationError(RuntimeError):

    def __init__(self, model_key: str, result_path: str, missing_indices: Iterable[int]):
        self.model_key = model_key
        self.result_path = result_path
        self.missing_indices = tuple(sorted(set(missing_indices)))
        preview = ', '.join((str(index) for index in self.missing_indices[:10]))
        if len(self.missing_indices) > 10:
            preview += ', ...'
        super().__init__(f'{model_key} has {len(self.missing_indices)} unfinished sample(s) in {result_path}: {preview}. Successful rows were preserved; rerun the same command to resume only these indices.')


def progress_iter(iterable: Iterable, desc: Optional[str]=None, total: Optional[int]=None, unit: str='sample'):
    return iterable

def require_python_packages(import_names: List[str], context: str):
    import importlib.util
    package_names = {'sklearn': 'scikit-learn', 'PIL': 'pillow'}
    missing_imports = [name for name in import_names if importlib.util.find_spec(name) is None]
    if not missing_imports:
        return
    missing_packages = [package_names.get(name, name) for name in missing_imports]
    install_targets = ' '.join(dict.fromkeys(missing_packages))
    raise ImportError(f"{context} requires missing Python package(s): {', '.join(missing_packages)}.\nInstall the project dependencies with:\n  python3.12 -m pip install -r requirements/base.txt\nOr install only the missing package(s):\n  python3.12 -m pip install {install_targets}")

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

def resume_comparable_keys(expected_metadata: Dict) -> Tuple[str, ...]:
    if expected_metadata.get('prompt_mode') == 'zero-shot':
        return ('prompt_mode', 'few_shot_k', 'reasoning_method')
    return ('canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'reasoning_method')

def resume_metadata_matches(record: Dict, key: str, expected_metadata: Dict) -> bool:
    expected_value = expected_metadata.get(key)
    if key in record:
        return record.get(key) == expected_value
    if key == 'reasoning_method':
        return expected_value is None
    if expected_metadata.get('prompt_mode') == 'zero-shot':
        if key == 'prompt_mode':
            return record.get('few_shot_k') in (None, 0)
        if key == 'few_shot_k':
            return record.get('prompt_mode') in (None, 'zero-shot')
    legacy_optional_keys = {'canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split'}
    legacy_strategy_modes = {LEGACY_SPLIT_CONTEXT['canonical_strategy_mode'], LEGACY_ALL_TRAIN_SPLIT_CONTEXT['canonical_strategy_mode']}
    if expected_metadata.get('canonical_strategy_mode') in legacy_strategy_modes and key in legacy_optional_keys:
        return True
    return False

def _preserve_before_resume_cleanup(path: str) -> str:
    backup_path = f'{path}.before-resume-cleanup'
    if not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f'Preserved original result file before cleanup: {backup_path}')
    return backup_path

def clean_result_jsonl_for_resume(path: str, expected_metadata: Dict) -> set:
    if not os.path.exists(path):
        return set()
    latest_by_index: Dict[int, Tuple[int, Dict]] = {}
    malformed_count = 0
    dropped_no_index = 0
    changed = False
    with open(path, 'r', encoding='utf-8') as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                malformed_count += 1
                changed = True
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                changed = True
                continue
            if not isinstance(record, dict):
                malformed_count += 1
                changed = True
                continue
            index = record.get('index')
            if isinstance(index, bool) or not isinstance(index, int):
                dropped_no_index += 1
                changed = True
                continue
            for key in resume_comparable_keys(expected_metadata):
                if not resume_metadata_matches(record, key, expected_metadata):
                    raise ValueError(f'Existing result file {path} has {key}={record.get(key)!r}, but this run expects {expected_metadata.get(key)!r}. The existing rows were left unchanged. Use the same run configuration or a different output file.')
            if index in latest_by_index:
                changed = True
            latest_by_index[index] = (line_number, record)
    cleaned_records = [item[1] for item in sorted(latest_by_index.values(), key=lambda x: x[0])]
    if changed:
        _preserve_before_resume_cleanup(path)
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(tmp_path, path)
        print('Evaluation state updated.', flush=True)
    return set(latest_by_index.keys())

def validate_resume_config(path: str, expected_metadata: Dict):
    if not os.path.exists(path):
        return
    comparable_keys = resume_comparable_keys(expected_metadata)
    for record in iter_jsonl(path):
        record_prompt_mode = record.get('prompt_mode')
        if expected_metadata.get('prompt_mode') == 'few-shot' and record_prompt_mode == 'few-shot' and (record.get('few_shot_sample_seed') is None):
            raise ValueError(f'Existing result file {path} contains few-shot rows without per-sample few_shot_sample_seed metadata. Use a different output directory or remove the old shared-example result file.')
        for key in comparable_keys:
            if not resume_metadata_matches(record, key, expected_metadata):
                raise ValueError(f'Existing result file {path} has {key}={record.get(key)!r}, but this run expects {expected_metadata.get(key)!r}.')

def append_to_jsonl(record: Dict, path: str):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

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

def detect_mime_type(raw_bytes: bytes) -> str:
    if raw_bytes[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if raw_bytes[:4] == b'\x89PNG':
        return 'image/png'
    if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/png'

def pil_image_to_png_data_uri(image: PILImage) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{b64}'

def raw_bytes_to_png_data_uri(raw_bytes: bytes) -> str:
    with PILImage.open(io.BytesIO(raw_bytes)) as image:
        image.load()
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        return pil_image_to_png_data_uri(image)


def query_images(query_fn: Callable, raw_bytes: bytes, model_key: str, sample_index: int, system: str, user: str, few_shot_examples: Optional[List[Dict]]=None) -> dict:
    return query_fn(image_url=raw_bytes_to_png_data_uri(raw_bytes), system=system, user=user, few_shot_examples=few_shot_examples, sample_index=sample_index)

def raw_bytes_to_resized_png_data_uri(raw_bytes: bytes, max_side: int) -> str:
    with PILImage.open(io.BytesIO(raw_bytes)) as image:
        image.load()
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        width, height = image.size
        largest_side = max(width, height)
        if largest_side > max_side:
            scale = max_side / float(largest_side)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, PILImage.LANCZOS)
        return pil_image_to_png_data_uri(image)

def image_part_for_gemini(image_url: str) -> Dict:
    if image_url.startswith('data:'):
        header, b64_data = image_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        return {'inlineData': {'mimeType': mime_type, 'data': b64_data}}
    return {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}

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
    if isinstance(image_value, PILImage.Image):
        buffer = io.BytesIO()
        image_value.save(buffer, format='PNG')
        return buffer.getvalue()
    raise KeyError("No 'image' field with raw bytes found in data item")

def dataset_image_is_readable(hf_dataset, index: int) -> bool:
    try:
        raw_bytes = get_raw_image_bytes(hf_dataset[index])
        with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
            pil_img.verify()
        return True
    except Exception:
        return False

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

def extract_json_candidate(response_text: str) -> Optional[str]:
    text = '' if response_text is None else str(response_text)
    fenced = re.search('```(?:json)?\\s*(.*?)\\s*```', text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    obj = re.search('\\{[^{}]*\\}', text, re.DOTALL)
    if obj:
        return obj.group(0)
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return stripped
    return None

def normalize_count_value(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        normalized = int(value)
        return normalized if normalized >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
            if number.is_integer() and number >= 0:
                return int(number)
        except Exception:
            pass
        try:
            normalized = int(text)
            return normalized if normalized >= 0 else None
        except Exception:
            return None
    return None

def parse_model_response_to_count(response) -> int:
    from evaluation.utils.scoring import scalar_count
    value = scalar_count(extract_response_content(response))
    return value if value is not None else -1

def get_ground_truth_count(data_item: Dict) -> int:
    objects = data_item.get('objects')
    if isinstance(objects, dict):
        bbox = objects.get('bbox')
        if isinstance(bbox, list):
            return len(bbox)
    count = data_item.get('count')
    normalized = normalize_count_value(count)
    if normalized is not None:
        return normalized
    label = data_item.get('label')
    normalized = normalize_count_value(label)
    if normalized is not None:
        return normalized
    return -1

def normalize_neon_objects(raw_objects, split: str, index: int) -> Tuple[Dict, int]:
    if not isinstance(raw_objects, Mapping):
        raise ValueError(f'Invalid annotations for split={split!r}, index={index}: expected a mapping.')
    objects = dict(raw_objects)
    bbox = objects.get('bbox')
    if not isinstance(bbox, list):
        raise ValueError(f'Invalid annotations.bbox for split={split!r}, index={index}: expected a list.')
    if 'category' in objects:
        objects['categories'] = objects.pop('category')
    elif 'categories' not in objects:
        raise ValueError(f'Invalid annotations for split={split!r}, index={index}: missing category labels.')
    return (objects, len(bbox))

def normalize_neon_dataset(dataset, split: str):
    column_names = set(dataset.column_names)
    if 'annotations' in column_names:
        if 'objects' in column_names:
            raise ValueError(f"Dataset split {split!r} contains both 'annotations' and 'objects'.")
        dataset = dataset.rename_column('annotations', 'objects')
        column_names.remove('annotations')
        column_names.add('objects')
    elif 'objects' not in column_names:
        raise ValueError(f"Dataset split {split!r} is missing the 'annotations' column.")
    if 'tile_name' in column_names:
        if 'path' in column_names:
            raise ValueError(f"Dataset split {split!r} contains both 'tile_name' and 'path'.")
        dataset = dataset.rename_column('tile_name', 'path')
        column_names.remove('tile_name')
        column_names.add('path')
    elif 'path' not in column_names:
        raise ValueError(f"Dataset split {split!r} is missing the 'tile_name' column.")
    normalized_objects = []
    counts = []
    for index, raw_objects in enumerate(dataset['objects']):
        objects, count = normalize_neon_objects(raw_objects, split=split, index=index)
        normalized_objects.append(objects)
        counts.append(count)
    dataset = dataset.remove_columns('objects').add_column('objects', normalized_objects)
    if 'count' in dataset.column_names:
        dataset = dataset.remove_columns('count')
    return dataset.add_column('count', counts)

def get_user_prompt() -> str:
    return USER_PROMPT

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

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str, max_side: int=1536, require_positive_count: bool=True):
    valid_candidates = []
    for item in metadata_items:
        answer = get_ground_truth_count(item)
        if answer < 0 or (require_positive_count and answer == 0):
            continue
        if not dataset_image_is_readable(hf_dataset, item['index']):
            print(f"Skipping invalid few-shot candidate #{item['index']}: image is not readable")
            continue
        valid_candidates.append({'index': item['index'], 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'image_url': raw_bytes_to_resized_png_data_uri(get_raw_image_bytes(hf_dataset[item['index']]), max_side=max_side), 'user_prompt': USER_PROMPT, 'answer': str(answer)})
    if len(valid_candidates) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}'. Found only {len(valid_candidates)} examples with readable images and positive counts.")
    return {'hf_dataset': hf_dataset, 'candidates': valid_candidates, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'max_side': max_side}

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidate_pool = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidate_pool = [candidate for candidate in candidate_pool if build_candidate_identity(candidate) != current_identity]
    if len(candidate_pool) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidate_pool)))
    sampled = random.Random(seed).sample(candidate_pool, few_shot_source['k'])
    examples = []
    for candidate in sampled:
        examples.append({'index': candidate['index'], 'split': candidate['split'], 'raw_split': candidate['raw_split'], 'canonical_split': candidate['canonical_split'], 'image_url': candidate['image_url'], 'user_prompt': candidate['user_prompt'], 'answer': candidate['answer']})
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


def resolve_canonical_pools(few_shot_k: int) -> Dict:
    raw_splits = ('train', 'validation', 'test')
    split_map = discover_available_splits_from_names(raw_splits)
    strategy = resolve_split_strategy(split_map)
    datasets_by_raw_split = {}
    metadata_by_raw_split = {}
    split_lengths = {}
    for raw_split in raw_splits:
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
        few_shot_items = select_metadata_items_by_indices(metadata_by_raw_split[train_assignment['raw_split']], train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(datasets_by_raw_split[train_assignment['raw_split']], few_shot_items, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train')
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw {} | Eval: canonical test <- raw {} | split seed={}'.format(train_assignment['raw_split'], eval_assignment['raw_split'], DEFAULT_SPLIT_SEED))
    return {'evaluation_dataset': datasets_by_raw_split[eval_assignment['raw_split']], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def get_query_function(model_key: str, api_key: str=None) -> Callable:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {model_key}')
    config = MODEL_CONFIGS[model_key]
    if config['api_type'] == 'hf_local':
        qwen_model_path = _QWEN_MODEL_PATHS.get(model_key)
        if not qwen_model_path:
            raise ValueError(f'Model {model_key} requires --qwen-model-path or --llava-model-path before local inference.')
        return make_qwen_query_callable(model_key, qwen_model_path)
    api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key)
    if not api_key:
        raise ValueError('No API key available for this model family. Please set OPENAI_API_KEY or GEMINI_API_KEY in .env.')
    if config['api_type'] == 'openai':
        return lambda image_url, system, user, few_shot_examples=None, sample_index=None: query_openai(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
    if config['api_type'] == 'google_native':
        return lambda image_url, system, user, few_shot_examples=None, sample_index=None: query_gemini(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, sample_index=sample_index)
    raise ValueError(f"Unknown API type: {config['api_type']}")

def get_models_to_test(model_arg: str) -> List[str]:
    return expand_model_selection(model_arg, MODEL_CONFIGS)

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
    candidates = result.get('candidates') or [{}]
    content_obj = candidates[0].get('content') or {}
    parts = content_obj.get('parts') or []
    content = ''
    for part in parts:
        if part.get('thought', False):
            continue
        content = part.get('text', '')
        break
    if not content and parts:
        content = parts[-1].get('text', '')
    return {'model': config['model_name'], 'choices': [{'message': {'content': content}}], 'original_response': result}

def download_and_prepare_dataset(split: str=HF_DATASET_SPLIT):
    from datasets import Image, DownloadConfig, load_dataset
    token = os.environ.get('HF_TOKEN')
    cache_dir = str(HF_CACHE_DIR)
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
    dataset = normalize_neon_dataset(dataset, split=split)
    print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c != 'image'])
    selected_indices = select_evaluation_indices(len(metadata_ds))
    print('Evaluation state updated.', flush=True)
    items = []
    for i in progress_iter(selected_indices, desc='Building metadata index'):
        row = metadata_ds[i]
        count = get_ground_truth_count(row)
        if count < 0:
            raise ValueError(f'Unable to derive ground-truth count for split={split!r}, index={i}.')
        items.append({'index': i, 'path': row.get('path'), 'objects': row.get('objects'), 'count': count, 'height': row.get('height'), 'width': row.get('width')})
    print('Evaluation state updated.', flush=True)
    return (dataset, items)

def parse_result_file_name(result_file: str) -> Optional[Tuple[str, str, int]]:
    stem = _remove_suffix(result_file, '.jsonl')
    match = re.match('^(?P<model>.+)-counting(?:-(?P<k>\\d+)shot)?(?:-CoT)?$', stem)
    if not match:
        return None
    model_key = match.group('model')
    few_shot_k = int(match.group('k')) if match.group('k') else 0
    prompt_mode = 'few-shot' if few_shot_k > 0 else 'zero-shot'
    return (model_key, prompt_mode, few_shot_k)

def result_path_for_run(output_dir: str, model_key: str, few_shot_k: int, reasoning_method: str) -> str:
    suffix = f'-{few_shot_k}shot' if few_shot_k > 0 else ''
    baseline_path = os.path.join(output_dir, f'{model_key}-counting{suffix}.jsonl')
    return str(result_path_for_reasoning(baseline_path, reasoning_method))


def validate_few_shot_resume_examples(path: str, few_shot_source: Optional[Dict], dataset_items: List[Dict]) -> None:
    if not few_shot_source or not os.path.exists(path):
        return
    items_by_index = {item['index']: item for item in dataset_items}
    for record in iter_jsonl(path):
        index = record.get('index')
        if index not in items_by_index:
            continue
        sample_seed = record.get('few_shot_sample_seed')
        recorded_indices = record.get('few_shot_indices')
        if sample_seed is None or not isinstance(recorded_indices, list):
            continue
        expected_examples = sample_few_shot_examples(few_shot_source, sample_seed, current_item=items_by_index[index])
        expected_indices = [example['index'] for example in expected_examples]
        if recorded_indices != expected_indices:
            raise ValueError(f'Existing Few-shot result {path} cannot be resumed with the resolved historical candidate pool: index={index} recorded demonstrations {recorded_indices}, expected {expected_indices}. Existing rows were left unchanged.')

def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, sample_parallel: int=4, reasoning_method: str='baseline'):
    few_shot_k = int((few_shot_source or {}).get('k', 0) or 0)
    if result_path is None:
        final_result_path = result_path_for_run(output_dir, model_key, few_shot_k, reasoning_method)
    else:
        final_result_path = str(result_path_for_reasoning(result_path, reasoning_method))
    os.makedirs(os.path.dirname(final_result_path) or '.', exist_ok=True)
    writer_lock = open(f'{final_result_path}.lock', 'a+', encoding='utf-8')
    try:
        try:
            fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is already writing {final_result_path}. Wait for it to finish, then rerun to resume its completed rows.') from exc
        return _query_model_on_dataset_locked(model_key=model_key, dataset_items=dataset_items, hf_dataset=hf_dataset, split_context=split_context, result_path=final_result_path, output_dir=output_dir, api_key=api_key, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, sample_parallel=sample_parallel, reasoning_method=reasoning_method)
    finally:
        writer_lock.close()

def _query_model_on_dataset_locked(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, output_dir: str=DEFAULT_RESULT_DIR, api_key: str=None, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, sample_parallel: int=4, reasoning_method: str='baseline'):
    prompt_metadata = few_shot_metadata(few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    if MODEL_CONFIGS[model_key].get('model_kind') == 'lora' and prompt_metadata['few_shot_k'] > 0:
        raise ValueError('The promoted NEON LoRA adapter is zero-shot only; --few-shot-k must be 0.')
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    if reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': reasoning_method})
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None)
    if result_path is None:
        os.makedirs(output_dir, exist_ok=True)
        result_path = result_path_for_run(output_dir, model_key, prompt_metadata['few_shot_k'], reasoning_method)
    else:
        result_path = str(result_path_for_reasoning(result_path, reasoning_method))
    completed_indices = clean_result_jsonl_for_resume(result_path, prompt_metadata)
    validate_resume_config(result_path, prompt_metadata)
    validate_few_shot_resume_examples(result_path, few_shot_source, dataset_items)
    target_indices = {item['index'] for item in dataset_items}
    completed_target_indices = completed_indices & target_indices
    print(f'[{model_key}] Starting, results -> {result_path}')
    print('Evaluation state updated.', flush=True)
    pending_items = [item for item in dataset_items if item['index'] not in completed_indices]
    total_pending = len(pending_items)
    total_target = len(target_indices)
    sample_parallel = max(1, int(sample_parallel or 1))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_parallel = 1
    print('Evaluation state updated.', flush=True)
    if not pending_items:
        print(f'[{model_key}] Complete; no model checkpoint or API connection was opened.')
        return result_path
    query_fn = get_query_function(model_key, api_key=api_key)

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[i])
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            task_prompt = get_user_prompt()
            query_prompt = effective_prompt(task_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)

            def call_model(user_prompt: str):
                return query_images(query_fn=query_fn, raw_bytes=raw_bytes, model_key=model_key, sample_index=i, system=query_system, user=user_prompt, few_shot_examples=sample_few_shot_examples_list)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='neon', context={}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            result_record = {'index': i, 'path': item.get('path'), 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'objects': item.get('objects'), 'count': item.get('count'), 'height': item.get('height'), 'width': item.get('width'), **result_model_metadata(model_key, MODEL_CONFIGS), 'response': response, 'prompt': task_prompt, 'parsed_answer': parse_model_response_to_count(answer_text)}
            result_record.update(prompt_metadata)
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            if reasoning_method == 'cot':
                result_record['answer_text'] = answer_text
                result_record.update(cot_row_metadata(base_prompt=task_prompt, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            return (result_record, response_content)
        except Exception as exc:
            print(f'[{model_key}] Error processing sample {i}: {exc}')
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS):
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
            print('Evaluation state updated.', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=sample_parallel) as pool:
            futures = {pool.submit(_process_single_sample, item): item['index'] for item in pending_items}
            for future in progress_iter(as_completed(futures), desc=f'{model_key} samples', unit='sample'):
                result = future.result()
                if result is None:
                    continue
                result_record, response_content = result
                append_to_jsonl(result_record, result_path)
                completed_indices.add(result_record['index'])
                print('Evaluation state updated.', flush=True)
    missing_indices = target_indices - completed_indices
    if missing_indices:
        raise IncompleteEvaluationError(model_key, result_path, missing_indices)
    print(f'[{model_key}] Done. Results saved to: {result_path}')
    return result_path

def evaluation(result_path: str, ax=None) -> Optional[Dict]:
    require_python_packages(['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'], 'Evaluation')
    import pandas as pd
    import matplotlib.pyplot as plt
    from counting import calculate_counting_metrics, plot_scatter
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {result_path}')
    print(f"{'=' * 60}")
    result_json = pd.read_json(result_path, lines=True)
    parsed_result = parse_result_file_name(os.path.basename(result_path))
    if parsed_result:
        base_model_key, filename_prompt_mode, filename_few_shot_k = parsed_result
    else:
        base_model_key, filename_prompt_mode, filename_few_shot_k = (_remove_suffix(os.path.basename(result_path), '.jsonl'), 'zero-shot', 0)
    result_json['model_response'] = result_json.apply(lambda row: row['answer_text'] if isinstance(row.get('answer_text'), str) else extract_response_content(row['response']), axis=1)

    def _scored_count(row):
        return parse_model_response_to_count(row['model_response'])
    result_json['parsed_response'] = result_json.apply(_scored_count, axis=1)
    original_count = result_json['count'].copy() if 'count' in result_json.columns else None
    if 'objects' in result_json.columns:
        result_json['count'] = result_json['objects'].apply(lambda x: len(x['bbox']) if isinstance(x, dict) and 'bbox' in x else None)
    elif 'count' not in result_json.columns:
        result_json['count'] = None
    if result_json['count'].isna().any():
        fallback_count = original_count
        if fallback_count is None:
            for count_column in ('label', 'gt_count', 'true_count', 'ground_truth', 'target', 'answer'):
                if count_column in result_json.columns:
                    fallback_count = result_json[count_column]
                    break
        if fallback_count is not None:
            result_json['count'] = result_json['count'].fillna(fallback_count)
    result_json['count'] = pd.to_numeric(result_json['count'], errors='coerce').fillna(-1).astype(int)
    result_json_no_refusal = result_json[result_json['parsed_response'] != -1].copy()
    rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal) = calculate_counting_metrics(result_json, result_json_no_refusal)
    prompt_values = result_json['prompt_mode'].dropna() if 'prompt_mode' in result_json.columns else []
    prompt_mode = str(prompt_values.iloc[0]) if len(prompt_values) else filename_prompt_mode
    few_shot_values = result_json['few_shot_k'].dropna() if 'few_shot_k' in result_json.columns else []
    few_shot_k = int(few_shot_values.iloc[0]) if len(few_shot_values) else filename_few_shot_k
    reasoning_method = str(result_json['reasoning_method'].dropna().iloc[0]) if 'reasoning_method' in result_json.columns and (not result_json['reasoning_method'].dropna().empty) else 'cot' if os.path.basename(result_path).endswith('-CoT.jsonl') else 'baseline'
    print(f'\nResults for {os.path.basename(result_path)}:')
    print('Evaluation state updated.', flush=True)
    print('Evaluation state updated.', flush=True)
    print(f'  Refusal Rate: {rr:.2%}')
    print(f'  Prompt mode: {prompt_mode}')
    if prompt_mode == 'few-shot':
        print(f'  Few-shot k: {few_shot_k}')
    print(f'\nMetrics (failed answers counted as zero):')
    print(f'  MAE: {mae:.3f}')
    print(f'  MAPE: {mape:.3f}')
    print(f'  R²: {r2:.3f}')
    print(f'\nLaTeX format:')
    print(f'MAE & MAPE & R² & Refusal Rate')
    print(f'{mae:.3f} & {mape:.3f} & {r2:.3f} & {rr:.2f}')
    try:
        plot_scatter(result_json, ax=ax)
        if ax:
            ax.set_title(_remove_suffix(os.path.basename(result_path), '.jsonl'))
    except Exception as exc:
        print(f'Warning: Could not generate scatter plot: {exc}')
    if result_path.endswith('.jsonl'):
        plt.savefig(result_path.replace('.jsonl', '.pdf'))
        plt.savefig(result_path.replace('.jsonl', '.png'))
    return {'model': _remove_suffix(os.path.basename(result_path), '.jsonl'), 'base_model_key': base_model_key, 'prompt_mode': prompt_mode, 'few_shot_k': few_shot_k, 'reasoning_method': reasoning_method, 'mae': mae, 'mape': mape, 'r2': r2, 'refusal_rate': rr}

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR):
    require_python_packages(['pandas', 'matplotlib', 'seaborn'], 'Model comparison plotting')
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    if not os.path.isdir(result_dir):
        print(f'No result directory found: {result_dir}')
        return
    result_files = [f for f in os.listdir(result_dir) if re.search('-counting(?:-\\d+shot)?(?:-CoT)?\\.jsonl$', f)]
    if not result_files:
        print(f'No result files found in {result_dir}')
        return
    all_metrics = []
    for result_file in progress_iter(result_files, desc='Evaluating models', total=len(result_files), unit='file'):
        result_path = os.path.join(result_dir, result_file)
        try:
            metrics = evaluation(result_path)
            if metrics is None:
                continue
            base_model_key = metrics['base_model_key']
            if base_model_key not in MODEL_CONFIGS:
                print(f'Warning: Model {base_model_key} not found in MODEL_CONFIGS')
                continue
            metrics['release_date'] = MODEL_CONFIGS[base_model_key]['release_date']
            metrics['family'] = MODEL_CONFIGS[base_model_key]['family']
            metrics['base_model_key'] = base_model_key
            metrics['result_file'] = result_file
            metrics.update(shot_from_metrics(metrics, result_file))
            if metrics.get('reasoning_method') == 'cot':
                metrics['shot_label'] = 'CoT'
            all_metrics.append(metrics)
        except Exception as exc:
            print(f'Error evaluating {result_file}: {exc}')
    if not all_metrics:
        print('No valid metrics collected, cannot generate plots')
        return
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics['release_date'] = pd.to_datetime(df_metrics['release_date'])
    df_metrics = df_metrics.sort_values(['release_date', 'base_model_key', 'shot_label'])
    write_shot_summary_json(all_metrics, os.path.join(result_dir, 'model_metrics_summary_all_shots.json'), dataset_name='NEON Trees')
    print(f'\nCollected metrics for {len(df_metrics)} experiments')
    print(df_metrics[['result_file', 'release_date', 'mae', 'r2', 'refusal_rate']])
    colors = FAMILY_COLORS
    plot_configs = [{'metric': 'mae', 'ylabel': 'MAE', 'title': 'MAE over Time'}, {'metric': 'r2', 'ylabel': 'R² Score', 'title': 'R² Score over Time'}, {'metric': 'mape', 'ylabel': 'MAPE', 'title': 'MAPE over Time'}, {'metric': 'refusal_rate', 'ylabel': 'Refusal Rate', 'title': 'Refusal Rate over Time'}]
    df_shot = df_metrics[df_metrics['shot_label'] == 'zero-shot'].copy()
    if df_shot.empty:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return df_metrics
    sns.set(font_scale=1.2, style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for pc, ax in zip(plot_configs, axes.flat):
        for family in PLOT_FAMILY_ORDER:
            df_family = df_shot[df_shot['family'] == family].sort_values('release_date')
            if df_family.empty:
                continue
            color = colors[family]
            ax.plot(df_family['release_date'], df_family[pc['metric']], marker='o', label=family, color=color, linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                ax.annotate(row['base_model_key'], (row['release_date'], row[pc['metric']]), textcoords='offset points', xytext=(5, 5), fontsize=10, color=color)
        ax.set_xlabel('Release Date')
        ax.set_ylabel(pc['ylabel'])
        ax.set_title(pc['title'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.suptitle('NEON Trees: Tree Counting (zero-shot)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path_pdf = os.path.join(result_dir, 'model_comparison_over_time.pdf')
    output_path_png = os.path.join(result_dir, 'model_comparison_over_time.png')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    csv_path = os.path.join(result_dir, 'model_metrics_summary.csv')
    df_shot.to_csv(csv_path, index=False)
    print(f'\nSaved zero-shot comparison outputs:')
    print(f'  {output_path_pdf}')
    print(f'  {output_path_png}')
    print(f'  {csv_path}')
    return df_metrics

def list_models(qwen_finetuned_root: Optional[str]=None, llava_model_root: Optional[str]=None):
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
            print(format_qwen_model_listing(name, config, qwen_finetuned_root))
    print('\nLLaVA Models:')
    for name, config in MODEL_CONFIGS.items():
        if config['family'] == 'LLaVA':
            print(format_llava_model_listing(name, config, llava_model_root))
    print('\nModel families:')
    print('  GPT-all    - Test all GPT models')
    print('  Gemini-all - Test all Gemini models')
    print('  Qwen-all   - Test all base Qwen models (excludes LoRA)')
    print('  LLaVA-all  - Test all LLaVA models')
    print('  (default)  - Test all GPT/Gemini models')
    print_runtime_model_summary(list(MODEL_CONFIGS.keys()), MODEL_CONFIGS)

def main():
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('neon-trees'):
        return
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate vision models on NEON Tree Counting dataset', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='\nExamples:\n  python neon_trees.py --model gpt-5.5\n  python neon_trees.py --model GPT-all\n  python neon_trees.py --few-shot\n  python neon_trees.py --few-shot-k 3\n  python neon_trees.py --evaluate-only results/inference/NeonTreeEvaluation/gpt-5.5-counting.jsonl\n  python neon_trees.py --plot-only\n        ')
    parser.add_argument('--model', type=str, default=None, help='Model to test (model key, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--plot-only', action='store_true', help='Only generate comparison plots from existing results')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel threads (default: auto-detect)')
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
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative.')

    os.makedirs(args.output_dir, exist_ok=True)
    if args.list_models:
        list_models(args.qwen_finetuned_root, args.llava_model_path)
        return
    if args.plot_only:
        plot_model_comparison(args.output_dir)
        return
    if args.evaluate_only:
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        if MODEL_CONFIGS[model_key].get('model_kind') == 'lora' and few_shot_k > 0:
            raise ValueError('The promoted NEON LoRA adapter is zero-shot only; --few-shot-k must be 0.')
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        print_provider_summary()
    pools = resolve_canonical_pools(few_shot_k)
    pools_by_model = {model_key: pools for model_key in models_to_test}
    print(f"\n{'=' * 60}")
    print(f'Models to test: {len(models_to_test)}')
    print_runtime_model_summary(models_to_test, MODEL_CONFIGS)
    print(f"Prompt mode: {('few-shot' if few_shot_k > 0 else 'zero-shot')}")
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
        model_pools = pools_by_model[model_key]
        if is_local_model(model_key, MODEL_CONFIGS):
            api_key = None
            print(f'  {model_key:<35} -> local checkpoint: {_QWEN_MODEL_PATHS[model_key]}')
        else:
            api_key = key_for_model_index(model_key, idx, MODEL_CONFIGS)
            key_display = safe_key_display(api_key)
            print(f'  {model_key:<35} -> API Key #{key_index_for_model(model_key, idx, MODEL_CONFIGS)} ({key_display})')
        tasks.append((model_key, api_key, args.output_dir, model_pools['few_shot_source'], args.few_shot_seed if model_pools['few_shot_source'] else None, args.sample_parallel, args.reasoning_method, model_pools['evaluation_items'], model_pools['evaluation_dataset'], model_pools['split_context']))
    failures = []
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
                except Exception as exc:
                    print(f'Failed: {model_key}: {exc}')
                    failures.append((model_key, exc))
    else:
        print(f"\n{'=' * 60}")
        print(f'Running sequentially...')
        print(f"{'=' * 60}\n")
        result_paths = []
        for task in progress_iter(tasks, desc='Testing models', total=len(tasks), unit='model'):
            try:
                result_paths.append(run_single_model(task))
            except Exception as exc:
                failures.append((task[0], exc))
    print(f"\n{'=' * 60}")
    print(f'Evaluating {len(result_paths)} result files')
    print(f"{'=' * 60}\n")
    evaluation_failures = []
    for result_path in progress_iter(result_paths, desc='Final evaluation', total=len(result_paths), unit='file'):
        try:
            evaluate_and_write_cot_metrics(evaluation, result_path)
        except Exception as exc:
            print(f'Error evaluating {result_path}: {exc}')
            evaluation_failures.append((result_path, exc))
    if len(result_paths) > 1:
        print(f'\nGenerating comparison plots...')
        try:
            plot_model_comparison(args.output_dir)
        except ImportError as exc:
            print(f'Skipping comparison plots: {exc}')
    if failures or evaluation_failures:
        details = [f'{model}: {error}' for model, error in failures]
        details.extend((f'{path}: {error}' for path, error in evaluation_failures))
        raise RuntimeError('NEON evaluation did not finish cleanly. Completed JSONL rows remain available for resume.\n' + '\n'.join(details))
    print(f"\n{'=' * 60}")
    print('All requested NEON tasks completed with full target coverage.')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")

def run_single_model(args_tuple):
    model_key, api_key, output_dir, few_shot_source, few_shot_seed, sample_parallel, reasoning_method, dataset_items, hf_dataset, split_context = args_tuple
    try:
        return query_model_on_dataset(model_key=model_key, dataset_items=dataset_items, hf_dataset=hf_dataset, split_context=split_context, result_path=None, output_dir=output_dir, api_key=api_key, few_shot_source=few_shot_source, few_shot_seed=few_shot_seed, sample_parallel=sample_parallel, reasoning_method=reasoning_method)
    except Exception as exc:
        print(f'\n[{model_key}] Error: {exc}')
        traceback.print_exc()
        raise
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
