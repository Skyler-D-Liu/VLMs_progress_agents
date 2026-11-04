"""Dior rsvg localization."""
import sys
import os
from pathlib import Path
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
UTILS_DIR = TASKS_DIR.parent / 'utils'
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(UTILS_DIR))
from vleo_runtime.localization import localization_prompt
from cot_prompting import REASONING_METHODS, call_with_reasoning, cot_row_metadata, evaluate_and_write_cot_metrics, effective_prompt, effective_system_message, local_generation_max_new_tokens, result_path_for_reasoning, validate_cot_run
import benchmark_environment as benchmark_environment_module
from benchmark_environment import APIStatusError, RESULTS_ROOT, api_host_for_model, api_keys_for_model, api_keys_for_models, first_api_key_for_model, load_benchmark_environment, post_gemini_json, post_openai_json, print_model_key_summary, print_provider_summary, random_api_key_for_model
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
LORA_TASK = 'dior-rsvg'
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
import json
import os
import random
import re
import math
import shutil
import http.client
import base64
import io
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, Callable, List, Dict, Any, Tuple
import pandas as pd
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch
from torchvision.ops import box_iou
from detection import get_mean_centroid_distance

def load_completed_indices(path):
    return clean_result_jsonl_for_resume(path)

def append_to_jsonl(record, path):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
DEFAULT_RESULT_DIR = str(RESULTS_ROOT / 'DIOR-RSVG')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
DEFAULT_SAMPLE_WORKERS = 8
HF_DATASET_NAME = 'danielz01/DIOR-RSVG'
HF_DATASET_SPLIT = 'test'
DIOR_RSVG_ALL_SPLITS = ('test', 'train', 'val')
HF_DATASET_URL = f'https://huggingface.co/datasets/{HF_DATASET_NAME}'
LOCAL_DIOR_RSVG_DIR_ENV = 'DIOR_RSVG_DIR'
MODEL_CONFIGS = {'gpt-6-astra': {'api_type': 'openai', 'model_name': 'gpt-6-astra', 'release_date': None, 'family': 'GPT'}, 'gpt-5.5': {'api_type': 'openai', 'model_name': 'gpt-5.5', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.4': {'api_type': 'openai', 'model_name': 'gpt-5.4', 'release_date': '2026-06-26', 'family': 'GPT'}, 'gpt-5.2-2025-12-11': {'api_type': 'openai', 'model_name': 'gpt-5.2-2025-12-11', 'release_date': '2025-12-11', 'family': 'GPT'}, 'gpt-5.1-2025-11-13': {'api_type': 'openai', 'model_name': 'gpt-5.1-2025-11-13', 'release_date': '2025-11-13', 'family': 'GPT'}, 'gpt-5-2025-08-07': {'api_type': 'openai', 'model_name': 'gpt-5-2025-08-07', 'release_date': '2025-08-07', 'family': 'GPT'}, 'o3-2025-04-16': {'api_type': 'openai', 'model_name': 'o3-2025-04-16', 'release_date': '2025-04-16', 'family': 'GPT'}, 'gpt-4.1-2025-04-14': {'api_type': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'release_date': '2025-04-14', 'family': 'GPT'}, 'o1-2024-12-17': {'api_type': 'openai', 'model_name': 'o1-2024-12-17', 'release_date': '2024-12-17', 'family': 'GPT'}, 'gpt-4o-2024-08-06': {'api_type': 'openai', 'model_name': 'gpt-4o-2024-08-06', 'release_date': '2024-08-06', 'family': 'GPT'}, 'gpt-4-turbo-2024-04-09': {'api_type': 'openai', 'model_name': 'gpt-4-turbo-2024-04-09', 'release_date': '2024-04-09', 'family': 'GPT'}, 'gemini-3.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-3.5-flash', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3.1-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3.1-pro-preview', 'release_date': '2026-02-01', 'family': 'Gemini'}, 'gemini-3.1-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-3.1-flash-lite', 'release_date': '2026-05-01', 'family': 'Gemini'}, 'gemini-3-flash-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-flash-preview', 'release_date': '2025-12-01', 'family': 'Gemini'}, 'gemini-3-pro-preview': {'api_type': 'google_native', 'model_name': 'gemini-3-pro-preview', 'release_date': '2025-11-01', 'family': 'Gemini'}, 'gemini-2.5-flash-lite': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash-lite', 'release_date': '2025-09-01', 'family': 'Gemini'}, 'gemini-2.5-pro': {'api_type': 'google_native', 'model_name': 'gemini-2.5-pro', 'release_date': '2025-08-01', 'family': 'Gemini'}, 'gemini-2.5-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.5-flash', 'release_date': '2025-07-01', 'family': 'Gemini'}, 'gemini-2.0-flash': {'api_type': 'google_native', 'model_name': 'gemini-2.0-flash', 'release_date': '2025-03-01', 'family': 'Gemini'}}
MODEL_CONFIGS.update(build_qwen_model_configs())
MODEL_CONFIGS.update(build_llava_model_configs())
MODEL_CONFIGS.update(build_lora_model_configs(LORA_TASK))
SYSTEM_MESSAGE = "You are a helpful image analyst that specializes in localizing objects from satellite and aerial images given a natural language instruction. You always truthfully answer the user's question. If you are not sure about something, don't answer false information."
from evaluation.utils.task_categories import DIOR_CATEGORIES as CATEGORY_NAME

def get_detection_prompt(h: int, w: int, description: str):
    return localization_prompt('dior', w, h, description)

class DIORDatasetAccessError(RuntimeError):
    pass

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

def get_image_size(raw_bytes: bytes):
    from PIL import Image as PILImage
    img = PILImage.open(BytesIO(raw_bytes))
    w, h = img.size
    del img
    return (h, w)

def get_raw_image_bytes(image_field: Dict[str, Any]) -> bytes:
    raw_bytes = image_field.get('bytes')
    if raw_bytes is not None:
        return raw_bytes
    image_path = image_field.get('path')
    if image_path:
        with open(image_path, 'rb') as f:
            return f.read()
    raise KeyError('No image bytes or path found in dataset row.')

class LocalDIORRSVGDataset:
    column_names = ['image', 'path', 'objects']

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx: int):
        return self._rows[idx]

    def select_columns(self, columns: List[str]):
        return LocalDIORRSVGDataset([{column: row[column] for column in columns if column in row} for row in self._rows])

    def cast_column(self, *_args, **_kwargs):
        return self

def local_dior_rsvg_candidates() -> List[Path]:
    candidates = []
    env_path = os.environ.get(LOCAL_DIOR_RSVG_DIR_ENV)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([PROJECT_ROOT / 'data' / 'DIOR-RSVG', PROJECT_ROOT.parent / 'data' / 'DIOR-RSVG', PROJECT_ROOT.parent / 'vleo-bench-main' / 'data' / 'DIOR-RSVG'])
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return unique_candidates

def is_raw_dior_rsvg_dir(path: Path) -> bool:
    return (path / 'JPEGImages').is_dir() and (path / 'Annotations').is_dir() and any(((path / f'{split}.txt').is_file() for split in ('train', 'val', 'test')))

def find_local_dior_rsvg_dir() -> Optional[Path]:
    for candidate in local_dior_rsvg_candidates():
        if is_raw_dior_rsvg_dir(candidate):
            return candidate
    return None

def discover_dior_raw_split_names() -> List[str]:
    local_root = find_local_dior_rsvg_dir()
    if local_root is not None:
        split_names = [split for split in ('train', 'val', 'validation', 'test', 'evaluation') if (local_root / f'{split}.txt').is_file()]
        if split_names:
            return split_names
    return list(DIOR_RSVG_ALL_SPLITS)

def _split_index_values(split_file: Path) -> set:
    values = set()
    with split_file.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                values.add(int(line))
            except ValueError:
                values.add(line)
    return values

def _xml_text(node, default: str='') -> str:
    return node.text.strip() if node is not None and node.text else default

def _parse_local_bbox(object_node) -> List[int]:
    bbox_node = object_node.find('bndbox')
    if bbox_node is None:
        raise ValueError('object has no bndbox')
    return [int(float(_xml_text(bbox_node.find('xmin'), '0'))), int(float(_xml_text(bbox_node.find('ymin'), '0'))), int(float(_xml_text(bbox_node.find('xmax'), '0'))), int(float(_xml_text(bbox_node.find('ymax'), '0')))]

def _parse_local_caption(object_node) -> str:
    for tag in ('text', 'caption', 'description', 'sentence', 'phrase'):
        value = _xml_text(object_node.find(tag))
        if value:
            return value
    children = list(object_node)
    if len(children) > 3:
        value = _xml_text(children[3])
        if value:
            return value
    return ''

def _local_annotation_files(root: Path) -> List[Path]:
    return [Path(directory_path) / filename for directory_path, _directory_names, filenames in os.walk(root / 'Annotations') for filename in filenames if filename.endswith('.xml')]

def load_local_dior_dataset(split: str, root: Path) -> LocalDIORRSVGDataset:
    split_file = root / f'{split}.txt'
    if not split_file.is_file():
        raise FileNotFoundError(f'Missing split file: {split_file}')
    split_values = _split_index_values(split_file)
    rows_by_image = {}
    object_counter = 0
    annotation_paths = _local_annotation_files(root)
    for annotation_path in annotation_paths:
        xml_root = ET.parse(annotation_path).getroot()
        filename = _xml_text(xml_root.find('filename'))
        if not filename:
            filename = f'{annotation_path.stem}.jpg'
        image_path = root / 'JPEGImages' / filename
        image_key = str(image_path)
        for object_node in xml_root.findall('object'):
            object_id_values = {object_counter, str(object_counter), annotation_path.stem, filename}
            if split_values and split_values.isdisjoint(object_id_values):
                object_counter += 1
                continue
            category = _xml_text(object_node.find('name'))
            if not category:
                object_counter += 1
                continue
            obj = {'bbox': _parse_local_bbox(object_node), 'categories': category, 'categories_normalized': CATEGORY_NAME.get(category, category), 'captions': _parse_local_caption(object_node)}
            row = rows_by_image.setdefault(image_key, {'image': {'path': image_key, 'bytes': None}, 'path': filename, 'objects': {'bbox': [], 'categories': [], 'categories_normalized': [], 'captions': []}})
            for key, value in obj.items():
                row['objects'][key].append(value)
            object_counter += 1
    rows = list(rows_by_image.values())
    if not rows:
        raise RuntimeError(f'No DIOR-RSVG rows found in local split {split!r} at {root}')
    return LocalDIORRSVGDataset(rows)

def gemini_image_part_from_url(image_url: str) -> Dict:
    if image_url.startswith('data:'):
        header, b64_data = image_url.split(',', 1)
        mime_type = header.split(':')[1].split(';')[0]
        return {'inlineData': {'mimeType': mime_type, 'data': b64_data}}
    return {'fileData': {'mimeType': 'image/png', 'fileUri': image_url}}

def dataset_cache_exists(dataset_name: str) -> bool:
    cache_slug = 'datasets--' + dataset_name.replace('/', '--')
    return (Path(HF_CACHE_DIR) / 'hub' / cache_slug).exists()

def _cached_dior_arrow_files(split: str) -> List[Path]:
    filename_pattern = f'dior-rsvg-{split}-*.arrow'
    roots = (Path(HF_CACHE_DIR) / 'danielz01___dior-rsvg', Path(HF_CACHE_DIR) / 'datasets' / 'danielz01___dior-rsvg')
    files = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(filename_pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return sorted(files)

def load_dior_dataset_from_cached_arrow(split: str, image_cls):
    files = _cached_dior_arrow_files(split)
    if not files:
        return None
    from datasets import Dataset, concatenate_datasets
    shards = [Dataset.from_file(str(path)) for path in files]
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    print(f'Using {len(files)} local Arrow shard(s) for {HF_DATASET_NAME} (split={split}); skipping Hugging Face builder/download.')
    return dataset.cast_column('image', image_cls(decode=False))

def hf_token_candidates() -> List[tuple]:
    tokens = []
    for name in ('HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
        token = os.environ.get(name)
        if token:
            tokens.append((name, token))
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        tokens.append((str(Path(hf_home) / 'token'), _read_hf_token_file(Path(hf_home) / 'token')))
    tokens.append((str(Path.home() / '.cache' / 'huggingface' / 'token'), _read_hf_token_file(Path.home() / '.cache' / 'huggingface' / 'token')))
    return [(source, token) for source, token in tokens if token]

def _read_hf_token_file(token_path: Path) -> Optional[str]:
    if not token_path.exists():
        return None
    token = token_path.read_text(encoding='utf-8').strip()
    return token or None


def raise_dior_dataset_access_error(exc: Exception):
    checked_paths = ', '.join((str(path) for path in local_dior_rsvg_candidates()))
    local_hint = f'You can also place the raw DIOR-RSVG files under ./data/DIOR-RSVG (with JPEGImages/, Annotations/, and split txt files) or set {LOCAL_DIOR_RSVG_DIR_ENV}.'
    raise DIORDatasetAccessError(f'Could not load {HF_DATASET_NAME}.\n- Dataset page: {HF_DATASET_URL}\n- This is a gated Hugging Face dataset. Log in with a valid token whose account has accepted/requested access to the dataset.\n- Cache directory: {HF_CACHE_DIR}\n- Local fallback: {local_hint}\n- Checked local paths: {checked_paths}\n- Underlying error: {type(exc).__name__}: {exc}') from exc

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

def dior_parquet_files_for_split(split: str, token: Optional[str]) -> List[str]:
    from huggingface_hub import HfApi
    info = HfApi(endpoint='https://huggingface.co').dataset_info(HF_DATASET_NAME, token=token)
    prefix = f'data/{split}-'
    parquet_files = sorted((sibling.rfilename for sibling in info.siblings if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith('.parquet')))
    if not parquet_files:
        available = sorted((sibling.rfilename for sibling in info.siblings if sibling.rfilename.startswith('data/') and sibling.rfilename.endswith('.parquet')))
        raise FileNotFoundError(f'No parquet files found for split={split!r} in {HF_DATASET_NAME}. Available parquet files: {available}')
    return parquet_files

def load_dior_dataset_from_parquet_files(split: str, token: Optional[str], image_cls):
    from datasets import Dataset
    parquet_files = dior_parquet_files_for_split(split, token)
    print(f'Loading {HF_DATASET_NAME} split={split!r} from {len(parquet_files)} exact parquet files.')
    local_paths = [download_hf_dataset_file_with_requests(filename, token, split) for filename in parquet_files]
    dataset_cache_dir = os.environ.get('HF_DATASETS_CACHE', str(Path(HF_CACHE_DIR) / 'datasets'))
    dataset = Dataset.from_parquet(local_paths, split=split, cache_dir=dataset_cache_dir)
    return dataset.cast_column('image', image_cls(decode=False))


def local_dior_parquet_path(filename: str, split: str) -> Path:
    return Path(HF_CACHE_DIR) / 'dior_rsvg_parquet' / split / filename.replace('/', '__')

def is_retryable_download_error(exc: Exception, requests_module) -> bool:
    retryable_request_errors = (requests_module.exceptions.ChunkedEncodingError, requests_module.exceptions.ConnectionError, requests_module.exceptions.ReadTimeout, requests_module.exceptions.Timeout)
    if isinstance(exc, retryable_request_errors):
        return True
    if isinstance(exc, requests_module.exceptions.HTTPError):
        status_code = getattr(exc.response, 'status_code', None)
        return status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, IOError)

def _download_hf_dataset_file_with_requests_once(filename: str, token: Optional[str], split: str, dest_path: Path, incomplete_path: Path) -> str:
    import contextlib
    import requests
    url = f'{HF_DATASET_URL}/resolve/main/{filename}'
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    head = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total_size = int(head.headers.get('content-length') or 0)
    if dest_path.exists():
        dest_size = dest_path.stat().st_size
        if total_size == 0 or dest_size == total_size:
            return str(dest_path)
        print(f'Discarding partial cached file with unexpected size: {dest_path}')
        dest_path.unlink()
    downloaded = incomplete_path.stat().st_size if incomplete_path.exists() else 0
    if total_size and downloaded >= total_size:
        if downloaded == total_size:
            incomplete_path.replace(dest_path)
            return str(dest_path)
        incomplete_path.unlink()
        downloaded = 0
    request_headers = dict(headers)
    if downloaded:
        request_headers['Range'] = f'bytes={downloaded}-'
    with requests.get(url, headers=request_headers, stream=True, allow_redirects=True, timeout=(30, 300)) as response:
        if downloaded and response.status_code != 206:
            downloaded = 0
            with contextlib.suppress(FileNotFoundError):
                incomplete_path.unlink()
        response.raise_for_status()
        mode = 'ab' if downloaded else 'wb'
        progress_total = total_size or downloaded + int(response.headers.get('content-length') or 0)
        with incomplete_path.open(mode) as f, tqdm(total=progress_total, initial=downloaded if progress_total else 0, desc=Path(filename).name, unit='B', unit_scale=True, unit_divisor=1024) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(len(chunk))
    if total_size and incomplete_path.stat().st_size != total_size:
        raise IOError(f'Incomplete download for {filename}: {incomplete_path.stat().st_size} of {total_size} bytes')
    incomplete_path.replace(dest_path)
    return str(dest_path)

def download_hf_dataset_file_with_requests(filename: str, token: Optional[str], split: str) -> str:
    import fcntl
    import requests
    import time
    output_dir = Path(HF_CACHE_DIR) / 'dior_rsvg_parquet' / split
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = local_dior_parquet_path(filename, split)
    incomplete_path = dest_path.with_suffix(dest_path.suffix + '.incomplete')
    lock_path = dest_path.with_suffix(dest_path.suffix + '.lock')
    max_attempts = 10
    with lock_path.open('w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    return _download_hf_dataset_file_with_requests_once(filename, token, split, dest_path, incomplete_path)
                except Exception as exc:
                    if not is_retryable_download_error(exc, requests) or attempt >= max_attempts:
                        raise
                    wait_seconds = min(60, 2 ** attempt)
                    downloaded = incomplete_path.stat().st_size if incomplete_path.exists() else 0
                    print(f'Download interrupted for {filename} ({type(exc).__name__}: {exc}). Cached {downloaded / 1024 ** 2:.1f} MiB; retrying {attempt + 1}/{max_attempts} in {wait_seconds}s.')
                    time.sleep(wait_seconds)
            raise RuntimeError(f'Failed to download {filename}')
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def get_dior_hf_token_for_download() -> Optional[str]:
    token_candidates = hf_token_candidates()
    if not token_candidates:
        print('No Hugging Face token found in environment or token cache; attempting anonymous download.')
        return None
    token_source, token = token_candidates[0]
    if token_source not in {'HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'}:
        print(f'Using Hugging Face token from {token_source}')
    return token

def ensure_dior_parquet_split_downloaded(split: str, token: Optional[str]) -> List[str]:
    parquet_files = dior_parquet_files_for_split(split, token)
    print(f'\nEnsuring DIOR-RSVG split={split!r}: {len(parquet_files)} parquet shard(s)')
    local_paths = []
    for shard_number, filename in enumerate(parquet_files, start=1):
        local_path = download_hf_dataset_file_with_requests(filename, token, split)
        local_paths.append(local_path)
        print(f'[{split}] shard {shard_number}/{len(parquet_files)} ready: {Path(local_path).name}')
    total_bytes = sum((Path(path).stat().st_size for path in local_paths))
    print(f'[{split}] complete: {len(local_paths)}/{len(parquet_files)} shard(s), {total_bytes / 1024 ** 3:.2f} GiB')
    return local_paths

def ensure_dior_parquet_splits_downloaded(splits: List[str]) -> Dict[str, List[str]]:
    invalid_splits = sorted(set(splits) - set(DIOR_RSVG_ALL_SPLITS))
    if invalid_splits:
        raise ValueError(f'Unknown DIOR-RSVG split(s): {invalid_splits}')
    token = get_dior_hf_token_for_download()
    downloaded = {}
    for split in splits:
        downloaded[split] = ensure_dior_parquet_split_downloaded(split, token)
    return downloaded

def set_hf_endpoint():
    endpoint = 'https://huggingface.co'
    os.environ['HF_ENDPOINT'] = endpoint
    try:
        import huggingface_hub.constants as hub_constants
        hub_constants.ENDPOINT = endpoint
    except Exception:
        pass
    try:
        import datasets.config as datasets_config
        datasets_config.HF_ENDPOINT = endpoint
    except Exception:
        pass

def clean_result_jsonl_for_resume(path: str) -> set:
    if not os.path.exists(path):
        return set()
    latest_by_index = {}
    malformed_count = 0
    duplicate_count = 0
    unkeyed_count = 0
    normalized_index_count = 0
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
            if not isinstance(record, dict):
                malformed_count += 1
                continue
            valid_count += 1
            index = _normalize_result_index(record.get('index'))
            if index is None:
                unkeyed_count += 1
                continue
            if record.get('index') != index:
                record['index'] = index
                normalized_index_count += 1
            if index in latest_by_index:
                duplicate_count += 1
            latest_by_index[index] = (line_number, record)
    cleaned_records = [item[1] for item in sorted(latest_by_index.values(), key=lambda item: item[0])]
    if malformed_count or duplicate_count or unkeyed_count or normalized_index_count:
        backup_path = f'{path}.before-resume-cleanup'
        if not os.path.exists(backup_path):
            shutil.copyfile(path, backup_path)
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(tmp_path, path)
        print('Evaluation state updated.', flush=True)
    return set(latest_by_index.keys())

def _normalize_result_index(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch('\\d+', value.strip()):
        return int(value.strip())
    return None


def iter_jsonl_records(path: str):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f'Warning: skipping malformed JSONL row in {path}')

def validate_resume_config(path: str, expected_metadata: Dict):
    if not os.path.exists(path):
        return
    keys_to_check = ['model', 'canonical_strategy_mode', 'canonical_split_seed', 'eval_raw_split', 'few_shot_source_raw_split', 'eval_canonical_split', 'few_shot_source_canonical_split', 'prompt_mode', 'few_shot_k', 'few_shot_split', 'few_shot_seed', 'task', 'reasoning_method']
    for record in iter_jsonl_records(path):
        if expected_metadata.get('prompt_mode') == 'few-shot' and record.get('prompt_mode') == 'few-shot' and (record.get('few_shot_sample_seed') is None):
            raise ValueError(f'Existing result file {path} contains few-shot rows without per-sample few_shot_sample_seed metadata. Use a different output file or remove the old shared-example result file.')
        for key in keys_to_check:
            existing = record.get(key)
            expected = expected_metadata.get(key)
            if existing is None and expected is not None:
                continue
            if existing != expected:
                raise ValueError(f'Existing result file {path} has incompatible {key}: {existing!r} != {expected!r}. Use a different output file or keep the same prompt configuration.')

def validate_resume_samples(path: str, dataset_items: List[Dict]) -> None:
    expected_by_index = {item['index']: item for item in dataset_items}
    for record in iter_jsonl_records(path):
        index = _normalize_result_index(record.get('index'))
        expected = expected_by_index.get(index)
        if expected is None:
            continue
        saved_path = record.get('path')
        expected_path = expected.get('path')
        if saved_path is not None and expected_path is not None and (saved_path != expected_path):
            raise ValueError(f'Existing result file {path} maps index {index} to path {saved_path!r}, expected {expected_path!r}.')
        saved_object = record.get('object')
        expected_object = expected.get('object')
        if not isinstance(saved_object, dict) or not isinstance(expected_object, dict):
            continue
        for key in ('bbox', 'captions', 'categories'):
            saved_value = saved_object.get(key)
            expected_value = expected_object.get(key)
            if saved_value is not None and expected_value is not None and (saved_value != expected_value):
                raise ValueError(f'Existing result file {path} maps index {index} to a different DIOR object ({key} mismatch).')


def _valid_bbox(coords: List[float], image_height: Optional[int]=None, image_width: Optional[int]=None) -> List[float]:
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return []
    try:
        values = [float(value) for value in coords]
    except (TypeError, ValueError):
        return []
    if any((not math.isfinite(value) or value < 0 for value in values)):
        return []
    xmin, ymin, xmax, ymax = values
    if not xmin < xmax or not ymin < ymax:
        return []
    width = None
    if image_width is not None:
        try:
            width = float(image_width)
        except (TypeError, ValueError):
            return []
        if xmax > width or xmin >= width:
            return []
    height = None
    if image_height is not None:
        try:
            height = float(image_height)
        except (TypeError, ValueError):
            return []
        if ymax > height or ymin >= height:
            return []
    return values

def _numeric_bbox_from_values(values: Any) -> List[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return []
    try:
        coords = [float(v) for v in values]
    except (TypeError, ValueError):
        return []
    return _valid_bbox(coords)

def _normalize_coord_key(key: Any) -> str:
    return re.sub('[^a-z0-9]', '', str(key).strip().lower())

def _bbox_from_named_mapping(value: Any) -> List[float]:
    if not isinstance(value, dict):
        return []
    aliases = {'xmin': ('xmin', 'xminimum', 'x1', 'left'), 'ymin': ('ymin', 'yminimum', 'y1', 'top'), 'xmax': ('xmax', 'xmaximum', 'x2', 'right'), 'ymax': ('ymax', 'ymaximum', 'y2', 'bottom')}
    normalized = {_normalize_coord_key(k): v for k, v in value.items()}
    coords = []
    for canonical in ('xmin', 'ymin', 'xmax', 'ymax'):
        raw = None
        for alias in aliases[canonical]:
            if alias in normalized:
                raw = normalized[alias]
                break
        if raw is None:
            return []
        try:
            coords.append(float(raw))
        except (TypeError, ValueError):
            return []
    return _valid_bbox(coords)

def _coerce_bbox(value: Any) -> List[float]:
    bbox = _numeric_bbox_from_values(value)
    if bbox:
        return bbox
    bbox = _bbox_from_named_mapping(value)
    if bbox:
        return bbox
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _bbox_from_named_mapping(value[0])
    return []

def parse_detection_response(text: str, image_height: Optional[int]=None, image_width: Optional[int]=None, model_key: Optional[str]=None) -> List[List[float]]:
    from evaluation.utils.response_protocol import parse_answer
    return parse_answer('dior_localization', text, {'image_width': image_width,
                        'image_height': image_height}) or []

def make_prompt_metadata(task: str, few_shot_source: Optional[Dict]=None, split_context: Optional[Dict]=None, few_shot_seed: Optional[int]=None) -> Dict:
    source = few_shot_source or {}
    split_context = split_context or {}
    k = int(source.get('k', 0) or 0)
    return {'canonical_strategy_mode': split_context.get('canonical_strategy_mode'), 'canonical_split_seed': split_context.get('canonical_split_seed'), 'eval_raw_split': split_context.get('eval_raw_split'), 'few_shot_source_raw_split': split_context.get('few_shot_source_raw_split'), 'eval_canonical_split': split_context.get('eval_canonical_split'), 'few_shot_source_canonical_split': split_context.get('few_shot_source_canonical_split'), 'prompt_mode': 'few-shot' if k > 0 else 'zero-shot', 'few_shot_k': k, 'few_shot_split': source.get('raw_split') if k > 0 else None, 'few_shot_seed': few_shot_seed if k > 0 else None, 'task': task}

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

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def query_openai(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, deterministic: bool=False) -> dict:
    del deterministic
    if api_key is None:
        api_key = first_api_key_for_model(model_key, MODEL_CONFIGS)
    config = MODEL_CONFIGS[model_key]
    messages = [{'role': 'system', 'content': system}]
    for example in few_shot_examples or []:
        messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': example['image_url']}}, {'type': 'text', 'text': example['prompt']}]})
        messages.append({'role': 'assistant', 'content': example['answer']})
    messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}, {'type': 'text', 'text': user}]})
    payload = json.dumps({'model': config['model_name'], 'messages': messages})
    status, data = post_openai_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key)
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
        raise APIStatusError(status, error_msg)
    return json.loads(data.decode('utf-8'))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def query_gemini(image_url: str, system: str, user: str, model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, deterministic: bool=False) -> dict:
    if api_key is None:
        api_key = first_api_key_for_model(model_key, MODEL_CONFIGS)
    config = MODEL_CONFIGS[model_key]
    contents = []
    for example in few_shot_examples or []:
        contents.append({'role': 'user', 'parts': [gemini_image_part_from_url(example['image_url']), {'text': example['prompt']}]})
        contents.append({'role': 'model', 'parts': [{'text': example['answer']}]})
    contents.append({'role': 'user', 'parts': [gemini_image_part_from_url(image_url), {'text': user}]})
    request_body = {'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents}
    if deterministic:
        request_body['generationConfig'] = {'temperature': 0, 'candidateCount': 1}
    payload = json.dumps(request_body)
    status, data = post_gemini_json(model_key=model_key, model_configs=MODEL_CONFIGS, payload=payload, api_key=api_key)
    if status != 200:
        error_msg = data.decode('utf-8')
        print(f'{model_key} API Error: {error_msg}')
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

def get_query_function(model_key: str, api_key: str=None, few_shot_examples: Optional[List[Dict]]=None, deterministic: bool=False) -> Callable:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f'Unknown model: {model_key}')
    config = MODEL_CONFIGS[model_key]
    if config['api_type'] == 'hf_local':
        qwen_model_path = _QWEN_MODEL_PATHS.get(model_key)
        if not qwen_model_path:
            raise ValueError(f'Model {model_key} requires --qwen-model-path or --llava-model-path before local inference.')
        qwen_query = make_qwen_query_callable(model_key, qwen_model_path)
        return lambda image_url, system, user: qwen_query(image_url=image_url, system=system, user=user, few_shot_examples=few_shot_examples)
    if config['api_type'] == 'openai':
        return lambda image_url, system, user: query_openai(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, deterministic=deterministic)
    elif config['api_type'] == 'google_native':
        return lambda image_url, system, user: query_gemini(image_url, system, user, model_key, api_key=api_key, few_shot_examples=few_shot_examples, deterministic=deterministic)
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")

def get_models_to_test(model_arg: str) -> List[str]:
    return expand_model_selection(model_arg, MODEL_CONFIGS)

def select_evaluation_indices(total_size: int, seed: int=DEFAULT_RANDOM_SEED) -> List[int]:
    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)
    return indices

def select_metadata_items_by_indices(metadata_items: List[Dict], selected_indices: List[int], raw_split: str, canonical_split: str) -> List[Dict]:
    metadata_by_index = {item['index']: item for item in metadata_items}
    selected_items = []
    for index in selected_indices:
        item = dict(metadata_by_index[index])
        item['raw_split'] = raw_split
        item['canonical_split'] = canonical_split
        selected_items.append(item)
    return selected_items

def resolve_canonical_pools(few_shot_k: int, task: str) -> Dict:
    raw_split_names = discover_dior_raw_split_names()
    split_map = discover_available_splits_from_names(raw_split_names)
    strategy = resolve_split_strategy(split_map)
    datasets_by_raw_split = {}
    metadata_by_raw_split = {}
    split_lengths = {}
    for raw_split in sorted(set(split_map.values())):
        hf_dataset, metadata_items = download_and_prepare_dataset(split=raw_split, task=task)
        datasets_by_raw_split[raw_split] = hf_dataset
        metadata_by_raw_split[raw_split] = metadata_items
        split_lengths[raw_split] = len(hf_dataset)
    assignments = build_split_assignments(split_map=split_map, split_lengths=split_lengths, seed=DEFAULT_SPLIT_SEED)
    eval_assignment = assignments['test']
    train_assignment = assignments['train']
    eval_selected_indices = eval_assignment['indices']
    evaluation_items = select_metadata_items_by_indices(metadata_by_raw_split[eval_assignment['raw_split']], eval_selected_indices, raw_split=eval_assignment['raw_split'], canonical_split='test')
    few_shot_source = None
    if few_shot_k > 0:
        few_shot_items = select_metadata_items_by_indices(metadata_by_raw_split[train_assignment['raw_split']], train_assignment['indices'], raw_split=train_assignment['raw_split'], canonical_split='train')
        few_shot_source = build_few_shot_source(datasets_by_raw_split[train_assignment['raw_split']], few_shot_items, k=few_shot_k, raw_split=train_assignment['raw_split'], canonical_split='train', task=task)
    split_context = {'canonical_strategy_mode': strategy['mode'], 'canonical_split_seed': DEFAULT_SPLIT_SEED, 'eval_raw_split': eval_assignment['raw_split'], 'few_shot_source_raw_split': train_assignment['raw_split'], 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': split_map}
    print('Detected raw splits: {}'.format(split_map))
    print('Canonical strategy mode: {}'.format(strategy['mode']))
    print('Few-shot source: canonical train <- raw {} | Eval: canonical test <- raw {} | split seed={}'.format(train_assignment['raw_split'], eval_assignment['raw_split'], DEFAULT_SPLIT_SEED))
    return {'evaluation_dataset': datasets_by_raw_split[eval_assignment['raw_split']], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def load_dior_dataset(split: str):
    try:
        from datasets import load_dataset, Image, DownloadConfig
    except ImportError as exc:
        local_root = find_local_dior_rsvg_dir()
        if local_root is not None:
            print(f'`datasets` is not installed; falling back to local raw DIOR-RSVG directory: {local_root}')
            return load_local_dior_dataset(split, local_root)
        raise_dior_dataset_access_error(exc)
    cached_arrow = load_dior_dataset_from_cached_arrow(split, Image)
    if cached_arrow is not None:
        return cached_arrow
    token = None
    cache_dir = HF_CACHE_DIR
    download_config = make_resume_download_config(DownloadConfig)
    print(f'\nHugging Face cache directory: {cache_dir}')
    if dataset_cache_exists(HF_DATASET_NAME):
        print(f'Found local dataset cache for {HF_DATASET_NAME}; reusing it when possible.')
    else:
        print(f'Local dataset cache for {HF_DATASET_NAME} not found; Hugging Face will download/cache it.')
    print('Hugging Face download resume: enabled (max_retries=10)')
    print(f'Loading DIOR-RSVG with datasets.load_dataset({HF_DATASET_NAME!r}).')
    try:
        set_hf_endpoint()
        token_candidates = hf_token_candidates()
        if token_candidates:
            token_source, token = token_candidates[0]
            if token_source not in {'HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN'}:
                print(f'Using Hugging Face token from {token_source}')
            if hasattr(download_config, 'token'):
                download_config.token = token
        else:
            print('No Hugging Face token found in environment or token cache; attempting anonymous load.')
        load_kwargs = {'cache_dir': cache_dir, 'download_mode': 'reuse_dataset_if_exists', 'download_config': download_config, 'writer_batch_size': 100}
        if token:
            load_kwargs['token'] = token
        loaded = load_dataset(HF_DATASET_NAME, split=split, **load_kwargs)
    except Exception as exc:
        load_error = exc
        print(f'Warning: direct load_dataset failed ({type(load_error).__name__}: {load_error}). Trying exact parquet-file download.')
        try:
            return load_dior_dataset_from_parquet_files(split, token, Image)
        except Exception as parquet_exc:
            load_error = parquet_exc
            print(f'Warning: exact parquet-file load failed ({type(load_error).__name__}: {load_error}).')
        local_root = find_local_dior_rsvg_dir()
        if local_root is not None:
            print(f'Warning: could not load {HF_DATASET_NAME} from Hugging Face ({type(load_error).__name__}: {load_error}).')
            print(f'Falling back to local raw DIOR-RSVG directory: {local_root}')
            return load_local_dior_dataset(split, local_root)
        raise_dior_dataset_access_error(load_error)
    if hasattr(loaded, 'keys'):
        if split not in loaded:
            raise ValueError(f'Split {split!r} not found in {HF_DATASET_NAME}. Available splits: {list(loaded.keys())}')
        dataset = loaded[split]
    else:
        dataset = loaded
    return dataset.cast_column('image', Image(decode=False))

def first_object(row: Dict) -> Dict:
    return {k: v[0] for k, v in row['objects'].items()}

def download_and_prepare_dataset(split: str='test', task: str='detection'):
    print(f'\nLoading dataset: {HF_DATASET_NAME} (split={split}) ...')
    dataset = load_dior_dataset(split)
    dataset_size = len(dataset)
    print('Evaluation state updated.', flush=True)
    selected_indices = select_evaluation_indices(dataset_size, seed=DEFAULT_RANDOM_SEED)
    print('Evaluation state updated.', flush=True)
    metadata_ds = dataset.select_columns([c for c in dataset.column_names if c != 'image'])
    items = []
    for rank, idx in enumerate(selected_indices):
        row = metadata_ds[idx]
        items.append({'index': idx, 'selected_index_rank': rank, 'object': first_object(row), 'path': row['path']})
    print('Evaluation state updated.', flush=True)
    return (dataset, items)

def make_task_prompt_and_answer(raw_bytes: bytes, obj: Dict, task: str) -> Optional[Dict]:
    h, w = get_image_size(raw_bytes)
    if task != 'detection':
        raise ValueError(f'Unknown task: {task}')
    bbox = _coerce_bbox(obj.get('bbox'))
    caption = obj.get('captions')
    if not bbox or not caption:
        return None
    prompt = get_detection_prompt(h, w, caption)
    answer = json.dumps({'bbox': [int(v) if float(v).is_integer() else v for v in bbox]}, ensure_ascii=False)
    return {'prompt': prompt, 'answer': answer}

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str, task: str):
    if k <= 0:
        return {'hf_dataset': hf_dataset, 'candidates': [], 'k': 0, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'task': task}
    valid_examples = []
    metadata_ds = hf_dataset.select_columns([c for c in hf_dataset.column_names if c != 'image'])
    for item in tqdm(metadata_items, desc='Scanning DIOR-RSVG few-shot candidates'):
        idx = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[idx]['image'])
            row = metadata_ds[idx]
            obj = first_object(row)
            prompt_answer = make_task_prompt_and_answer(raw_bytes, obj, task)
            if prompt_answer is None:
                continue
            valid_examples.append({'index': idx, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'task': task, 'object': obj, 'path': row.get('path'), **prompt_answer})
        except Exception:
            continue
    if len(valid_examples) < k:
        raise ValueError(f"Could not find {k} valid few-shot examples in raw split '{raw_split}' for task '{task}'. Found {len(valid_examples)}.")
    return {'hf_dataset': hf_dataset, 'candidates': valid_examples, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'task': task}

def sample_few_shot_examples(few_shot_source: Dict, seed: int, current_item: Optional[Dict]=None) -> List[Dict]:
    candidate_pool = list(few_shot_source['candidates'])
    if current_item is not None:
        current_identity = build_candidate_identity(current_item)
        candidate_pool = [candidate for candidate in candidate_pool if build_candidate_identity(candidate) != current_identity]
    if len(candidate_pool) < few_shot_source['k']:
        raise ValueError('Not enough non-overlapping few-shot candidates after filtering the current evaluation sample. Need {}, found {}.'.format(few_shot_source['k'], len(candidate_pool)))
    selected = random.Random(seed).sample(candidate_pool, few_shot_source['k'])
    examples = []
    hf_dataset = few_shot_source['hf_dataset']
    for source_example in selected:
        example = dict(source_example)
        raw_bytes = get_raw_image_bytes(hf_dataset[example['index']]['image'])
        example['image_url'] = raw_bytes_to_base64_uri(raw_bytes)
        examples.append(example)
    return examples


def query_model_on_dataset(model_key: str, dataset_items: List[Dict], hf_dataset, split_context: Dict, result_path: Optional[str]=None, api_key: str=None, task: str='detection', output_dir: str=DEFAULT_RESULT_DIR, few_shot_source: Optional[Dict]=None, few_shot_seed: Optional[int]=None, sample_workers: int=DEFAULT_SAMPLE_WORKERS, reasoning_method: str='baseline'):
    prompt_metadata = make_prompt_metadata(task, few_shot_source=few_shot_source, split_context=split_context, few_shot_seed=few_shot_seed)
    validate_cot_run(reasoning_method=reasoning_method, model_key=model_key, few_shot_k=prompt_metadata['few_shot_k'], model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    prompt_metadata['model'] = model_key
    if reasoning_method == 'cot':
        prompt_metadata.update({'reasoning_method': reasoning_method})
    few_shot_seed_by_index = build_few_shot_seed_by_index(dataset_items, few_shot_seed if few_shot_source else None)
    sample_workers = max(1, int(sample_workers))
    if is_local_model(model_key, MODEL_CONFIGS):
        sample_workers = 1
    if result_path is None:
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"-{prompt_metadata['few_shot_k']}shot" if prompt_metadata['prompt_mode'] == 'few-shot' else ''
        result_path = os.path.join(output_dir, f'{model_key}-dior-rsvg-{task}{suffix}.jsonl')
    result_path = result_path_for_reasoning(result_path, reasoning_method)
    print(f'[{model_key}] Starting (task={task}), results -> {result_path}')
    completed_indices = clean_result_jsonl_for_resume(result_path)
    validate_resume_config(result_path, prompt_metadata)
    validate_resume_samples(result_path, dataset_items)
    target_indices = {item['index'] for item in dataset_items}
    completed_target_indices = completed_indices & target_indices
    print('Evaluation state updated.', flush=True)
    pending_items = [item for item in dataset_items if item['index'] not in completed_indices]
    print('Evaluation state updated.', flush=True)
    if not pending_items:
        print(f'[{model_key}] Result file is complete; no model will be loaded or queried.')
        return result_path
    local_qwen_query = None
    if is_local_model(model_key, MODEL_CONFIGS):
        qwen_model_path = _QWEN_MODEL_PATHS.get(model_key)
        if not qwen_model_path:
            raise ValueError(f'Model {model_key} requires --qwen-model-path or --llava-model-path before local inference.')
        local_qwen_query = make_qwen_query_callable(model_key, qwen_model_path)
    write_lock = threading.Lock()

    def _process_single_sample(item):
        i = item['index']
        try:
            raw_bytes = get_raw_image_bytes(hf_dataset[i]['image'])
            image_data_uri = raw_bytes_to_base64_uri(raw_bytes)
            h, w = get_image_size(raw_bytes)
            if task != 'detection':
                raise ValueError(f'Unknown task: {task}')
            user_prompt = get_detection_prompt(h, w, item['object']['captions'])
            query_prompt = effective_prompt(user_prompt, reasoning_method)
            query_system = effective_system_message(SYSTEM_MESSAGE, reasoning_method)
            request_api_key = random_api_key_for_model(model_key, MODEL_CONFIGS, fallback=api_key or '')
            few_shot_sample_seed = few_shot_seed_by_index.get(i)
            sample_few_shot_examples_list = sample_few_shot_examples(few_shot_source, few_shot_sample_seed, current_item=item) if few_shot_source and few_shot_sample_seed is not None else []
            request_query_fn = None
            if local_qwen_query is None:
                request_query_fn = get_query_function(model_key, api_key=request_api_key, few_shot_examples=sample_few_shot_examples_list)

            def call_model(user_prompt: str):
                if local_qwen_query is not None:
                    return local_qwen_query(image_url=image_data_uri, system=query_system, user=user_prompt, few_shot_examples=sample_few_shot_examples_list)
                return request_query_fn(image_url=image_data_uri, system=query_system, user=user_prompt)
            response, response_content, reasoning_response = call_with_reasoning(call_model, reasoning_method, task='dior_localization', context={'image_width': w, 'image_height': h}, prompt=query_prompt)
            answer_text = reasoning_response['answer_text']
            parsed_answer = parse_detection_response(answer_text, h, w, model_key=model_key)
            response_space = 'pixel'
            result_record = {'index': i, 'selected_index_rank': item.get('selected_index_rank'), 'raw_split': item.get('raw_split'), 'canonical_split': item.get('canonical_split'), 'object': item['object'], 'response': response, 'prompt': user_prompt, 'parsed_answer': parsed_answer, 'path': item['path'], 'image_height': h, 'image_width': w, **result_model_metadata(model_key, MODEL_CONFIGS), 'task': task, 'random_seed': DEFAULT_RANDOM_SEED}
            result_record.update(prompt_metadata)
            result_record.update(few_shot_sample_metadata(sample_few_shot_examples_list, few_shot_sample_seed))
            if reasoning_method == 'cot':
                result_record['answer_text'] = answer_text
                result_record.update(cot_row_metadata(base_prompt=user_prompt, effective_prompt_text=query_prompt, parsed_response=reasoning_response, base_system_message=SYSTEM_MESSAGE, effective_system_message_text=query_system))
            with write_lock:
                append_to_jsonl(result_record, result_path)
                completed_indices.add(i)
                print('Evaluation state updated.', flush=True)
        except Exception as e:
            print(f'[{model_key}] Error processing sample {i}: {e}')
            import traceback
            traceback.print_exc()
            if is_local_model(model_key, MODEL_CONFIGS):
                raise
    with ThreadPoolExecutor(max_workers=sample_workers) as pool:
        futures = {pool.submit(_process_single_sample, item): item['index'] for item in pending_items}
        for future in as_completed(futures):
            future.result()
    missing_target_indices = target_indices - completed_indices
    if missing_target_indices:
        raise RuntimeError(f'[{model_key}] incomplete result file; refusing to evaluate partial API output')
    print('Evaluation state updated.', flush=True)
    return result_path

def evaluation(result_path: str) -> Optional[Dict]:
    print(f"\n{'=' * 60}")
    print(f'Evaluating: {result_path}')
    print(f"{'=' * 60}")
    result_json = pd.read_json(result_path, lines=True)
    result_json['model_response'] = result_json.apply(lambda row: row['answer_text'] if isinstance(row.get('answer_text'), str) else row['response']['choices'][0]['message']['content'] if isinstance(row['response'], dict) and 'choices' in row['response'] else str(row['response']), axis=1)
    result_json['model_answer'] = result_json.apply(lambda row: parse_detection_response(row['model_response'], row.get('image_height'), row.get('image_width'), model_key=row.get('model')), axis=1)
    result_json['gt'] = result_json['object'].apply(lambda x: x['bbox'])
    result_json['is_refusal'] = result_json['model_answer'].apply(lambda x: len(x) == 0)
    rr = result_json['is_refusal'].mean()
    result_json_no_refusal = result_json[~result_json['is_refusal']].copy()
    if len(result_json_no_refusal) == 0:
        print(f'All refused for {result_path}; mean IoU is 0')
        return {'model': _remove_suffix(os.path.basename(result_path), '.jsonl'), 'mean_iou': 0.0, 'precision_05': 0.0, 'mean_centroid_distance': None, 'refusal_rate': rr}
    result_json_no_refusal['model_answer'] = result_json_no_refusal['model_answer'].apply(lambda x: np.array(x).reshape(-1))
    predicted_bbox = np.stack(result_json_no_refusal['model_answer'].to_list())
    gt_bbox = np.stack(result_json_no_refusal['gt'].to_list())
    ious = torch.diag(box_iou(torch.from_numpy(predicted_bbox), torch.from_numpy(gt_bbox)))
    mean_iou = ious.sum().item() / len(result_json)
    pr_05 = (ious > 0.5).float().sum().item() / len(result_json)
    mean_dist = get_mean_centroid_distance(predicted_bbox, gt_bbox)
    model_name = _remove_suffix(os.path.basename(result_path), '.jsonl')
    print(f'\nResults for {model_name}:')
    print('Evaluation state updated.', flush=True)
    print('Evaluation state updated.', flush=True)
    print(f'  Refusal Rate: {rr:.2%}')
    print(f'  Mean IoU: {mean_iou:.4f}')
    print(f'  Precision@0.5: {pr_05:.4f}')
    print(f'  Mean Centroid Distance: {mean_dist:.4f}')
    print(f'\nLaTeX format:')
    print(f'Mean IoU & Precision@0.5 & Mean Centroid Dist & Refusal Rate')
    print(f'{mean_iou:.4f} & {pr_05:.4f} & {mean_dist:.4f} & {rr:.2f}')
    return {'model': model_name, 'mean_iou': mean_iou, 'precision_05': pr_05, 'mean_centroid_distance': mean_dist, 'refusal_rate': rr}

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR, task: str='detection'):
    print(f"\n{'=' * 60}")
    print(f'Generating model comparison plots')
    print(f"{'=' * 60}\n")
    suffix_pattern = f'-dior-rsvg-{task}(?:-\\d+shot)?\\.jsonl$'
    result_files = [f for f in os.listdir(result_dir) if re.search(suffix_pattern, f)]
    if not result_files:
        print(f'No result files found in {result_dir} for task {task}')
        return
    print(f'Found {len(result_files)} result files')
    all_metrics = []
    for result_file in tqdm(result_files, desc='Evaluating models'):
        result_path = os.path.join(result_dir, result_file)
        try:
            metrics = evaluation(result_path)
            if metrics is not None:
                model_key = re.sub(suffix_pattern, '', result_file)
                if model_key in MODEL_CONFIGS:
                    metrics['release_date'] = MODEL_CONFIGS[model_key]['release_date']
                    metrics['family'] = MODEL_CONFIGS[model_key]['family']
                    metrics['model_key'] = model_key
                    metrics['result_file'] = result_file
                    metrics.update(shot_from_metrics(metrics, result_file))
                    all_metrics.append(metrics)
        except Exception as e:
            print(f'Error evaluating {result_file}: {e}')
            continue
    if not all_metrics:
        print('No valid metrics collected, cannot generate plots')
        return
    write_shot_summary_json(all_metrics, os.path.join(result_dir, f'model_metrics_{task}_all_shots.json'), dataset_name='DIOR-RSVG', summary_type=f'{task}_model_comparison')
    zero_shot_metrics = [metrics for metrics in all_metrics if is_zero_shot_metrics(metrics, metrics.get('result_file'))]
    if not zero_shot_metrics:
        print('No zero-shot metrics collected; skipping model comparison plots and zero-shot CSV.')
        return pd.DataFrame(all_metrics)
    df_metrics = pd.DataFrame(zero_shot_metrics)
    df_metrics['release_date'] = pd.to_datetime(df_metrics['release_date'])
    df_metrics = df_metrics.sort_values('release_date')
    print(f'\nCollected metrics for {len(df_metrics)} models')
    print(df_metrics[['model_key', 'release_date', 'mean_iou', 'precision_05', 'refusal_rate']])
    from adjustText import adjust_text
    sns.set(font_scale=1.2, style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = FAMILY_COLORS
    ax1 = axes[0, 0]
    texts1 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family].sort_values('release_date')
        if len(df_family) > 0:
            color = colors[family]
            ax1.plot(df_family['release_date'], df_family['mean_iou'], marker='o', label=family, color=color, linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts1.append(ax1.text(row['release_date'], row['mean_iou'], label, fontsize=12, color=color))
    ax1.set_xlabel('Release Date')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Mean IoU over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if texts1:
        adjust_text(texts1, ax=ax1, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax2 = axes[0, 1]
    texts2 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family].sort_values('release_date')
        if len(df_family) > 0:
            color = colors[family]
            ax2.plot(df_family['release_date'], df_family['precision_05'], marker='o', label=family, color=color, linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts2.append(ax2.text(row['release_date'], row['precision_05'], label, fontsize=12, color=color))
    ax2.set_xlabel('Release Date')
    ax2.set_ylabel('Precision@0.5')
    ax2.set_title('Precision@0.5 over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if texts2:
        adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax3 = axes[1, 0]
    texts3 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family].sort_values('release_date')
        if len(df_family) > 0:
            color = colors[family]
            ax3.plot(df_family['release_date'], df_family['mean_centroid_distance'], marker='o', label=family, color=color, linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts3.append(ax3.text(row['release_date'], row['mean_centroid_distance'], label, fontsize=12, color=color))
    ax3.set_xlabel('Release Date')
    ax3.set_ylabel('Mean Centroid Distance')
    ax3.set_title('Mean Centroid Distance over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if texts3:
        adjust_text(texts3, ax=ax3, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax4 = axes[1, 1]
    texts4 = []
    for family in PLOT_FAMILY_ORDER:
        df_family = df_metrics[df_metrics['family'] == family].sort_values('release_date')
        if len(df_family) > 0:
            color = colors[family]
            ax4.plot(df_family['release_date'], df_family['refusal_rate'], marker='o', label=family, color=color, linewidth=2, markersize=8)
            for _, row in df_family.iterrows():
                label = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', row['model_key'])
                texts4.append(ax4.text(row['release_date'], row['refusal_rate'], label, fontsize=12, color=color))
    ax4.set_xlabel('Release Date')
    ax4.set_ylabel('Refusal Rate')
    ax4.set_title('Refusal Rate over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    if texts4:
        adjust_text(texts4, ax=ax4, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    fig.suptitle('DIOR-RSVG: Object Detection', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path_pdf = os.path.join(result_dir, f'model_comparison_{task}.pdf')
    output_path_png = os.path.join(result_dir, f'model_comparison_{task}.png')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f'\nPlots saved to:')
    print(f'  {output_path_pdf}')
    print(f'  {output_path_png}')
    csv_path = os.path.join(result_dir, f'model_metrics_{task}.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f'Metrics summary saved to: {csv_path}\n')
    return df_metrics

def main():
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('dior-refdet'):
        return
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate vision models on DIOR-RSVG dataset', epilog='\nAvailable models:\n  Individual: gpt-5.2-2025-12-11, gemini-3-pro-preview, Qwen2.5-VL-7B-Instruct, etc.\n  Families:   GPT-all, Gemini-all, Qwen-all\n  Default:    Test all GPT/Gemini models\n\nTask:\n  detection: predict bbox from description\n\nExamples:\n  python evaluation/tasks/dior_rsvg_localization.py\n  python evaluation/tasks/dior_rsvg_localization.py --model Gemini-all\n  python evaluation/tasks/dior_rsvg_localization.py --list-models\n  python evaluation/tasks/dior_rsvg_localization.py --evaluate-only results/inference/DIOR-RSVG/gpt-5.2-2025-12-11-dior-rsvg-detection.jsonl\n  python evaluation/tasks/dior_rsvg_localization.py --plot-only\n  python evaluation/tasks/dior_rsvg_localization.py --download-only --download-all-splits\n        ', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None, help='Model to test (model name, GPT-all, Gemini-all, Qwen-all, or None for default API models)')
    parser.add_argument('--task', type=str, default='detection', choices=['detection'], help='Task to evaluate (default: detection)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_RESULT_DIR, help='Output directory for results')
    parser.add_argument('--cache-dir', type=str, default=None, help='Optional cache root for Hugging Face / datasets / torch')
    parser.add_argument('--evaluate-only', type=str, default=None, help='Only run evaluation on existing results file')
    parser.add_argument('--plot-only', action='store_true', help='Only generate comparison plots from existing results')
    parser.add_argument('--download-only', action='store_true', help='Only download/cache the dataset and build the metadata index, without API calls')
    parser.add_argument('--download-all-splits', action='store_true', help='With --download-only, download all DIOR-RSVG parquet shards for test/train/val')
    parser.add_argument('--list-models', action='store_true', help='List available models and exit')
    parser.add_argument('--parallel', type=int, default=None, help='Number of parallel threads (default: auto-detect)')
    parser.add_argument('--sample-workers', type=int, default=DEFAULT_SAMPLE_WORKERS, help=f'Concurrent API calls per model (default: {DEFAULT_SAMPLE_WORKERS})')
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
    if args.download_all_splits and (not args.download_only):
        raise ValueError('--download-all-splits must be used together with --download-only.')
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
        plot_model_comparison(args.output_dir, task=args.task)
        return
    if args.evaluate_only:
        evaluate_and_write_cot_metrics(evaluation, args.evaluate_only)
        return
    few_shot_k = args.few_shot_k
    if args.few_shot and few_shot_k <= 0:
        few_shot_k = DEFAULT_FEW_SHOT_K
    if few_shot_k < 0:
        raise ValueError('--few-shot-k must be non-negative.')
    if args.download_only:
        if args.download_all_splits:
            ensure_dior_parquet_splits_downloaded(list(DIOR_RSVG_ALL_SPLITS))
            print('Download/cache check complete for all DIOR-RSVG splits. No model API calls were made.')
            return
        canonical_pools = resolve_canonical_pools(few_shot_k, args.task)
        if canonical_pools['few_shot_source']:
            print('Evaluation state updated.', flush=True)
        print('Download/cache check complete. No model API calls were made.')
        return
    models_to_test = get_models_to_test(args.model)
    for model_key in models_to_test:
        validate_cot_run(reasoning_method=args.reasoning_method, model_key=model_key, few_shot_k=few_shot_k, model_kind=MODEL_CONFIGS[model_key].get('model_kind'))
    global _QWEN_MODEL_PATHS
    _QWEN_MODEL_PATHS = validate_qwen_model_paths(models_to_test, MODEL_CONFIGS, args.qwen_model_path, args.qwen_finetuned_root, args.llava_model_path)
    if has_api_models(models_to_test, MODEL_CONFIGS):
        print_provider_summary()
    canonical_pools = resolve_canonical_pools(few_shot_k, args.task)
    global _shared_dataset_items, _shared_hf_dataset, _shared_task
    global _shared_output_dir, _shared_few_shot_source, _shared_few_shot_seed, _shared_sample_workers, _shared_split_context, _shared_reasoning_method
    _shared_hf_dataset = canonical_pools['evaluation_dataset']
    _shared_dataset_items = canonical_pools['evaluation_items']
    _shared_task = args.task
    _shared_output_dir = args.output_dir
    _shared_sample_workers = args.sample_workers
    _shared_few_shot_source = canonical_pools['few_shot_source']
    _shared_few_shot_seed = args.few_shot_seed if _shared_few_shot_source else None
    _shared_split_context = canonical_pools['split_context']
    _shared_reasoning_method = args.reasoning_method
    if _shared_few_shot_source:
        print('Evaluation state updated.', flush=True)
    print(f"\n{'=' * 60}")
    print(f'Models to test: {len(models_to_test)}')
    print(f'Task: {args.task}')
    print(f"Prompt mode: {('few-shot' if _shared_few_shot_source else 'zero-shot')}")
    print(f'Sample workers per model: {_shared_sample_workers}')
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
    for model_key in models_to_test:
        if is_local_model(model_key, MODEL_CONFIGS):
            print(f'  {model_key:<35} -> local checkpoint: {_QWEN_MODEL_PATHS[model_key]}')
            tasks.append((model_key, None))
            continue
        keys = api_keys_for_model(model_key, MODEL_CONFIGS)
        if not keys:
            raise ValueError(f"No API key configured for {MODEL_CONFIGS[model_key]['family']} model {model_key}.")
        print(f"  {model_key:<35} -> random {MODEL_CONFIGS[model_key]['family']} API key from {len(keys)} available key(s) per sample request")
        tasks.append((model_key, None))
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
                    raise
    else:
        print(f"\n{'=' * 60}")
        print(f'Running sequentially...')
        print(f"{'=' * 60}\n")
        result_paths = []
        for task in tqdm(tasks, desc='Testing models'):
            result_path = run_single_model(task)
            if result_path is not None:
                result_paths.append(result_path)
    print(f"\n{'=' * 60}")
    print(f'Evaluating {len(result_paths)} result files')
    print(f"{'=' * 60}\n")
    for result_path in result_paths:
        try:
            evaluate_and_write_cot_metrics(evaluation, result_path)
        except Exception as e:
            print(f'Error evaluating {result_path}: {e}')
            continue
    if len(result_paths) > 1:
        print(f'\nGenerating comparison plots...')
        plot_model_comparison(args.output_dir, task=args.task)
    print(f"\n{'=' * 60}")
    print(f'All tasks completed!')
    print(f'Results saved to: {args.output_dir}')
    print(f"{'=' * 60}\n")
_shared_dataset_items = None
_shared_hf_dataset = None
_shared_task = 'detection'
_shared_output_dir = DEFAULT_RESULT_DIR
_shared_few_shot_source = None
_shared_few_shot_seed = None
_shared_sample_workers = DEFAULT_SAMPLE_WORKERS
_shared_split_context = None
_shared_reasoning_method = 'baseline'

def run_single_model(args_tuple):
    model_key, api_key = args_tuple
    try:
        result_path = query_model_on_dataset(model_key=model_key, dataset_items=_shared_dataset_items, hf_dataset=_shared_hf_dataset, split_context=_shared_split_context, result_path=None, api_key=api_key, task=_shared_task, output_dir=_shared_output_dir, few_shot_source=_shared_few_shot_source, few_shot_seed=_shared_few_shot_seed, sample_workers=_shared_sample_workers, reasoning_method=_shared_reasoning_method)
        return result_path
    except Exception as e:
        print(f'\n[{model_key}] Error: {e}')
        import traceback
        traceback.print_exc()
        raise
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    try:
        main()
    except DIORDatasetAccessError as exc:
        sys.stdout.flush()
        print(f'\n{exc}', file=sys.stderr)
        sys.exit(1)
