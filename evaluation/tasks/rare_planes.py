"""Rare planes."""
from __future__ import annotations
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple
TASKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASKS_DIR.parent.parent
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import aerial_animal as base
from finetune.utils.rare_planes_data import DATASET_CONFIG, DATASET_ID, DEFAULT_RAW_DATASET_ROOT, SPLIT_SEED, RarePlanesBundle, RarePlanesRecord, build_official_test_bundle, build_rare_planes_bundle, download_rare_planes_images
DEFAULT_RESULT_DIR = str(base.RESULTS_ROOT / 'aircraft_counting' / 'test_selection')
DEFAULT_RANDOM_SEED = 0
DEFAULT_FEW_SHOT_K = 4
DEFAULT_FEW_SHOT_SEED = 0
LORA_TASK = 'rare-planes'
DATASET_DISPLAY_NAME = 'RarePlanes Real Aircraft Counting'
PLOT_TITLE = 'RarePlanes Real Satellite Imagery: General Aircraft Counting'
ANSWER_SCHEMA = '{"Answer": <nonnegative integer>}'
SYSTEM_MESSAGE = 'You are a careful remote-sensing image analyst. Count every visible aircraft in the satellite image exactly once. Return the requested JSON object and do not classify aircraft types.'
USER_PROMPT = 'How many aircraft are visible in this image?\nAnswer in JSON format, for example, {"Answer": 5}'
ANSWER_FORMAT_HINT = '{"Answer": 5} (replace 5 with the non-negative integer number of all visible aircraft)'
PROMPT_PROVENANCE = {'benchmark': 'GEOBench-VLM General Aircraft Counting', 'source_dataset': 'RarePlanes real observed satellite imagery', 'synthetic_imagery_used': False, 'aircraft_type_classification_used': False}
USER_PROMPT_SHA256 = hashlib.sha256(USER_PROMPT.encode('utf-8')).hexdigest()
PROMPT_PROTOCOL_SHA256 = hashlib.sha256(json.dumps({'system': SYSTEM_MESSAGE, 'user': USER_PROMPT, 'answer_schema': ANSWER_SCHEMA, 'provenance': PROMPT_PROVENANCE}, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
_BASE_CONFIGURE_CACHE = base.configure_cache_environment_from_args
_RAW_DATASET_ROOT = DEFAULT_RAW_DATASET_ROOT
_BUNDLE_CACHE: dict[tuple[Path, int], RarePlanesBundle] = {}

def configure_cache_environment_from_args(cache_dir=None):
    global _RAW_DATASET_ROOT
    _BASE_CONFIGURE_CACHE(None)
    raw_root = cache_dir or os.environ.get('RARE_PLANES_DATA_ROOT') or DEFAULT_RAW_DATASET_ROOT
    _RAW_DATASET_ROOT = Path(raw_root).expanduser().resolve()
    _RAW_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    return str(_RAW_DATASET_ROOT)

def format_count_answer(count: int) -> str:
    value = int(count)
    if value < 0:
        raise ValueError('RarePlanes aircraft count must be non-negative')
    return json.dumps({'Answer': value}, ensure_ascii=False)

def parse_count_response(response_text: str) -> Tuple[int, Optional[Dict[str, int]]]:
    from evaluation.utils.scoring import scalar_count
    value = scalar_count(response_text, 'Answer')
    return (-1, None) if value is None else (value, {'Answer': value})

def total_count_from_objects(objects, fallback_label: int=-1) -> int:
    if isinstance(objects, Mapping):
        boxes = objects.get('bbox')
        if isinstance(boxes, list):
            return len(boxes)
        count = objects.get('count')
        if type(count) is int and count >= 0:
            return count
    return int(fallback_label)

def squared_pearson_r2(y_true: List[float], y_pred: List[float]) -> float:
    return base._squared_pearson_corr([float(v) for v in y_true], [float(v) for v in y_pred])

class RarePlanesDataset:

    def __init__(self, bundle: RarePlanesBundle, split: str):
        self.bundle = bundle
        self.split = 'validation' if split in {'val', 'valid'} else split
        self.records = bundle.records(self.split)
        self._records_by_index = {record.raw_index: record for record in self.records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            record = self._records_by_index[index]
        except KeyError as exc:
            raise IndexError(f'RarePlanes {self.split} raw index is unavailable: {index}') from exc
        return {'image': {'path': str((self.bundle.dataset / record.image_relative_path).resolve())}, 'label': record.count, 'objects': {'bbox': [list(box) for box in record.bboxes], 'categories': ['aircraft'] * record.count, 'count': record.count}, 'source_id': record.source_id, 'official_split': record.official_split, 'file_name': record.file_name, 'source_scene_id': record.source_scene_id}

    def record_for_index(self, index: int) -> RarePlanesRecord:
        try:
            return self._records_by_index[index]
        except KeyError as exc:
            raise IndexError(f'RarePlanes {self.split} raw index is unavailable: {index}') from exc

def _load_bundle() -> RarePlanesBundle:
    root = _RAW_DATASET_ROOT.resolve()
    selection_seed = int(base.DEFAULT_RANDOM_SEED)
    cache_key = (root, selection_seed)
    bundle = _BUNDLE_CACHE.get(cache_key)
    if bundle is None:
        bundle = build_official_test_bundle(root, seed=selection_seed, download_images=True, strict=True)
        _BUNDLE_CACHE[cache_key] = bundle
    return bundle

def _metadata_item(record: RarePlanesRecord, index: int) -> dict[str, Any]:
    return {'index': index, 'label': record.count, 'objects': {'bbox': [list(box) for box in record.bboxes], 'categories': ['aircraft'] * record.count, 'count': record.count}, 'source_id': record.source_id, 'official_split': record.official_split, 'file_name': record.file_name, 'source_scene_id': record.source_scene_id}

def download_and_prepare_dataset(split: str='test'):
    bundle = _load_bundle()
    normalized = 'validation' if split in {'val', 'valid'} else split
    dataset = RarePlanesDataset(bundle, normalized)
    records = list(dataset.records)
    items = [_metadata_item(record, record.raw_index) for record in records]
    return (dataset, items)

def build_few_shot_source(hf_dataset, metadata_items: List[Dict], k: int, raw_split: str, canonical_split: str) -> Dict:
    candidates = []
    for item in metadata_items:
        count = int(item['label'])
        candidates.append({'index': item['index'], 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'image_path': hf_dataset[item['index']]['image']['path'], 'record': hf_dataset.record_for_index(item['index']), 'mime_type': 'image/png', 'count': count, 'answer': format_count_answer(count), 'source_id': item['source_id']})
    if len(candidates) < k:
        raise ValueError(f'RarePlanes train split has only {len(candidates)} few-shot candidates; need {k}')
    return {'candidates': candidates, 'k': k, 'split': raw_split, 'raw_split': raw_split, 'canonical_split': canonical_split, 'materialize_candidate': _materialize_few_shot_candidate}

def _materialize_few_shot_candidate(candidate: Dict) -> Dict:
    materialized = dict(candidate)
    path = Path(materialized['image_path'])
    if not path.is_file():
        record = materialized.get('record')
        if record is None:
            raise FileNotFoundError(f'RarePlanes Few-shot image is unavailable: {path}')
        download_rare_planes_images(_RAW_DATASET_ROOT, [record])
    raw = path.read_bytes()
    materialized['base64_image'] = base.raw_bytes_to_base64(raw)
    materialized['mime_type'] = base.detect_mime_type(raw)
    return materialized

def build_candidate_identity(item: Dict) -> Tuple[str, Any]:
    return (str(item.get('raw_split')), item.get('source_id', item.get('index')))

def completed_dataset_indices(path: str, dataset_items: List[Dict]) -> set[int]:
    expected_by_source = {str(item['source_id']): int(item['index']) for item in dataset_items if item.get('source_id') is not None}
    completed: set[int] = set()
    for record in base.iter_jsonl(path):
        if not isinstance(record, dict) or record.get('official_split') != 'test':
            continue
        source_id = record.get('source_id')
        if source_id is None:
            continue
        expected_index = expected_by_source.get(str(source_id))
        if expected_index is not None:
            completed.add(expected_index)
    return completed

def validate_resume_samples(path: str, dataset_items: List[Dict], model_key: str) -> None:
    if not os.path.exists(path):
        return
    expected_by_index: dict[int, Dict] = {}
    expected_by_source: dict[str, Dict] = {}
    for item in dataset_items:
        index = item.get('index')
        if type(index) is not int:
            raise ValueError(f'RarePlanes evaluation item has invalid index: {index!r}')
        if index in expected_by_index:
            raise ValueError(f'RarePlanes evaluation items contain duplicate index {index}')
        expected_by_index[index] = item
        source_id = item.get('source_id')
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f'RarePlanes evaluation item has invalid source_id: {source_id!r}')
        if source_id in expected_by_source:
            raise ValueError(f'RarePlanes evaluation items contain duplicate source_id {source_id}')
        expected_by_source[source_id] = item
    seen_sources: set[str] = set()
    for record in base.iter_jsonl(path):
        if not isinstance(record, dict):
            continue
        raw_index = record.get('index')
        index = base.normalize_result_index(raw_index)
        if raw_index is not None and index is None:
            raise ValueError(f'Existing result file {path} contains a non-integral sample index {raw_index!r}.')
        source_id = record.get('source_id')
        if not isinstance(source_id, str) or source_id not in expected_by_source:
            raise ValueError(f'Existing result file {path} contains an unselected or unidentified RarePlanes source_id={source_id!r}.')
        if source_id in seen_sources:
            raise ValueError(f'Existing result file {path} contains duplicate source_id={source_id!r}.')
        seen_sources.add(source_id)
        expected = expected_by_source[source_id]
        if index != expected.get('index'):
            raise ValueError(f"Existing result file {path} has index={index!r} for source_id={source_id!r}, but the current official test index is {expected.get('index')!r}.")
        existing_model = record.get('model')
        if existing_model != model_key:
            raise ValueError(f'Existing result file {path} has model={existing_model!r} at index {index}, but this run expects {model_key!r}.')
        for key in ('source_id', 'official_split', 'file_name', 'raw_split', 'canonical_split'):
            existing_value = record.get(key)
            expected_value = expected.get(key)
            if existing_value != expected_value:
                raise ValueError(f'Existing result file {path} has {key}={existing_value!r} at index {index}, but the current RarePlanes sample has {key}={expected_value!r}. Refusing to skip a different sample under the same index.')
        existing_label = record.get('label')
        expected_label = expected.get('label')
        if existing_label != expected_label:
            raise ValueError(f'Existing result file {path} has label={existing_label!r} at index {index}, but the current RarePlanes sample has label={expected_label!r}.')


def resolve_canonical_pools(few_shot_k: int) -> Dict:
    selection_seed = int(base.DEFAULT_RANDOM_SEED)
    bundle = _load_bundle()
    datasets = {split: RarePlanesDataset(bundle, split) for split in ('train', 'test')}
    evaluation_items = []
    for record in datasets['test'].records:
        item = _metadata_item(record, record.raw_index)
        item.update({'raw_split': 'test', 'canonical_split': 'test'})
        evaluation_items.append(item)
    few_shot_source = None
    if few_shot_k > 0:
        train_items = []
        for record in datasets['train'].records:
            item = _metadata_item(record, record.raw_index)
            item.update({'raw_split': 'train', 'canonical_split': 'train'})
            train_items.append(item)
        few_shot_source = build_few_shot_source(datasets['train'], train_items, few_shot_k, 'train', 'train')
    split_context = {'canonical_strategy_mode': 'official-test-random-selection', 'canonical_split_seed': int(base.DEFAULT_RANDOM_SEED), 'eval_raw_split': 'test', 'few_shot_source_raw_split': 'train', 'eval_canonical_split': 'test', 'few_shot_source_canonical_split': 'train', 'raw_split_map': {'train': 'train', 'val': 'validation', 'test': 'test'}}
    print('Evaluation state updated.', flush=True)
    return {'evaluation_dataset': datasets['test'], 'evaluation_items': evaluation_items, 'few_shot_source': few_shot_source, 'split_context': split_context}

def _configure_base_module() -> None:
    base.DEFAULT_RESULT_DIR = DEFAULT_RESULT_DIR
    base.DEFAULT_RANDOM_SEED = DEFAULT_RANDOM_SEED
    base.DEFAULT_FEW_SHOT_K = DEFAULT_FEW_SHOT_K
    base.DEFAULT_FEW_SHOT_SEED = DEFAULT_FEW_SHOT_SEED
    base.LORA_TASK = LORA_TASK
    base.DATASET_DISPLAY_NAME = DATASET_DISPLAY_NAME
    base.PLOT_TITLE = PLOT_TITLE
    base.SYSTEM_MESSAGE = SYSTEM_MESSAGE
    base.USER_PROMPT = USER_PROMPT
    base.ANSWER_FORMAT_HINT = ANSWER_FORMAT_HINT
    base.configure_cache_environment_from_args = configure_cache_environment_from_args
    base.format_count_answer = format_count_answer
    base.parse_count_response = parse_count_response
    base.total_count_from_objects = total_count_from_objects
    base.build_candidate_identity = build_candidate_identity
    base.completed_dataset_indices = completed_dataset_indices
    base.validate_resume_samples = validate_resume_samples
    base.build_few_shot_source = build_few_shot_source
    base.download_and_prepare_dataset = download_and_prepare_dataset
    base.resolve_canonical_pools = resolve_canonical_pools
    base.MODEL_CONFIGS = {name: dict(config) for name, config in base.MODEL_CONFIGS.items() if config.get('model_kind') != 'lora'}
    base.MODEL_CONFIGS.update(base.build_lora_model_configs(LORA_TASK))
    gpt_models = [name for name, config in base.MODEL_CONFIGS.items() if config.get('family') == 'GPT' and config.get('model_kind') != 'lora' and (config.get('api_type') == 'openai')]
    gemini_models = [name for name, config in base.MODEL_CONFIGS.items() if config.get('family') == 'Gemini' and config.get('model_kind') != 'lora' and (config.get('api_type') == 'google_native')]
_BASE_CONFIGURATION_NAMES = ('DEFAULT_RESULT_DIR', 'DEFAULT_RANDOM_SEED', 'DEFAULT_FEW_SHOT_K', 'DEFAULT_FEW_SHOT_SEED', 'LORA_TASK', 'DATASET_DISPLAY_NAME', 'PLOT_TITLE', 'SYSTEM_MESSAGE', 'USER_PROMPT', 'ANSWER_FORMAT_HINT', 'configure_cache_environment_from_args', 'format_count_answer', 'parse_count_response', 'total_count_from_objects', 'build_candidate_identity', 'completed_dataset_indices', 'validate_resume_config', 'validate_resume_samples', 'build_few_shot_source', 'download_and_prepare_dataset', 'resolve_canonical_pools', 'MODEL_CONFIGS')
_BASE_CONFIGURATION_ENV_KEYS = ()
_MISSING = object()

@contextmanager
def _base_module_configuration():
    saved = {name: getattr(base, name, _MISSING) for name in _BASE_CONFIGURATION_NAMES}
    saved_env = {key: os.environ.get(key, _MISSING) for key in _BASE_CONFIGURATION_ENV_KEYS}
    _configure_base_module()
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                try:
                    delattr(base, name)
                except AttributeError:
                    pass
            else:
                setattr(base, name, value)
        for key, value in saved_env.items():
            if value is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def _build_model_configs() -> Dict[str, Dict[str, Any]]:
    configs = {name: dict(config) for name, config in base.MODEL_CONFIGS.items() if config.get('model_kind') != 'lora'}
    configs.update(base.build_lora_model_configs(LORA_TASK))
    return configs
MODEL_CONFIGS = _build_model_configs()

def evaluation(result_path: str, ax=None) -> Dict:
    with _base_module_configuration():
        return base.evaluation(result_path, ax=ax)

def plot_model_comparison(result_dir: str=DEFAULT_RESULT_DIR, evaluate_all_methods: bool=False):
    with _base_module_configuration():
        return base.plot_model_comparison(result_dir, evaluate_all_methods=evaluate_all_methods)

def main() -> None:
    from reasoning_effort import run_reasoning_task_cli
    if run_reasoning_task_cli('rare-planes'):
        return
    with _base_module_configuration():
        base.main()
from reasoning_effort import register_reasoning_models
register_reasoning_models(MODEL_CONFIGS)
if __name__ == '__main__':
    main()
