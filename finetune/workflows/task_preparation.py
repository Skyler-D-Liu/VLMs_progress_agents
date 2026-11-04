"""Task preparation."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from finetune.config import load_base_lora_config, load_dataset_config, resolve_experiment_config
from finetune.config.runtime_context import active_base_config, active_dataset_config, active_registry
from finetune.datasets.prepared_task_adapter import COUNT_KEYS, load_authoritative_task_semantics
from finetune.model_backends import get_model_backend
from finetune.datasets.task_registry import get_task_dataset
from finetune.workflows.pipeline_catalog import base_config_for_task, dataset_config_for_task, get_pipeline_spec
from finetune.training.checkpoint_manager import atomic_write_json
from finetune.utils.dataset_splits import build_split_assignments, discover_available_splits_from_names
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = active_registry(PROJECT_ROOT / 'finetune/experiment_registry.yaml')
SUPPORTED_TASKS = ('cowc', 'dior-rsvg', 'dior-rsvg-recognition', 'fpcd', 'geobench-refdet', 'landmarks', 'neon-tree', 'pattern-net', 'xview2', 'rare-planes')

@dataclass(frozen=True)
class SourceDescriptor:
    dataset_key: str
    raw_split: str
    raw_index: int
    source_qualifier: str = ''

def parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=SUPPORTED_TASKS)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--dataset-config', type=Path, default=active_dataset_config())
    return parser.parse_args(argv)

def _load_parquet(paths: Sequence[Path], image_columns: Sequence[str]):
    from datasets import Dataset, Image, load_dataset
    try:
        dataset = load_dataset('parquet', data_files={'train': [str(path) for path in paths]}, split='train', cache_dir='./raw_dataset', download_mode='reuse_dataset_if_exists')
    except OSError as exc:
        if getattr(exc, 'errno', None) != 122:
            raise
        import pyarrow as pa
        import pyarrow.parquet as pq
        tables = [pq.read_table(path) for path in paths]
        dataset = Dataset(pa.concat_tables(tables) if len(tables) > 1 else tables[0])
    for column in image_columns:
        dataset = dataset.cast_column(column, Image(decode=False))
    return dataset

def _snapshot(config) -> Path:
    slug = 'datasets--' + config.dataset.id.replace('/', '--')
    root = config.paths.raw_dataset_root / slug / 'snapshots'
    candidates = sorted((path for path in root.iterdir() if path.is_dir()))
    if len(candidates) != 1:
        raise FileNotFoundError(f'expected exactly one cached snapshot for {config.dataset.id}, found {len(candidates)}')
    return candidates[0]

def _load_sources(config, authority):
    datasets: dict[str, Any] = {}
    descriptors: dict[str, list[SourceDescriptor]] = {}
    shard_paths: list[Path] = []
    snapshot = _snapshot(config)
    if config.task == 'cowc':
        descriptors['train'] = []
        for city in config.dataset.configs:
            paths = tuple(sorted((snapshot / city).glob('train-*.parquet')))
            if not paths:
                raise FileNotFoundError(f'missing COWC shards for {city}: {snapshot / city}')
            dataset = _load_parquet(paths, ('image',))
            key = f'city:{city}'
            datasets[key] = dataset
            descriptors['train'].extend((SourceDescriptor(key, 'train', index, city) for index in range(len(dataset))))
            shard_paths.extend(paths)
        return (datasets, descriptors, shard_paths)
    if config.task == 'neon-tree':
        from datasets import Dataset, Image, concatenate_datasets
        root = config.paths.raw_dataset_root / 'CanopyRS___neon_tree_evaluation/default'
        for split in ('train', 'validation', 'test'):
            paths = tuple(sorted(root.rglob(f'neon_tree_evaluation-{split}*.arrow')))
            if not paths:
                raise FileNotFoundError(f'missing Neon Tree Arrow files for {split}: {root}')
            parts = [Dataset.from_file(str(path)) for path in paths]
            dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
            dataset = dataset.cast_column('image', Image(decode=False))
            datasets[split] = dataset
            descriptors[split] = [SourceDescriptor(split, split, index) for index in range(len(dataset))]
            shard_paths.extend(paths)
        return (datasets, descriptors, shard_paths)
    config_dir = config.dataset.config or 'data'
    data_root = snapshot / (config_dir if config.task in {'landmarks', 'xview2'} else 'data')
    raw_splits = {'dior-rsvg': ('train', 'val', 'test'), 'landmarks': ('train',), 'pattern-net': ('train',), 'xview2': ('train', 'test')}[config.task]
    image_columns = ('image1', 'image2') if config.task == 'xview2' else ('image',)
    for split in raw_splits:
        paths = tuple(sorted(data_root.glob(f'{split}-*.parquet')))
        if not paths:
            raise FileNotFoundError(f'missing {config.task} shards for {split}: {data_root}')
        dataset = _load_parquet(paths, image_columns)
        datasets[split] = dataset
        descriptors[split] = [SourceDescriptor(split, split, index) for index in range(len(dataset))]
        shard_paths.extend(paths)
    return (datasets, descriptors, shard_paths)

def _split_assignments(config, descriptors, datasets=None, authority=None):
    split_map = discover_available_splits_from_names(descriptors)
    stratify = None
    if config.task == 'cowc':
        stratify = {'train': [descriptor.source_qualifier for descriptor in descriptors['train']]}
    assignments = build_split_assignments(split_map=split_map, split_lengths={split: len(values) for split, values in descriptors.items()}, seed=config.splits.seed, stratify_values_by_split=stratify)
    output = {}
    for canonical in ('train', 'validation', 'test'):
        key = 'val' if canonical == 'validation' else canonical
        assignment = assignments[key]
        output[canonical] = [descriptors[assignment['raw_split']][index] for index in assignment['indices']]
    if config.task == 'landmarks':
        if datasets is None or authority is None:
            raise ValueError('Landmarks split filtering requires dataset metadata and authority')
        from finetune.workflows.canonical_cohorts import load_cohort
        canonical_test = {str(entry['key']) for entry in load_cohort('landmarks', config.paths.raw_dataset_root)}
        metadata = datasets['train'].select_columns([column for column in datasets['train'].column_names if column != 'image'])
        for canonical in output:
            output[canonical] = [descriptor for descriptor in output[canonical] if f'{descriptor.raw_split}:{int(descriptor.raw_index)}' in canonical_test or len(authority.build_landmark_options(metadata[descriptor.raw_index], num_distractors=4, seed=config.splits.seed) or []) == 5]

        def _landmark_key(descriptor: SourceDescriptor) -> str:
            return f'{descriptor.raw_split}:{int(descriptor.raw_index)}'
        available = {_landmark_key(descriptor) for split_descriptors in output.values() for descriptor in split_descriptors}
        missing = sorted(canonical_test - available)
        if missing:
            raise RuntimeError(f'landmarks canonical test cohort is unavailable after filtering: {missing[:5]}')
        test_rows = [descriptor for split_descriptors in output.values() for descriptor in split_descriptors if _landmark_key(descriptor) in canonical_test]
        test_keys = {_landmark_key(descriptor) for descriptor in test_rows}
        validation_rows = [descriptor for descriptor in output['validation'] if _landmark_key(descriptor) not in test_keys]
        training_pool = [descriptor for descriptor in output['train'] if _landmark_key(descriptor) not in test_keys]
        output = {'train': training_pool, 'validation': validation_rows, 'test': test_rows}
    return output

def _raw_image_bytes(value: Any) -> bytes:
    if isinstance(value, Mapping):
        if value.get('bytes') is not None:
            return bytes(value['bytes'])
        if value.get('path'):
            return Path(value['path']).read_bytes()
    if isinstance(value, bytes):
        return value
    raise ValueError('image field has neither bytes nor path')

def _materialize_image(raw: bytes, destination_stem: Path) -> dict[str, Any]:
    from PIL import Image
    with Image.open(BytesIO(raw)) as image:
        width, height = image.size
        image_format = (image.format or 'PNG').casefold()
        image.verify()
    materialized = raw
    if image_format in {'tiff', 'tif'}:
        converted = BytesIO()
        with Image.open(BytesIO(raw)) as image:
            image.convert('RGB').save(converted, format='PNG')
        materialized = converted.getvalue()
        image_format = 'png'
    extension = {'jpeg': '.jpg', 'jpg': '.jpg', 'png': '.png', 'webp': '.webp'}.get(image_format)
    if extension is None:
        raise ValueError(f'unsupported image format: {image_format}')
    path = destination_stem.with_suffix(extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_bytes() != materialized:
        temporary = path.with_name(f'.{path.name}.preparing-{str(os.getpid())}')
        temporary.write_bytes(materialized)
        if temporary.read_bytes() != materialized:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f'prepared image copy differs: {temporary}')
        os.replace(temporary, path)
    return {'path': str(path.resolve()), 'width': width, 'height': height}

def _cowc_count(row, authority) -> int:
    objects = row.get('objects')
    count = len(objects.get('bbox') or []) if isinstance(objects, Mapping) else 0
    if count <= 0:
        count = sum((int(row.get(field, 0) or 0) for field in authority.COUNT_HEADER))
    return int(count)

def _record_semantics(task: str, row: Mapping[str, Any], authority, images, seed: int):
    system = authority.SYSTEM_MESSAGE
    metadata: dict[str, Any] = {'width': images[0]['width'], 'height': images[0]['height'], 'images': images}
    if task == 'cowc':
        truth = _cowc_count(row, authority)
        return (system, authority.get_user_prompt(), authority.format_count_answer(truth), truth, metadata)
    if task == 'dior-rsvg':
        obj = authority.first_object(row)
        bbox = authority._coerce_bbox(obj.get('bbox'))
        description = obj.get('captions')
        if not bbox or not description:
            raise ValueError('DIOR-RSVG row has no valid first bbox/caption')
        prompt = authority.get_detection_prompt(images[0]['height'], images[0]['width'], description)
        normalized = [int(value) if float(value).is_integer() else float(value) for value in bbox]
        metadata['description'] = description
        return (system, prompt, json.dumps({'bbox': normalized}, ensure_ascii=False), normalized, metadata)
    if task == 'landmarks':
        options = authority.build_landmark_options(row, num_distractors=4, seed=seed)
        name = str(row.get('name') or '').strip()
        if options is None or len(options) < 2 or name not in options:
            raise ValueError('Landmarks row does not have a valid option set')
        metadata['options'] = list(options)
        truth = {'name': name, 'options': list(options)}
        return (system, authority.get_user_prompt(options), name, truth, metadata)
    if task == 'neon-tree':
        annotations = row.get('annotations')
        if not isinstance(annotations, Mapping) or not isinstance(annotations.get('bbox'), list):
            raise ValueError('Neon Tree annotations.bbox is missing')
        truth = len(annotations['bbox'])
        return (system, authority.USER_PROMPT, str(truth), truth, metadata)
    if task == 'pattern-net':
        category = authority.reassign_label(row.get('label'))
        if category is None:
            raise ValueError(f"PatternNet label is not canonical: {row.get('label')!r}")
        return (system, authority.get_user_prompt(), authority.format_category_answer(category), category, metadata)
    if task == 'xview2':
        truth = authority.parse_ground_truth(row.get('objects1'), row.get('objects2'))
        truth = {key: int(truth[key]) for key in COUNT_KEYS}
        return (system, authority.USER_PROMPT, authority.format_count_answer(truth), truth, metadata)
    raise ValueError(f'unsupported task: {task}')

def _source_id(config, descriptor: SourceDescriptor) -> str:
    qualifier = f':{descriptor.source_qualifier}' if descriptor.source_qualifier else ''
    return f'{config.dataset.id}:{descriptor.raw_split}{qualifier}:{descriptor.raw_index:08d}'

def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.preparing-{str(os.getpid())}')
    with temporary.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def main(argv: Sequence[str] | None=None) -> int:
    args = parse_args(argv)
    registration = get_task_dataset(args.task)
    spec = get_pipeline_spec(args.task)
    base_path = args.base_config or active_base_config(base_config_for_task(spec))
    base = load_base_lora_config(base_path, args.registry)
    get_model_backend(base).validate_config(base)
    dataset_path = args.dataset_config or dataset_config_for_task(spec)
    dataset_config = load_dataset_config(dataset_path)
    resolved = resolve_experiment_config(base, dataset_config, args.registry)
    if resolved.dataset.task != args.task:
        raise ValueError('dataset config task does not match --task')
    config = resolved.dataset
    semantics = load_authoritative_task_semantics(config)
    authority = semantics.module
    if config.task == 'fpcd':
        from finetune.workflows.fpcd_preparation import prepare_fpcd
        return prepare_fpcd(config, semantics)
    if config.task == 'rare-planes':
        from finetune.workflows.rare_planes_preparation import prepare_rare_planes
        return prepare_rare_planes(config, semantics)
    if config.task == 'dior-rsvg-recognition':
        from finetune.workflows.dior_rsvg_recognition_preparation import prepare_dior_rsvg_recognition
        return prepare_dior_rsvg_recognition(config, semantics)
    if config.task == 'geobench-refdet':
        from finetune.workflows.geobench_refdet_preparation import prepare_geobench_refdet
        return prepare_geobench_refdet(config, semantics)
    datasets, descriptors, shard_paths = _load_sources(config, authority)
    assignments = _split_assignments(config, descriptors, datasets, authority)
    records_root = config.paths.prepared_dir / 'records'
    image_root = config.paths.prepared_dir / 'images'
    manifest_entries: list[dict[str, Any]] = []
    all_source_ids: set[str] = set()
    for canonical_split, split_descriptors in assignments.items():
        records = []
        for position, descriptor in enumerate(split_descriptors):
            row = datasets[descriptor.dataset_key][descriptor.raw_index]
            source_id = _source_id(config, descriptor)
            if source_id in all_source_ids:
                raise RuntimeError(f'source leakage across canonical splits: {source_id}')
            all_source_ids.add(source_id)
            image_fields = ('image1', 'image2') if config.task == 'xview2' else ('image',)
            images = [_materialize_image(_raw_image_bytes(row[field]), image_root / descriptor.raw_split / f"{source_id.split(':')[-1]}-{position:08d}-{image_index}") for image_index, field in enumerate(image_fields)]
            system, user, assistant, truth, metadata = _record_semantics(config.task, row, authority, images, config.splits.seed)
            sample_id = f'{config.task}:{canonical_split}:{position:08d}'
            record = {'task': config.task, 'dataset_id': config.dataset.id, 'canonical_split': canonical_split, 'raw_split': descriptor.raw_split, 'raw_index': descriptor.raw_index, 'source_qualifier': descriptor.source_qualifier, 'sample_id': sample_id, 'source_id': source_id, 'image_paths': [image['path'] for image in images], 'system_message': system, 'user_prompt': user, 'assistant_answer': assistant, 'ground_truth': truth, 'metadata': metadata}
            records.append(record)
            manifest_entries.append({'sample_id': sample_id, 'source_id': source_id, 'canonical_split': canonical_split, 'raw_split': descriptor.raw_split, 'raw_index': descriptor.raw_index})
        record_path = records_root / f'{canonical_split}.jsonl'
        _write_jsonl(record_path, records)
    manifest_dir = config.paths.manifest_dir
    prompt_manifest = {'task': config.task, 'authoritative_script': str(semantics.script), 'system_message': semantics.system_message, 'system_symbol': config.prompt.system_symbol, 'user_symbol': config.prompt.user_symbol, 'dynamic': config.prompt.dynamic, 'dynamic_builder_symbols': list(config.prompt.dynamic_builder_symbols)}
    split_manifest = {'task': config.task, 'dataset': config.dataset.model_dump(mode='json'), 'split_strategy': config.splits.strategy.value, 'split_seed': config.splits.seed, 'entries': manifest_entries, 'test_data_used_for_tuning': False}
    prompt_path = manifest_dir / 'prompt_manifest.json'
    split_path = manifest_dir / 'split_manifest.json'
    atomic_write_json(prompt_path, prompt_manifest)
    atomic_write_json(split_path, split_manifest)
    marker = {'task': config.task, 'source_shards': [{'path': str(path), 'size_bytes': path.stat().st_size} for path in shard_paths], 'training_started': False, 'validation_generation_run': False, 'test_inference_run': False}
    atomic_write_json(manifest_dir / 'PREPARATION_COMPLETE.json', marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return 0
