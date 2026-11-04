"""Dataset splits."""
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping
DEFAULT_SPLIT_SEED = 0
STANDARD_SPLITS = ('train', 'val', 'test')
_VAL_ALIASES = frozenset(('val', 'validation'))

def normalize_split_name(raw_name):
    if raw_name is None:
        return None
    normalized = str(raw_name).strip().lower()
    if normalized == 'train':
        return 'train'
    if normalized in _VAL_ALIASES:
        return 'val'
    if normalized in ('test', 'evaluation'):
        return 'test'
    return None

def discover_available_splits_from_names(raw_split_names):
    ordered_raw_names = []
    seen_raw_names = set()
    for raw_name in raw_split_names:
        raw_name_str = str(raw_name).strip()
        if not raw_name_str or raw_name_str in seen_raw_names:
            continue
        seen_raw_names.add(raw_name_str)
        ordered_raw_names.append(raw_name_str)
    if not ordered_raw_names:
        raise ValueError('No raw split names were provided')
    if len(ordered_raw_names) == 1:
        raw_name = ordered_raw_names[0]
        normalized = normalize_split_name(raw_name)
        if normalized is None:
            normalized = 'train'
        return {normalized: raw_name}
    split_map = {}
    unrecognized = []
    for raw_name in ordered_raw_names:
        normalized = normalize_split_name(raw_name)
        if normalized is None:
            unrecognized.append(raw_name)
            continue
        previous = split_map.get(normalized)
        if previous is not None and previous != raw_name:
            raise ValueError('Conflicting raw splits map to {!r}: {!r}, {!r}'.format(normalized, previous, raw_name))
        split_map[normalized] = raw_name
    if unrecognized:
        raise ValueError('Unsupported raw split names: {}. Detected raw splits: {}'.format(', '.join(sorted(unrecognized)), ', '.join(sorted(ordered_raw_names))))
    return split_map

def discover_available_splits(snapshot_path):
    snapshot_root = Path(snapshot_path)
    parquet_files = sorted(snapshot_root.rglob('*.parquet'))
    if not parquet_files:
        raise FileNotFoundError('No parquet files found under {}'.format(snapshot_root))
    raw_split_names = []
    seen_raw_splits = set()
    for parquet_path in parquet_files:
        raw_name = parquet_path.name.split('-', 1)[0]
        if raw_name not in seen_raw_splits:
            seen_raw_splits.add(raw_name)
            raw_split_names.append(raw_name)
    return discover_available_splits_from_names(raw_split_names)

def resolve_split_strategy(split_map):
    if not split_map:
        raise ValueError('split_map is empty')
    split_keys = set(split_map)
    if split_keys == set(STANDARD_SPLITS):
        return {'mode': 'reuse_all', 'output_to_source': {'train': split_map['train'], 'val': split_map['val'], 'test': split_map['test']}, 'resplit_source': None, 'fixed_output_source': {'train': split_map['train'], 'val': split_map['val'], 'test': split_map['test']}, 'ratios': {}}
    if split_keys == set(('train', 'test')):
        return {'mode': 'resplit_train_keep_test', 'output_to_source': {'train': split_map['train'], 'val': split_map['train'], 'test': split_map['test']}, 'resplit_source': split_map['train'], 'fixed_output_source': {'test': split_map['test']}, 'ratios': {'train': 0.9, 'val': 0.1}}
    if split_keys == set(('train', 'val')):
        return {'mode': 'resplit_train_promote_val_to_test', 'output_to_source': {'train': split_map['train'], 'val': split_map['train'], 'test': split_map['val']}, 'resplit_source': split_map['train'], 'fixed_output_source': {'test': split_map['val']}, 'ratios': {'train': 0.9, 'val': 0.1}}
    if len(split_map) == 1:
        raw_name = next(iter(split_map.values()))
        return {'mode': 'resplit_single_source', 'output_to_source': {'train': raw_name, 'val': raw_name, 'test': raw_name}, 'resplit_source': raw_name, 'fixed_output_source': {}, 'ratios': {'train': 0.8, 'val': 0.1}}
    raise ValueError('Unsupported split combination: {} (raw mapping: {})'.format(', '.join(sorted(split_keys)), split_map))

def _validate_ratios(train_ratio, val_ratio):
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio < 0.0 or val_ratio < 0.0:
        raise ValueError('Split ratios must be non-negative.')
    if train_ratio + val_ratio > 1.0:
        raise ValueError('train_ratio + val_ratio must be <= 1.0.')
    return (train_ratio, val_ratio)

def random_split_indices(indices, seed=DEFAULT_SPLIT_SEED, train_ratio=0.8, val_ratio=0.1):
    train_ratio, val_ratio = _validate_ratios(train_ratio, val_ratio)
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    if abs(train_ratio + val_ratio - 1.0) < 1e-12:
        val_end = total
    else:
        val_end = train_end + int(total * val_ratio)
    return {'train': sorted(shuffled[:train_end]), 'val': sorted(shuffled[train_end:val_end]), 'test': sorted(shuffled[val_end:])}

def build_seeded_80_10_10_splits(indices, seed=DEFAULT_SPLIT_SEED):
    split = random_split_indices(indices, seed=seed, train_ratio=0.8, val_ratio=0.1)
    return {'train': split['train'], 'validation': split['val'], 'test': split['test']}

def stratified_split_indices(indices, group_values, seed=DEFAULT_SPLIT_SEED, train_ratio=0.8, val_ratio=0.1):
    train_ratio, val_ratio = _validate_ratios(train_ratio, val_ratio)
    index_list = list(indices)
    group_list = list(group_values)
    if len(index_list) != len(group_list):
        raise ValueError('indices and group_values must have the same length')
    grouped_indices = defaultdict(list)
    for index, group_value in zip(index_list, group_list):
        grouped_indices[str(group_value)].append(index)
    merged = {'train': [], 'val': [], 'test': []}
    for group_name in sorted(grouped_indices):
        split_indices = random_split_indices(grouped_indices[group_name], seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
        for split_name in STANDARD_SPLITS:
            merged[split_name].extend(split_indices[split_name])
    for split_name in STANDARD_SPLITS:
        merged[split_name] = sorted(merged[split_name])
    return merged

def build_split_assignments(split_map, split_lengths, seed=DEFAULT_SPLIT_SEED, stratify_values_by_split=None):
    strategy = resolve_split_strategy(split_map)
    if strategy['mode'] == 'reuse_all':
        assignments = {}
        for output_split in STANDARD_SPLITS:
            raw_split = strategy['output_to_source'][output_split]
            if raw_split not in split_lengths:
                raise KeyError('Missing split length for raw split {!r}'.format(raw_split))
            assignments[output_split] = {'raw_split': raw_split, 'indices': list(range(int(split_lengths[raw_split])))}
        return assignments
    assignments = {'train': None, 'val': None, 'test': None}
    if stratify_values_by_split is None:
        stratify_values_by_split = {}
    for output_split, raw_split in (strategy.get('fixed_output_source') or {}).items():
        if raw_split not in split_lengths:
            raise KeyError('Missing split length for raw split {!r}'.format(raw_split))
        assignments[output_split] = {'raw_split': raw_split, 'indices': list(range(int(split_lengths[raw_split])))}
    resplit_source = strategy.get('resplit_source')
    if resplit_source is not None:
        if resplit_source not in split_lengths:
            raise KeyError('Missing split length for raw split {!r}'.format(resplit_source))
        total_size = int(split_lengths[resplit_source])
        all_indices = list(range(total_size))
        ratios = strategy.get('ratios') or {}
        group_values = stratify_values_by_split.get(resplit_source)
        if group_values is not None:
            split_indices = stratified_split_indices(all_indices, group_values, seed=seed, train_ratio=ratios['train'], val_ratio=ratios['val'])
        else:
            split_indices = random_split_indices(all_indices, seed=seed, train_ratio=ratios['train'], val_ratio=ratios['val'])
        for output_split in STANDARD_SPLITS:
            if strategy['output_to_source'].get(output_split) != resplit_source:
                continue
            assignments[output_split] = {'raw_split': resplit_source, 'indices': list(split_indices[output_split])}
    for output_split in STANDARD_SPLITS:
        if assignments[output_split] is None:
            raise ValueError('Failed to build canonical {!r} assignment for split_map {}'.format(output_split, split_map))
    return assignments

def build_leakage_aware_source_id_splits(train_source_ids: Iterable[str], test_source_ids: Iterable[str], *, seed=DEFAULT_SPLIT_SEED, train_ratio=0.9, val_ratio=0.1, excluded_train_source_ids: Iterable[str]=(), excluded_test_source_ids: Iterable[str]=()):
    train_ratio, val_ratio = _validate_ratios(train_ratio, val_ratio)
    if abs(train_ratio + val_ratio - 1.0) >= 1e-12:
        raise ValueError('source-ID train/validation ratios must sum to 1.0')
    native_train = [str(value) for value in train_source_ids]
    native_test = [str(value) for value in test_source_ids]
    if len(native_train) != len(set(native_train)):
        raise ValueError('train_source_ids contains duplicate identities')
    if len(native_test) != len(set(native_test)):
        raise ValueError('test_source_ids contains duplicate identities')
    excluded_train = {str(value) for value in excluded_train_source_ids}
    excluded_test = {str(value) for value in excluded_test_source_ids}
    native_test_set = set(native_test)
    overlap = set(native_train) & native_test_set
    candidates = [source_id for source_id in native_train if source_id not in native_test_set and source_id not in excluded_train]
    canonical_test = [source_id for source_id in native_test if source_id not in excluded_test]
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    return {'train': sorted(shuffled[:train_end]), 'val': sorted(shuffled[train_end:]), 'test': sorted(canonical_test), 'overlap': sorted(overlap), 'excluded_train': sorted(excluded_train & set(native_train)), 'excluded_test': sorted(excluded_test & native_test_set)}

def infer_cowc_city(row):
    if not isinstance(row, Mapping):
        raise ValueError('Expected a mapping-like row when inferring COWC city')
    city = row.get('city')
    if city is not None:
        city_str = str(city).strip()
        if city_str:
            return city_str
    for file_name_key in ('file_name', 'File_Name'):
        file_name = row.get(file_name_key)
        if file_name is None:
            continue
        prefix = str(file_name).strip().split('_', 1)[0].strip()
        if prefix:
            return prefix
    raise ValueError("Unable to infer COWC city from row; expected 'city', 'file_name', or 'File_Name'")
