"""Canonical cohorts."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT_ROOT = PROJECT_ROOT / 'raw_dataset/manifests/cohorts'
TASK_ALIASES = {'dior-rsvg-recognition': 'dior-recognition', 'dior-rsvg': 'dior-refdet', 'neon-tree': 'neon-trees'}
TASK_NAMES = ('landmarks', 'dior-rsvg-recognition', 'pattern-net', 'rare-planes', 'cowc', 'xview2', 'neon-tree', 'aerial-animal', 'dior-rsvg', 'geobench-refdet', 'fpcd')

def _number(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)

def cohort_key(task: str, value: Mapping[str, Any] | Any) -> str:
    if not isinstance(value, Mapping):
        return str(value)
    source = str(value.get('source_id') or '')
    sample = str(value.get('sample_id') or '')
    if task in {'landmarks', 'neon-tree', 'aerial-animal', 'rare-planes', 'xview2'}:
        raw_split = value.get('raw_split') or 'test'
        raw_index = value.get('raw_index', value.get('index'))
        if raw_index is not None:
            return f'{raw_split}:{_number(raw_index)}'
        match = re.search('(?:^|:)(train|test):0*(\\d+)$', source or sample)
        if match:
            return f'{match.group(1)}:{int(match.group(2))}'
    if task == 'pattern-net':
        raw_index = value.get('raw_index', value.get('index'))
        if raw_index is not None:
            return f"{value.get('raw_split') or 'train'}:{_number(raw_index)}"
        match = re.search(':(train|test):0*(\\d+)$', source or sample)
        if match:
            return f'{match.group(1)}:{int(match.group(2))}'
    if task == 'cowc':
        city = value.get('city') or value.get('source_qualifier')
        index = value.get('index', value.get('raw_index'))
        if city is not None and index is not None:
            return f'{city}:{_number(index)}'
        match = re.search(':(train|test):([^:]+):0*(\\d+)$', source)
        if match:
            return f'{match.group(2)}:{int(match.group(3))}'
    if task == 'dior-rsvg-recognition':
        if source and re.fullmatch('(?:train|test):\\d+:\\d+', source):
            return source
        if sample and re.fullmatch('(?:train|test):\\d+:\\d+', sample):
            return sample
        if value.get('row_index') is not None:
            return f"{value.get('raw_split', 'test')}:{int(value['row_index'])}:{int(value.get('object_index', 0))}"
    if task == 'dior-rsvg':
        index = value.get('index', value.get('raw_index'))
        if index is not None:
            return _number(index)
    if task == 'geobench-refdet':
        question_id = value.get('question_id')
        if question_id is not None:
            return f'refdet:{question_id}'
        match = re.search('(?:^|:)refdet:\\d+$', source or sample)
        if match:
            return match.group(0).lstrip(':')
    if task == 'fpcd':
        return source or sample
    return source or sample

def canonical_truth(task: str, row: Mapping[str, Any]) -> Any:
    if task == 'landmarks':
        return str(row.get('name') or row.get('ground_truth') or '').strip()
    if task == 'dior-rsvg-recognition':
        return str(row.get('label') or row.get('ground_truth') or '').strip()
    if task in {'rare-planes', 'cowc', 'neon-tree', 'aerial-animal'}:
        if task == 'aerial-animal' and isinstance(row.get('objects'), Mapping):
            return int(row.get('label', sum((len(row['objects'].get(k, [])) for k in ('bbox',)))))
        return int(row.get('label', row.get('count', row.get('ground_truth', 0))))
    if task == 'xview2':
        value = row.get('gt', row.get('ground_truth', {}))
        return tuple((float(value.get(key, 0) or 0) for key in ('count_before', 'no_damage', 'minor_damage', 'major_damage', 'destroyed'))) if isinstance(value, Mapping) else value
    if task == 'fpcd':
        return str(row.get('ground_truth') or row.get('label') or '').strip().upper()
    if task == 'dior-rsvg':
        obj = row.get('object') if isinstance(row.get('object'), Mapping) else row
        return tuple((float(x) for x in obj.get('bbox') or []))
    if task == 'geobench-refdet':
        value = row.get('ground_truth_polygons') or row.get('ground_truth_bboxes', row.get('ground_truth', []))
        return json.loads(json.dumps(value))
    return row.get('ground_truth')

def build_cohort(task: str, data_root=PROJECT_ROOT / 'raw_dataset') -> tuple[dict[str, Any], ...]:
    from evaluation.utils.dataset_catalog import dataset_rows
    rows = dataset_rows(TASK_ALIASES.get(task, task), Path(data_root))
    entries = [{'key': cohort_key(task, row), 'ground_truth': canonical_truth(task, row)} for row in rows]
    if len({entry['key'] for entry in entries}) != len(entries):
        raise ValueError('Dataset identities are not unique')
    return tuple(entries)

def write_cohort(task: str, path: Path | None=None, data_root=PROJECT_ROOT / 'raw_dataset') -> Path:
    destination = path or Path(data_root) / 'manifests/cohorts' / f'{task}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {'task': task, 'entries': list(build_cohort(task, data_root))}
    destination.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return destination

def load_cohort(task: str, data_root=PROJECT_ROOT / 'raw_dataset') -> tuple[dict[str, Any], ...]:
    current = build_cohort(task, data_root)
    path = Path(data_root) / 'manifests/cohorts' / f'{task}.json'
    if path.is_file():
        payload = json.loads(path.read_text(encoding='utf-8'))
        expected = {'task':task, 'entries':list(current)}
        if json.dumps(payload, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f'Saved cohort differs from dataset annotations: {path}')
    return current

def select_examples(task: str, examples: Iterable[Any], data_root=PROJECT_ROOT / 'raw_dataset') -> tuple[Any, ...]:
    wanted = {str(entry['key']): entry for entry in load_cohort(task, data_root)}
    selected = [example for example in examples if cohort_key(task, {'source_id': example.source_id, 'sample_id': example.sample_id, **dict(example.metadata)}) in wanted]
    observed = [cohort_key(task, {'source_id': example.source_id, 'sample_id': example.sample_id, **dict(example.metadata)}) for example in selected]
    if len(selected) != len(wanted) or set(observed) != set(wanted):
        missing = sorted(set(wanted) - set(observed))
        raise ValueError(f'prepared {task} test split does not cover canonical cohort: missing {missing[:5]}')
    return tuple(selected)

def main(argv: list[str] | None=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'raw_dataset')
    parser.add_argument('--tasks', default=','.join(TASK_NAMES))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    tasks = [item.strip() for item in args.tasks.split(',') if item.strip()]
    for task in tasks:
        write_cohort(task, args.output_dir / f'{task}.json' if args.output_dir else None, args.data_root)
    return 0
