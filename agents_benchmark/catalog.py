"""Catalog."""
from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict
from urllib.parse import quote
from .storage import atomic_json

TASKS = {
    "Landmarks Recognition": "landmarks",
    "PatternNet Scene Recognition": "patternnet",
    "DIOR-RSVG Recognition": "dior_classification",
    "RarePlanes Aircraft Counting": "rareplanes",
    "COWC Car Counting": "cowc",
    "xView2 Pre-disaster Building Counting": "xview2",
    "NEON Tree Counting": "neon",
    "Aerial Animal Counting": "animals",
    "xView2 Change Detection": "xview2",
    "FPCD Change Detection": "fpcd",
    "DIOR-RSVG Ref-Det": "dior_localization",
    "DOTA Localization": "dota",
    "FAIR1M Localization": "fair1m",
}


DATASET_TASKS = {'landmarks':'landmarks', 'patternnet':'pattern-net', 'dior_classification':'dior-recognition',
    'rareplanes':'rare-planes', 'cowc':'cowc', 'xview2':'xview2', 'neon':'neon-trees',
    'animals':'aerial-animal', 'fpcd':'fpcd', 'dior_localization':'dior-refdet',
    'dota':'geobench-refdet', 'fair1m':'geobench-refdet'}

def response_text(row):
    return str(row.get('answer_text') or '')

def options_from_prompt(prompt):
    return [s.strip().removesuffix(';') for s in re.findall(r"^\d+\.\s*(.+)$", prompt, re.M)]

@dataclass(frozen=True)
class Sample:
    task: str
    sample_id: str
    source: str
    position: int
    row: dict

    @property
    def prompt(self):
        return self.row['prompt']

    def directory(self, output):
        return Path(output) / 'samples' / self.task / quote(self.sample_id, safe='')

    def identity(self):
        return {'task': self.task, 'sample_id': self.sample_id, 'source': self.source,
                'position': self.position, 'row': self.row}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()


def stratum(task, row):
    """Return the dataset annotation used for proportional sampling."""
    if task == 'landmarks':
        return str(row['name'])
    if task in {'patternnet', 'dior_classification', 'fpcd'}:
        return str(row['ground_truth'] if task == 'fpcd' else row['label'])
    if task in {'rareplanes', 'cowc', 'animals'}:
        return str(int(row['label']))
    if task == 'neon':
        return str(int(row['count']))
    if task == 'xview2':
        return str(int(row['gt']['count_before']))
    if task == 'dior_localization':
        return 'one_target'
    if task in {'dota', 'fair1m'}:
        return str(len(row['ground_truth_bboxes']))
    raise ValueError(f'Unknown task: {task}')


def stratified_sample(task, rows, maximum, seed=0):
    if maximum < 1 or seed < 0:
        raise ValueError('Sample limit must be positive and seed non-negative')
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        identity = str(row['sample_id'])
        if identity in ids:
            raise ValueError(f'Duplicate sample identity in {task}: {identity}')
        ids.add(identity)
        groups[stratum(task, row)].append(row)
    total = len(rows)
    if not total:
        raise ValueError(f'Empty task: {task}')
    size = min(maximum, total)
    # Largest-remainder allocation preserves the total and proportional strata.
    quotas = {key: size * len(group) // total for key, group in groups.items()}
    def tie_break(key):
        value = json.dumps([seed, task, key, 'quota'], ensure_ascii=True)
        return hashlib.sha256(value.encode()).hexdigest(), key
    order = sorted(groups, key=lambda key: (-(size * len(groups[key]) % total), tie_break(key)))
    for key in order[:size - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        def rank(row):
            value = json.dumps([seed, task, key, str(row['sample_id'])], ensure_ascii=True)
            return hashlib.sha256(value.encode()).hexdigest(), str(row['sample_id'])
        selected.extend(sorted(groups[key], key=rank)[:quotas[key]])
    selected.sort(key=lambda row: str(row['sample_id']))
    details = {'available': total, 'selected': size,
               'strata': {key: {'available': len(groups[key]), 'selected': quotas[key]}
                          for key in sorted(groups)}}
    return selected, details


def prepare_catalog(output, maximum, data_root=Path('raw_dataset'), seed=0):
    from evaluation.utils.dataset_catalog import dataset_rows
    output = Path(output).resolve()
    if (output / 'cohort.json').exists():
        raise FileExistsError('Use the existing immutable cohort or choose a new experiment directory')
    samples = []
    selection = {'seed': seed, 'max_per_task': maximum, 'tasks': {}}
    for dataset_task in dict.fromkeys(DATASET_TASKS.values()):
        rows = dataset_rows(dataset_task, data_root)
        grouped = defaultdict(list)
        for row in rows:
            task = row['source_dataset'].lower() if dataset_task == 'geobench-refdet' else next(
                t for t, d in DATASET_TASKS.items() if d == dataset_task)
            grouped[task].append(row)
        selected = []
        for task, group in sorted(grouped.items()):
            chosen, selection['tasks'][task] = stratified_sample(task, group, maximum, seed)
            selected.extend(chosen)
        rows = selected
        from evaluation.utils.reasoning_tasks import TaskInputs
        inputs = TaskInputs(dataset_task, rows, data_root)
        for position, source in enumerate(rows):
            row, uris = inputs.prepare(source)
            if dataset_task == 'geobench-refdet':
                task = row['source_dataset'].lower()
                row['target_boxes'] = row['ground_truth_bboxes']
            else:
                task = next(t for t, d in DATASET_TASKS.items() if d == dataset_task)

            row['image_files'] = []
            for uri in uris:
                header, encoded = uri.split(',', 1)
                raw = base64.b64decode(encoded)
                digest = hashlib.sha256(raw).hexdigest()
                suffix = '.png' if 'image/png' in header else '.webp' if 'image/webp' in header else '.jpg'
                name = Path('inputs') / (digest + suffix)
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                row['image_files'].append({'path': str(name), 'sha256': digest})
            samples.append(Sample(task, row['sample_id'], dataset_task, position, row))
    expected = set(DATASET_TASKS)
    if set(selection['tasks']) != expected:
        raise ValueError(f'Missing or unexpected tasks: {sorted(expected ^ set(selection["tasks"]))}')
    selection['selected_ids'] = {task: sorted(s.sample_id for s in samples if s.task == task)
                                 for task in sorted(expected)}
    atomic_json(output / 'sampling.json', selection)
    atomic_json(output / 'cohort.json', [s.identity() for s in samples])
    return samples


def load_frozen_catalog(output):
    path = Path(output) / 'cohort.json'
    if not path.is_file():
        raise FileNotFoundError('Run python -m agents_benchmark prepare for this experiment first')
    samples = [Sample(**row) for row in json.loads(path.read_text())]
    keys = [(s.task, s.sample_id) for s in samples]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Cohort identities must be nonempty and unique')
    metadata_path = Path(output) / 'sampling.json'
    if not metadata_path.is_file():
        raise ValueError('Cohort has no sampling provenance; prepare a new experiment directory')
    metadata = json.loads(metadata_path.read_text())
    selected = {task: sorted(s.sample_id for s in samples if s.task == task) for task in sorted(DATASET_TASKS)}
    if ({s.task for s in samples} != set(DATASET_TASKS)
            or metadata.get('selected_ids') != selected
            or any(not ids or len(ids) > metadata.get('max_per_task', 0) for ids in selected.values())):
        raise ValueError('Saved cohort differs from its sampling provenance')
    return samples

load_catalog = load_frozen_catalog
