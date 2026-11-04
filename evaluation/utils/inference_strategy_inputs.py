"""Inference strategy inputs."""
from __future__ import annotations
import random
from pathlib import Path
from reasoning_effort import ROOT, GPT_MODEL, GEMINI_MODEL
from reasoning_tasks import COPY_FIELDS, TaskInputs, reference_rows
OUTPUT_ROOT = ROOT / 'results/inference/inference_strategies'

def matched_rows(task, cache_root=Path('raw_dataset')):
    return reference_rows(task, cache_root)

def image_keys(row, default_split='test'):
    keys = set()
    split = row.get('raw_split') or default_split
    index = row.get('row_index', row.get('index'))
    if index is not None:
        keys.add(('row', row.get('city', ''), split, str(index)))
    for field in ('path', 'image_path', 'file_name'):
        if row.get(field):
            keys.add(('image', row.get('city', ''), Path(str(row[field])).name))
    if row.get('source_id'):
        keys.add(('source', str(row['source_id'])))
    return keys

def exclude_test_images(candidates, test_rows, default_split='test'):
    forbidden = set().union(*(image_keys(row, default_split) for row in test_rows))
    return [row for row in candidates if not image_keys(row, 'train') & forbidden]

def normalized_example(example, default_prompt):
    images = example.get('image_urls') or [example.get('image_url')]
    if not images or any((not isinstance(uri, str) or not uri.startswith('data:image/') for uri in images)):
        raise ValueError('Demonstration has no inline image')
    identity = example.get('source_id', example.get('sample_id'))
    if identity is None:
        identity = ':'.join((str(example.get(k, '')) for k in ('city', 'raw_split', 'index')))
    return dict(sample_id=str(identity), images=images, prompt=example.get('user_prompt') or example.get('prompt') or default_prompt, answer=str(example['answer']))

class StrategyInputs:

    def __init__(self, task, rows, cache_root, strategy, support_checkpoint=None):
        self.base = TaskInputs(task, rows, cache_root)
        self.task, self.rows, self.strategy = (task, rows, strategy)
        self.module = m = self.base.module
        self.source, self.fixed = (None, None)
        rng = random.Random(0)
        self.seeds = {r['sample_id']: rng.randrange(2 ** 32) for r in rows}
        if strategy == 'cot':
            return
        if task == 'fpcd':
            records = m.select_few_shot_records(self.base.bundle.records('train'), 4)
            if {r.source_id for r in records} & {r['source_id'] for r in rows}:
                raise ValueError('FPCD support overlaps test')
            self.fixed = m.build_few_shot_examples(self.base.bundle, records)
        elif task == 'geobench-refdet':
            # DOTA/FAIR1M samples are divided 80:10:10 (image-disjoint); one fixed
            # set of four training samples with distinct images (seed-0 shuffle)
            # is used for every test query (Supplementary Material S.2-A).
            from evaluation.tasks.create_geobench_refdet_manifest import partition_rows
            _, manifest = m.geo.load_manifest(str(Path(cache_root) / 'geobench_spatial'))
            candidates = m.build_items(manifest, 'refdet')
            train_ids = set(partition_rows(candidates)['train'])
            pool = exclude_test_images([row for row in candidates if row['sample_id'] in train_ids], rows)
            selected = m.geo.few_shot_support_rows(pool, 'refdet', support_size=4, seed=0)
            self.fixed = m._build_few_shot_examples(self.base.data_root, selected, 'refdet', 4)
        elif task == 'cowc':
            cities = {}
            for city in m.DEFAULT_CITIES:
                payload = m.load_canonical_city_payload(city)
                assignment = payload['assignments']['train']
                items = m.select_metadata_items_by_indices(payload['metadata_items'], assignment['indices'], raw_split=assignment['raw_split'], canonical_split='train')
                cities[city] = dict(hf_dataset=payload['hf_dataset'], metadata_items=items)
            self.source = m.build_few_shot_source(cities, 4, 'train', 'train')
        elif task == 'neon-trees':
            self.source = m.resolve_canonical_pools(4)['few_shot_source']
        elif task == 'dior-refdet':
            self.source = m.resolve_canonical_pools(4, 'detection')['few_shot_source']
        elif task == 'dior-recognition':
            self.source = m.resolve_canonical_pools(4)['few_shot_source']
        elif task == 'landmarks' and support_checkpoint is not None:
            self.source = m.resolve_canonical_pools(4, support_checkpoint=support_checkpoint)['few_shot_source']
        else:
            self.source = m.resolve_canonical_pools(4)['few_shot_source']
        if self.source is not None:
            candidates = self.source['candidates']
            dataset = self.source.get('hf_dataset')
            if task in {'dior-recognition', 'neon-trees'} and 'path' in dataset.column_names:
                paths = dataset.select_columns(['path'])
                candidates = [dict(c, path=c.get('path') or paths[c.get('row_index', c['index'])]['path']) for c in candidates]
            default = 'train' if task in {'landmarks', 'pattern-net', 'cowc'} else 'test'
            if task == 'neon-trees':
                default = 'train'
            self.source['candidates'] = exclude_test_images(candidates, rows, default)
            if len(self.source['candidates']) < 4:
                raise ValueError(f'{task}: fewer than four image-disjoint support candidates')
            print(f"{task}: {len(self.source['candidates'])} image-disjoint support candidates", flush=True)

    def prepare(self, source):
        from cot_prompting import effective_prompt, effective_system_message
        prepared, images = self.base.prepare(source)
        examples = []
        if self.strategy == 'cot':
            prepared['base_prompt'] = prepared['prompt']
            prepared['base_system_message'] = prepared['system_message']
            prepared['prompt'] = effective_prompt(prepared['prompt'], 'cot')
            prepared['system_message'] = effective_system_message(prepared['system_message'], 'cot')
            prepared['reasoning_method'] = 'cot'
        else:
            m = self.module
            if self.fixed is not None:
                selected = self.fixed
            elif self.task == 'rare-planes':
                selected = m.base.sample_few_shot_examples(self.source, self.seeds[source['sample_id']])
                selected = [dict(e, user_prompt=m.USER_PROMPT) for e in selected]
            else:
                selected = m.sample_few_shot_examples(self.source, self.seeds[source['sample_id']])
            examples = [normalized_example(e, prepared['prompt']) for e in selected]
            if len(examples) != 4 or len({e['sample_id'] for e in examples}) != 4:
                raise ValueError('Exactly four distinct demonstrations required')
            prepared.update(few_shot_k=4, few_shot_seed=0, prompt_mode='few-shot', few_shot_sample_seed=self.seeds[source['sample_id']], demonstrations=[{k: v for k, v in e.items() if k != 'images'} for e in examples])
        prepared.update(strategy=self.strategy, reasoning_effort='default')
        return (prepared, images, examples)

