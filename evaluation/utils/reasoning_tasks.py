"""Reasoning tasks."""
from __future__ import annotations
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from reasoning_effort import ROOT
TASKS_DIR = str(ROOT / 'evaluation/tasks')
if TASKS_DIR not in sys.path:
    sys.path.insert(0, TASKS_DIR)
LOCALIZATION_TASKS = frozenset({'dior-refdet', 'geobench-refdet'})


@dataclass(frozen=True)
class Task:
    module: str
    scores: tuple[tuple[str, str, str], ...]

TASKS = {
    'landmarks': Task('landmarks', (('Landmarks Recognition', 'landmarks_accuracy', 'Accuracy'),)),
    'pattern-net': Task('pattern_net', (('PatternNet Scene Recognition', 'patternnet_accuracy', 'Accuracy'),)),
    'dior-recognition': Task('dior_rsvg_recognition', (('DIOR-RSVG Recognition', 'dior_recognition_accuracy', 'Accuracy'),)),
    'rare-planes': Task('rare_planes', (('RarePlanes Aircraft Counting', 'rareplanes_r2', 'R²'),)),
    'cowc': Task('cowc', (('COWC Car Counting', 'cowc_r2', 'R²'),)),
    'xview2': Task('xview2', (('xView2 Pre-disaster Building Counting', 'xview2_predisaster_r2', 'R²'), ('xView2 Change Detection', 'xview2_change_r2', 'R²'))),
    'neon-trees': Task('neon_trees', (('NEON Tree Counting', 'neon_tree_r2', 'R²'),)),
    'aerial-animal': Task('aerial_animal', (('Aerial Animal Counting', 'aerial_animal_r2', 'R²'),)),
    'fpcd': Task('fpcd', (('FPCD Change Detection', 'fpcd_accuracy', 'Accuracy'),)),
    'dior-refdet': Task('dior_rsvg_localization', (('DIOR-RSVG Ref-Det', 'dior_refdet_miou', 'mIoU'),)),
    'geobench-refdet': Task('geobench_localization', (('DOTA Localization', 'geobench_refdet_miou', 'mIoU'), ('FAIR1M Localization', 'geobench_refdet_miou', 'mIoU'))),
}
RAW_ROOT = Path("raw_dataset")
RARE_ROOT = Path('./raw_dataset')
COPY_FIELDS = ('index', 'sample_id', 'source_id', 'question_id', 'sample_key', 'row_index', 'object_index', 'city', 'name', 'options', 'label', 'count', 'gt', 'object', 'objects', 'objects1', 'objects2', 'ground_truth', 'ground_truth_bboxes', 'path', 'image_path', 'image_width', 'image_height', 'height', 'width', 'raw_split', 'canonical_split', 'file_name', 'official_split', 'official_prompt', 'bbox', 'instanceOfLabels', 'lat', 'lon', 'source_scene_id', 'source_dataset', 'ground_truth_polygons')

def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reference_rows(task, root=RAW_ROOT, *, reference_path=None):
    from evaluation.utils.dataset_catalog import dataset_rows
    if reference_path is not None:
        raise ValueError("Evaluation inputs must come from dataset annotations, not predictions")
    return dataset_rows(task, root)

class TaskInputs:

    def __init__(self, task, rows, cache_root=RAW_ROOT, model=None):
        self.task, self.rows, self.model = (task, rows, model)
        self.module = importlib.import_module(TASKS[task].module)
        m = self.module
        self.datasets = {}
        self.geo_items = {}
        if task == 'rare-planes':
            m.configure_cache_environment_from_args(str(cache_root))
            self.dataset, items = m.download_and_prepare_dataset('test')
            if not {r['source_id'] for r in rows} <= {x['source_id'] for x in items}:
                raise ValueError('RarePlanes official test identities differ from the reference')
            self.rare_indices = {x['source_id']: x['index'] for x in items}
        elif task == 'fpcd':
            m.pin_canonical_fpcd_source()
            self.bundle = m.build_fpcd_bundle(Path(cache_root), strict=True)
            self.fpcd_records = {r.source_id: r for r in self.bundle.records('test')}
            if not {r['source_id'] for r in rows} <= set(self.fpcd_records):
                raise ValueError('FPCD test identities differ from the reference')
        elif task == 'geobench-refdet':
            self.data_root, manifest = m.geo.load_manifest(str(Path(cache_root) / 'geobench_spatial'))
            self.geo_items = {str(r['sample_id']): r for r in m.build_items(manifest, 'refdet')}
        elif task == 'cowc':
            m.configure_cache_environment_from_args(str(cache_root))
            for city in sorted({r['city'] for r in rows}):
                payload = m.load_canonical_city_payload(city)
                self.datasets[city] = payload['hf_dataset']
                labels = {x['index']: x['label'] for x in payload['metadata_items']}
                allowed = set(payload['assignments']['test']['indices'])
                for r in rows:
                    if r['city'] == city and (r['index'] not in allowed or labels[r['index']] != r['label']):
                        raise ValueError('COWC split or label mismatch')
        elif task == 'neon-trees':
            m.configure_cache_environment_from_args(str(cache_root))
            self.dataset = m.resolve_canonical_pools(0)['evaluation_dataset']
            if rows:
                for row in rows:
                    if m.get_ground_truth_count(self.dataset[row['index']]) != m.get_ground_truth_count(row):
                        raise ValueError('NEON count differs from dataset annotation')
        elif task == 'dior-recognition':
            m.dior.configure_cache_environment_from_args(str(cache_root))
            self.dataset = m.dior.load_dior_dataset('test')
            if rows:
                actual = {r['sample_id']: r for r in m.build_object_items(self.dataset, sorted({r['row_index'] for r in rows}), 'test', 'test')}
                for row in rows:
                    current = actual.get(row['sample_id'])
                    if current is None or any((current[k] != row[k] for k in ('row_index', 'object_index', 'bbox', 'label'))):
                        raise ValueError('DIOR reference target differs from source annotations')
        elif task == 'dior-refdet':
            m.configure_cache_environment_from_args(str(cache_root))
            self.dataset = m.load_dior_dataset('test')
        else:
            m.configure_cache_environment_from_args(str(cache_root))
            raw_split = 'train' if task in {'landmarks', 'pattern-net'} else 'test'
            self.dataset, items = m.download_and_prepare_dataset(split=raw_split)
            by_index = {r['index']: r for r in items}
            for r in rows:
                current = by_index[r['index']]
                for field in ('label', 'name'):
                    if field in current and field in r and (current[field] != r[field]):
                        raise ValueError(f"{task} sample {r['sample_id']} {field} mismatch")

    def prepare(self, source):
        m, task = (self.module, self.task)
        row = {k: v for k, v in source.items() if not k.startswith('reference_')}
        generation = {}
        if task == 'fpcd':
            record = self.fpcd_records[source['source_id']]
            if record.answer != source['ground_truth']:
                raise ValueError('FPCD ground truth changed')
            images = m.provider_backend.image_data_uris_from_raw_bytes_list(list(self.bundle.read_image_bytes(record)))
            prompt = m.USER_PROMPT
        elif task == 'geobench-refdet':
            item = self.geo_items[source['sample_id']]
            if item['ground_truth_bboxes'] != source['ground_truth_bboxes']:
                raise ValueError('GEOBench target boxes changed')
            path = (self.data_root / item['image_path']).resolve()
            if not path.is_relative_to(self.data_root.resolve()):
                raise ValueError('Image is outside the GEOBench dataset')
            images = [m._image_uri(path.read_bytes())]
            width, height = (int(item['image_width']), int(item['image_height']))
            source_meta = m.geo.source_metadata(item)
            dataset = m.geo.source_dataset(source) or m.geo.source_dataset(item)
            if dataset is None:
                raise ValueError('Localization input requires source_dataset')
            prompt = m._bbox_prompt(str(item['prompt']), width, height, dataset)
            row.update(image_width=width, image_height=height, source_dataset=dataset)
            if dataset == 'FAIR1M':
                row['ground_truth_polygons'] = m.geo.ground_truth_polygons(item)
            generation = {'temperature': 0, 'candidateCount': 1}
        elif task == 'dior-recognition':
            raw = m.dior.get_raw_image_bytes(self.dataset[source['row_index']]['image'])
            uri, height, width = m.annotate_target_box(raw, source['bbox'])
            images, prompt = ([uri], m.USER_PROMPT)
            row.update(image_height=height, image_width=width)
        elif task == 'dior-refdet':
            raw = m.get_raw_image_bytes(self.dataset[source['index']]['image'])
            height, width = m.get_image_size(raw)
            prompt = m.get_detection_prompt(height, width, source['object']['captions'])
            images = [m.raw_bytes_to_base64_uri(raw)]
            row.update(image_height=height, image_width=width)
        elif task == 'xview2':
            current = self.dataset[source['index']]
            if m.parse_ground_truth(current['objects1'], current['objects2']) != source['gt']:
                raise ValueError('xView2 ground truth differs from dataset annotations')
            images = m.image_data_uris_from_raw_bytes_list([m.read_image_raw_bytes(self.dataset, source['index'], column) for column in ('image1', 'image2')])
            prompt = m.USER_PROMPT
        elif task == 'landmarks':
            raw = m.get_raw_image_bytes(self.dataset[source['index']])
            images = [m.image_data_uri_from_raw_bytes(raw)]
            prompt = m.get_user_prompt(source['options'])
        elif task == 'neon-trees':
            raw = m.get_raw_image_bytes(self.dataset[source['index']])
            images, prompt = ([m.raw_bytes_to_png_data_uri(raw)], m.get_user_prompt())
        elif task == 'rare-planes':
            item = self.dataset[self.rare_indices[source['source_id']]]
            if item['label'] != source['label']:
                raise ValueError('RarePlanes count changed')
            images = [m.base.raw_bytes_to_base64_uri(m.base.get_raw_image_bytes(item))]
            prompt = m.USER_PROMPT
        elif task == 'cowc':
            raw = m.get_raw_image_bytes(self.datasets[source['city']][source['index']])
            images, prompt = ([m.raw_bytes_to_base64_uri(raw)], m.get_user_prompt())
        else:
            raw = m.get_raw_image_bytes(self.dataset[source['index']])
            images, prompt = ([m.raw_bytes_to_base64_uri(raw)], m.USER_PROMPT)
        backend = m.base if task == 'rare-planes' else m
        system = m.SYSTEM_MESSAGE if hasattr(m, 'SYSTEM_MESSAGE') else backend.SYSTEM_MESSAGE
        from PIL import Image
        import base64
        import io
        image_settings = []
        for uri in images:
            header, encoded = uri.split(',', 1)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                image_settings.append({'mime_type': header[5:-7], 'width': image.width, 'height': image.height})
        row.update(prompt=prompt, system_message=system, image_settings=image_settings, generation_settings=generation, prompt_mode='zero-shot', few_shot_k=0, reasoning_method='baseline', task=task)
        return (row, images)

def parse_result(task, row):
    from evaluation.utils.response_protocol import parse_answer, format_valid
    text = row.get('answer_text', '')
    parsed = parse_answer(task, text, row)
    valid = format_valid(task, text, row)
    row['parsed_answer'] = parsed
    row['parse_failed'] = not valid
    if row.get('status') in {'ok', 'parse_failed'}:
        row['status'] = 'ok' if valid else 'parse_failed'
    return row

def task_metrics(task, rows):
    from task_scoring import SCORERS
    summaries = []
    for name, scorer, metric in TASKS[task].scores:
        subset = rows
        expected = len(rows)
        if task == 'geobench-refdet':
            dataset = name.split()[0]
            subset = [r for r in rows if r['source_dataset'] == dataset]
            expected = len(subset)
        answered = subset
        summaries.append({'dataset_task': name, 'metric': metric, 'score': SCORERS[scorer](answered) if answered else None, 'complete': len(subset) == expected and all((r['status'] != 'request_failed' or r.get('retry_exhausted', False) for r in subset)), 'elapsed_seconds': sum((r.get('elapsed_seconds', 0) for r in subset)), **{k: sum((r.get('token_usage', {}).get(k, 0) for r in subset)) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')}})
    return summaries
