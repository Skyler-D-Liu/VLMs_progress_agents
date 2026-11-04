"""Task scoring."""
from functools import partial
from agents_benchmark.catalog import Sample, DATASET_TASKS
from evaluation.utils.benchmark_evaluation import response_text
from evaluation.utils.scoring import score_group


def prepared_samples(task, rows):
    samples = []
    for index, source in enumerate(rows):
        row = dict(source)
        slug = row.get('source_dataset', '').lower() if task == 'geobench-refdet' else task
        if slug in {'patternnet', 'dior_classification'} and not row.get('prompt'):
            import importlib
            module = importlib.import_module('pattern_net' if slug == 'patternnet' else 'dior_rsvg_recognition')
            row['prompt'] = module.USER_PROMPT
        if slug == 'neon' and 'count' not in row:
            row['count'] = len(row['objects']['bbox']) if 'objects' in row else row['label']
        if slug == 'xview2' and 'gt' not in row:
            import xview2
            row['gt'] = xview2.parse_ground_truth_from_record(row)
        if slug in {'fair1m','dota'}:
            row['target_boxes'] = row['ground_truth_bboxes']
        samples.append(Sample(slug, str(row.get('sample_id', index)), DATASET_TASKS[slug], index, row))
    return samples


def score(task, metric, rows):
    if not rows:
        return 0.0
    import sys
    root = str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'tasks')
    if root not in sys.path:
        sys.path.insert(0, root)
    answers = [None if row.get('error') or row.get('status') in
               {'error','failed','request_failed','incomplete'} else response_text(row) for row in rows]
    samples = prepared_samples(task, rows)
    if task == 'geobench-refdet':
        values = [score_group([sample],[answer],metric)['score'] for sample,answer in zip(samples,answers)]
        return sum(values)/len(values)
    return score_group(samples, answers, metric)['score']


_NAMES = {
 'landmarks_accuracy':('landmarks','Landmarks Recognition'),
 'dior_recognition_accuracy':('dior_classification','DIOR-RSVG Recognition'),
 'patternnet_accuracy':('patternnet','PatternNet Scene Recognition'),
 'rareplanes_r2':('rareplanes','RarePlanes Aircraft Counting'),
 'cowc_r2':('cowc','COWC Car Counting'),
 'xview2_predisaster_r2':('xview2','xView2 Pre-disaster Building Counting'),
 'neon_tree_r2':('neon','NEON Tree Counting'),
 'aerial_animal_r2':('animals','Aerial Animal Counting'),
 'dior_refdet_miou':('dior_localization','DIOR-RSVG Ref-Det'),
 'geobench_refdet_miou':('geobench-refdet','Localization'),
 'xview2_change_r2':('xview2','xView2 Change Detection'),
 'fpcd_accuracy':('fpcd','FPCD Change Detection'),
}
SCORERS = {name:partial(score, task, metric) for name,(task,metric) in _NAMES.items()}
