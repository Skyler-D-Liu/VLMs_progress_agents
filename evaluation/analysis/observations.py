"""Observations."""
from agents_benchmark.catalog import TASKS
from evaluation.utils.scoring import parse, truth, matching_iou, COUNT_KEYS

CATEGORIES = {
    'dior_classification':'Object Classification', 'landmarks':'Scene Understanding',
    'patternnet':'Scene Understanding', 'dior_localization':'Object Localization',
    'dota':'Object Localization', 'fair1m':'Object Localization', 'fpcd':'Change Detection',
    'rareplanes':'Object Counting', 'cowc':'Object Counting', 'neon':'Object Counting', 'animals':'Object Counting',
}


def observations(samples, answers):
    if not answers or any(len(values)!=len(samples) for values in answers.values()):
        raise ValueError('Every arm must provide an aligned answer for each input')
    rows = []
    for i, sample in enumerate(samples):
        target = truth(sample)
        predicted = {arm:parse(sample, values[i]) if values[i] is not None else None for arm,values in answers.items()}
        task = sample.task
        row = sample.row
        cluster = row.get('source_scene_id') or row.get('image_path') or row.get('path') or row.get('file_name')
        if not cluster:
            cluster = f"{row.get('city','')}:{row.get('raw_split','')}:{row.get('row_index',row.get('index',sample.sample_id))}"
        dataset = 'dior' if task.startswith('dior_') else task
        for metric_name, metric_task in TASKS.items():
            if metric_task != task:
                continue
            kind = 'mean'
            category = CATEGORIES.get(task)
            if task in {'landmarks','patternnet','dior_classification','fpcd'}:
                y = 1.
                p = {a:float(v == target) for a,v in predicted.items()}
            elif task in {'dior_localization','dota','fair1m'}:
                y = 1.
                p = {a:matching_iou(v or [], target, oriented=task=='fair1m') for a,v in predicted.items()}
            else:
                kind = 'r2'
                if task == 'xview2':
                    change = 'Change' in metric_name
                    keys = COUNT_KEYS[1:] if change else COUNT_KEYS[:1]
                    category = 'Change Detection' if change else 'Object Counting'
                    y = [target[k] for k in keys]
                    p = {a:[v[k] if v and v.get(k) is not None else 0 for k in keys] for a,v in predicted.items()}
                elif task == 'animals':
                    y = sum(target.values())
                    p = {a:sum(v.values()) if v and all(x is not None for x in v.values()) else 0 for a,v in predicted.items()}
                else:
                    y = target
                    p = {a:v if v is not None else 0 for a,v in predicted.items()}
            rows.append({'task':metric_name, 'category':category, 'dataset':dataset,
                         'sample_id':sample.sample_id, 'cluster_id':str(cluster), 'metric':kind,
                         'truth':y, 'predictions':p})
    return rows
