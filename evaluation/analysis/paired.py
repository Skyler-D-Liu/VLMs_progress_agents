"""Paired."""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def metric_score(truth, prediction, metric, variant='primary'):
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite aligned truth and prediction arrays are required')
    if not len(y):
        return float('nan')
    if metric == 'mean':
        if np.any((p < 0) | (p > 1)):
            raise ValueError('Accuracy/IoU scores must be in [0, 1]')
        return float(p.mean())
    if y.ndim not in (1, 2):
        raise ValueError('Expected scalar counts or vectors of damage counts')
    if np.any(y < 0) or np.any(p < 0):
        raise ValueError('Counts must be nonnegative')
    if metric != 'r2':
        raise ValueError(f'Unknown metric: {metric}')
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    if variant in {'mae', 'rmse'}:
        if variant == 'mae':
            error = np.abs(p-y).mean(0)
            reference = np.abs(y-np.median(y, axis=0)).mean(0)
        else:
            error = np.sqrt(np.square(p-y).mean(0))
            reference = np.sqrt(np.square(y-y.mean(0)).mean(0))
        skill = np.where(reference > 0, 1-np.divide(error, reference, out=np.zeros_like(error), where=reference>0), (error == 0).astype(float))
        return float(np.maximum(0, skill).mean())
    a, b = y-y.mean(0), p-p.mean(0)
    denominator = np.square(a).sum(0)*np.square(b).sum(0)
    squared = np.divide(np.square((a*b).sum(0)), denominator, out=np.zeros_like(denominator), where=denominator>0)
    return float(np.clip(squared, 0, 1).mean())


class PairedData:
    def __init__(self, rows):
        self.rows = rows
        if not rows:
            raise ValueError('No observations')
        self.arms = sorted(rows[0]['predictions'])
        if not self.arms:
            raise ValueError('At least one model prediction is required')
        self.tasks = sorted({r['task'] for r in rows})
        seen = set()
        self.categories = {}
        self.clusters = {}
        self.indices = {}
        for i, row in enumerate(rows):
            if any(row.get(field) is None or str(row[field]) == '' for field in ('task','category','dataset','sample_id','cluster_id')):
                raise ValueError('Task, category, dataset, sample and cluster identities are required')
            key = row['task'], row['sample_id']
            if key in seen or sorted(row['predictions']) != self.arms:
                raise ValueError('Duplicate identity or incomplete paired predictions')
            seen.add(key)
            previous = self.categories.setdefault(row['task'], row['category'])
            if previous != row['category']:
                raise ValueError('Task category changed')
            cluster = row['dataset'], str(row['cluster_id'])
            self.clusters.setdefault(cluster, []).append(i)
            self.indices.setdefault(row['task'], []).append(i)
        self.strata = {}
        for key in sorted(self.clusters):
            coverage = tuple(sorted({rows[i]['task'] for i in self.clusters[key]}))
            self.strata.setdefault((key[0], coverage), []).append(key)
        self.metrics = {}
        for task in self.tasks:
            kinds = {rows[i]['metric'] for i in self.indices[task]}
            if len(kinds) != 1:
                raise ValueError('Task metric changed')
            self.metrics[task] = kinds.pop()
        self.arrays = {}
        self.positions = {}
        for task, indices in self.indices.items():
            self.positions[task] = {index:position for position,index in enumerate(indices)}
            self.arrays[task] = (np.asarray([rows[i]['truth'] for i in indices], dtype=float),
                                {arm:np.asarray([rows[i]['predictions'][arm] for i in indices], dtype=float) for arm in self.arms})

        self.scores()

    def scores(self, selected=None, swapped=None, variant='primary'):
        groups = self.indices if selected is None else {t: [] for t in self.tasks}
        if selected is not None:
            for i in selected:
                groups[self.rows[i]['task']].append(i)
        result = {}
        arms = self.arms if swapped is None else list(swapped[:2])
        for task, indices in groups.items():
            positions = [self.positions[task][i] for i in indices]
            truth_array, prediction_arrays = self.arrays[task]
            y = truth_array[positions]
            result[task] = {}
            mask = None
            if swapped is not None:
                left, right, keys = swapped
                mask = np.asarray([(self.rows[i]['dataset'], str(self.rows[i]['cluster_id'])) in keys for i in indices], dtype=bool)
            for arm in arms:
                p = prediction_arrays[arm][positions]
                if mask is not None:
                    other = right if arm == left else left
                    condition = mask if p.ndim == 1 else mask[:, None]
                    p = np.where(condition, prediction_arrays[other][positions], p)
                result[task][arm] = metric_score(y, p, self.metrics[task], variant)
        return result

    def bootstrap_indices(self, rng):
        indices = []
        for clusters in self.strata.values():
            for j in rng.integers(len(clusters), size=len(clusters)):
                indices.extend(self.clusters[clusters[j]])
        return indices

    def aggregate(self, scores):
        arms = list(scores[self.tasks[0]])
        result = {f'task:{t}': scores[t] for t in self.tasks}
        category_scores = {}
        for category in sorted(set(self.categories.values())):
            tasks = [t for t in self.tasks if self.categories[t] == category]
            category_scores[category] = {a: float(np.mean([scores[t][a] for t in tasks])) for a in arms}
            result['category:'+category] = category_scores[category]
        result['overall'] = {a: float(np.mean([scores[t][a] for t in self.tasks])) for a in arms}
        result['category_balanced'] = {a: float(np.mean([r[a] for r in category_scores.values()])) for a in arms}
        return result


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values)-rank)*values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def interval(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {'lower':None, 'upper':None, 'valid_draw_fraction':0.}
    low, high = np.quantile(valid, [.025,.975])
    return {'lower':float(low), 'upper':float(high), 'valid_draw_fraction':float(len(valid)/len(values))}


def count_errors(data, selected=None):
    groups = data.indices if selected is None else {t: [] for t in data.tasks}
    if selected is not None:
        for i in selected:
            groups[data.rows[i]['task']].append(i)
    result = {}
    for task, indices in groups.items():
        if data.metrics[task] != 'r2':
            continue
        result[task] = {}
        for arm in data.arms:
            y = np.asarray([data.rows[i]['truth'] for i in indices], dtype=float)
            pred = np.asarray([data.rows[i]['predictions'][arm] for i in indices], dtype=float)
            if not len(y):
                values = {name: float('nan') for name in ('mae','rmse','percent_mae')}
            else:
                absolute = np.abs(pred-y)
                values = {'mae':float(absolute.mean()),
                          'rmse':float(np.sqrt(np.square(pred-y).mean(axis=0)).mean()),
                          'percent_mae':float(np.mean(100*absolute.sum(axis=0)/y.sum(axis=0))) if np.all(y.sum(axis=0)>0) else float('nan')}
            result[task][arm] = values
    return result


def analyze(rows, draws=2000, seed=0, resampling_unit="sample"):
    if resampling_unit not in {"cluster", "sample"}:
        raise ValueError("Resampling unit must be cluster or sample")
    if resampling_unit == "sample":
        rows = [dict(row, cluster_id=json.dumps([row["task"], row["sample_id"]])) for row in rows]
    if draws < 2:
        raise ValueError('At least two resampling draws are required')
    data = PairedData(rows)
    point = data.aggregate(data.scores())
    rng = np.random.default_rng(seed)
    bootstrap, error_draws = [], []
    variant_draws = {'mae':[], 'rmse':[]}
    for _ in range(draws):
        selected = data.bootstrap_indices(rng)
        bootstrap.append(data.aggregate(data.scores(selected)))
        error_draws.append(count_errors(data, selected))
        for variant in variant_draws:
            variant_draws[variant].append(data.aggregate(data.scores(selected, variant=variant)))
    estimates = {level:{arm:{'estimate':value, 'ci95':interval([r[level][arm] for r in bootstrap])}
                        for arm,value in values.items()} for level,values in point.items()}
    comparisons = []
    cluster_keys = list(data.clusters)
    for left, right in combinations(data.arms, 2):
        exceed = {level:0 for level in point}
        observed = {level:point[level][left]-point[level][right] for level in point}
        for _ in range(draws):
            keys = {key for key, flag in zip(cluster_keys, rng.integers(2, size=len(cluster_keys))) if flag}
            permuted = data.aggregate(data.scores(swapped=(left,right,keys)))
            for level in point:
                if abs(permuted[level][left]-permuted[level][right]) >= abs(observed[level])-1e-12:
                    exceed[level] += 1
        for level in point:
            differences = np.asarray([r[level][left]-r[level][right] for r in bootstrap])
            differences = differences[np.isfinite(differences)]
            bootstrap_p = float(min(1., 2*min(np.mean(differences <= 0), np.mean(differences >= 0)))) if len(differences) else None
            comparisons.append({'level':level, 'left':left, 'right':right, 'difference':observed[level],
                'ci95':interval([r[level][left]-r[level][right] for r in bootstrap]),
                'bootstrap_p':bootstrap_p,
                'permutation_p':(exceed[level]+1)/(draws+1)})
    for row, adjusted in zip(comparisons, holm([r['permutation_p'] for r in comparisons])):
        row['holm_p'] = adjusted
    base = point['overall']
    variants = {'category_balanced':point['category_balanced']}
    for variant in ('mae','rmse'):
        variants[variant+'_skill'] = data.aggregate(data.scores(variant=variant))['overall']
    if len(data.tasks)>1:
        for omitted in data.tasks:
            scores = data.scores()
            variants['without:'+omitted] = {a:float(np.mean([scores[t][a] for t in data.tasks if t!=omitted])) for a in data.arms}
    robustness = {}
    primary_rank = sorted(base, key=lambda arm: (-base[arm], arm))
    primary_top5 = set(primary_rank[:5])
    for name, values in variants.items():
        x,y = [base[a] for a in data.arms], [values[a] for a in data.arms]
        rho = float(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
        variant_top5 = set(sorted(values, key=lambda arm: (-values[arm], arm))[:5])
        robustness[name] = {'scores':values, 'spearman_with_primary':rho,
                            'top5_overlap_with_primary':len(primary_top5 & variant_top5) if len(base) >= 5 else None}
    error_intervals = {}
    for task, values in count_errors(data).items():
        error_intervals[task] = {
            arm: {metric: {'estimate':value if np.isfinite(value) else None,
                          'ci95':interval([r[task][arm][metric] for r in error_draws])}
                  for metric,value in scores.items()} for arm,scores in values.items()}
        error_intervals[task]['paired_differences'] = [
            {'left':left, 'right':right, 'metric':metric,
             'difference':values[left][metric]-values[right][metric] if np.isfinite(values[left][metric]-values[right][metric]) else None,
             'ci95':interval([r[task][left][metric]-r[task][right][metric] for r in error_draws])}
            for left,right in combinations(data.arms,2) for metric in ('mae','rmse','percent_mae')]
    alternative_intervals = {}
    for variant, results in variant_draws.items():
        alternative_point = data.aggregate(data.scores(variant=variant))
        alternative_intervals[variant] = {
            level:{arm:{'estimate':value, 'ci95':interval([r[level][arm] for r in results])}
                   for arm,value in values.items()} for level,values in alternative_point.items()}
    return {'method':{'bootstrap':'paired percentile bootstrap, stratified by dataset and task coverage',
                      'resampling_unit':resampling_unit,
                      'sampling_assumption':'independent scenes/images' if resampling_unit == 'cluster' else 'independent observations within and across tasks',
                      'test':'two-sided paired cluster label-swap permutation; exchangeability under the null is required',
                      'holm_family':'all reported level-by-arm-pair tests', 'scale':'0-1',
                      'degenerate_correlation':'zero when either variance is zero',
                      'uncertainty':'dataset sampling only; not repeated model runs',
                      'included_tasks':data.tasks, 'included_models':data.arms, 'draws':draws, 'seed':seed},
            'estimates':estimates, 'comparisons':comparisons, 'robustness':robustness,
            'counting_errors':error_intervals, 'alternative_metric_intervals':alternative_intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resampling-unit', choices=('cluster','sample'), default='sample')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = analyze(rows, args.draws, args.seed, args.resampling_unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
