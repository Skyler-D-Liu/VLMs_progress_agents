"""Trends."""
from __future__ import annotations
import argparse
import csv
from datetime import date
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import f as f_distribution
from scipy.stats import spearmanr


def export_score_csv(input_files, output, model_metadata):
    rows = []
    for model, path in input_files.items():
        if model not in model_metadata:
            raise ValueError(f'Missing release metadata for {model}')
        score_data = json.loads(Path(path).read_text())
        score = score_data.get('overall_score', score_data.get('overall', score_data.get('score')))
        if isinstance(score, dict):
            score = score.get('score')
        if score is None:
            raise ValueError(f'No overall score in {path}')
        rows.append({'model': model, **model_metadata[model], 'score': float(score)})
    if len({r['model'] for r in rows}) != len(rows):
        raise ValueError('Duplicate models in score inventory')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('model','family','release_date','score'))
        writer.writeheader()
        writer.writerows(rows)
    return output
from .paired import interval


def decimal_year(text):
    value = date.fromisoformat(text)
    start, end = date(value.year,1,1), date(value.year+1,1,1)
    return value.year + (value-start).days/(end-start).days


def fit(rows, separate_slopes=False):
    if not rows:
        raise ValueError('No model observations')
    families = sorted({r['family'] for r in rows})
    dates = np.array([decimal_year(r['release_date']) for r in rows])
    origin = float(dates.mean())
    time = dates-origin
    indicators = np.array([[float(r['family']==f) for f in families] for r in rows])
    x = np.column_stack([indicators, indicators*time[:,None] if separate_slopes else time])
    y = np.array([float(r['score']) for r in rows])
    if not np.isfinite(y).all() or np.any((y<0)|(y>100)):
        raise ValueError('Scores must be finite and between 0 and 100')
    if np.linalg.matrix_rank(x) != x.shape[1] or len(y)<=x.shape[1]:
        raise ValueError('Insufficient within-family time variation or residual degrees of freedom')
    beta = np.linalg.lstsq(x,y,rcond=None)[0]
    residual = float(np.square(y-x@beta).sum())
    total = float(np.square(y-y.mean()).sum())
    slope = float(beta[-1]) if not separate_slopes else None
    average = float(beta[:len(families)].mean())
    crossing = origin+(90-average)/slope if slope is not None and slope>0 else None
    return {'slope':slope, 'year_at_90':crossing,
            'fitted_latest':average+slope*(dates.max()-origin) if slope is not None else None,
            'r_squared':1-residual/total if total else None,
            'origin_year':origin, 'average_family_intercept':average,
            'residual_sum_squares':residual, 'residual_df':len(y)-x.shape[1]}


def analyze(rows, draws=2000, seed=0, cutoff='2024-05-01', bin_origin='2023-07-01'):
    if len({r['model'] for r in rows}) != len(rows):
        raise ValueError('Each model must appear once in a trend analysis')
    if draws<2:
        raise ValueError('At least two bootstrap draws are required')
    point = fit(rows)
    rng = np.random.default_rng(seed)
    groups = [[r for r in rows if r['family']==f] for f in sorted({r['family'] for r in rows})]
    slopes, crossings, curves = [], [], []
    grid = np.linspace(min(decimal_year(r['release_date']) for r in rows), max(decimal_year(r['release_date']) for r in rows)+3, 80)
    for _ in range(draws):
        sampled = [group[i] for group in groups for i in rng.integers(len(group),size=len(group))]
        try:
            result = fit(sampled)
            slopes.append(result['slope'])
            curves.append(result['average_family_intercept'] + result['slope']*(grid-result['origin_year']))
            crossings.append(result['year_at_90'] if result['year_at_90'] is not None else float('nan'))
        except ValueError:
            slopes.append(float('nan')); crossings.append(float('nan'))
            curves.append(np.full_like(grid, np.nan))
    sensitivity = {}
    try:
        sensitivity['released_from:'+cutoff] = fit([r for r in rows if r['release_date']>=cutoff])
    except ValueError:
        sensitivity['released_from:'+cutoff] = None
    for family in sorted({r['family'] for r in rows}):
        try:
            sensitivity['without:'+family] = fit([r for r in rows if r['family']!=family])
        except ValueError:
            sensitivity['without:'+family] = None
    try:
        unrestricted = fit(rows, separate_slopes=True)
        q = point['residual_df']-unrestricted['residual_df']
        denominator = unrestricted['residual_sum_squares']/unrestricted['residual_df']
        tolerance = np.finfo(float).eps * max(1., sum(float(r['score'])**2 for r in rows))
        if q <= 0:
            pvalue = None
        elif unrestricted['residual_sum_squares'] <= tolerance:
            pvalue = 0. if point['residual_sum_squares'] > tolerance else None
        else:
            stat = max(0., (point['residual_sum_squares']-unrestricted['residual_sum_squares'])/q/denominator)
            pvalue = float(f_distribution.sf(stat,q,unrestricted['residual_df']))
    except ValueError:
        pvalue = None
    bins = {}
    start = date.fromisoformat(bin_origin)
    for row in rows:
        released = date.fromisoformat(row['release_date'])
        month = (released.year-start.year)*12 + released.month-start.month
        bins.setdefault(month//8, []).append(row)
    points = [{'bin':key, 'year':float(np.mean([decimal_year(r['release_date']) for r in group])),
               'score':float(np.mean([float(r['score']) for r in group]))} for key,group in sorted(bins.items())]
    bin_slope, bin_intercept = (tuple(float(v) for v in np.polyfit([r['year'] for r in points], [r['score'] for r in points], 1))
                                if len(points)>1 else (None, None))
    bin_crossing = (90-bin_intercept)/bin_slope if bin_slope is not None and bin_slope>0 else None
    band = [{'year':float(year), 'ci95':interval([curve[i] for curve in curves])} for i,year in enumerate(grid)]
    score_order = sorted(rows, key=lambda row: (-float(row['score']), str(row['model'])))
    top5 = [str(row['model']) for row in score_order[:5]]
    release_pairs = [(left,right) for left,right in combinations(rows,2)
                     if left['release_date'] != right['release_date']]
    newer_rate = (sum(float((right if right['release_date'] > left['release_date'] else left)['score']) >
                      float((left if right['release_date'] > left['release_date'] else right)['score'])
                      for left,right in release_pairs) / len(release_pairs)) if release_pairs else None
    date_rho = float(spearmanr([date.fromisoformat(row['release_date']).toordinal() for row in rows],
                               [float(row['score']) for row in rows]).statistic) if len(rows)>1 else None
    if date_rho is not None and not np.isfinite(date_rho):
        date_rho = None
    first_latest = {'first':points[0], 'latest':points[-1],
                    'difference':points[-1]['score']-points[0]['score']} if points else None
    return {'method':'OLS with family intercepts; within-family model bootstrap; exploratory extrapolation',
            'seed':seed, 'draws':draws, 'fit':point, 'slope_ci95':interval(slopes),
            'year_at_90_ci95_conditional_on_positive_slope':interval(crossings),
            'family_slope_f_test_p':pvalue, 'sensitivity':sensitivity,
            'f_test_assumptions':'Nested Gaussian homoskedastic independent-error OLS models; null for unidentifiable or jointly perfect fits',
            'trend_band':band, 'release_bins':{'origin':bin_origin, 'points':points, 'slope':bin_slope, 'year_at_90':bin_crossing},
            'robustness_summary':{'top5_model_ids':top5, 'date_rho':date_rho,
                                  'newer_models_win_fraction':newer_rate,
                                  'first_latest_release_bin':first_latest},
            'limitations':'Sampling uncertainty conditional on selected models; family releases are not independent replications. Crossing intervals exclude nonpositive slopes and are not forecasts.'}


def analyze_paired_summary(summary, metadata, **kwargs):
    primary = {model: value['estimate'] for model,value in summary['estimates']['overall'].items()}
    variants = {'primary': primary, **{name:value['scores'] for name,value in summary['robustness'].items()},
                **{level:{model:value['estimate'] for model,value in values.items()}
                   for level,values in summary['estimates'].items() if level.startswith('category:')}}
    if set(primary) != set(metadata):
        raise ValueError('Release metadata must match the paired model inventory exactly')
    output = {}
    for name,scores in variants.items():
        if set(scores) != set(primary):
            raise ValueError('All variants must use the same model inventory')
        rows = [{'model':model, **metadata[model], 'score':100*float(score)} for model,score in scores.items()]
        output[name] = analyze(rows, **kwargs)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Prepared CSV model,family,release_date,score')
    parser.add_argument('--paired', type=Path, help='Paired analysis JSON; analyze primary and every robustness variant with --metadata')
    parser.add_argument('--score', action='append', default=[], help='MODEL=score.json; repeat for every model')
    parser.add_argument('--metadata', type=Path, help='CSV model,family,release_date for --score entries')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cutoff', default='2024-05-01')
    parser.add_argument('--bin-origin', default='2023-07-01')
    args=parser.parse_args()
    if args.paired is not None:
        if args.metadata is None or args.input or args.score:
            parser.error('--paired requires --metadata and cannot be combined with --input or --score')
        metadata_rows = list(csv.DictReader(args.metadata.open()))
        if len({row['model'] for row in metadata_rows}) != len(metadata_rows):
            parser.error('duplicate model in release metadata')
        metadata = {row['model']:{'family':row['family'], 'release_date':row['release_date']} for row in metadata_rows}
        result = analyze_paired_summary(json.loads(args.paired.read_text()), metadata, draws=args.draws, seed=args.seed, cutoff=args.cutoff, bin_origin=args.bin_origin)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return
    if args.input is None:
        if not args.score or args.metadata is None:
            parser.error('provide --input, or provide --score and --metadata')
        if any('=' not in entry for entry in args.score):
            parser.error('--score requires MODEL=score.json')
        pairs = [entry.split('=',1) for entry in args.score]
        metadata_rows = list(csv.DictReader(args.metadata.open()))
        if len({row['model'] for row in metadata_rows}) != len(metadata_rows):
            parser.error('duplicate model in release metadata')
        metadata = {row['model']:{'family':row['family'],'release_date':row['release_date']} for row in metadata_rows}
        if len({name for name,_ in pairs}) != len(pairs):
            parser.error('duplicate --score model')
        args.input = export_score_csv(dict(pairs), args.output.with_suffix('.input.csv'), metadata)
    with args.input.open() as stream:
        result=analyze(list(csv.DictReader(stream)),args.draws,args.seed,args.cutoff,args.bin_origin)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    main()
