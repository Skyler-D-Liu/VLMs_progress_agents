"""Statistics."""
import argparse
import json
from pathlib import Path
from evaluation.analysis.observations import observations
from evaluation.analysis.paired import analyze
from .catalog import load_frozen_catalog
from .report import paired_answers
from .storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/agents/comparison'))
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resampling-unit', choices=('cluster','sample'), default='sample')
    args = parser.parse_args()
    samples = load_frozen_catalog(args.output)
    rows = observations(samples, paired_answers(samples, args.output))
    (args.output / 'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(args.output / 'statistics.json', analyze(rows, args.draws, args.seed, args.resampling_unit))

if __name__ == '__main__':
    main()
