"""Samples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from evaluation.utils.dataset_catalog import dataset_rows

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / 'results/analysis/specialist_reference/samples'


def prepared_rows(task, data_root):
    rows = dataset_rows(task, data_root)
    if task != 'geobench-refdet':
        return rows
    from evaluation.utils.reasoning_tasks import TaskInputs
    inputs = TaskInputs(task, rows, data_root)
    result = []
    for source in rows:
        row, _ = inputs.prepare(source)
        row['task'] = row['source_dataset'] + ' Localization'
        row['image'] = str((inputs.data_root / row['image_path']).resolve())
        result.append(row)
    return result


def export(data_root, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, name in [('landmarks','landmarks'), ('geobench-refdet','refdet'), ('xview2','xview2')]:
        rows = prepared_rows(task, data_root)
        (out_dir / (name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT/'raw_dataset')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    export(args.data_root, args.out_dir)

if __name__ == '__main__':
    main()
