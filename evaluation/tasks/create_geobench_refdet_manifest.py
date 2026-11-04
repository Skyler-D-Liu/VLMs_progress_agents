"""Create geobench refdet manifest."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from vleo_runtime.localization import source_dataset
from evaluation.utils.dataset_splits import DEFAULT_SPLIT_SEED

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / 'raw_dataset/GEOBench-VLM/Ref-Det/qa.json'
DEFAULT_OUTPUT = REPO_ROOT / 'raw_dataset/geobench_spatial/splits.json'


def partition_rows(rows, seed=DEFAULT_SPLIT_SEED):
    groups = {}
    identities = set()
    for position, row in enumerate(rows):
        dataset = source_dataset(row)
        if dataset is None:
            raise ValueError('Every Ref-Det query requires dataset provenance')
        if dataset == 'DIOR':
            continue
        if dataset not in {'DOTA', 'FAIR1M'}:
            raise ValueError('Unknown Ref-Det source dataset')
        identity = str(row.get('sample_id') or f"refdet:{row.get('question_id', position)}")
        if identity in identities:
            raise ValueError('Duplicate Ref-Det query identity')
        identities.add(identity)
        image = row.get('image_ref') or row.get('image_path')
        if not image:
            raise ValueError('Image identity is required for disjoint partitions')
        groups.setdefault(dataset, {}).setdefault(str(image), []).append(identity)
    if not groups:
        raise ValueError('Dataset provenance is required; run prepare_refdet_sources first')
    splits = {name: [] for name in ('train', 'validation', 'test')}
    for dataset, images in sorted(groups.items()):
        keys = sorted(images)
        random.Random(f'{seed}:{dataset}').shuffle(keys)
        train_end = int(len(keys) * .8)
        val_end = train_end + int(len(keys) * .1)
        for name, selected in zip(splits, (keys[:train_end], keys[train_end:val_end], keys[val_end:])):
            splits[name].extend(identity for image in selected for identity in images[image])
    return {key: sorted(values) for key, values in splits.items()}


def build_manifest(source, output, seed=DEFAULT_SPLIT_SEED):
    rows = json.loads(Path(source).read_text())
    payload = {'policy': 'dataset-image-disjoint', 'seed': seed, 'splits': partition_rows(rows, seed)}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2) + '\n')
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_manifest(args.source, args.output))

if __name__ == '__main__':
    main()
