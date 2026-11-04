"""Refdet nonoverlap."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .samples import prepared_rows, PROJECT, OUT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=PROJECT/'raw_dataset')
    parser.add_argument('--training-annotations', type=Path, required=True)
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT_DIR/'refdet_nonoverlap.jsonl')
    args = parser.parse_args()
    training = json.loads(args.training_annotations.read_text())
    if not isinstance(training, list) or not training:
        raise ValueError('A nonempty list of documented training records is required')
    trained = set()
    for row in training:
        candidates = row.get('original_image_candidates') or row.get('image_files') or row.get('images')
        if isinstance(candidates, str):
            candidates = [candidates]
        if not candidates:
            candidates = [row.get('image')]
        if not all(candidates):
            raise ValueError('Training annotations must identify their source images')
        trained.update(Path(path).stem for path in candidates)
    metadata = json.loads(args.source_metadata.read_text())
    tiles = {str(r['question_id']):r for r in metadata}
    if len(tiles) != len(metadata):
        raise ValueError('Duplicate source metadata question IDs')
    selected = []
    for row in prepared_rows('geobench-refdet', args.data_root):
        if str(row['question_id']) not in tiles:
            raise ValueError('Source metadata is incomplete for the Ref-Det dataset')
        tile = tiles[str(row['question_id'])]
        candidates = tile['original_image_candidates']
        if not candidates:
            raise ValueError('Original tile provenance is required')
        if any(Path(path).stem in trained for path in candidates):
            continue
        selected.append(row)
    if not selected:
        raise ValueError('No held-out Ref-Det samples remain after training-image exclusion')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row)+'\n' for row in selected))

if __name__ == '__main__':
    main()
