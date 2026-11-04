"""Prepare refdet sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def image_digest(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def prepare(qa_path, image_root, sources, output):
    index = {}
    for dataset, root in sources.items():
        for path in sorted(Path(root).rglob('*')):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}:
                continue
            key = image_digest(path)
            index.setdefault(key, []).append((dataset, str(path.relative_to(root))))
    rows = json.loads(Path(qa_path).read_text())
    mapping = []
    for row in rows:
        path = Path(image_root) / row['image_path']
        matches = index.get(image_digest(path), [])
        datasets = {dataset for dataset, _ in matches}
        if len(datasets) != 1:
            raise ValueError(f"Missing or ambiguous source image: {row['image_path']}")
        mapping.append({'question_id': row['question_id'], 'image_path': row['image_path'],
                        'dataset': next(iter(datasets)),
                        'original_image_candidates': sorted(name for _, name in matches)})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(mapping, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--source', action='append', required=True, help='DATASET=directory of original tiles; include every source dataset')
    parser.add_argument('--output', type=Path, default=Path('raw_dataset/geobench_spatial/source_metadata.json'))
    args = parser.parse_args()
    sources = {name: Path(path) for name, path in (s.split('=', 1) for s in args.source)}
    prepare(args.qa, args.image_root, sources, args.output)

if __name__ == '__main__':
    main()
