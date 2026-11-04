"""Geobench refdet preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finetune.training.checkpoint_manager import atomic_write_json


def _xyxy(value: Any) -> list[float]:
    points = value[0] if isinstance(value, list) and len(value) == 1 else value
    if isinstance(points, list) and len(points) == 4 and all(isinstance(p, (int, float)) for p in points):
        return [float(p) for p in points]
    raise ValueError(f"invalid GEOBench bbox: {value!r}")


def prepare_geobench_refdet(config, semantics) -> int:
    from PIL import Image

    root = config.paths.raw_dataset_root / "GEOBench-VLM"
    qa_path = root / "Ref-Det" / "qa.json"
    from evaluation.tasks.create_geobench_refdet_manifest import partition_rows, DEFAULT_SPLIT_SEED
    if config.splits.seed != DEFAULT_SPLIT_SEED:
        raise ValueError("Fine-tuning and evaluation must use the same Ref-Det split seed")
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    manifest = {"splits": partition_rows(payload)}
    by_id = {f"refdet:{row.get('question_id', i)}": row for i, row in enumerate(payload) if isinstance(row, dict)}
    splits = {name: [str(value) for value in manifest["splits"][name]] for name in ("train", "validation", "test")}
    split_sets = {name: set(values) for name, values in splits.items()}
    if len(set().union(*split_sets.values())) != sum(len(values) for values in split_sets.values()):
        raise RuntimeError("GEOBench split manifest contains leakage")
    missing = set().union(*split_sets.values()) - set(by_id)
    if missing:
        raise RuntimeError(f"GEOBench split manifest contains unknown IDs: {sorted(missing)[:5]}")
    records_root = config.paths.prepared_dir / "records"
    entries = []
    for split_name in ("train", "validation", "test"):
        records = []
        for position, sample_id in enumerate(splits[split_name]):
            row = by_id[str(sample_id)]
            ref = Path(str(row["image_path"]))
            image = ref if ref.is_absolute() else root / ref
            if not image.is_file() and not ref.is_absolute():
                image = root / 'Ref-Det' / ref
            with Image.open(image) as image_handle:
                width, height = image_handle.size
            from vleo_runtime.localization import localization_prompt, source_dataset, polygon_boxes
            dataset = row.get('source_dataset') or source_dataset(row)
            if not dataset:
                raise ValueError('Ref-Det records require source_dataset')
            targets = (polygon_boxes(row.get('ground_truth', []))
                       if dataset == 'FAIR1M' else [([min(p[0] for p in box), min(p[1] for p in box), max(p[0] for p in box), max(p[1] for p in box)] if isinstance(box[0], list) else _xyxy(box)) for box in row.get('ground_truth', [])])
            expression = str((row.get('prompts') or [''])[0])
            user = localization_prompt(dataset, width, height, expression)
            canonical_id = f"geobench-refdet:{split_name}:{position:08d}"
            source_id = str(sample_id)
            records.append({
                "task": config.task,
                "dataset_id": config.dataset.id,
                "canonical_split": split_name,
                "raw_split": getattr(config.splits, split_name).source_split,
                "raw_index": int(row.get("question_id", position)),
                "sample_id": canonical_id,
                "source_id": source_id,
                "image_paths": [str(image.resolve())],
                "system_message": semantics.system_message,
                "user_prompt": user,
                "assistant_answer": json.dumps({"bboxes": targets}, ensure_ascii=True),
                "ground_truth": targets,
                "metadata": {
                    "expression": expression,
                    "source_dataset": dataset,
                    "question_id": row.get("question_id"),
                    "images": [{"path": str(image.resolve()), "width": width, "height": height}],
                },
            })
            entries.append({"sample_id": canonical_id, "source_id": source_id, "canonical_split": split_name})
        _write_jsonl(records_root / f"{split_name}.jsonl", records)
    manifest_dir = config.paths.manifest_dir
    atomic_write_json(manifest_dir / "prompt_manifest.json", {"task": config.task, "authoritative_script": str(semantics.script), "system_message": semantics.system_message})
    atomic_write_json(manifest_dir / "split_manifest.json", {"task": config.task, "split_strategy": config.splits.strategy.value, "split_seed": config.splits.seed, "entries": entries, "test_data_used_for_tuning": False})
    atomic_write_json(manifest_dir / "PREPARATION_COMPLETE.json", {"task": config.task, "training_started": False, "validation_generation_run": False, "test_inference_run": False})
    return 0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
