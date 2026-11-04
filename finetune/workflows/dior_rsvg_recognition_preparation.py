"""Dior rsvg recognition preparation."""

from __future__ import annotations

import io
import json
from pathlib import Path

from finetune.training.checkpoint_manager import atomic_write_json


def prepare_dior_rsvg_recognition(config, semantics) -> int:
    from datasets import Dataset, Image, concatenate_datasets

    import dior_rsvg_recognition as authority

    split_map = authority.dior.discover_available_splits_from_names(authority.dior.discover_dior_raw_split_names())


    cache_root = config.paths.raw_dataset_root / "danielz01___dior-rsvg/default/0.0.0"
    snapshots = sorted(path for path in cache_root.iterdir() if path.is_dir())
    if len(snapshots) != 1:
        raise FileNotFoundError(f"expected one cached DIOR snapshot under {cache_root}")
    snapshot = snapshots[0]
    datasets = {}
    for raw in sorted(set(split_map.values())):
        shards = sorted(snapshot.glob(f"dior-rsvg-{raw}-*.arrow"))
        if not shards:
            raise FileNotFoundError(f"missing cached DIOR Arrow shards for {raw}: {snapshot}")
        parts = [Dataset.from_file(str(path)) for path in shards]
        dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        datasets[raw] = dataset.cast_column("image", Image(decode=False))
    lengths = {raw: len(dataset) for raw, dataset in datasets.items()}
    assignments = authority.dior.build_split_assignments(
        split_map=split_map,
        split_lengths=lengths,
        seed=authority.dior.DEFAULT_SPLIT_SEED,
    )
    records_root = config.paths.prepared_dir / "records"
    image_root = config.paths.prepared_dir / "images"
    entries = []
    for canonical, raw_key in (("train", "train"), ("validation", "val"), ("test", "test")):
        assignment = assignments[raw_key]
        candidates = authority.build_object_items(datasets[assignment["raw_split"]], assignment["indices"], assignment["raw_split"], canonical)
        selected = candidates
        output = []
        for position, item in enumerate(selected):
            row = datasets[assignment["raw_split"]][item["row_index"]]
            raw = authority.dior.get_raw_image_bytes(row["image"])
            marked = _mark(raw, item["bbox"])
            image_path = image_root / canonical / f"{position:08d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(marked)
            from PIL import Image

            with Image.open(image_path) as image_handle:
                width, height = image_handle.size
            source_id = f"{assignment['raw_split']}:{item['row_index']}:{item['object_index']}"
            sample_id = f"dior-rsvg-recognition:{canonical}:{position:08d}"
            output.append({"task": config.task, "dataset_id": config.dataset.id, "canonical_split": canonical, "raw_split": assignment["raw_split"], "raw_index": item["row_index"], "sample_id": sample_id, "source_id": source_id, "image_paths": [str(image_path.resolve())], "system_message": semantics.system_message, "user_prompt": authority.USER_PROMPT, "assistant_answer": authority.format_category_answer(item["label"]), "ground_truth": item["label"], "metadata": {"bbox": item["bbox"], "description": item.get("caption"), "object_index": item["object_index"], "images": [{"path": str(image_path.resolve()), "width": width, "height": height}]}})
            entries.append({"sample_id": sample_id, "source_id": source_id, "canonical_split": canonical})
        _write_jsonl(records_root / f"{canonical}.jsonl", output)
    manifest_dir = config.paths.manifest_dir
    atomic_write_json(manifest_dir / "prompt_manifest.json", {"task": config.task, "authoritative_script": str(semantics.script), "system_message": semantics.system_message})
    atomic_write_json(manifest_dir / "split_manifest.json", {"task": config.task, "split_strategy": config.splits.strategy.value, "split_seed": config.splits.seed, "entries": entries, "test_data_used_for_tuning": False})
    atomic_write_json(manifest_dir / "PREPARATION_COMPLETE.json", {"task": config.task, "training_started": False, "validation_generation_run": False, "test_inference_run": False})
    return 0


def _mark(raw: bytes, bbox) -> bytes:
    import base64
    import dior_rsvg_recognition as authority

    data_uri, _, _ = authority.annotate_target_box(raw, bbox)
    return base64.b64decode(data_uri.split(",", 1)[1])


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
