"""Rare planes preparation."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from finetune.training.checkpoint_manager import atomic_write_json
from finetune.utils.rare_planes_data import build_rare_planes_bundle


def _materialize_image(raw: bytes, destination_stem: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        width, height = image.size
        image_format = (image.format or "PNG").casefold()
        image.verify()
    if image_format != "png" or (width, height) != (512, 512):
        raise ValueError(
            f"RarePlanes prepared image must be a 512x512 PNG, got {image_format} {width}x{height}"
        )
    path = destination_stem.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_bytes() != raw:
        temporary = path.with_name(f".{path.name}.preparing-{str(os.getpid())}")
        temporary.write_bytes(raw)
        if temporary.read_bytes() != raw:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"prepared RarePlanes image copy differs: {temporary}")
        os.replace(temporary, path)
    return {"path": str(path.resolve()), "width": width, "height": height}


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.preparing-{str(os.getpid())}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare_rare_planes(config, semantics) -> int:
    authority = semantics.module
    bundle = build_rare_planes_bundle(config.paths.raw_dataset_root, strict=True)
    records_root = config.paths.prepared_dir / "records"
    image_root = config.paths.prepared_dir / "images"
    entries: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    scenes_by_split: dict[str, set[str]] = {
        "train": set(), "validation": set(), "test": set()
    }

    for canonical_split in ("train", "validation", "test"):
        prepared_records = []
        for position, source in enumerate(bundle.records(canonical_split)):
            if source.source_id in seen_source_ids:
                raise RuntimeError(f"RarePlanes source tile leakage: {source.source_id}")
            seen_source_ids.add(source.source_id)
            scenes_by_split[canonical_split].add(source.source_scene_id)
            image = _materialize_image(
                bundle.read_image_bytes(source),
                image_root / canonical_split / f"{position:08d}",
            )
            sample_id = f"rare-planes:{canonical_split}:{position:08d}"
            metadata = {
                "width": image["width"],
                "height": image["height"],
                "images": [image],
                "official_split": source.official_split,
                "coco_image_id": source.coco_image_id,
                "file_name": source.file_name,
                "source_scene_id": source.source_scene_id,
                "aircraft_bboxes": [list(box) for box in source.bboxes],
                "ground_truth_authority": "all_official_aircraft_annotations_in_tile",
                "synthetic_imagery_used": False,
                "aircraft_type_labels_used": False,
                "answer_schema": authority.ANSWER_SCHEMA,
            }
            record = {
                "task": config.task,
                "dataset_id": config.dataset.id,
                "canonical_split": canonical_split,
                "raw_split": source.official_split,
                "raw_index": source.raw_index,
                "source_qualifier": "real-observed-scene-disjoint-seed0-sampled",
                "sample_id": sample_id,
                "source_id": source.source_id,
                "image_paths": [image["path"]],
                "system_message": authority.SYSTEM_MESSAGE,
                "user_prompt": authority.USER_PROMPT,
                "assistant_answer": authority.format_count_answer(source.count),
                "ground_truth": source.count,
                "metadata": metadata,
            }
            prepared_records.append(record)
            entries.append(
                {
                    "sample_id": sample_id,
                    "source_id": source.source_id,
                    "canonical_split": canonical_split,
                    "official_split": source.official_split,
                    "raw_index": source.raw_index,
                    "source_scene_id": source.source_scene_id,
                    "count": source.count,
                }
            )
        path = records_root / f"{canonical_split}.jsonl"
        _write_jsonl(path, prepared_records)

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = scenes_by_split[left] & scenes_by_split[right]
        if overlap:
            raise RuntimeError(f"RarePlanes source-scene leakage between {left} and {right}")

    manifest_dir = config.paths.manifest_dir
    prompt_manifest = {
        "task": config.task,
        "authoritative_script": str(semantics.script),
        "system_message": authority.SYSTEM_MESSAGE,
        "user_prompt": authority.USER_PROMPT,
        "answer_schema": authority.ANSWER_SCHEMA,
        "system_symbol": config.prompt.system_symbol,
        "user_symbol": config.prompt.user_symbol,
        "dynamic": config.prompt.dynamic,
        "dynamic_builder_symbols": list(config.prompt.dynamic_builder_symbols),
        "provenance": authority.PROMPT_PROVENANCE,
    }
    split_manifest = {
        "task": config.task,
        "dataset": config.dataset.model_dump(mode="json"),
        "split_strategy": config.splits.strategy.value,
        "split_seed": config.splits.seed,
        "source_audit": dict(bundle.audit),
        "entries": entries,
        "synthetic_imagery_used": False,
        "test_data_used_for_tuning": False,
    }
    prompt_path = manifest_dir / "prompt_manifest.json"
    split_path = manifest_dir / "split_manifest.json"
    atomic_write_json(prompt_path, prompt_manifest)
    atomic_write_json(split_path, split_manifest)
    marker = {
        "task": config.task,
        "source_shards": [],
        "synthetic_imagery_used": False,
        "source_scene_overlap_across_splits": False,
        "training_started": False,
        "validation_generation_run": False,
        "test_inference_run": False,
    }
    atomic_write_json(manifest_dir / "PREPARATION_COMPLETE.json", marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return 0
