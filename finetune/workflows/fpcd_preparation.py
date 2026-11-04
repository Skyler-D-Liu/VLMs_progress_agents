"""Fpcd preparation."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from finetune.training.checkpoint_manager import atomic_write_json
from finetune.utils.fpcd_data import build_fpcd_bundle


def _materialize_image(raw: bytes, destination_stem: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        width, height = image.size
        image_format = (image.format or "JPEG").casefold()
        image.verify()
    extension = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(
        image_format
    )
    if extension is None:
        raise ValueError(f"unsupported FPCD image format: {image_format}")
    path = destination_stem.with_suffix(extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.read_bytes() != raw:
        temporary = path.with_name(f".{path.name}.preparing-{str(os.getpid())}")
        temporary.write_bytes(raw)
        if temporary.read_bytes() != raw:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"prepared FPCD image copy differs: {temporary}")
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


def prepare_fpcd(config, semantics) -> int:
    authority = semantics.module
    bundle = build_fpcd_bundle(config.paths.raw_dataset_root, strict=True)
    records_root = config.paths.prepared_dir / "records"
    image_root = config.paths.prepared_dir / "images"
    manifest_entries: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    for canonical_split in ("train", "validation", "test"):
        prepared_records = []
        for position, source in enumerate(bundle.records(canonical_split)):
            if source.source_id in seen_source_ids:
                raise RuntimeError(f"FPCD source leakage across splits: {source.source_id}")
            seen_source_ids.add(source.source_id)
            raw_images = bundle.read_image_bytes(source)
            images = [
                _materialize_image(
                    raw,
                    image_root / canonical_split / f"{source.source_id}-{image_index}",
                )
                for image_index, raw in enumerate(raw_images)
            ]
            sample_id = f"fpcd:{canonical_split}:{position:08d}"
            metadata = {
                "width": images[0]["width"],
                "height": images[0]["height"],
                "images": images,
                "image_order": ["T0/pre-image", "T1/post-image"],
                "mask_classes": list(source.mask_classes),
                "class_name": source.class_name,
                "answer_schema": authority.ANSWER_SCHEMA,
            }
            record = {
                "task": config.task,
                "dataset_id": config.dataset.id,
                "canonical_split": canonical_split,
                "raw_split": source.raw_split,
                "raw_index": source.raw_index,
                "source_qualifier": "mask-single-label",
                "sample_id": sample_id,
                "source_id": source.source_id,
                "image_paths": [image["path"] for image in images],
                "system_message": authority.SYSTEM_MESSAGE,
                "user_prompt": authority.USER_PROMPT,
                "assistant_answer": authority.format_answer(source.answer),
                "ground_truth": source.answer,
                "metadata": metadata,
            }
            prepared_records.append(record)
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "source_id": source.source_id,
                    "canonical_split": canonical_split,
                    "raw_split": source.raw_split,
                    "raw_index": source.raw_index,
                    "mask_classes": list(source.mask_classes),
                    "answer": source.answer,
                }
            )
        path = records_root / f"{canonical_split}.jsonl"
        _write_jsonl(path, prepared_records)

    manifest_dir = config.paths.manifest_dir
    prompt_manifest = {
        "task": config.task,
        "authoritative_script": str(semantics.script),
        "system_message": authority.SYSTEM_MESSAGE,
        "user_prompt": authority.USER_PROMPT,
        "answer_schema": authority.ANSWER_SCHEMA,
        "valid_answers": list("ABCDE"),
        "option_order": authority.OPTION_TO_CLASS,
        "mask_to_option": {str(key): value for key, value in authority.MASK_TO_OPTION.items()},
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
        "entries": manifest_entries,
        "test_data_used_for_tuning": False,
    }
    prompt_path = manifest_dir / "prompt_manifest.json"
    split_path = manifest_dir / "split_manifest.json"
    atomic_write_json(prompt_path, prompt_manifest)
    atomic_write_json(split_path, split_manifest)
    marker = {
        "task": config.task,
        "source_shards": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in bundle.archive_paths.values()
        ],
        "training_started": False,
        "validation_generation_run": False,
        "test_inference_run": False,
    }
    atomic_write_json(manifest_dir / "PREPARATION_COMPLETE.json", marker)
    print(json.dumps({"task": config.task, "preparation_complete": True}, sort_keys=True), flush=True)
    return 0
