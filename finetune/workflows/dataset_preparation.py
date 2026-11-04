#!/usr/bin/env python3
"""Dataset preparation."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from finetune.config import (
    load_base_lora_config,
    load_dataset_config,
    resolve_experiment_config,
)
from finetune.model_backends import get_model_backend
from finetune.config.runtime_context import active_base_config, active_dataset_config, active_registry
from finetune.datasets.aerial_animal_adapter import (
    RAW_SPLIT_BY_CANONICAL,
    counts_from_objects,
    load_authoritative_semantics,
    parquet_files,
    snapshot_path,
)
from finetune.datasets.adapter_contract import CanonicalSplit
from finetune.training.checkpoint_manager import atomic_write_json
from finetune.utils.dataset_splits import (
    build_split_assignments,
    discover_available_splits_from_names,
    resolve_split_strategy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "finetune/experiment_specs/base_lora.yaml"
DEFAULT_REGISTRY = PROJECT_ROOT / "finetune/experiment_registry.yaml"
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "finetune/experiment_specs/datasets/aerial_animal.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="aerial-animal", choices=("aerial-animal",))
    parser.add_argument("--base-config", type=Path, default=active_base_config(DEFAULT_BASE_CONFIG))
    parser.add_argument("--registry", type=Path, default=active_registry(DEFAULT_REGISTRY))
    parser.add_argument("--dataset-config", type=Path, default=active_dataset_config(DEFAULT_DATASET_CONFIG))
    return parser.parse_args()


def _image_bytes(row: Mapping[str, Any]) -> bytes:
    image = row.get("image")
    if not isinstance(image, Mapping):
        raise ValueError("dataset row is missing image bytes/path")
    raw = image.get("bytes")
    if raw is not None:
        return bytes(raw)
    source_path = image.get("path")
    if source_path:
        return Path(source_path).read_bytes()
    raise ValueError("dataset image has neither bytes nor path")


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True))


def _materialize_image(
    *, raw_bytes: bytes, image_root: Path, raw_split: str, raw_index: int
) -> tuple[Path, int, int]:
    from PIL import Image

    with Image.open(BytesIO(raw_bytes)) as image:
        width, height = image.size
        image_format = (image.format or "JPEG").casefold()
        image.verify()
    extension = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(
        image_format
    )
    if extension is None:
        raise ValueError(f"unsupported image format {image_format!r} at {raw_split}[{raw_index}]")
    destination_dir = image_root / raw_split
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{raw_index:08d}{extension}"
    if destination.is_file() and destination.read_bytes() == raw_bytes:
        return destination.absolute(), width, height
    temporary = destination.with_name(
        f".{destination.name}.preparing-{os.getpid()}"
    )
    temporary.write_bytes(raw_bytes)
    if temporary.read_bytes() != raw_bytes:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"prepared image copy differs: {temporary}")
    os.replace(temporary, destination)
    return destination.absolute(), width, height


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.preparing-{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


_GENERATED_IMAGE_NAME = re.compile(r"^\d{8}\.(?:jpe?g|png|webp)$", re.IGNORECASE)


def _clean_prepared_image_dir(
    destination_dir: Path, referenced_paths: set[Path]
) -> int:
    destination_dir = destination_dir.expanduser().absolute()
    expected = {path.expanduser().absolute() for path in referenced_paths}
    if any(path.parent != destination_dir for path in expected):
        raise RuntimeError(
            f"referenced image escapes canonical split directory: {destination_dir}"
        )

    removed = 0
    for child in destination_dir.iterdir():
        child = child.absolute()
        if child in expected:
            continue
        if child.is_dir():
            raise RuntimeError(
                f"unexpected directory in generated image split: {child}"
            )
        is_pipeline_residue = (
            child.name.startswith("._")
            or child.name.startswith(".") and ".preparing-" in child.name
            or _GENERATED_IMAGE_NAME.fullmatch(child.name) is not None
        )
        if not is_pipeline_residue:
            raise RuntimeError(
                f"unexpected file in generated image split; refusing to delete: {child}"
            )
        child.unlink()
        removed += 1

    actual = {path.absolute() for path in destination_dir.iterdir() if path.is_file()}
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise RuntimeError(
            f"prepared image directory is not canonical: missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )
    return removed


def _clean_generated_container(
    directory: Path,
    *,
    expected_files: set[str],
    expected_directories: set[str],
    legacy_files: set[str] | None = None,
) -> int:
    directory = directory.expanduser().absolute()
    legacy_files = legacy_files or set()
    removed = 0
    for child in directory.iterdir():
        if child.name in expected_files:
            if not child.is_file():
                raise RuntimeError(f"expected generated file is not a file: {child}")
            continue
        if child.name in expected_directories:
            if not child.is_dir():
                raise RuntimeError(f"expected generated directory is not a directory: {child}")
            continue
        removable = (
            child.name.startswith("._")
            or child.name == ".DS_Store"
            or child.name in legacy_files
            or child.name.startswith(".") and ".preparing-" in child.name
        )
        if not removable or child.is_dir():
            raise RuntimeError(
                f"unexpected entry in generated directory; refusing to delete: {child}"
            )
        child.unlink()
        removed += 1

    actual_files = {child.name for child in directory.iterdir() if child.is_file()}
    actual_directories = {child.name for child in directory.iterdir() if child.is_dir()}
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError(
            f"generated directory is not canonical: {directory}; "
            f"files={sorted(actual_files)}, directories={sorted(actual_directories)}"
        )
    return removed


def _load_local_split(config, raw_split: str):
    from datasets import Image, load_dataset

    shards = parquet_files(config, raw_split)
    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in shards]},
        split="train",
        cache_dir=str(config.paths.raw_dataset_root),
        download_mode="reuse_dataset_if_exists",
    )
    return dataset.cast_column("image", Image(decode=False)), shards


def main() -> int:
    args = parse_args()
    base = load_base_lora_config(args.base_config, args.registry)
    get_model_backend(base).validate_config(base)
    dataset_config = load_dataset_config(args.dataset_config)
    resolved = resolve_experiment_config(base, dataset_config, args.registry)
    if resolved.dataset.task != args.task:
        raise ValueError(f"dataset config task mismatch: {resolved.dataset.task} != {args.task}")

    config = resolved.dataset
    snapshot = snapshot_path(config)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"dataset snapshot is missing: {snapshot}")

    semantics = load_authoritative_semantics(config)
    split_map = discover_available_splits_from_names(("train", "val", "test"))
    strategy = resolve_split_strategy(split_map)
    if strategy["mode"] != "reuse_all":
        raise RuntimeError(f"Aerial Animal must reuse all native splits: {strategy}")

    datasets_by_split = {}
    shards_by_split = {}
    split_lengths = {}
    for raw_split in ("train", "val", "test"):
        dataset, shards = _load_local_split(config, raw_split)
        datasets_by_split[raw_split] = dataset
        shards_by_split[raw_split] = shards
        split_lengths[raw_split] = len(dataset)

    assignments = build_split_assignments(
        split_map=split_map,
        split_lengths=split_lengths,
        seed=config.splits.seed,
    )

    prepared = config.paths.prepared_dir.expanduser().absolute()
    records_root = prepared / "records"
    image_root = prepared / "images"
    manifest_entries: list[dict[str, Any]] = []
    removed_stale_image_count = 0

    for canonical_split in CanonicalSplit:
        canonical_name = canonical_split.value
        assignment_key = "val" if canonical_split is CanonicalSplit.VALIDATION else canonical_name
        assignment = assignments[assignment_key]
        raw_split = assignment["raw_split"]
        expected_raw = RAW_SPLIT_BY_CANONICAL[canonical_split]
        if raw_split != expected_raw:
            raise RuntimeError(
                f"canonical {canonical_name} unexpectedly maps to raw {raw_split}, expected {expected_raw}"
            )
        indices = assignment["indices"]
        if indices != list(range(len(datasets_by_split[raw_split]))):
            raise RuntimeError(f"canonical {canonical_name} is not the full native split")
        prepared_records: list[dict[str, Any]] = []
        for raw_index in indices:
            row = datasets_by_split[raw_split][raw_index]
            objects = _json_value(row.get("objects"))
            counts = counts_from_objects(semantics, objects)
            raw_bytes = _image_bytes(row)
            image_path, width, height = _materialize_image(
                raw_bytes=raw_bytes,
                image_root=image_root,
                raw_split=raw_split,
                raw_index=raw_index,
            )
            source_id = f"{config.dataset.id}:{raw_split}:{raw_index:08d}"
            sample_id = f"aerial-animal:{canonical_name}:{raw_index:08d}"
            record = {
                "dataset_id": config.dataset.id,
                "dataset_config": config.dataset.config,
                "canonical_split": canonical_name,
                "raw_split": raw_split,
                "raw_index": raw_index,
                "sample_id": sample_id,
                "source_id": source_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "pixels": width * height,
                "bbox_count": len(objects["bbox"]),
                "ground_truth": counts,
                "total_count": sum(counts.values()),
            }
            prepared_records.append(record)
            manifest_entries.append(record)
        record_name = "validation" if canonical_split is CanonicalSplit.VALIDATION else canonical_name
        record_path = records_root / f"{record_name}.jsonl"
        _write_jsonl_atomic(record_path, prepared_records)
        removed_stale_image_count += _clean_prepared_image_dir(
            image_root / raw_split,
            {Path(record["image_path"]) for record in prepared_records},
        )

    shard_manifest = {
        split: [
            {"name": path.name, "size_bytes": path.stat().st_size}
            for path in shards
        ]
        for split, shards in shards_by_split.items()
    }
    split_manifest = {
        "task": config.task,
        "dataset": {
            "id": config.dataset.id,
            "config": config.dataset.config,
            "parquet_shards": shard_manifest,
        },
        "split_strategy": config.splits.strategy.value,
        "split_seed": config.splits.seed,
        "raw_to_canonical": {"train": "train", "val": "validation", "test": "test"},
        "authoritative_script": str(semantics.authoritative_script),
        "test_data_used_for_tuning": False,
        "test_split_materialized_for_manifest_only": True,
        "entries": manifest_entries,
    }
    prompt_manifest = {
        "authoritative_script": str(semantics.authoritative_script),
        "system_symbol": config.prompt.system_symbol,
        "system_message": semantics.system_message,
        "user_symbol": config.prompt.user_symbol,
        "user_prompt": semantics.user_prompt,
    }

    permanent_manifest_dir = config.paths.manifest_dir.resolve()
    permanent_manifest_dir.mkdir(parents=True, exist_ok=True)
    split_manifest_path = permanent_manifest_dir / "split_manifest.json"
    prompt_manifest_path = permanent_manifest_dir / "prompt_manifest.json"
    atomic_write_json(split_manifest_path, split_manifest)
    atomic_write_json(prompt_manifest_path, prompt_manifest)
    atomic_write_json(prepared / "split_manifest.json", split_manifest)
    removed_stale_image_count += _clean_generated_container(
        records_root,
        expected_files={"train.jsonl", "validation.jsonl", "test.jsonl"},
        expected_directories=set(),
    )
    removed_stale_image_count += _clean_generated_container(
        image_root,
        expected_files=set(),
        expected_directories={"train", "val", "test"},
    )
    removed_stale_image_count += _clean_generated_container(
        prepared,
        expected_files={"split_manifest.json"},
        expected_directories={"images", "records"},
        legacy_files={"train.jsonl", "val.jsonl", "test.jsonl"},
    )
    marker = {"task": config.task, "training_started": False, "test_inference_run": False}
    atomic_write_json(permanent_manifest_dir / "PREPARATION_COMPLETE.json", marker)
    print(
        f"Removed {removed_stale_image_count} stale generated image artifact(s).",
        flush=True,
    )
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
