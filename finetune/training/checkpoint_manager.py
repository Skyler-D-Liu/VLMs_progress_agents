"""Checkpoint manager."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from transformers import TrainerCallback


CHECKPOINT_COMPLETE = "CHECKPOINT_COMPLETE.json"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    CHECKPOINT_MANIFEST,
    CHECKPOINT_COMPLETE,
)


class CheckpointError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Invalid JSON checkpoint file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Checkpoint JSON must be an object: {path}")
    return value


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise CheckpointError(f"Invalid checkpoint directory name: {path}")
    return int(match.group(1))


def _validate_safetensors(path: Path) -> None:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise CheckpointError(f"Safetensors file has no tensors: {path}")
            for key in keys:
                handle.get_slice(key)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"Unreadable safetensors checkpoint file {path}: {exc}") from exc


def _validate_torch_archive(path: Path) -> None:
    try:
        import torch

        try:


            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CheckpointError(f"Unreadable PyTorch checkpoint file {path}: {exc}") from exc
    if value is None:
        raise CheckpointError(f"PyTorch checkpoint file contains no state: {path}")


def validate_checkpoint_files(path: Path) -> None:
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint directory does not exist: {path}")
    missing = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    ]
    if missing:
        raise CheckpointError(f"Incomplete checkpoint {path}; missing or empty: {', '.join(missing)}")

    step = checkpoint_step(path)
    adapter_config = _load_json_object(path / "adapter_config.json")
    trainer_state = _load_json_object(path / "trainer_state.json")
    manifest = _load_json_object(path / CHECKPOINT_MANIFEST)
    marker = _load_json_object(path / CHECKPOINT_COMPLETE)
    if not adapter_config:
        raise CheckpointError(f"Empty adapter configuration: {path}")
    for label, value in (
        ("trainer state", trainer_state.get("global_step")),
        ("checkpoint manifest", manifest.get("global_step")),
        ("completion marker", marker.get("global_step")),
    ):
        if value != step:
            raise CheckpointError(
                f"Checkpoint {label} step={value!r}, directory step={step}: {path}"
            )
    _validate_safetensors(path / "adapter_model.safetensors")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _validate_torch_archive(path / name)


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    validate_checkpoint_files(path)
    return _load_json_object(Path(path) / CHECKPOINT_MANIFEST)


def validate_resume_checkpoint(
    path: Path, expected_compatibility: Mapping[str, Any]
) -> None:
    manifest = load_checkpoint_manifest(path)
    actual = manifest.get("compatibility")
    if not isinstance(actual, Mapping):
        raise CheckpointError(f"Checkpoint compatibility is missing at {path}")
    required = ("task", "stage", "learning_rate", "seed", "resolution_candidate")
    mismatches = {
        field: (actual.get(field), expected_compatibility.get(field))
        for field in required
        if field in expected_compatibility
        and actual.get(field) != expected_compatibility.get(field)
    }
    if mismatches:
        raise CheckpointError(
            f"Resume semantic parameters mismatch at {path}: {mismatches!r}"
        )


def find_resume_checkpoint(
    trial_dir: str | Path,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> Path | None:
    root = Path(trial_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(child.name)
        if not child.is_dir() or match is None or not (child / CHECKPOINT_COMPLETE).is_file():
            continue
        try:
            validate_checkpoint_files(child)
        except CheckpointError:
            continue
        candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: item[0])[1]
    if expected_compatibility is not None:
        validate_resume_checkpoint(selected, expected_compatibility)
    return selected


def resolve_resume_checkpoint(
    resume: str,
    trial_dir: Path,
    expected_compatibility: Mapping[str, Any],
) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        checkpoint = find_resume_checkpoint(trial_dir, expected_compatibility)
        if checkpoint is None:
            raise CheckpointError(f"No complete compatible checkpoint found under {trial_dir}")
        return checkpoint
    checkpoint = Path(resume).expanduser().resolve()
    try:
        checkpoint.relative_to(trial_dir.resolve())
    except ValueError as exc:
        raise CheckpointError(
            f"Resume checkpoint must remain inside trial directory {trial_dir}: {checkpoint}"
        ) from exc
    validate_resume_checkpoint(checkpoint, expected_compatibility)
    return checkpoint


class CompleteCheckpointCallback(TrainerCallback):

    def __init__(self, compatibility: Mapping[str, Any]) -> None:
        self.compatibility = dict(compatibility)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        atomic_write_json(
            checkpoint / CHECKPOINT_MANIFEST,
            {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "compatibility": self.compatibility,
            },
        )
        atomic_write_json(
            checkpoint / CHECKPOINT_COMPLETE,
            {"global_step": state.global_step},
        )
        validate_checkpoint_files(checkpoint)
        return control
