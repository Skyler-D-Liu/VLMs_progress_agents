"""Promotion."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from finetune.training.checkpoint_manager import atomic_write_json, validate_checkpoint_files
from finetune.model_backends import get_model_backend
from .selection import (
    SELECTION_COMPLETE,
    SELECTION_FILE,
    TrialDurationPolicy,
    load_checkpoint_score,
    select_best_checkpoint,
    verify_training_completion_marker,
)


PROMOTION_MANIFEST = "PROMOTION_MANIFEST.json"
PROMOTION_COMPLETE = "PROMOTION_COMPLETE.json"


class PromotionError(RuntimeError):
    pass


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"JSON artifact must be an object: {path}")
    return value


def _verify_promoted_directory(
    path: Path, expected_manifest: Mapping[str, Any], base_config: Any
) -> None:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "inference_config.json",
        SELECTION_FILE,
        PROMOTION_MANIFEST,
        PROMOTION_COMPLETE,
    }
    missing = sorted(
        name
        for name in required
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    )
    if missing:
        raise PromotionError(f"promoted adapter is incomplete at {path}; missing={missing}")
    manifest = _load_json_object(path / PROMOTION_MANIFEST)
    marker = _load_json_object(path / PROMOTION_COMPLETE)
    if manifest != dict(expected_manifest):
        raise PromotionError(f"promoted adapter manifest differs: {path}")
    files = manifest.get("files")
    if not isinstance(files, list) or sorted(files) != sorted(
        {"adapter_config.json", "adapter_model.safetensors", "inference_config.json", SELECTION_FILE}
    ):
        raise PromotionError(f"promoted adapter manifest has an invalid file list: {path}")
    for name in files:
        file_path = path / name
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise PromotionError(f"promoted adapter file is missing or empty: {file_path}")
    if marker != {
        "task": manifest["task"],
        "learning_rate": manifest["selected_learning_rate"],
        "seed": manifest["selected_seed"],
        "epoch": manifest["selected_epoch"],
        "promotion_source": "canonical_validation_selection",
    }:
        raise PromotionError(f"promoted adapter completion marker is inconsistent: {path}")
    _load_json_object(path / "adapter_config.json")
    _load_json_object(path / "inference_config.json")
    _load_json_object(path / SELECTION_FILE)
    get_model_backend(base_config).validate_adapter_directory(base_config, path)


def _load_selection(
    selection_path: Path, *, base_config: Any, require_formal_duration: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_path = selection_path.parent / SELECTION_COMPLETE
    selection = _load_json_object(selection_path)
    marker = _load_json_object(marker_path)
    if selection.get("selection_source") != "canonical_validation_only":
        raise PromotionError("adapter promotion requires canonical-validation-only selection")
    if marker.get("selection_source") != "canonical_validation_only":
        raise PromotionError("selection completion marker is not validation-only")
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise PromotionError("selection is missing selected checkpoint")
    marker_checks = {
        "task": selected.get("task"),
        "learning_rate": selected.get("learning_rate"),
        "seed": selected.get("seed"),
        "epoch": selected.get("epoch"),
        "checkpoint_dir": selected.get("checkpoint_dir"),
        "selection_source": "canonical_validation_only",
    }
    if marker != marker_checks:
        raise PromotionError("selection completion marker is inconsistent")
    duration_policy = None
    configured_halving = base_config.search.successive_halving
    if require_formal_duration and (
        configured_halving is None or not configured_halving.enabled
    ):
        try:
            duration_policy = TrialDurationPolicy(**selection["duration_policy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionError("invalid selection duration policy") from exc
        expected = TrialDurationPolicy.from_search(base_config.search)
        if duration_policy != expected:
            raise PromotionError("selection duration policy differs from configuration")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PromotionError("selection has no validation candidates")
    verified_scores = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise PromotionError("selection candidate must be an object")
        try:
            verified = load_checkpoint_score(
                Path(candidate["validation_dir"]) / "VALIDATION_COMPLETE.json"
            )
        except Exception as exc:
            raise PromotionError(
                f"selection validation candidate cannot be verified: {candidate.get('validation_dir')}"
            ) from exc
        if asdict(verified) != candidate:
            raise PromotionError(f"selection validation candidate changed: {candidate['validation_dir']}")
        verified_scores.append(verified)
    required_trials = selection.get("required_trials")
    if not isinstance(required_trials, list) or not required_trials:
        raise PromotionError("selection required_trials is invalid")
    required_pairs: set[tuple[float, int]] = set()
    for entry in required_trials:
        if not isinstance(entry, dict):
            raise PromotionError("required trial parameters must be objects")
        try:
            pair = (float(entry["learning_rate"]), int(entry["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionError("required trial parameters are incomplete") from exc
        if pair in required_pairs:
            raise PromotionError("required trial parameters contain duplicates")
        required_pairs.add(pair)
    if {score.trial_key for score in verified_scores} != required_pairs:
        raise PromotionError("selection candidates do not exactly cover required trials")
    completion_inputs = selection.get("input_training_complete")
    if not isinstance(completion_inputs, list) or len(completion_inputs) != len(required_pairs):
        raise PromotionError("selection training completion inputs are missing or incomplete")
    verified_pairs: set[tuple[float, int]] = set()
    for entry in completion_inputs:
        if not isinstance(entry, dict):
            raise PromotionError("training completion input must be an object")
        pair = (float(entry.get("learning_rate")), int(entry.get("seed")))
        if pair not in required_pairs or pair in verified_pairs:
            raise PromotionError("training completion input parameters are invalid")
        verified_entry = verify_training_completion_marker(
            learning_rate=pair[0],
            seed=pair[1],
            scores=verified_scores,
            marker_path=entry.get("path", ""),
            duration_policy=duration_policy,
        )
        if verified_entry != entry:
            raise PromotionError("training completion input changed")
        verified_pairs.add(pair)
    winner = select_best_checkpoint(verified_scores)
    if selected != asdict(winner):
        raise PromotionError("selected checkpoint is not the validation-only ranked winner")
    if selection.get("task") != winner.task:
        raise PromotionError("selection task does not match the winning checkpoint")
    return selection, marker


def _copy_verified(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise PromotionError(f"promotion source is missing or empty: {source}")
    shutil.copy2(source, destination)
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        raise PromotionError(f"promotion copy is incomplete: {source} -> {destination}")
    if source.suffix == ".json":
        if _load_json_object(source) != _load_json_object(destination):
            raise PromotionError(f"promotion JSON copy differs: {source} -> {destination}")


def _planned_manifest(*, selection: Mapping[str, Any], base_config: Any) -> dict[str, Any]:
    selected = selection["selected"]
    manifest: dict[str, Any] = {
        "promotion_source": "canonical_validation_selection",
        "base_model": base_config.base_model.name,
        "task": selection["task"],
        "selected_epoch": selected["epoch"],
        "selected_learning_rate": selected["learning_rate"],
        "selected_seed": selected["seed"],
        "selected_checkpoint_dir": selected["checkpoint_dir"],
        "files": [
            "adapter_config.json",
            "adapter_model.safetensors",
            "inference_config.json",
            SELECTION_FILE,
        ],
    }
    if "duration_policy" in selection:
        manifest["duration_policy"] = selection["duration_policy"]
    return manifest


def promote_selected_adapter(
    *,
    base_config: Any,
    selection_path: str | Path,
    final_adapter_dir: str | Path,
    small_job: bool = False,
    scratch_root: str | Path | None = None,
) -> Path:
    selection_file = Path(selection_path).expanduser().resolve()
    if selection_file.name != SELECTION_FILE:
        raise PromotionError(f"selection path must end with {SELECTION_FILE}")
    selection, _ = _load_selection(
        selection_file,
        base_config=base_config,
        require_formal_duration=not small_job,
    )
    selected = selection["selected"]
    checkpoint = Path(selected["checkpoint_dir"]).expanduser().resolve()
    validate_checkpoint_files(checkpoint)
    get_model_backend(base_config).validate_adapter_directory(base_config, checkpoint)

    destination = Path(final_adapter_dir).expanduser().resolve()
    formal_root = Path(base_config.paths.final_adapter_template).parent.resolve()
    if small_job:
        if scratch_root is None:
            raise PromotionError("small-job promotion requires scratch_root")
        scratch = Path(scratch_root).expanduser().resolve()
        try:
            destination.relative_to(scratch)
        except ValueError as exc:
            raise PromotionError(
                f"small-job promotion must remain under scratch root {scratch}: {destination}"
            ) from exc
        try:
            destination.relative_to(formal_root)
        except ValueError:
            pass
        else:
            raise PromotionError("small-job promotion cannot write under the formal adapter root")
    else:
        expected = Path(
            base_config.paths.final_adapter_template.format(
                adapter_name=selection["task"], task=selection["task"]
            )
        ).resolve()
        if destination != expected:
            raise PromotionError(f"formal adapter path must equal configured path {expected}")

    inference_config: dict[str, Any] = {
        "base_model": base_config.base_model.name,
        "base_model_path": str(base_config.base_model.path),
        "task": selection["task"],
        "resolution_candidate": selection["resolution_candidate"],
        "generation": selection["generation"],
        "few_shot_k": 0,
        "learning_rate": selected["learning_rate"],
        "seed": selected["seed"],
        "epoch": selected["epoch"],
    }
    if "duration_policy" in selection:
        inference_config["duration_policy"] = selection["duration_policy"]
    staging = destination.parent / f".{destination.name}.promoting-{os.getpid()}"
    if staging.exists():
        raise PromotionError(f"promotion staging directory already exists: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    manifest = _planned_manifest(selection=selection, base_config=base_config)
    try:
        _copy_verified(checkpoint / "adapter_config.json", staging / "adapter_config.json")
        _copy_verified(
            checkpoint / "adapter_model.safetensors",
            staging / "adapter_model.safetensors",
        )
        _copy_verified(selection_file, staging / SELECTION_FILE)
        atomic_write_json(staging / "inference_config.json", inference_config)
        atomic_write_json(staging / PROMOTION_MANIFEST, manifest)
        atomic_write_json(
            staging / PROMOTION_COMPLETE,
            {
                "task": selection["task"],
                "learning_rate": selected["learning_rate"],
                "seed": selected["seed"],
                "epoch": selected["epoch"],
                "promotion_source": "canonical_validation_selection",
            },
        )
        _verify_promoted_directory(staging, manifest, base_config)
        if destination.exists():
            existing = _load_json_object(destination / PROMOTION_MANIFEST)
            if existing != manifest:
                raise PromotionError(
                    f"existing adapter differs and will not be overwritten: {destination}"
                )
            _verify_promoted_directory(destination, manifest, base_config)
            shutil.rmtree(staging)
            return destination
        os.replace(staging, destination)
        _verify_promoted_directory(destination, manifest, base_config)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
