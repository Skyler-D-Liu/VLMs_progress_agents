"""Training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from transformers import TrainerCallback

from .checkpoint_manager import atomic_write_json


INCOMPATIBLE_TRAINING_EXIT_CODE = 78


class IncompatibleTrainingManifestError(RuntimeError):
    pass


def _matches_existing_semantic_training(
    training_dir: Path,
    existing: Mapping[str, Any],
    requested: Mapping[str, Any],
) -> bool:
    if "compatibility" in existing:
        return False
    required = ("task", "learning_rate", "seed")
    if any(
        field not in existing
        or field not in requested
        or existing[field] != requested[field]
        for field in required
    ):
        return False
    stage = requested.get("stage")
    if stage not in {"screening", "final"} or training_dir.parent.name != stage:
        return False
    if "stage" in existing and existing["stage"] != stage:
        return False
    for field in ("selection_source", "test_data_used"):
        if field in existing and existing[field] != requested.get(field):
            return False
    return True


def initialize_training_files(
    training_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = training_dir / "training_manifest.json"
    requested = dict(manifest)
    if manifest_path.exists():
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IncompatibleTrainingManifestError(
                f"Invalid training manifest: {manifest_path}"
            ) from exc
        if current != requested and not _matches_existing_semantic_training(
            training_dir, current, requested
        ):
            raise IncompatibleTrainingManifestError(
                f"Existing training parameters are incompatible: {manifest_path}"
            )
        return
    atomic_write_json(manifest_path, requested)


class TrainingBudgetCallback(TrainerCallback):
    def __init__(self, budget_steps):
        self.budget_steps = budget_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.budget_steps:
            control.should_save = True
            control.should_training_stop = True
        return control
