"""Runtime context."""

from __future__ import annotations

import os
from pathlib import Path


BASE_CONFIG_ENV = "VLEO_FINETUNE_BASE_CONFIG"
REGISTRY_ENV = "VLEO_FINETUNE_REGISTRY"
DATASET_CONFIG_ENV = "VLEO_FINETUNE_DATASET_CONFIG"
TASK_QUEUE_ENV = "VLEO_FINETUNE_TASK_QUEUE"
MAX_CONCURRENT_GPU_JOBS_ENV = "VLEO_FINETUNE_MAX_CONCURRENT_GPU_JOBS"
GPU_EXECUTION_PROFILE_ENV = "VLEO_FINETUNE_GPU_EXECUTION_PROFILE"


def _active_path(variable: str, default: Path | str | None) -> Path | None:
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        return None
    return Path(default).expanduser().resolve()


def active_base_config(default: Path | str) -> Path:
    return _active_path(BASE_CONFIG_ENV, default)


def active_registry(default: Path | str) -> Path:
    return _active_path(REGISTRY_ENV, default)


def active_dataset_config(default: Path | str | None = None) -> Path | None:
    return _active_path(DATASET_CONFIG_ENV, default)


