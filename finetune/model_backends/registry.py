"""Registry."""

from __future__ import annotations

from typing import Any

from finetune.config.schema import BaseLoraConfig

from .base import ModelBackend
from .llava_onevision2 import LLAVA_ONEVISION2_BACKEND
from .qwen3_vl import QWEN3_VL_BACKEND


class BackendNotAvailableError(ValueError):
    pass


_BACKENDS: dict[str, ModelBackend] = {
    QWEN3_VL_BACKEND.model_type: QWEN3_VL_BACKEND,
    LLAVA_ONEVISION2_BACKEND.model_type: LLAVA_ONEVISION2_BACKEND,
}


def _configured_model_type(config_or_model_type: BaseLoraConfig | str | Any) -> str:
    if isinstance(config_or_model_type, str):
        value = config_or_model_type
    else:
        base_model = getattr(config_or_model_type, "base_model", None)
        value = getattr(base_model, "model_type", None)
    if not isinstance(value, str) or not value.strip():
        raise BackendNotAvailableError("base_model.model_type must be a non-empty string")
    return value.strip().casefold()


def get_model_backend(config_or_model_type: BaseLoraConfig | str | Any) -> ModelBackend:
    model_type = _configured_model_type(config_or_model_type)
    backend = _BACKENDS.get(model_type)
    if backend is None:
        available = ", ".join(sorted(_BACKENDS))
        raise BackendNotAvailableError(
            f"No fine-tuning backend is registered for model_type={model_type!r}; "
            f"available: {available}"
        )
    return backend
