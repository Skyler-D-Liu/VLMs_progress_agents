"""  init  ."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "QwenAssistantOnlyCollator": (
        "finetune.model_backends.qwen3_vl.collator",
        "QwenAssistantOnlyCollator",
    ),
    "build_qwen_messages": ("finetune.model_backends.qwen3_vl.messages", "build_qwen_messages"),
    "create_lora_trainer": ("finetune.training.trainer_builder", "create_lora_trainer"),
    "find_resume_checkpoint": (
        "finetune.training.checkpoint_manager",
        "find_resume_checkpoint",
    ),
    "load_qlora_model": ("finetune.model_backends.qwen3_vl.model", "load_qlora_model"),
    "load_qwen_processor": (
        "finetune.model_backends.qwen3_vl.processor",
        "load_qwen_processor",
    ),
    "reload_adapter": ("finetune.model_backends.qwen3_vl.model", "reload_adapter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
