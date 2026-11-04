"""  init  ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune.config.schema import BaseLoraConfig, ResolutionCandidate
from finetune.datasets import AdapterExample

from .collator import QwenAssistantOnlyCollator
from .messages import build_qwen_messages
from .model import (
    LANGUAGE_LORA_TARGET_PATTERN,
    _validate_adapter_directory,
    _validate_frozen_model_settings,
    load_qlora_model,
    reload_adapter,
)
from .processor import load_qwen_processor


@dataclass(frozen=True)
class Qwen3VLBackend:

    model_type: str = "qwen3_vl"
    lora_target_pattern: str = LANGUAGE_LORA_TARGET_PATTERN

    def validate_config(self, base_config: BaseLoraConfig) -> None:
        _validate_frozen_model_settings(base_config)

    def load_training_model(self, base_config: BaseLoraConfig):
        return load_qlora_model(base_config)

    def reload_adapter(
        self, base_config: BaseLoraConfig, adapter_path: str | Path
    ) -> Any:
        return reload_adapter(base_config, adapter_path)

    def load_processor(
        self,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> Any:
        return load_qwen_processor(base_config, resolution_candidate)

    def save_processor(
        self,
        processor: Any,
        destination: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        del base_config, resolution_candidate
        processor.save_pretrained(Path(destination).expanduser().resolve())

    def validate_processor_artifact(
        self,
        directory: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        del base_config, resolution_candidate
        target = Path(directory).expanduser().resolve()
        if not target.is_dir() or not any(path.is_file() for path in target.iterdir()):
            raise RuntimeError(f"Qwen processor artifact is empty: {target}")

    def create_collator(self, processor: Any, max_length: int) -> Any:
        return QwenAssistantOnlyCollator(processor, max_length)

    def build_messages(self, example: AdapterExample) -> list[dict[str, Any]]:
        return build_qwen_messages(example)

    def prepare_generation_inputs(
        self,
        processor: Any,
        example: AdapterExample,
        *,
        add_generation_prompt: bool = True,
    ) -> Any:
        messages = self.build_messages(example)
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )

    def validate_adapter_directory(
        self, base_config: BaseLoraConfig, adapter_path: Path
    ) -> None:
        _validate_adapter_directory(base_config, adapter_path)


QWEN3_VL_BACKEND = Qwen3VLBackend()

__all__ = [
    "LANGUAGE_LORA_TARGET_PATTERN",
    "QWEN3_VL_BACKEND",
    "Qwen3VLBackend",
    "QwenAssistantOnlyCollator",
    "build_qwen_messages",
    "load_qlora_model",
    "load_qwen_processor",
    "reload_adapter",
]
