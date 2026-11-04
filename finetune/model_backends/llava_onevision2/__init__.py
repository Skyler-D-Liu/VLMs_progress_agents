"""  init  ."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune.config.schema import BaseLoraConfig, ResolutionCandidate
from finetune.datasets import AdapterExample

from .collator import LlavaOneVision2AssistantOnlyCollator
from .messages import build_llava_messages
from .model import (
    LANGUAGE_LORA_TARGET_PATTERN,
    LANGUAGE_LORA_TARGETS,
    _validate_adapter_directory,
    _validate_frozen_model_settings,
    load_qlora_model,
    reload_adapter,
)
from .processor import (
    _validate_processor_artifact,
    load_llava_processor,
    save_llava_processor,
)


@dataclass(frozen=True)
class LLaVAOneVision2Backend:
    model_type: str = "llava_onevision2"
    lora_target_pattern: str = LANGUAGE_LORA_TARGET_PATTERN

    def validate_config(self, base_config: BaseLoraConfig) -> None:
        _validate_frozen_model_settings(base_config)

    def load_training_model(self, base_config: BaseLoraConfig):
        return load_qlora_model(base_config)

    def reload_adapter(self, base_config: BaseLoraConfig, adapter_path: str | Path) -> Any:
        return reload_adapter(base_config, adapter_path)

    def load_processor(self, base_config: BaseLoraConfig, resolution_candidate: str | ResolutionCandidate):
        return load_llava_processor(base_config, resolution_candidate)

    def save_processor(
        self,
        processor: Any,
        destination: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        save_llava_processor(processor, destination, base_config, resolution_candidate)

    def validate_processor_artifact(
        self,
        directory: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        _validate_processor_artifact(directory, base_config, resolution_candidate)

    def create_collator(self, processor: Any, max_length: int):
        return LlavaOneVision2AssistantOnlyCollator(processor, max_length)

    def build_messages(self, example: AdapterExample):
        return build_llava_messages(example)

    def prepare_generation_inputs(
        self,
        processor: Any,
        example: AdapterExample,
        *,
        add_generation_prompt: bool = True,
    ) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("LLaVA generation requires Pillow") from exc

        messages = self.build_messages(example)
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        images = []
        for raw_path in example.image_paths:
            with Image.open(Path(raw_path).expanduser().resolve()) as image:
                images.append(image.convert("RGB"))
        return processor(
            text=[rendered],
            images=images,
            return_tensors="pt",
            padding=True,
        )

    def validate_adapter_directory(self, base_config: BaseLoraConfig, adapter_path: Path) -> None:
        _validate_adapter_directory(base_config, adapter_path)


LLAVA_ONEVISION2_BACKEND = LLaVAOneVision2Backend()

__all__ = [
    "LANGUAGE_LORA_TARGET_PATTERN",
    "LANGUAGE_LORA_TARGETS",
    "LLAVA_ONEVISION2_BACKEND",
    "LLaVAOneVision2Backend",
]
