"""Base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from finetune.config.schema import BaseLoraConfig, ResolutionCandidate
from finetune.datasets import AdapterExample


@runtime_checkable
class ModelBackend(Protocol):

    model_type: str
    lora_target_pattern: str

    def validate_config(self, base_config: BaseLoraConfig) -> None:
        pass

    def load_training_model(self, base_config: BaseLoraConfig) -> tuple[Any, Any]:
        pass

    def reload_adapter(
        self, base_config: BaseLoraConfig, adapter_path: str | Path
    ) -> Any:
        pass

    def load_processor(
        self,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> Any:
        pass

    def save_processor(
        self,
        processor: Any,
        destination: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        pass

    def validate_processor_artifact(
        self,
        directory: str | Path,
        base_config: BaseLoraConfig,
        resolution_candidate: str | ResolutionCandidate,
    ) -> None:
        pass

    def create_collator(self, processor: Any, max_length: int) -> Any:
        pass

    def build_messages(self, example: AdapterExample) -> list[dict[str, Any]]:
        pass

    def prepare_generation_inputs(
        self,
        processor: Any,
        example: AdapterExample,
        *,
        add_generation_prompt: bool = True,
    ) -> Any:
        pass

    def validate_adapter_directory(
        self, base_config: BaseLoraConfig, adapter_path: Path
    ) -> None:
        pass
