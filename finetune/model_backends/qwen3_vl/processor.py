"""Processor."""

from __future__ import annotations

from finetune.config.schema import BaseLoraConfig, ResolutionCandidate
from finetune.training.input_errors import TrainingDataError


def _resolution_candidate(
    config: BaseLoraConfig, candidate: str | ResolutionCandidate
) -> ResolutionCandidate:
    if isinstance(candidate, ResolutionCandidate):
        selected = candidate
    else:
        matches = [item for item in config.resolution.candidates_high_to_low if item.id == candidate]
        if len(matches) != 1:
            available = ", ".join(item.id for item in config.resolution.candidates_high_to_low)
            raise TrainingDataError(
                f"Unknown resolution candidate {candidate!r}; expected one of: {available}"
            )
        selected = matches[0]
    if selected not in config.resolution.candidates_high_to_low:
        raise TrainingDataError(
            f"Resolution candidate {selected.id!r} is not frozen in base_lora.yaml"
        )
    return selected


def load_qwen_processor(
    base_config: BaseLoraConfig,
    resolution_candidate: str | ResolutionCandidate,
):
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Qwen processing requires Transformers. Install it through "
            "the required dependencies."
        ) from exc

    selected = _resolution_candidate(base_config, resolution_candidate)
    processor = AutoProcessor.from_pretrained(
        str(base_config.base_model.path),
        local_files_only=True,
        trust_remote_code=False,
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise TrainingDataError("The local Base processor has no image_processor")
    current_size = dict(getattr(image_processor, "size", {}) or {})
    min_pixels = int(current_size.get("shortest_edge", 65_536))
    if min_pixels > selected.max_pixels_per_image:
        raise TrainingDataError(
            f"Processor min_pixels={min_pixels} exceeds selected max_pixels="
            f"{selected.max_pixels_per_image}"
        )
    image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": selected.max_pixels_per_image,
    }
    return processor
