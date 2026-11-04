"""Processor."""

import json
from pathlib import Path
import shutil

from finetune.config.schema import BaseLoraConfig, ResolutionCandidate
from finetune.training.input_errors import TrainingDataError


PROCESSOR_ARTIFACT_NAMES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "processor_config.json",
    "image_processor_config.json",
    "chat_template.jinja",
    "processing_llava_onevision2.py",
    "video_processing_llava_onevision2.py",
    "codec_video_processing_llava_onevision2.py",
)
REQUIRED_PROCESSOR_ARTIFACTS = frozenset(
    {
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "processing_llava_onevision2.py",
        "video_processing_llava_onevision2.py",
    }
)


def load_llava_processor(base_config: BaseLoraConfig, resolution_candidate: str | ResolutionCandidate):
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("LLaVA processing requires Transformers 5.7 or newer") from exc
    selected = _selected_resolution(base_config, resolution_candidate)
    if not base_config.resolution.preserve_aspect_ratio:
        raise TrainingDataError("LLaVA training requires preserve_aspect_ratio=true")
    if base_config.resolution.crop or base_config.resolution.tile or base_config.resolution.stretch:
        raise TrainingDataError(
            "LLaVA image training does not permit crop, tile, or stretch transforms"
        )
    processor = AutoProcessor.from_pretrained(
        str(base_config.base_model.path), local_files_only=True, trust_remote_code=True
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise TrainingDataError("The local LLaVA processor has no image_processor")
    size = dict(getattr(image_processor, "size", {}) or {})
    shortest = int(
        getattr(image_processor, "min_pixels", size.get("shortest_edge", 3136))
    )
    max_pixels = int(selected.max_pixels_per_image)
    if shortest > max_pixels:
        raise TrainingDataError(
            f"Processor min_pixels={shortest} exceeds selected max_pixels={max_pixels}"
        )
    image_processor.size = {
        "shortest_edge": shortest,
        "longest_edge": max_pixels,
    }
    if hasattr(image_processor, "default_to_square"):
        image_processor.default_to_square = False
    if hasattr(image_processor, "do_center_crop"):
        image_processor.do_center_crop = False


    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = shortest
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels
    if hasattr(image_processor, "do_resize"):
        image_processor.do_resize = True
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None and hasattr(video_processor, "max_pixels"):


        video_processor.max_pixels = max_pixels
    if not callable(getattr(processor, "save_pretrained", None)):


        setattr(processor, "_llava_local_processor_saver", True)
        setattr(
            processor,
            "save_pretrained",
            lambda destination: save_llava_processor(
                processor, destination, base_config, selected
            ),
        )
    return processor


def save_llava_processor(
    processor,
    destination: str | Path,
    base_config: BaseLoraConfig,
    resolution_candidate: str | ResolutionCandidate,
) -> None:
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    source_root = Path(base_config.base_model.path).expanduser().resolve()
    saver = getattr(processor, "save_pretrained", None)
    if callable(saver) and not getattr(processor, "_llava_local_processor_saver", False):
        saver(target)


        _copy_missing_processor_assets(source_root, target)
        _write_resolution_cap(target, base_config, resolution_candidate)
        _validate_processor_artifact(target, base_config, resolution_candidate)
        return
    available = _copy_missing_processor_assets(source_root, target)
    _write_resolution_cap(target, base_config, resolution_candidate)
    if available == 0:
        raise TrainingDataError(
            f"No local LLaVA processor assets were found under {source_root}"
        )
    _validate_processor_artifact(target, base_config, resolution_candidate)


def _copy_missing_processor_assets(source_root: Path, target: Path) -> int:
    available = 0
    for name in PROCESSOR_ARTIFACT_NAMES:
        source = source_root / name
        destination = target / name
        if not source.is_file():
            continue
        available += 1
        if not destination.is_file() or destination.stat().st_size == 0:
            shutil.copy2(source, destination)
    return available


def _selected_resolution(
    base_config: BaseLoraConfig,
    resolution_candidate: str | ResolutionCandidate,
) -> ResolutionCandidate:
    if isinstance(resolution_candidate, ResolutionCandidate):
        selected = resolution_candidate
    else:
        matches = [
            item
            for item in base_config.resolution.candidates_high_to_low
            if item.id == resolution_candidate
        ]
        if len(matches) != 1:
            raise TrainingDataError(f"Unknown resolution candidate {resolution_candidate!r}")
        selected = matches[0]
    if selected not in base_config.resolution.candidates_high_to_low:
        raise TrainingDataError(
            f"Resolution candidate {selected.id!r} is not frozen in the base configuration"
        )
    return selected


def _write_resolution_cap(
    target: Path,
    base_config: BaseLoraConfig,
    resolution_candidate: str | ResolutionCandidate,
) -> None:
    selected = _selected_resolution(base_config, resolution_candidate)
    config_path = target / "preprocessor_config.json"
    if not config_path.is_file():
        return
    try:
        processor_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(processor_config, dict):
            raise ValueError("processor config is not an object")
        processor_config["max_pixels"] = int(selected.max_pixels_per_image)
        processor_config["default_to_square"] = False
        processor_config["do_center_crop"] = False
        size = dict(processor_config.get("size", {}) or {})
        raw_min_pixels = processor_config.get("min_pixels", size.get("shortest_edge"))
        if raw_min_pixels is None:
            raise ValueError("processor config has no min_pixels/shortest_edge")
        processor_config["min_pixels"] = int(raw_min_pixels)
        if processor_config["min_pixels"] <= 0:
            raise ValueError("processor min_pixels must be positive")
        size["longest_edge"] = int(selected.max_pixels_per_image)
        size.setdefault("shortest_edge", processor_config["min_pixels"])
        processor_config["size"] = size
        config_path.write_text(
            json.dumps(processor_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingDataError(
            f"Unable to persist the LLaVA processor configuration in {target}"
        ) from exc


def _validate_processor_artifact(
    directory: str | Path,
    base_config: BaseLoraConfig,
    resolution_candidate: str | ResolutionCandidate,
) -> None:
    target = Path(directory).expanduser().resolve()
    if not target.is_dir() or not any(path.is_file() for path in target.iterdir()):
        raise TrainingDataError(f"LLaVA processor artifact is empty: {target}")
    missing = sorted(
        name
        for name in REQUIRED_PROCESSOR_ARTIFACTS
        if not (target / name).is_file() or (target / name).stat().st_size == 0
    )
    if missing:
        raise TrainingDataError(
            f"LLaVA processor artifact is incomplete at {target}; missing: "
            + ", ".join(missing)
        )
    selected = _selected_resolution(base_config, resolution_candidate)
    config_path = target / "preprocessor_config.json"
    if not config_path.is_file():
        raise TrainingDataError(f"LLaVA processor artifact lacks preprocessor_config.json: {target}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Invalid persisted LLaVA processor config: {config_path}") from exc
    if not isinstance(config, dict):
        raise TrainingDataError(f"Persisted LLaVA processor config must be an object: {config_path}")
    if config.get("do_center_crop") is True or config.get("default_to_square") is True:
        raise TrainingDataError(
            "Persisted LLaVA processor enables a crop/square transform; "
            "aspect-ratio-preserving training requires both disabled"
        )
    expected_max = int(selected.max_pixels_per_image)
    configured_max = config.get("max_pixels")
    size = config.get("size")
    configured_min = config.get("min_pixels")
    size_min = size.get("shortest_edge") if isinstance(size, dict) else None
    if configured_min is None and size_min is None:
        raise TrainingDataError(
            f"Persisted LLaVA processor config has no minimum pixel size: {config_path}"
        )
    size_max = size.get("longest_edge") if isinstance(size, dict) else None
    if configured_max is None and size_max is None:
        raise TrainingDataError(
            f"Persisted LLaVA processor config has no resolution cap: {config_path}"
        )
    try:
        if configured_min is not None and int(configured_min) <= 0:
            raise TrainingDataError(
                f"Persisted LLaVA min_pixels={configured_min!r} is not positive"
            )
        if size_min is not None and int(size_min) <= 0:
            raise TrainingDataError(
                f"Persisted LLaVA shortest_edge={size_min!r} is not positive"
            )
        if configured_max is not None and int(configured_max) != expected_max:
            raise TrainingDataError(
                f"Persisted LLaVA max_pixels={configured_max!r} does not match {expected_max}"
            )
        if size_max is not None and int(size_max) != expected_max:
            raise TrainingDataError(
                f"Persisted LLaVA longest_edge={size_max!r} does not match {expected_max}"
            )
    except (TypeError, ValueError) as exc:
        raise TrainingDataError(
            f"Persisted LLaVA processor resolution cap is not an integer: {config_path}"
        ) from exc


__all__ = [
    "PROCESSOR_ARTIFACT_NAMES",
    "REQUIRED_PROCESSOR_ARTIFACTS",
    "load_llava_processor",
    "save_llava_processor",
    "_validate_processor_artifact",
]
