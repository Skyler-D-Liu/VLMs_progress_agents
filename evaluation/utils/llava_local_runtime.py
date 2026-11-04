"""Llava local runtime."""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


LLAVA_MODEL_ROOT_DEFAULT = "./downloaded_LLaVA"

LLAVA_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "LLaVA-OneVision-0.5B": {
        "hf_repo": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "release_date": "2024-08-05",
        "generation": "onevision",
    },
    "LLaVA-OneVision-7B": {
        "hf_repo": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "release_date": "2024-08-05",
        "generation": "onevision",
    },
    "LLaVA-OneVision-72B": {
        "hf_repo": "llava-hf/llava-onevision-qwen2-72b-ov-hf",
        "release_date": "2024-08-05",
        "generation": "onevision",
    },
    "LLaVA-OV-1.5-4B": {
        "hf_repo": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        "release_date": "2025-09-26",
        "generation": "onevision-1.5",
    },
    "LLaVA-OV-1.5-8B": {
        "hf_repo": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "release_date": "2025-09-16",
        "generation": "onevision-1.5",
    },
    "LLaVA-OV-2-8B": {
        "hf_repo": "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct",
        "release_date": "2026-05-11",
        "generation": "onevision-2",
    },
}

_LLAVA_MODEL_CACHE: Dict[tuple[str, "ResolvedLlavaModel"], Dict[str, Any]] = {}


_LLAVA_1_5_KEY_MAPPING = {
    r"^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}


def build_llava_model_configs() -> Dict[str, Dict[str, str]]:
    return {
        model_key: {
            "api_type": "hf_local",
            "model_name": model_key,
            "release_date": spec["release_date"],
            "family": "LLaVA",
            "model_kind": "base",
            "hf_repo": spec["hf_repo"],
            "llava_generation": spec["generation"],
        }
        for model_key, spec in LLAVA_MODEL_SPECS.items()
    }


def is_llava_config(config: Mapping[str, Any]) -> bool:
    return config.get("family") == "LLaVA"


@dataclass(frozen=True)
class ResolvedLlavaModel:
    model_key: str
    base_path: Path
    hf_repo: str
    generation: str

    def __str__(self) -> str:
        return str(self.base_path)


def _configured_root(explicit: Optional[str]) -> Path:
    value = (
        explicit
        or os.environ.get("LLAVA_MODEL_PATH")
        or os.environ.get("LLAVA_MODEL_ROOT")
        or LLAVA_MODEL_ROOT_DEFAULT
    )
    return Path(value).expanduser()


def _resolve_checkpoint(root: Path, model_name: str, *, single_model: bool) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"LLaVA model path does not exist: {root}")
    candidate = root / model_name
    if candidate.is_dir():
        return candidate
    if single_model and root.is_dir():
        return root
    raise FileNotFoundError(
        f"Unable to resolve local checkpoint for {model_name!r}. Expected directory: {candidate}"
    )


def resolve_llava_model_specs(
    model_keys: Iterable[str],
    model_configs: Mapping[str, Mapping[str, Any]],
    llava_model_path: Optional[str] = None,
) -> Dict[str, ResolvedLlavaModel]:
    local_keys = [
        key
        for key in model_keys
        if model_configs[key].get("api_type") == "hf_local"
        and is_llava_config(model_configs[key])
    ]
    if not local_keys:
        return {}

    root = _configured_root(llava_model_path)
    single_model = len(local_keys) == 1
    resolved: Dict[str, ResolvedLlavaModel] = {}
    for model_key in local_keys:
        config = model_configs[model_key]
        model_name = str(config.get("model_name") or model_key)
        resolved[model_key] = ResolvedLlavaModel(
            model_key=model_key,
            base_path=_resolve_checkpoint(root, model_name, single_model=single_model),
            hf_repo=str(config["hf_repo"]),
            generation=str(config["llava_generation"]),
        )
    return resolved


def _checkpoint_config(checkpoint_path: Path) -> Dict[str, Any]:
    config_path = checkpoint_path / "config.json"
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_llava_model_classes(transformers_module: Any, checkpoint_path: Path):
    config = _checkpoint_config(checkpoint_path)
    architecture_names = [str(name) for name in config.get("architectures", [])]
    auto_map = config.get("auto_map") or {}

    class_names = list(architecture_names)
    if "AutoModelForImageTextToText" in auto_map:
        class_names.append("AutoModelForImageTextToText")
    if "AutoModelForCausalLM" in auto_map:
        class_names.append("AutoModelForCausalLM")
    class_names.extend(
        (
            "LlavaOnevisionForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        )
    )

    seen = set()
    for class_name in class_names:
        if class_name in seen:
            continue
        seen.add(class_name)
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            yield class_name, model_class


def _validate_loaded_model_architecture(
    model: Any, checkpoint_path: Path, model_key: str
) -> None:
    checkpoint_config = _checkpoint_config(checkpoint_path)
    expected_model_type = str(checkpoint_config.get("model_type") or "").strip()
    loaded_config = getattr(model, "config", None)
    loaded_model_type = str(getattr(loaded_config, "model_type", "") or "").strip()
    if expected_model_type and loaded_model_type and loaded_model_type != expected_model_type:
        raise RuntimeError(
            f"Refusing to run {model_key}: checkpoint model_type is "
            f"{expected_model_type!r}, but Transformers instantiated "
            f"{loaded_model_type!r} ({model.__class__.__name__})."
        )


def _load_model_with_attempts(
    model_classes: Iterable[tuple[str, Any]],
    checkpoint_path: str,
    *,
    config: Any = None,
    key_mapping: Optional[Mapping[str, str]] = None,
    disable_mmap: bool = False,
) -> tuple[Any, list[str]]:
    load_attempts = (
        {
            "device_map": "auto",
            "dtype": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        {
            "device_map": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        },
        {"device_map": "auto", "trust_remote_code": True},
    )
    errors = []
    for class_name, model_class in model_classes:
        for kwargs in load_attempts:
            try:
                if config is not None:
                    kwargs = {**kwargs, "config": config}
                if key_mapping is not None:
                    kwargs = {
                        **kwargs,
                        "key_mapping": dict(key_mapping),
                        "output_loading_info": True,
                    }
                if disable_mmap:
                    kwargs = {**kwargs, "disable_mmap": True}
                loaded = model_class.from_pretrained(
                    checkpoint_path,
                    local_files_only=True,
                    **kwargs,
                )
                if key_mapping is not None:
                    model, loading_info = loaded
                    _validate_mapped_loading_info(loading_info)
                else:
                    model = loaded
                return model, errors
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")
    return None, errors


def _validate_mapped_loading_info(loading_info: Mapping[str, Any]) -> None:
    problems = {
        name: list(loading_info.get(name) or [])
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(name)
    }
    if problems:
        summary = "; ".join(
            f"{name}={values[:8]}" for name, values in problems.items()
        )
        raise RuntimeError(f"Mapped checkpoint load was incomplete: {summary}")


def _load_llava_1_5_compat_config(
    transformers_module: Any,
    runtime: ResolvedLlavaModel,
    checkpoint_path: str,
    processor: Any,
) -> Any:
    if runtime.generation != "onevision-1.5":
        return None

    config = transformers_module.AutoConfig.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    generation_path = runtime.base_path / "generation_config.json"
    try:
        generation_config = json.loads(generation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        generation_config = {}
    tokenizer = getattr(processor, "tokenizer", None)
    token_defaults = {
        "bos_token_id": generation_config.get("bos_token_id"),
        "eos_token_id": generation_config.get("eos_token_id"),
        "pad_token_id": generation_config.get("pad_token_id"),
    }
    for name, value in list(token_defaults.items()):
        if value is None and tokenizer is not None:
            token_defaults[name] = getattr(tokenizer, name, None)


    for target in (config, getattr(config, "text_config", None)):
        if target is None:
            continue
        for name, value in token_defaults.items():
            if value is not None and getattr(target, name, None) is None:
                setattr(target, name, int(value))
    return config


def _legacy_default_rope_parameters(
    config: Any,
    device: Any = None,
    seq_len: Optional[int] = None,
    **_: Any,
) -> tuple[Any, float]:
    del seq_len
    import torch

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    partial_rotary_factor = rope_parameters.get(
        "partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0)
    )
    dim = int(head_dim * float(partial_rotary_factor))
    inv_freq = 1.0 / (
        float(base)
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64)
            .to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def _install_llava_1_5_rope_compat(rope_init_functions: Dict[str, Any]) -> None:
    rope_init_functions.setdefault("default", _legacy_default_rope_parameters)


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", str(device)) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _is_cuda_device(device: Any) -> bool:
    if isinstance(device, int):
        return device >= 0
    normalized = str(device).strip().lower()
    return normalized.isdigit() or normalized.startswith("cuda")


def load_llava_local_model(
    model_key: str, runtime: ResolvedLlavaModel
) -> Dict[str, Any]:
    if not isinstance(runtime, ResolvedLlavaModel):
        raise TypeError(
            f"Expected ResolvedLlavaModel for {model_key}, got {type(runtime).__name__}"
        )
    cache_key = (model_key, runtime)
    if cache_key in _LLAVA_MODEL_CACHE:
        return _LLAVA_MODEL_CACHE[cache_key]

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Running local LLaVA inference requires torch and a supported "
            "Transformers runtime (>=5.7, or 4.53.x for OneVision-1.5)."
        ) from exc

    if runtime.generation == "onevision-1.5":
        version_parts = str(transformers.__version__).split(".")[:2]
        if version_parts != ["4", "53"]:
            raise RuntimeError(
                f"{model_key} requires Transformers 4.53.x, but "
                f"{transformers.__version__} is active. Use the required dependencies "
                "or prepend the runtime installed by "
                "the required dependencies to PYTHONPATH."
            )

    allow_cpu = os.environ.get("LLAVA_ALLOW_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            f"CUDA is unavailable, so {model_key} would fall back to impractical CPU "
            "inference. Run it in a runtime GPU allocation, or set LLAVA_ALLOW_CPU=1 "
            "only for an intentional CPU diagnostic."
        )

    checkpoint_path = str(runtime.base_path)
    processor_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if runtime.generation == "onevision-1.5":


        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        _install_llava_1_5_rope_compat(ROPE_INIT_FUNCTIONS)
        processor_kwargs["fix_mistral_regex"] = True
    processor = transformers.AutoProcessor.from_pretrained(
        checkpoint_path,
        **processor_kwargs,
    )
    compat_config = _load_llava_1_5_compat_config(
        transformers, runtime, checkpoint_path, processor
    )
    model, errors = _load_model_with_attempts(
        _iter_llava_model_classes(transformers, runtime.base_path),
        checkpoint_path,
        config=compat_config,
        key_mapping=(
            _LLAVA_1_5_KEY_MAPPING
            if runtime.generation == "onevision-1.5"
            else None
        ),
        disable_mmap=(model_key == "LLaVA-OneVision-72B"),
    )
    if model is None:
        raise RuntimeError(
            f"Failed to load local checkpoint for {model_key} at {checkpoint_path}:\n"
            + "\n".join(errors)
        )
    _validate_loaded_model_architecture(model, runtime.base_path, model_key)
    model = model.eval()

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        raw_devices = set(device_map.values())
        devices = {str(device) for device in raw_devices}
        if not any(_is_cuda_device(device) for device in raw_devices) and not allow_cpu:
            raise RuntimeError(
                f"{model_key} loaded without CUDA layers (device map: {sorted(devices)})."
            )
        print(f"[{model_key}] Loaded on devices: {', '.join(sorted(devices))}")
    else:
        print(f"[{model_key}] Loaded on device: {_model_device(model)}")

    bundle = {
        "model": model,
        "processor": processor,
        "runtime": runtime,
        "runtime_family": "LLaVA",
    }
    _LLAVA_MODEL_CACHE[cache_key] = bundle
    return bundle


def _load_image(image_ref: Any) -> Any:
    from PIL import Image

    if hasattr(image_ref, "convert"):
        return image_ref.convert("RGB")
    if not isinstance(image_ref, str):
        raise TypeError(f"Unsupported LLaVA image reference: {type(image_ref)!r}")
    if image_ref.startswith("data:"):
        header, separator, encoded = image_ref.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("LLaVA received an invalid image data URI")
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    return Image.open(image_ref).convert("RGB")


def _few_shot_image_refs(example: Mapping[str, Any]) -> list[Any]:
    if "image_urls" in example:
        return list(example["image_urls"])
    if "image_url" in example:
        return [example["image_url"]]
    raise ValueError("Few-shot example is missing image_url/image_urls.")


def _few_shot_prompt_text(example: Mapping[str, Any], default_user: str) -> str:
    return str(example.get("user_prompt") or example.get("prompt") or default_user)


def _build_messages_and_images(
    image_refs: Sequence[Any],
    system: Optional[str],
    user: str,
    few_shot_examples: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    messages: list[dict[str, Any]] = []
    images: list[Any] = []
    if system:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": str(system)}],
            }
        )

    def user_content(refs: Sequence[Any], prompt: str) -> list[dict[str, str]]:
        content: list[dict[str, str]] = []
        for image_ref in refs:
            images.append(_load_image(image_ref))
            content.append({"type": "image"})
        content.append({"type": "text", "text": str(prompt)})
        return content

    for example in few_shot_examples or []:
        messages.append(
            {
                "role": "user",
                "content": user_content(
                    _few_shot_image_refs(example),
                    _few_shot_prompt_text(example, user),
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(example.get("answer", ""))}
                ],
            }
        )
    messages.append(
        {"role": "user", "content": user_content(image_refs, str(user))}
    )
    return messages, images


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    target = _model_device(model)
    if hasattr(inputs, "to"):
        return inputs.to(target)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(target) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _remove_unsupported_processor_outputs(
    inputs: Any, runtime: ResolvedLlavaModel
) -> Any:
    if runtime.generation == "onevision-1.5" and isinstance(inputs, Mapping):
        inputs.pop("mm_token_type_ids", None)
    return inputs


def run_llava_chat(
    bundle: Mapping[str, Any],
    image_refs: Sequence[Any],
    system: Optional[str],
    user: str,
    few_shot_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    max_new_tokens: int = 128,
) -> str:
    import torch

    model = bundle["model"]
    processor = bundle["processor"]
    messages, images = _build_messages_and_images(
        image_refs, system, user, few_shot_examples
    )
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = _remove_unsupported_processor_outputs(inputs, bundle["runtime"])
    inputs = _move_inputs_to_model(inputs, model)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

    input_ids = inputs["input_ids"] if isinstance(inputs, Mapping) else inputs.input_ids
    trimmed = [
        output_ids[len(prompt_ids) :]
        for prompt_ids, output_ids in zip(input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(decoded[0] if decoded else "").strip()


def local_response_metadata(runtime: ResolvedLlavaModel) -> Dict[str, str]:
    metadata = {
        "local_model_path": str(runtime.base_path),
        "huggingface_repo": runtime.hf_repo,
        "llava_generation": runtime.generation,
    }
    try:
        import transformers

        metadata["transformers_version"] = str(transformers.__version__)
    except (ImportError, AttributeError):
        pass
    return metadata


def format_llava_model_listing(
    model_key: str,
    config: Mapping[str, Any],
    llava_model_root: Optional[str] = None,
) -> str:
    root = _configured_root(llava_model_root)
    return (
        f"  {model_key} (released: {config['release_date']}, "
        f"HF: {config['hf_repo']}, local: {root / model_key})"
    )
