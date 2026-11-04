"""Model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from finetune.config.schema import BaseLoraConfig

LANGUAGE_LORA_TARGET_PATTERN = (
    r"^model\.language_model\.layers\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
LANGUAGE_LORA_TARGETS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
EXPECTED_MODEL_NAME = "LLaVA-OneVision-2-8B-Instruct"
EXPECTED_MODEL_PATH = Path("./downloaded_LLaVA/LLaVA-OV-2-8B")
EXPECTED_PATHS = {
    "paths.temporary_trial_template": (
        "./llava_ov2_8b_lora_trials/"
        "{task}/lr-{learning_rate}-seed-{seed}"
    ),
    "paths.final_adapter_template": (
        "./finetuned_llava/{adapter_name}"
    ),
    "paths.permanent_result_root": Path(
        "./results/finetune/llava-ov2-8b"
    ),
    "paths.validation_result_template": (
        "./results/finetune/llava-ov2-8b/"
        "validation/{task}"
    ),
    "paths.test_result_template": (
        "./results/finetune/llava-ov2-8b/test/{task}"
    ),
    "paths.manifest_template": (
        "./results/finetune/llava-ov2-8b/"
        "manifests/{task}"
    ),
}


class ModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainableParameterSummary:
    trainable: int
    total: int
    ratio: float
    trainable_names: tuple[str, ...]


def _disable_model_cache(model: Any) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    language_config = getattr(language_model, "config", None)
    if language_config is not None and hasattr(language_config, "use_cache"):
        language_config.use_cache = False


def _imports():
    try:
        import torch
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        import transformers
    except ImportError as exc:
        raise ImportError(
            "LLaVA QLoRA dependencies are missing; use the LLaVA fine-tuning runtime environment."
        ) from exc
    model_class = getattr(
        transformers, "LlavaOnevision2ForConditionalGeneration", None
    )
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:


        model_class = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_class is None:
        raise ImportError(
            "The installed Transformers release has no LLaVA multimodal model class"
        )
    bits_and_bytes_config = getattr(transformers, "BitsAndBytesConfig", None)
    if bits_and_bytes_config is None:
        raise ImportError("The installed Transformers release has no BitsAndBytesConfig")
    return (
        torch,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        model_class,
        bits_and_bytes_config,
    )


def _validate_frozen_model_settings(config: BaseLoraConfig) -> None:
    if config.base_model.model_type != "llava_onevision2":
        raise ModelConfigurationError(
            "LLaVA backend requires base_model.model_type='llava_onevision2'"
        )
    if config.base_model.architecture != "LlavaOnevision2ForConditionalGeneration":
        raise ModelConfigurationError(
            "LLaVA backend requires architecture="
            "'LlavaOnevision2ForConditionalGeneration'"
        )
    if config.base_model.name != EXPECTED_MODEL_NAME:
        raise ModelConfigurationError(
            f"LLaVA backend requires base_model.name={EXPECTED_MODEL_NAME!r}"
        )
    if config.base_model.path.expanduser().resolve() != EXPECTED_MODEL_PATH.resolve():
        raise ModelConfigurationError(
            f"LLaVA backend requires the local checkpoint at {EXPECTED_MODEL_PATH}"
        )
    if config.hardware.gpu_type.casefold() != "h200":
        raise ModelConfigurationError("LLaVA QLoRA requires the frozen H200 execution profile")
    if config.hardware.gpus_per_trial != 1:
        raise ModelConfigurationError("LLaVA QLoRA requires exactly one GPU per trial")
    training = config.training
    quant = training.quantization
    lora = training.lora
    expected = {
        "training.method": (training.method, "qlora"),
        "training.loss_scope": (training.loss_scope, "assistant_only"),
        "training.mask_non_assistant_tokens": (training.mask_non_assistant_tokens, True),
        "training.truncate_gold_answer": (training.truncate_gold_answer, False),
        "training.overflow_action": (training.overflow_action, "reject"),
        "training.data_augmentation": (training.data_augmentation, "none"),
        "training.max_length": (training.max_length, 16384),
        "training.attention_implementation": (
            training.attention_implementation,
            "sdpa",
        ),
        "quantization.bits": (quant.bits, 4),
        "quantization.method": (quant.method, "bnb"),
        "quantization.quant_type": (quant.quant_type, "nf4"),
        "quantization.double_quantization": (quant.double_quantization, True),
        "quantization.compute_dtype": (quant.compute_dtype, "bfloat16"),
        "lora.freeze_vision_encoder": (lora.freeze_vision_encoder, True),
        "lora.target_modules": (lora.target_modules, "language_decoder_attention_mlp"),
        "lora.rank": (lora.rank, 8),
        "lora.alpha": (lora.alpha, 32),
        "lora.dropout": (lora.dropout, 0.05),
        "batch.per_device_train_batch_size": (
            training.batch.per_device_train_batch_size,
            1,
        ),
        "batch.per_device_eval_batch_size": (
            training.batch.per_device_eval_batch_size,
            1,
        ),
        "batch.gradient_accumulation_steps": (
            training.batch.gradient_accumulation_steps,
            8,
        ),
        "batch.effective_global_batch_size": (
            training.batch.effective_global_batch_size,
            8,
        ),
        "training.gradient_checkpointing": (training.gradient_checkpointing, True),
        "optimizer.name": (training.optimizer.name, "adamw_torch_fused"),
        "optimizer.weight_decay": (training.optimizer.weight_decay, 0.1),
        "optimizer.beta1": (training.optimizer.beta1, 0.9),
        "optimizer.beta2": (training.optimizer.beta2, 0.95),
        "optimizer.epsilon": (training.optimizer.epsilon, 1e-8),
        "optimizer.max_grad_norm": (training.optimizer.max_grad_norm, 1.0),
        "scheduler.name": (training.scheduler.name, "cosine"),
        "scheduler.warmup_ratio": (training.scheduler.warmup_ratio, 0.03),
        "resolution.preserve_aspect_ratio": (
            config.resolution.preserve_aspect_ratio,
            True,
        ),
        "resolution.crop": (config.resolution.crop, False),
        "resolution.tile": (config.resolution.tile, False),
        "resolution.stretch": (config.resolution.stretch, False),
        "resolution.offline_resize": (config.resolution.offline_resize, False),
        "resolution.augmentation": (config.resolution.augmentation, False),
    }
    for field, expected_value in EXPECTED_PATHS.items():
        actual_value = getattr(config.paths, field.split(".", 1)[1])
        if isinstance(expected_value, Path):
            actual_value = Path(actual_value).expanduser().resolve()
            expected_value = expected_value.expanduser().resolve()
        expected[field] = (actual_value, expected_value)
    mismatches = [
        f"{field}={actual!r}, expected {wanted!r}"
        for field, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ModelConfigurationError("Frozen QLoRA configuration mismatch: " + "; ".join(mismatches))


def _load_quantized_base(config: BaseLoraConfig):
    torch, _, _, _, _, model_class, BitsAndBytesConfig = _imports()
    _validate_frozen_model_settings(config)
    if not torch.cuda.is_available():
        raise ModelConfigurationError("LLaVA QLoRA requires a CUDA runtime allocation")
    if torch.cuda.device_count() != 1:
        raise ModelConfigurationError(
            f"Frozen training uses exactly one GPU, found {torch.cuda.device_count()}"
        )
    quant = config.training.quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant.quant_type,
        bnb_4bit_use_double_quant=quant.double_quantization,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return model_class.from_pretrained(
        str(config.base_model.path),
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        dtype=torch.bfloat16,
        attn_implementation=config.training.attention_implementation,
        low_cpu_mem_usage=True,
    )


def _matching_language_target_modules(model: Any) -> tuple[str, ...]:
    matcher = re.compile(LANGUAGE_LORA_TARGET_PATTERN)
    matching = tuple(
        name
        for name, _module in model.named_modules()
        if matcher.fullmatch(name)
    )
    matched_targets = {name.rsplit(".", 1)[-1] for name in matching}
    missing_targets = LANGUAGE_LORA_TARGETS - matched_targets
    if missing_targets:
        raise ModelConfigurationError(
            "The LLaVA checkpoint is missing language projection targets: "
            + ", ".join(sorted(missing_targets))
        )
    return matching


def summarize_and_validate_trainable_parameters(model: Any) -> TrainableParameterSummary:
    total = trainable = 0
    names: list[str] = []
    invalid: list[str] = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
            names.append(name)
            normalized = name.casefold()
            lora_base_name = normalized.split(".lora_", 1)[0]
            if (
                "lora_" not in normalized
                or re.search(
                    r"model\.language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
                    lora_base_name,
                )
                is None
                or ".visual." in normalized
                or ".embed_tokens." in normalized
                or ".lm_head." in normalized
                or not any(
                    f".{target}.lora_" in normalized for target in LANGUAGE_LORA_TARGETS
                )
            ):
                invalid.append(name)
    if not trainable:
        raise ModelConfigurationError("LoRA mounting produced zero trainable parameters")
    if invalid:
        raise ModelConfigurationError(
            "Only language-model LoRA parameters may be trainable; invalid: "
            + ", ".join(invalid[:10])
        )
    covered_targets = {
        target
        for target in LANGUAGE_LORA_TARGETS
        if any(f".{target}.lora_" in name.casefold() for name in names)
    }
    missing_targets = LANGUAGE_LORA_TARGETS - covered_targets
    if missing_targets:
        raise ModelConfigurationError(
            "LoRA mounting did not cover language projection targets: "
            + ", ".join(sorted(missing_targets))
        )
    return TrainableParameterSummary(trainable, total, trainable / total, tuple(names))


def load_qlora_model(base_config: BaseLoraConfig):
    _, LoraConfig, _, get_peft_model, prepare_model_for_kbit_training, _, _ = _imports()
    model = _load_quantized_base(base_config)
    matching_modules = _matching_language_target_modules(model)
    if not matching_modules:
        raise ModelConfigurationError(
            "The LLaVA checkpoint has no language-layer LoRA targets matching "
            f"{LANGUAGE_LORA_TARGET_PATTERN!r}"
        )
    _disable_model_cache(model)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=base_config.training.gradient_checkpointing
    )
    if base_config.training.gradient_checkpointing:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ModelConfigurationError(
                "The loaded LLaVA model does not support gradient checkpointing"
            )
        if not bool(getattr(model, "is_gradient_checkpointing", False)):
            enable_checkpointing()
        if getattr(model, "is_gradient_checkpointing", False) is not True:
            raise ModelConfigurationError("LLaVA gradient checkpointing could not be enabled")
    lora = base_config.training.lora
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=LANGUAGE_LORA_TARGET_PATTERN,
            bias="none",
        ),
    )
    for name, parameter in model.named_parameters():
        if "visual" in name.casefold() or "embed_tokens" in name.casefold() or "lm_head" in name.casefold():
            parameter.requires_grad = False
    return model, summarize_and_validate_trainable_parameters(model)


def _validate_adapter_directory(config: BaseLoraConfig, adapter_path: Path) -> None:
    _validate_frozen_model_settings(config)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Adapter directory is incomplete: {adapter_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Invalid adapter config: {config_path}") from exc
    if str(data.get("peft_type", "")).upper() != "LORA":
        raise ModelConfigurationError("Adapter peft_type must be LORA")
    if str(data.get("task_type", "")).upper() != "CAUSAL_LM":
        raise ModelConfigurationError("Adapter task_type must be CAUSAL_LM")
    adapter_lora_settings = {
        "r": (data.get("r"), config.training.lora.rank),
        "lora_alpha": (data.get("lora_alpha"), config.training.lora.alpha),
        "lora_dropout": (data.get("lora_dropout"), config.training.lora.dropout),
        "bias": (data.get("bias"), "none"),
    }
    lora_mismatches = [
        f"{field}={actual!r}, expected {wanted!r}"
        for field, (actual, wanted) in adapter_lora_settings.items()
        if actual != wanted
    ]
    if lora_mismatches:
        raise ModelConfigurationError(
            "Adapter LoRA settings differ from the frozen configuration: "
            + "; ".join(lora_mismatches)
        )
    modules_to_save = data.get("modules_to_save")
    if modules_to_save not in (None, [], ()):
        raise ModelConfigurationError("Adapter modules_to_save must be empty for language-only LoRA")
    adapter_base_value = data.get("base_model_name_or_path")
    if not isinstance(adapter_base_value, str) or not adapter_base_value.strip():
        raise ModelConfigurationError("Adapter config is missing base_model_name_or_path")
    adapter_base = Path(adapter_base_value).name
    expected_bases = {
        config.base_model.name.casefold(),
        config.base_model.path.name.casefold(),
    }
    if adapter_base.casefold() not in expected_bases:
        raise ModelConfigurationError(
            f"Adapter targets {adapter_base!r}, expected one of {sorted(expected_bases)!r}"
        )
    target_modules = data.get("target_modules")

    def allowed_target(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if value == LANGUAGE_LORA_TARGET_PATTERN:
            return True


        return bool(re.fullmatch(LANGUAGE_LORA_TARGET_PATTERN, value))

    if isinstance(target_modules, list):
        if not target_modules or any(not allowed_target(target) for target in target_modules):
            raise ModelConfigurationError(
                "Adapter target_modules must contain only language projection modules"
            )
        if LANGUAGE_LORA_TARGET_PATTERN not in target_modules:
            covered_targets = {
                target.rsplit(".", 1)[-1].casefold()
                for target in target_modules
            }
            missing_targets = LANGUAGE_LORA_TARGETS - covered_targets
            if missing_targets:
                raise ModelConfigurationError(
                    "Adapter target_modules do not cover language projection targets: "
                    + ", ".join(sorted(missing_targets))
                )
    elif isinstance(target_modules, str):
        if not allowed_target(target_modules):
            raise ModelConfigurationError(
                "Adapter target_modules do not match the frozen language-layer pattern"
            )
        if target_modules != LANGUAGE_LORA_TARGET_PATTERN:
            raise ModelConfigurationError(
                "Adapter target_modules must cover all frozen language projections"
            )
    else:
        raise ModelConfigurationError("Adapter config is missing target_modules")
    _validate_adapter_weight_keys(weights_path)


def _validate_adapter_weight_keys(weights_path: Path) -> None:
    try:
        from safetensors import safe_open

        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
    except Exception as exc:
        raise ModelConfigurationError(
            f"Unreadable adapter weights: {weights_path}"
        ) from exc
    if not keys:
        raise ModelConfigurationError("Adapter weights contain no tensors")
    target_pattern = re.compile(
        r"(?:^|\.)model\.language_model\.layers\..*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
        r"lora_([AB])(?:\.[^.]+)?\.weight$",
        re.IGNORECASE,
    )
    matched: list[tuple[str, str, str]] = []
    invalid = []
    for key in keys:
        match = target_pattern.search(key)
        if match is None:
            invalid.append(key)
        else:
            module_key = key[: match.start(2)].rstrip(".")
            matched.append(
                (module_key, match.group(1).casefold(), match.group(2).casefold())
            )
    if invalid:
        raise ModelConfigurationError(
            "Adapter weights must contain language-layer LoRA A/B tensors only; invalid: "
            + ", ".join(invalid[:10])
        )
    groups: dict[str, set[str]] = {}
    for base_name, _target, side in matched:
        groups.setdefault(base_name, set()).add(side)
    incomplete = [base_name for base_name, sides in groups.items() if sides != {"a", "b"}]
    if incomplete:
        raise ModelConfigurationError(
            "Adapter weights must contain both LoRA A and B tensors for every target; "
            + ", ".join(incomplete[:10])
        )
    covered_targets = {target for _base_name, target, _side in matched}
    missing_targets = LANGUAGE_LORA_TARGETS - covered_targets
    if missing_targets:
        raise ModelConfigurationError(
            "Adapter weights do not cover language projection targets: "
            + ", ".join(sorted(missing_targets))
        )


def reload_adapter(base_config: BaseLoraConfig, adapter_path: str | Path):
    _, _, PeftModel, _, _, _, _ = _imports()
    path = Path(adapter_path).expanduser().resolve()
    _validate_adapter_directory(base_config, path)
    model = PeftModel.from_pretrained(
        _load_quantized_base(base_config), str(path), is_trainable=False, local_files_only=True
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ModelConfigurationError("Reloaded inference adapter unexpectedly has trainable parameters")
    return model
