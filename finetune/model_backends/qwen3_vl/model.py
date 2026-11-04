"""Model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from finetune.config.schema import BaseLoraConfig


LANGUAGE_LORA_TARGET_PATTERN = (
    r"^model\.language_model\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class ModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainableParameterSummary:
    trainable: int
    total: int
    ratio: float
    trainable_names: tuple[str, ...]


def _imports():
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen QLoRA dependencies are missing. Run "
            "the required dependencies."
        ) from exc
    return (
        torch,
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )


def _validate_frozen_model_settings(config: BaseLoraConfig) -> None:
    training = config.training
    quant = training.quantization
    lora = training.lora
    expected = {
        "training.method": (training.method, "qlora"),
        "quantization.bits": (quant.bits, 4),
        "quantization.quant_type": (quant.quant_type, "nf4"),
        "quantization.double_quantization": (quant.double_quantization, True),
        "quantization.compute_dtype": (quant.compute_dtype, "bfloat16"),
        "lora.target_modules": (lora.target_modules, "language_decoder_attention_mlp"),
        "lora.freeze_vision_encoder": (lora.freeze_vision_encoder, True),
    }
    mismatches = [f"{field}={actual!r}, expected {wanted!r}" for field, (actual, wanted) in expected.items() if actual != wanted]
    if mismatches:
        raise ModelConfigurationError("Frozen QLoRA configuration mismatch: " + "; ".join(mismatches))


def _load_quantized_base(config: BaseLoraConfig):
    (
        torch,
        _,
        _,
        _,
        _,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    ) = _imports()
    _validate_frozen_model_settings(config)
    if not torch.cuda.is_available():
        raise ModelConfigurationError("Qwen3-VL-32B QLoRA requires a CUDA runtime allocation")
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
    return Qwen3VLForConditionalGeneration.from_pretrained(
        str(config.base_model.path),
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        dtype=torch.bfloat16,
        attn_implementation=config.training.attention_implementation,
        low_cpu_mem_usage=True,
    )


def summarize_and_validate_trainable_parameters(model: Any) -> TrainableParameterSummary:
    total = 0
    trainable = 0
    names = []
    invalid = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append(name)
            normalized = name.casefold()
            if (
                "lora_" not in normalized
                or ".language_model." not in normalized
                or ".visual." in normalized
                or ".lm_head." in normalized
                or ".embed_tokens." in normalized
            ):
                invalid.append(name)
    if trainable == 0:
        raise ModelConfigurationError("LoRA mounting produced zero trainable parameters")
    if invalid:
        raise ModelConfigurationError(
            "Only language-model LoRA parameters may be trainable; invalid: "
            + ", ".join(invalid[:10])
        )
    return TrainableParameterSummary(
        trainable=trainable,
        total=total,
        ratio=trainable / total,
        trainable_names=tuple(names),
    )


def load_qlora_model(
    base_config: BaseLoraConfig,
) -> tuple[Any, TrainableParameterSummary]:
    (
        _,
        LoraConfig,
        _,
        get_peft_model,
        prepare_model_for_kbit_training,
        _,
        _,
    ) = _imports()
    model = _load_quantized_base(base_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=base_config.training.gradient_checkpointing,
    )
    lora = base_config.training.lora
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=LANGUAGE_LORA_TARGET_PATTERN,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    for name, parameter in model.named_parameters():
        if ".visual." in name.casefold():
            parameter.requires_grad = False
    summary = summarize_and_validate_trainable_parameters(model)
    return model, summary


def _validate_adapter_directory(config: BaseLoraConfig, adapter_path: Path) -> None:
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Invalid adapter config {config_path}: {exc}") from exc
    base_name = Path(str(adapter_config.get("base_model_name_or_path", ""))).name
    if base_name.casefold() != config.base_model.name.casefold():
        raise ModelConfigurationError(
            f"Adapter targets {base_name or '<missing>'!r}, expected {config.base_model.name!r}"
        )
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ModelConfigurationError("Adapter peft_type must be LORA")


def reload_adapter(base_config: BaseLoraConfig, adapter_path: str | Path):
    _, _, PeftModel, _, _, _, _ = _imports()
    path = Path(adapter_path).expanduser().resolve()
    _validate_adapter_directory(base_config, path)
    base_model = _load_quantized_base(base_config)
    model = PeftModel.from_pretrained(
        base_model,
        str(path),
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ModelConfigurationError(
            "Reloaded inference adapter unexpectedly has trainable parameters: "
            + ", ".join(trainable[:10])
        )
    return model
