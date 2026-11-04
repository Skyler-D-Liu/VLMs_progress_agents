#!/usr/bin/env python3
"""Resolution selection."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from finetune.config import (
    load_base_lora_config,
    load_dataset_config,
    resolve_experiment_config,
)
from finetune.config.runtime_context import active_base_config, active_dataset_config, active_registry
from finetune.datasets import CanonicalSplit
from finetune.datasets.task_registry import get_task_dataset
from finetune.model_backends import get_model_backend
from finetune.workflows.pipeline_catalog import (
    base_config_for_task,
    dataset_config_for_task,
    get_pipeline_spec,
)
from finetune.workflows.validation import BackendValidationPredictor
from finetune.training.checkpoint_manager import atomic_write_json
from finetune.workflows.artifact_paths import (
    resolution_manifest_root,
    resolution_statistics_path,
)
from finetune.workflows.resolution_statistics import load_resolution_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = active_registry(PROJECT_ROOT / "finetune/experiment_registry.yaml")
DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "finetune/experiment_specs/datasets/aerial_animal.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="aerial-animal")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset-config", type=Path, default=active_dataset_config())
    return parser.parse_args()


def require_gpus(gpus=1, gpu_type="h200") -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != gpus:
        raise RuntimeError(f"resolution selection requires exactly {gpus} CUDA GPUs")
    names = [torch.cuda.get_device_name(i) for i in range(gpus)]
    if any(not re.search(r"\b" + re.escape(gpu_type) + r"\b", name, re.IGNORECASE)
           for name in names):
        raise RuntimeError(
            f"resolution selection requires configured {gpu_type.upper()} GPUs; found {names}"
        )


def _find_example(examples, sample_id: str):
    matches = [example for example in examples if example.sample_id == sample_id]
    if len(matches) != 1:
        raise RuntimeError(f"stress sample is not unique: {sample_id}")
    return matches[0]


def _optimizer(model, base):
    import torch

    settings = base.training.optimizer
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=base.search.learning_rate.initial_values[0],
        betas=(settings.beta1, settings.beta2),
        eps=settings.epsilon,
        weight_decay=settings.weight_decay,
        fused=True,
    )


def _capture_trainable_state(model) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model, initial_state: dict[str, Any]) -> None:
    import torch

    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(initial_state):
        raise RuntimeError("trainable parameter set changed during resolution selection")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(initial_state[name].to(device=parameter.device, dtype=parameter.dtype))
            parameter.grad = None


def _run_candidate(base, backend, model, candidate_id: str, examples, generation) -> dict[str, Any]:
    import torch

    processor = backend.load_processor(base, candidate_id)
    collator = backend.create_collator(processor, base.training.max_length)
    optimizer = _optimizer(model, base)
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    sequences = []
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for example in examples:
            batch = collator([example])
            sequences.append(int(batch["input_ids"].shape[1]))
            batch = batch.to(next(model.parameters()).device)
            output = model(**batch)
            (output.loss / base.training.batch.gradient_accumulation_steps).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

        predictor = BackendValidationPredictor(backend, model, processor, generation)
        p95_response = predictor(examples[0])
        max_response_one = predictor(examples[1])
        max_response_two = predictor(examples[1])
        if max_response_one != max_response_two:
            raise RuntimeError("greedy validation generation was not deterministic")
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        headroom_fraction = min(1 - torch.cuda.max_memory_reserved(i) /
                                torch.cuda.get_device_properties(i).total_memory
                                for i in range(torch.cuda.device_count()))
        required = base.resolution.required_memory_headroom_fraction
        return {
            "candidate": candidate_id,
            "status": "stable" if headroom_fraction >= required else "insufficient_headroom",
            "sequence_lengths": sequences,
            "optimizer_step_completed": True,
            "deterministic_validation_generation": True,
            "p95_generated_characters": len(p95_response),
            "max_generated_characters": len(max_response_one),
        }
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if isinstance(exc, RuntimeError) and "out of memory" not in message.casefold():
            known_data_failure = (
                "max_length" in message
                or "sequence length" in message
                or "deterministic" in message
            )
            if not known_data_failure:
                raise
        return {
            "candidate": candidate_id,
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure_message": message,
            "sequence_lengths": sequences,
            "optimizer_step_completed": False,
            "deterministic_validation_generation": False,
        }
    finally:
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    spec = get_pipeline_spec(args.task)
    base_path = args.base_config or active_base_config(base_config_for_task(spec))
    base = load_base_lora_config(base_path, args.registry)
    require_gpus(base.hardware.gpus_per_trial, base.hardware.gpu_type)
    backend = get_model_backend(base)
    backend.validate_config(base)
    registration = get_task_dataset(args.task)
    dataset_path = args.dataset_config or dataset_config_for_task(spec)
    dataset = load_dataset_config(dataset_path)
    resolved = resolve_experiment_config(base, dataset, args.registry)
    if resolved.dataset.task != args.task:
        raise RuntimeError("resolution selector task/config mismatch")

    dataset_manifest_dir = resolved.dataset.paths.manifest_dir.resolve()
    manifest_dir = resolution_manifest_root(
        base,
        task=args.task,
        dataset_manifest_dir=dataset_manifest_dir,
    )
    if (manifest_dir / "RESOLUTION_SELECTED.json").exists():
        raise RuntimeError(
            f"{args.task} resolution is already frozen; refusing to rerun selection"
        )
    processor_path = (base.base_model.path / "preprocessor_config.json").resolve()
    statistics_path = resolution_statistics_path(
        base,
        task=args.task,
        dataset_manifest_dir=dataset_manifest_dir,
    )
    statistics = load_resolution_statistics(
        statistics_path,
        task=args.task,
        processor_path=processor_path,
    )

    from finetune.datasets.task_registry import get_task_dataset
    adapter = get_task_dataset(resolved.dataset.task).adapter_class(resolved.dataset)
    train = adapter.materialize_split(CanonicalSplit.TRAIN)
    validation = adapter.materialize_split(CanonicalSplit.VALIDATION)
    if adapter.materialized_splits != {CanonicalSplit.TRAIN, CanonicalSplit.VALIDATION}:
        raise RuntimeError("resolution selection must read train and validation only")
    if any(example.canonical_split is not CanonicalSplit.TRAIN for example in train):
        raise RuntimeError("resolution selection train examples contain a non-train split")
    if any(
        example.canonical_split is not CanonicalSplit.VALIDATION
        for example in validation
    ):
        raise RuntimeError("resolution selection validation examples contain a non-validation split")
    if {example.source_id for example in train} & {
        example.source_id for example in validation
    }:
        raise RuntimeError("resolution selection train and validation identities overlap")
    stress = statistics["stress_examples"]
    combined = (*train, *validation)
    p95 = _find_example(combined, stress["p95"]["sample_id"])
    maximum = _find_example(combined, stress["max"]["sample_id"])

    model, parameter_summary = backend.load_training_model(base)
    initial_trainable_state = _capture_trainable_state(model)
    generation = resolved.dataset.generation.model_dump(mode="json")
    results = []
    selected = None
    for candidate in base.resolution.candidates_high_to_low:
        _restore_trainable_state(model, initial_trainable_state)
        result = _run_candidate(
            base, backend, model, candidate.id, (p95, maximum), generation
        )
        results.append(result)
        if result["status"] == "stable":
            selected = candidate
            break
    if selected is None:
        raise RuntimeError("no frozen resolution candidate was stable on the allocated GPUs")

    selection = {
        "task": args.task,
        "selection_basis": "gpu_memory_only",
        "selected_candidate": selected.id,
        "selected_max_pixels_per_image": selected.max_pixels_per_image,
        "selected_max_visual_tokens_per_image": selected.max_visual_tokens_per_image,
        "required_memory_headroom_fraction": base.resolution.required_memory_headroom_fraction,
        "candidate_order": [candidate.id for candidate in base.resolution.candidates_high_to_low],
        "candidate_results": results,
        "stress_examples": {
            "p95": p95.sample_id,
            "max": maximum.sample_id,
        },
        "trainable_parameter_count": parameter_summary.trainable,
        "materialized_splits": ["train", "validation"],
        "validation_performance_used": False,
        "test_data_used": False,
        "test_inference_run": False,
    }
    selection_path = manifest_dir / "resolution_selection.json"
    atomic_write_json(selection_path, selection)
    atomic_write_json(
        manifest_dir / "RESOLUTION_SELECTED.json",
        {
            "task": args.task,
            "selected_candidate": selected.id,
            "selection_basis": "gpu_memory_only",
            "test_data_used": False,
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
