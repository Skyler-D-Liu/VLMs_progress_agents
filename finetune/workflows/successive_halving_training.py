"""Successive-halving learning-rate selection and final training.

Stages (Supplementary Material S.3-C):

* ``screening``: train one learning-rate candidate up to a rung budget
  (32, 64, or 128 optimizer steps) and score it on the fixed screening subset
  (the first 256 validation samples in a deterministic ordering).
* ``select``: rank the rung results (primary validation metric, then lower
  parse-failure/refusal rate, earlier epoch, lower learning rate), keep the top
  two after 32 steps and the top one after 64 steps, and, if the 128-step
  winner is 1e-5 or 5e-5, compare it with a 5e-6 or 1e-4 candidate trained for
  128 steps. Writes ``selection_plan.json``.
* ``final``: restart from the pretrained model and train a new adapter for one
  epoch (cosine schedule spanning that epoch) on at most 8,192 training
  samples, evaluate the end-of-epoch checkpoint once on the full validation
  partition, and write ``selection.json`` for adapter promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
from typing import Any, Sequence

from finetune.config import load_base_lora_config, load_dataset_config, resolve_experiment_config
from finetune.config.runtime_context import active_base_config, active_dataset_config, active_registry
from finetune.datasets import CanonicalSplit
from finetune.datasets.task_registry import get_task_dataset
from finetune.model_backends import get_model_backend
from finetune.training import (
    create_lora_trainer,
    find_resume_checkpoint,
)
from finetune.training.checkpoint_manager import (
    atomic_write_json,
    checkpoint_step,
    resolve_resume_checkpoint,
    validate_checkpoint_files,
)
from finetune.training.epoch_validation_callback import EpochValidationCallback
from finetune.training.signals import (
    SignalCheckpointCallback,
    install_signal_handlers,
    signal_requested,
)
from finetune.training.training_state import initialize_training_files
from finetune.workflows.artifact_paths import (
    resolution_manifest_root,
    trial_root,
    validation_root,
)
from finetune.workflows.pipeline_catalog import (
    dataset_config_for_task,
    get_pipeline_spec,
    implemented_dataset_names,
)
from finetune.workflows.selection import (
    SELECTION_COMPLETE,
    SELECTION_FILE,
    TrialDurationPolicy,
    TrialSpec,
    discover_validation_scores,
    plan_boundary_expansion,
    rank_scores,
    select_trial_checkpoint,
    trial_completion_reason,
    verify_training_completion_marker,
)
from finetune.workflows.successive_halving import (
    FINAL_TRAINING_COMPLETE,
    FixedValidationSubsetAdapter,
    RungResult,
    deterministic_training_subset,
    load_rung_result,
    require_policy,
    rung_marker_name,
    rung_plans,
    screening_validation_subset,
)
from finetune.workflows.validation import (
    BackendValidationPredictor,
    VALIDATION_COMPLETE,
    ValidationIdentity,
    ValidationInterrupted,
    run_validation,
    validation_index_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = active_base_config(PROJECT_ROOT / "finetune/experiment_specs/base_lora.yaml")
REGISTRY = active_registry(PROJECT_ROOT / "finetune/experiment_registry.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("screening", "select", "final"))
    parser.add_argument("--task", required=True, choices=implemented_dataset_names())
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rung", type=int)
    parser.add_argument("--direct-final-rung", action="store_true")
    parser.add_argument("--selection-plan", type=Path)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--dataset-config", type=Path, default=active_dataset_config())
    return parser.parse_args(argv)


def _immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid immutable artifact: {path}") from exc
        if current != value:
            raise RuntimeError(f"immutable artifact drift: {path}")
        return
    atomic_write_json(path, value)


def _example_index(examples) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": example.sample_id,
            "source_id": example.source_id,
            "canonical_split": example.canonical_split.value,
            "ground_truth": example.ground_truth,
        }
        for example in examples
    ]


def _resolution_contract(manifest_dir: Path, base, *, task: str) -> str:
    marker_path = manifest_dir / "RESOLUTION_SELECTED.json"
    selection_path = manifest_dir / "resolution_selection.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid resolution selection under {manifest_dir}") from exc
    if not isinstance(marker, dict) or not isinstance(selection, dict):
        raise RuntimeError("resolution selection artifacts must be JSON objects")
    selected = selection.get("selected_candidate")
    checks = {
        "selected_candidate": selected,
        "selection_basis": "gpu_memory_only",
        "test_data_used": False,
    }
    if marker.get("task") != task:
        raise RuntimeError("resolution completion marker task does not match the training task")
    for field, expected in checks.items():
        if marker.get(field) != expected:
            raise RuntimeError(f"resolution completion marker mismatch for {field}")
    if selection.get("task") != task:
        raise RuntimeError("resolution selection task does not match the training task")
    if selection.get("selection_basis") != "gpu_memory_only":
        raise RuntimeError("resolution was not selected from memory only")
    if selection.get("validation_performance_used") is not False:
        raise RuntimeError("resolution selection used validation performance")
    if selection.get("test_data_used") is not False:
        raise RuntimeError("resolution selection used test data")
    if selection.get("test_inference_run") is not False:
        raise RuntimeError("resolution selection ran test inference")
    materialized_splits = selection.get("materialized_splits")
    if (
        not isinstance(materialized_splits, list)
        or len(materialized_splits) != 2
        or set(materialized_splits) != {"train", "validation"}
    ):
        raise RuntimeError("resolution selection must materialize train and validation only")
    if not isinstance(selected, str) or not selected:
        raise RuntimeError("resolution selection is missing selected_candidate")
    allowed_candidates = {
        candidate.id for candidate in base.resolution.candidates_high_to_low
    }
    if selected not in allowed_candidates:
        raise RuntimeError(
            f"resolution candidate {selected!r} is not frozen in the base model config"
        )
    return selected


def _load_inputs(args):
    spec = get_pipeline_spec(args.task)
    base = load_base_lora_config(args.base_config, args.registry)
    get_model_backend(base).validate_config(base)
    policy = require_policy(base.search.successive_halving)
    dataset_path = args.dataset_config or dataset_config_for_task(spec)
    dataset = load_dataset_config(dataset_path)
    resolved = resolve_experiment_config(base, dataset, args.registry)
    if resolved.dataset.task != args.task:
        raise RuntimeError("task and dataset config differ")
    allowed_lrs = {
        *base.search.learning_rate.initial_values,
        *(
            (
                base.search.learning_rate.lower_expansion_value,
                base.search.learning_rate.upper_expansion_value,
            )
            if base.search.learning_rate.maximum_expansion_rounds
            else ()
        ),
    }
    if args.learning_rate not in allowed_lrs:
        raise ValueError("learning rate is outside the configured search grid")
    allowed_seeds = {base.training.seed, *base.training.additional_training_seeds}
    if args.seed not in allowed_seeds:
        raise ValueError("seed is outside the configured training seeds")
    adapter = get_task_dataset(resolved.dataset.task).adapter_class(resolved.dataset)
    train_examples = adapter.materialize_split(CanonicalSplit.TRAIN)
    validation_examples = adapter.materialize_split(CanonicalSplit.VALIDATION)
    if adapter.materialized_splits != {CanonicalSplit.TRAIN, CanonicalSplit.VALIDATION}:
        raise RuntimeError("training stages must materialize train and validation only")
    if any(example.canonical_split is not CanonicalSplit.TRAIN for example in train_examples):
        raise RuntimeError("training examples contain a non-train canonical split")
    if any(
        example.canonical_split is not CanonicalSplit.VALIDATION
        for example in validation_examples
    ):
        raise RuntimeError("validation examples contain a non-validation canonical split")
    train_sources = {example.source_id for example in train_examples}
    validation_sources = {example.source_id for example in validation_examples}
    if train_sources & validation_sources:
        raise RuntimeError("train and validation source identities overlap")
    for split_name in ("train", "validation"):
        source_split = getattr(resolved.dataset.splits, split_name).source_split.casefold()
        if source_split in {"test", "evaluation"}:
            raise RuntimeError(f"training input {split_name} source split is forbidden: {source_split}")
    resolution_manifest = resolution_manifest_root(
        base,
        task=args.task,
        dataset_manifest_dir=resolved.dataset.paths.manifest_dir,
    )
    resolution = _resolution_contract(resolution_manifest, base, task=args.task)
    return base, policy, resolved, adapter, train_examples, validation_examples, resolution


def _prepare_model_and_processor(base, resolution: str, training_dir: Path):
    backend = get_model_backend(base)
    processor = backend.load_processor(base, resolution)
    processor_dir = training_dir / "processor"
    if not processor_dir.exists():
        backend.save_processor(processor, processor_dir, base, resolution)
    if not processor_dir.is_dir() or not any(processor_dir.iterdir()):
        raise RuntimeError(f"processor artifact was not saved: {processor_dir}")
    validator = getattr(backend, "validate_processor_artifact", None)
    if callable(validator):
        validator(processor_dir, base, resolution)
    model, parameter_summary = backend.load_training_model(base)
    _immutable_json(
        training_dir / "trainable_parameters.json",
        {
            "trainable": parameter_summary.trainable,
            "total": parameter_summary.total,
            "ratio": parameter_summary.ratio,
            "names": list(parameter_summary.trainable_names),
        },
    )
    return model, processor


def _reload_validation_model(base, backend, checkpoint: Path):
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return backend.reload_adapter(base, checkpoint)


def _base_compatibility(args, base, resolved, resolution: str) -> dict[str, Any]:
    backend = get_model_backend(base)
    return {
        "task": args.task,
        "stage": args.stage,
        "base_model": base.base_model.model_dump(mode="json"),
        "split_seed": resolved.dataset.splits.seed,
        "resolution_candidate": resolution,
        "training": base.training.model_dump(mode="json"),
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "lora_target_pattern": backend.lora_target_pattern,
        "generation": resolved.dataset.generation.model_dump(mode="json"),
        "test_data_used": False,
    }


def _training_manifest(args, compatibility: dict[str, Any], policy) -> dict[str, Any]:
    return {
        "task": args.task,
        "stage": args.stage,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "compatibility": compatibility,
        "successive_halving": policy.model_dump(mode="json"),
        "selection_source": "canonical_validation_only",
        "test_data_used": False,
    }


def _run_screening(args, inputs) -> int:
    base, policy, resolved, adapter, train_examples, validation_examples, resolution = inputs
    plans = rung_plans(policy)
    if args.rung is None or args.rung < 1 or args.rung > len(plans):
        raise ValueError("screening requires a valid --rung")
    plan = plans[args.rung - 1]
    parameter_name = TrialSpec(args.learning_rate, args.seed).directory_name
    training_dir = trial_root(base, task=args.task) / "screening" / parameter_name
    marker_path = training_dir / rung_marker_name(plan.rung)
    if marker_path.is_file():
        load_rung_result(marker_path)
        print(f"Screening rung already complete: {marker_path}", flush=True)
        return 0
    if plan.rung > 1 and not args.direct_final_rung:
        previous = training_dir / rung_marker_name(plan.rung - 1)
        if not previous.is_file():
            raise RuntimeError(f"screening rung {plan.rung} requires {previous.name}")
        load_rung_result(previous)
    if args.direct_final_rung and plan.rung != len(plans):
        raise ValueError("--direct-final-rung is valid only for the last rung")

    screening_examples = screening_validation_subset(
        validation_examples, maximum=policy.screening_validation_max_examples
    )
    compatibility = _base_compatibility(args, base, resolved, resolution)
    compatibility.update(
        {
            "scheduler_horizon_steps": policy.rung_steps[-1],
            "validation_scope": plan.validation_scope,
            "screening_validation_examples": len(screening_examples),
        }
    )
    initialize_training_files(
        training_dir, _training_manifest(args, compatibility, policy)
    )
    _immutable_json(training_dir / "train_index.json", _example_index(train_examples))
    _immutable_json(
        training_dir / "validation_index.json", _example_index(validation_examples)
    )
    _immutable_json(
        training_dir / "screening_validation_index.json",
        validation_index_manifest(screening_examples),
    )
    resume_checkpoint = (
        resolve_resume_checkpoint(args.resume, training_dir, compatibility)
        if args.resume != "none"
        else None
    )
    if args.resume == "none" and find_resume_checkpoint(training_dir) is not None:
        raise RuntimeError(f"screening checkpoints exist; use --resume auto: {training_dir}")

    install_signal_handlers()
    resume_step = checkpoint_step(resume_checkpoint) if resume_checkpoint else 0
    if resume_step > plan.budget_steps:
        raise RuntimeError("resume checkpoint is beyond the requested screening rung")
    if resume_step == plan.budget_steps:
        backend = get_model_backend(base)
        processor = backend.load_processor(base, resolution)
        processor_dir = training_dir / "processor"
        validator = getattr(backend, "validate_processor_artifact", None)
        if not processor_dir.is_dir() or not any(processor_dir.iterdir()):
            raise RuntimeError(f"complete screening checkpoint lacks processor artifact: {processor_dir}")
        if callable(validator):
            validator(processor_dir, base, resolution)
        prediction_model = _reload_validation_model(base, backend, resume_checkpoint)
        checkpoint = resume_checkpoint
    else:
        model, processor = _prepare_model_and_processor(base, resolution, training_dir)
        trainer = create_lora_trainer(
            base_config=base,
            model=model,
            processor=processor,
            train_examples=train_examples,
            output_dir=training_dir,
            learning_rate=args.learning_rate,
            max_steps=policy.rung_steps[-1],
            compatibility=compatibility,
            budget_steps=plan.budget_steps,
            extra_callbacks=(SignalCheckpointCallback(),),
            save_steps=policy.rung_steps[-1],
            save_total_limit=16,
        )
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
        if signal_requested() or trainer.state.global_step < plan.budget_steps:
            print("Screening stopped at a safe checkpoint for runtime recovery", flush=True)
            return 75
        checkpoint = find_resume_checkpoint(training_dir, compatibility)
        if checkpoint is None:
            raise RuntimeError("training completed without a complete checkpoint")
        backend = get_model_backend(base)
        del trainer
        del model
        prediction_model = _reload_validation_model(base, backend, checkpoint)
    if checkpoint is None or checkpoint_step(checkpoint) != plan.budget_steps:
        raise RuntimeError(f"missing complete checkpoint for rung {plan.rung}")

    validation_adapter = FixedValidationSubsetAdapter(adapter, screening_examples)
    output = (
        validation_root(base, task=args.task)
        / "screening"
        / parameter_name
        / f"rung-{plan.rung}"
    )
    identity = ValidationIdentity(
        task=args.task,
        checkpoint_dir=str(checkpoint),
        epoch=plan.rung,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resolution_candidate=resolution,
        generation=resolved.dataset.generation.model_dump(mode="json"),
        validation_scope=plan.validation_scope,
    )
    try:
        artifact = run_validation(
            adapter=validation_adapter,
            checkpoint_dir=checkpoint,
            output_dir=output,
            identity=identity,
            predictor=BackendValidationPredictor(
                get_model_backend(base), prediction_model, processor, identity.generation
            ),
            should_interrupt=signal_requested,
        )
    except ValidationInterrupted as exc:
        print(str(exc), flush=True)
        return 75
    from finetune.workflows.selection import load_checkpoint_score

    score = load_checkpoint_score(artifact.output_dir / VALIDATION_COMPLETE)
    atomic_write_json(
        marker_path,
        {
            "task": args.task,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "rung": plan.rung,
            "budget_steps": plan.budget_steps,
            "scheduler_horizon_steps": policy.rung_steps[-1],
            "validation_scope": plan.validation_scope,
            "validation_marker": str(artifact.output_dir / VALIDATION_COMPLETE),
            "score": asdict(score),
            "selection_source": "canonical_validation_only",
            "test_data_used": False,
        },
    )
    load_rung_result(marker_path)
    print(json.dumps({"completed_rung": plan.rung, "score": asdict(score)}, indent=2), flush=True)
    return 0


SELECTION_PLAN = "selection_plan.json"
INCOMPLETE_EXIT_CODE = 3


def selection_plan_path(base, *, task: str, seed: int) -> Path:
    return trial_root(base, task=task) / "screening" / f"seed-{seed}" / SELECTION_PLAN


def _rung_result(base, *, task: str, learning_rate: float, seed: int, rung: int) -> RungResult | None:
    marker = (
        trial_root(base, task=task)
        / "screening"
        / TrialSpec(learning_rate, seed).directory_name
        / rung_marker_name(rung)
    )
    if not marker.is_file():
        return None
    result = load_rung_result(marker)
    if result.task != task or result.learning_rate != learning_rate or result.seed != seed:
        raise RuntimeError(f"rung marker identity mismatch: {marker}")
    return result


def _pending(task: str, seed: int, learning_rates: Sequence[float], rung: int, *, direct: bool = False) -> int:
    print(
        f"Learning-rate selection is waiting for rung {rung} of: "
        + ", ".join(f"{lr:g}" for lr in learning_rates),
        flush=True,
    )
    for lr in learning_rates:
        command = (
            "python -m finetune.workflows.successive_halving_training --stage screening "
            f"--task {task} --learning-rate {lr:g} --seed {seed} --rung {rung}"
        )
        if direct:
            command += " --direct-final-rung"
        elif rung > 1:
            command += " --resume auto"
        print("  " + command, flush=True)
    return INCOMPLETE_EXIT_CODE


def _candidate_record(result: RungResult) -> dict[str, Any]:
    return {
        **asdict(result.score),
        "rung": result.rung,
        "budget_steps": result.budget_steps,
        "marker_path": result.marker_path,
        "selection_source": "canonical_validation_only",
    }


def _run_select(args) -> int:
    """Successive-halving learning-rate selection on the screening subset."""
    base = load_base_lora_config(args.base_config, args.registry)
    policy = require_policy(base.search.successive_halving)
    if args.seed not in {base.training.seed, *base.training.additional_training_seeds}:
        raise ValueError("seed is outside the configured training seeds")
    plans = rung_plans(policy)
    active = list(base.search.learning_rate.initial_values)
    rungs: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    final_results: dict[float, RungResult] = {}
    for plan in plans:
        results = {
            lr: _rung_result(base, task=args.task, learning_rate=lr, seed=args.seed, rung=plan.rung)
            for lr in active
        }
        missing = [lr for lr, result in results.items() if result is None]
        if missing:
            return _pending(args.task, args.seed, missing, plan.rung)
        ranked = rank_scores([result.score for result in results.values()])
        candidates.extend(_candidate_record(results[score.learning_rate]) for score in ranked)
        survivors = [score.learning_rate for score in ranked[: plan.survivors]]
        rungs.append(
            {
                "rung": plan.rung,
                "budget_steps": plan.budget_steps,
                "ranked_learning_rates": [score.learning_rate for score in ranked],
                "survivors": survivors,
            }
        )
        final_results = {lr: results[lr] for lr in survivors}
        active = survivors
    if len(active) != 1:
        raise RuntimeError("successive halving must end with exactly one learning rate")
    incumbent = final_results[active[0]]
    selected = incumbent.score
    expansion_record: dict[str, Any] | None = None
    expansion = plan_boundary_expansion(selected.learning_rate, base.search, args.seed)
    if expansion is not None:
        last_rung = plans[-1].rung
        challenger = _rung_result(
            base,
            task=args.task,
            learning_rate=expansion.learning_rate,
            seed=args.seed,
            rung=last_rung,
        )
        if challenger is None:
            return _pending(
                args.task, args.seed, [expansion.learning_rate], last_rung, direct=True
            )
        if challenger.budget_steps != incumbent.budget_steps:
            raise RuntimeError("expansion candidate and incumbent used different budgets")
        ranked = rank_scores([incumbent.score, challenger.score])
        candidates.append(_candidate_record(challenger))
        selected = ranked[0]
        expansion_record = {
            "incumbent_learning_rate": incumbent.score.learning_rate,
            "expansion_learning_rate": expansion.learning_rate,
            "budget_steps": challenger.budget_steps,
            "winner": selected.learning_rate,
        }
    plan_value = {
        "task": args.task,
        "seed": args.seed,
        "selected_learning_rate": selected.learning_rate,
        "selected_score": asdict(selected),
        "rungs": rungs,
        "boundary_expansion": expansion_record,
        "candidates": candidates,
        "tie_break_order": list(base.search.global_tie_break_order),
        "screening_validation_max_examples": policy.screening_validation_max_examples,
        "selection_source": "canonical_validation_only",
        "test_data_used": False,
    }
    path = selection_plan_path(base, task=args.task, seed=args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    _immutable_json(path, plan_value)
    print(json.dumps({"selection_plan": str(path), "selected_learning_rate": selected.learning_rate}, indent=2), flush=True)
    return 0


def _load_selection_plan(path: Path, *, task: str, learning_rate: float, seed: int) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid learning-rate selection plan: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("learning-rate selection plan must be an object")
    expected = {"task": task, "selected_learning_rate": learning_rate, "seed": seed}
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise RuntimeError(f"learning-rate selection plan mismatch for {field}")
    if value.get("selection_source") != "canonical_validation_only":
        raise RuntimeError("final training requires a canonical-validation-only selection plan")
    if value.get("test_data_used") is not False:
        raise RuntimeError("final training selection must not use test data")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("learning-rate selection plan has no validation candidates")
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("selection_source") not in {
            None,
            "canonical_validation_only",
        }:
            raise RuntimeError("learning-rate selection candidate is not validation-only")
    return value


def _run_final(args, inputs) -> int:
    base, policy, resolved, adapter, train_examples, validation_examples, resolution = inputs
    selection_plan_file = args.selection_plan or selection_plan_path(
        base, task=args.task, seed=args.seed
    )
    selection_plan = _load_selection_plan(
        selection_plan_file,
        task=args.task,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    parameter_name = TrialSpec(args.learning_rate, args.seed).directory_name
    training_dir = trial_root(base, task=args.task) / "final" / parameter_name
    final_validation_root = validation_root(base, task=args.task) / "final"
    if (training_dir / FINAL_TRAINING_COMPLETE).is_file():
        _write_final_selection(
            args, training_dir, final_validation_root / parameter_name, selection_plan_file
        )
        print(f"Final training already complete: {training_dir}", flush=True)
        return 0
    selected_train = deterministic_training_subset(
        train_examples,
        maximum=policy.final_training_max_examples,
        seed=args.seed,
    )
    duration = TrialDurationPolicy(
        initial_max_epochs=policy.final_training_max_epochs,
        extended_max_epochs=policy.final_training_max_epochs,
        scheduler_horizon_epochs=policy.final_training_scheduler_horizon_epochs,
        patience=policy.final_training_patience_validation_events,
        min_delta=base.search.early_stopping.min_delta,
    )
    compatibility = _base_compatibility(args, base, resolved, resolution)
    compatibility.update(
        {
            "duration_policy": asdict(duration),
            "scheduler_horizon_epochs": policy.final_training_scheduler_horizon_epochs,
            "final_training_max_examples": policy.final_training_max_examples,
            "final_training_max_optimizer_steps": policy.final_training_max_optimizer_steps,
            "selected_from_screening": {
                "learning_rate": selection_plan["selected_learning_rate"],
                "seed": selection_plan["seed"],
            },
        }
    )
    manifest = _training_manifest(args, compatibility, policy)
    manifest["duration_policy"] = asdict(duration)
    initialize_training_files(training_dir, manifest)
    _immutable_json(training_dir / "train_index.json", _example_index(selected_train))
    _immutable_json(
        training_dir / "validation_index.json", _example_index(validation_examples)
    )
    resume_checkpoint = (
        resolve_resume_checkpoint(args.resume, training_dir, compatibility)
        if args.resume != "none"
        else None
    )
    if args.resume == "none" and find_resume_checkpoint(training_dir) is not None:
        raise RuntimeError(f"final checkpoints exist; use --resume auto: {training_dir}")

    install_signal_handlers()
    model, processor = _prepare_model_and_processor(base, resolution, training_dir)
    callback = EpochValidationCallback(
        adapter=adapter,
        validation_root=final_validation_root,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resolution_candidate=resolution,
        patience=duration.patience,
        min_delta=duration.min_delta,
        initial_max_epochs=duration.initial_max_epochs,
        extended_max_epochs=duration.extended_max_epochs,
        hard_max_epochs=duration.extended_max_epochs,
        generation=resolved.dataset.generation.model_dump(mode="json"),
        validation_predictor_factory=lambda _epoch, current_model, current_processor: BackendValidationPredictor(
            get_model_backend(base), current_model, current_processor,
            resolved.dataset.generation.model_dump(mode="json"),
        ),
    )
    trainer = create_lora_trainer(
        base_config=base,
        model=model,
        processor=processor,
        train_examples=selected_train,
        output_dir=training_dir,
        learning_rate=args.learning_rate,
        max_steps=policy.final_training_max_optimizer_steps,
        num_train_epochs=float(policy.final_training_max_epochs),
        epoch_checkpoints=True,
        compatibility=compatibility,
        extra_callbacks=(SignalCheckpointCallback(), callback),
        save_total_limit=max(2, policy.final_training_max_epochs + 1),
    )
    try:
        trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
        )
    except ValidationInterrupted as exc:
        print(str(exc), flush=True)
        return 75
    if signal_requested():
        print("Final training stopped at a safe checkpoint for runtime recovery", flush=True)
        return 75

    scores = tuple(
        score
        for score in discover_validation_scores(
            final_validation_root / parameter_name
        )
        if score.learning_rate == args.learning_rate and score.seed == args.seed
    )
    if not scores or any(score.validation_scope != "canonical_full" for score in scores):
        raise RuntimeError("final training lacks complete canonical validation")
    if len(scores) != 1:
        raise RuntimeError(
            "final training must produce exactly one end-of-epoch checkpoint evaluation"
        )
    best = select_trial_checkpoint(scores)
    best_checkpoint = Path(best.checkpoint_dir).expanduser().resolve()
    validate_checkpoint_files(best_checkpoint)


    del trainer
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    backend = get_model_backend(base)
    reloaded = backend.reload_adapter(base, best_checkpoint)
    del reloaded
    gc.collect()
    validator = getattr(backend, "validate_processor_artifact", None)
    if callable(validator):
        validator(training_dir / "processor", base, resolution)
    atomic_write_json(
        training_dir / FINAL_TRAINING_COMPLETE,
        {
            "task": args.task,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "selection_source": "canonical_validation_only",
            "validation_epochs": sorted(score.epoch for score in scores),
            "best_checkpoint": asdict(best),
            "duration_policy": asdict(duration),
            "completion_reason": trial_completion_reason(scores, duration),
            "test_data_used": False,
        },
    )
    selection_path = _write_final_selection(
        args, training_dir, final_validation_root / parameter_name, selection_plan_file
    )
    print(
        json.dumps(
            {"final_training": str(training_dir), "best": asdict(best), "selection": str(selection_path)},
            indent=2,
        ),
        flush=True,
    )
    return 0


def _write_final_selection(
    args, training_dir: Path, validation_dir: Path, selection_plan_file: Path
) -> Path:
    """Record the single end-of-epoch checkpoint as the adapter to promote.

    Validation-based selection is restricted to the learning rate; the final
    checkpoint is used for the test evaluation without checkpoint selection.
    """
    scores = tuple(
        score
        for score in discover_validation_scores(validation_dir)
        if score.learning_rate == args.learning_rate and score.seed == args.seed
    )
    if len(scores) != 1 or scores[0].validation_scope != "canonical_full":
        raise RuntimeError("final training must have one full-validation checkpoint evaluation")
    selected = scores[0]
    completion = verify_training_completion_marker(
        learning_rate=args.learning_rate,
        seed=args.seed,
        scores=scores,
        marker_path=training_dir / FINAL_TRAINING_COMPLETE,
    )
    selection_path = training_dir / SELECTION_FILE
    _immutable_json(
        selection_path,
        {
            "task": args.task,
            "selected": asdict(selected),
            "candidates": [asdict(selected)],
            "required_trials": [{"learning_rate": args.learning_rate, "seed": args.seed}],
            "input_training_complete": [completion],
            "learning_rate_selection_plan": str(Path(selection_plan_file).expanduser().resolve()),
            "checkpoint_selection": "none_single_end_of_epoch_checkpoint",
            "selection_source": "canonical_validation_only",
            "test_data_used": False,
        },
    )
    _immutable_json(
        training_dir / SELECTION_COMPLETE,
        {
            "task": selected.task,
            "learning_rate": selected.learning_rate,
            "seed": selected.seed,
            "epoch": selected.epoch,
            "checkpoint_dir": selected.checkpoint_dir,
            "selection_source": "canonical_validation_only",
        },
    )
    return selection_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "select":
        return _run_select(args)
    if args.learning_rate is None:
        raise ValueError(f"--stage {args.stage} requires --learning-rate")
    inputs = _load_inputs(args)
    if args.stage == "screening":
        return _run_screening(args, inputs)
    return _run_final(args, inputs)


if __name__ == "__main__":
    raise SystemExit(main())
