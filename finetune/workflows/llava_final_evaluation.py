"""Llava final evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from finetune.config import load_base_lora_config, load_dataset_config, resolve_experiment_config
from finetune.config.runtime_context import active_base_config, active_registry
from finetune.datasets import CanonicalSplit
from finetune.datasets.task_registry import get_task_dataset
from finetune.model_backends import get_model_backend
from finetune.training.checkpoint_manager import atomic_write_json
from finetune.workflows.artifact_paths import model_artifact_paths
from finetune.workflows.pipeline_catalog import (
    base_config_for_task,
    dataset_config_for_task,
    get_pipeline_spec,
    implemented_dataset_names,
)
from finetune.workflows.canonical_cohorts import select_examples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_COMPLETE = "TEST_COMPLETED.json"
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"


class LlavaFinalEvaluationError(RuntimeError):
    pass


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LlavaFinalEvaluationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LlavaFinalEvaluationError(f"{label} must be a JSON object: {path}")
    return value


def _decode(processor: Any, generated: Any, input_ids: Any) -> str:
    trimmed = generated[:, input_ids.shape[1] :]
    decoder = getattr(processor, "batch_decode", None)
    if not callable(decoder):
        decoder = processor.tokenizer.batch_decode
    return str(
        decoder(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )


def _generate(backend: Any, model: Any, processor: Any, example: Any, generation: Mapping[str, Any]) -> str:
    import torch

    inputs = backend.prepare_generation_inputs(processor, example, add_generation_prompt=True)
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    input_ids = inputs["input_ids"]
    with torch.inference_mode():
        generated = model.generate(**inputs, **dict(generation))
    return _decode(processor, generated, input_ids)


def _read_existing_predictions(
    path: Path,
    *,
    task: str,
    expected_examples: Mapping[str, Any],
    parse_prediction: Any,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):


        last_newline = raw.rfind(b"\n")
        raw = raw[: last_newline + 1] if last_newline >= 0 else b""
        path.write_bytes(raw)
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LlavaFinalEvaluationError(
                f"invalid LLaVA test prediction at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise LlavaFinalEvaluationError("LLaVA test prediction must be an object")
        sample_id = value.get("sample_id")
        if sample_id not in expected_examples:
            raise LlavaFinalEvaluationError(
                f"LLaVA test prediction is not from canonical test: {sample_id!r}"
            )
        if sample_id in rows:
            raise LlavaFinalEvaluationError(f"duplicate LLaVA test prediction: {sample_id}")
        required = {
            "task": task,
            "model_kind": "lora",
            "adapter_name": task,
            "prompt_mode": "zero-shot",
            "few_shot_k": 0,
            "canonical_split": "test",
        }
        for field, expected in required.items():
            if value.get(field) != expected:
                raise LlavaFinalEvaluationError(
                    f"LLaVA test prediction {sample_id} has {field}={value.get(field)!r}"
                )
        if not isinstance(value.get("response_text"), str):
            raise LlavaFinalEvaluationError("LLaVA test response_text must be text")
        if type(value.get("parse_failed")) is not bool:
            raise LlavaFinalEvaluationError("LLaVA test parse_failed must be boolean")
        example = expected_examples[sample_id]
        if value.get("source_id") != example.source_id:
            raise LlavaFinalEvaluationError(
                f"LLaVA test prediction {sample_id} source_id does not match canonical test"
            )
        try:
            parsed = parse_prediction(value["response_text"], example)
            expected_failed = False
        except (TypeError, ValueError):
            parsed = None
            expected_failed = True
        if value.get("parse_failed") is not expected_failed or value.get("parsed_prediction") != parsed:
            raise LlavaFinalEvaluationError(
                f"LLaVA test prediction {sample_id} parser fields are inconsistent"
            )
        rows[sample_id] = value
    return rows


def _quarantine_stale_result(result_dir: Path) -> None:
    for name in (PREDICTIONS, METRICS, TEST_COMPLETE):
        source = result_dir / name
        if not source.exists():
            continue
        destination = result_dir / f"{name}.noncanonical"
        if destination.exists():
            raise LlavaFinalEvaluationError(
                f"stale LLaVA result already quarantined; manual review required: {destination}"
            )
        source.replace(destination)


def run(task: str, *, base_config: Path | None = None, registry: Path | None = None) -> int:
    if task not in implemented_dataset_names():
        raise LlavaFinalEvaluationError(f"unsupported task: {task!r}")
    base_path = base_config or active_base_config(
        PROJECT_ROOT / "finetune/experiment_specs/models/llava_ov2_8b.yaml"
    )
    registry_path = registry or active_registry(PROJECT_ROOT / "finetune/experiment_registry.yaml")
    base = load_base_lora_config(base_path, registry_path)
    backend = get_model_backend(base)
    backend.validate_config(base)
    if base.base_model.model_type.casefold() != "llava_onevision2":
        raise LlavaFinalEvaluationError("LLaVA final evaluator requires llava_onevision2")
    spec = get_pipeline_spec(task)
    dataset = load_dataset_config(dataset_config_for_task(spec))
    resolved = resolve_experiment_config(base, dataset, registry_path)
    model_dataset = model_artifact_paths(base, resolved.dataset)
    adapter_dir = model_dataset.paths.final_adapter_dir.expanduser().resolve()
    backend.validate_adapter_directory(base, adapter_dir)
    inference = _json_object(adapter_dir / "inference_config.json", "inference configuration")
    expected_inference = {
        "task": task,
        "base_model": base.base_model.name,
        "few_shot_k": 0,
    }
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise LlavaFinalEvaluationError(f"inference configuration drift for {field}")
    generation = inference.get("generation")
    if not isinstance(generation, dict) or generation.get("do_sample") is not False:
        raise LlavaFinalEvaluationError("LLaVA final generation must be greedy")
    resolution = inference.get("resolution_candidate")
    if not isinstance(resolution, str) or not resolution:
        raise LlavaFinalEvaluationError("promoted adapter has no resolution candidate")

    adapter = get_task_dataset(task).adapter_class(resolved.dataset)
    all_test_examples = adapter.materialize_split(CanonicalSplit.TEST)


    try:
        examples = select_examples(task, all_test_examples, resolved.dataset.paths.raw_dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise LlavaFinalEvaluationError(str(exc)) from exc
    expected_count = len(examples)
    result_dir = model_dataset.paths.test_result_dir.expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = result_dir / PREDICTIONS
    expected_examples = {example.sample_id: example for example in examples}
    marker_path = result_dir / TEST_COMPLETE
    try:
        existing = _read_existing_predictions(
            predictions_path,
            task=task,
            expected_examples=expected_examples,
            parse_prediction=adapter.parse_example_prediction,
        )
    except LlavaFinalEvaluationError:
        if marker_path.is_file():
            _quarantine_stale_result(result_dir)
            existing = {}
        else:
            raise
    if marker_path.is_file():
        marker = _json_object(marker_path, "test completion marker")
        if marker.get("task") != task:
            raise LlavaFinalEvaluationError("existing LLaVA test marker does not match the task")
        if len(existing) != expected_count or set(existing) != set(expected_examples):
            raise LlavaFinalEvaluationError("test marker exists but predictions are incomplete")
        return 0

    processor = backend.load_processor(base, resolution)
    model = backend.reload_adapter(base, adapter_dir)
    mode = "a" if predictions_path.exists() else "x"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for example in examples:
            if example.sample_id in existing:
                continue
            from dataclasses import replace
            from evaluation.utils.response_protocol import validated_call
            user_index = max(i for i, message in enumerate(example.messages) if message.role == 'user')
            original_prompt = example.messages[user_index].content
            context = dict(example.metadata)
            image = (context.get('images') or [{}])[0]
            context.update(image_width=context.get('width', image.get('width')),
                           image_height=context.get('height', image.get('height')))
            def call(prompt):
                messages = list(example.messages)
                messages[user_index] = replace(messages[user_index], content=prompt)
                value = _generate(backend, model, processor, replace(example, messages=tuple(messages)), generation)
                return {'answer_text': value}
            attempt_result, _, answer = validated_call(call, 'baseline', task=task,
                                                       context=context, prompt=original_prompt)
            response = answer['answer_text']
            try:
                parsed = adapter.parse_example_prediction(response, example)
                parse_failed = False
            except (TypeError, ValueError):
                parsed = None
                parse_failed = True
            handle.write(
                json.dumps(
                    {
                        "task": task,
                        "model": base.base_model.name,
                        "model_kind": "lora",
                        "adapter_name": task,
                        "prompt_mode": "zero-shot",
                        "few_shot_k": 0,
                        "canonical_split": "test",
                        "sample_id": example.sample_id,
                        "source_id": example.source_id,
                        "response_text": response,
                        "attempt_count": attempt_result["attempt_count"],
                        "parsed_prediction": parsed,
                        "parse_failed": parse_failed,
                        "ground_truth": example.ground_truth,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        handle.flush()
        os.fsync(handle.fileno())

    by_id = _read_existing_predictions(
        predictions_path,
        task=task,
        expected_examples=expected_examples,
        parse_prediction=adapter.parse_example_prediction,
    )
    predictions = list(by_id.values())
    if len(predictions) != expected_count or set(by_id) != {example.sample_id for example in examples}:
        raise LlavaFinalEvaluationError("LLaVA test predictions do not cover the canonical test split")
    from finetune.datasets import ValidationPrediction

    metric = adapter.compute_validation_metric(
        tuple(
            ValidationPrediction(
                sample_id=example.sample_id,
                ground_truth=example.ground_truth,
                parsed_prediction=by_id[example.sample_id]["parsed_prediction"],
                parse_failed=bool(by_id[example.sample_id]["parse_failed"]),
            )
            for example in examples
        )
    )
    atomic_write_json(
        result_dir / METRICS,
        {
            "task": task,
            "metric_name": resolved.dataset.validation.metric.value,
            "direction": resolved.dataset.validation.direction.value,
            "primary_metric": float(metric.primary_metric),
            "parse_failure_rate": float(metric.parse_failure_rate),
            "details": dict(metric.details),
            "test_split": "canonical_test",
        },
    )
    atomic_write_json(
        result_dir / TEST_COMPLETE,
        {
            "task": task,
            "test_split": "canonical_test",
            "complete_canonical_test_split": True,
            "resolution_candidate": resolution,
            "generation": generation,
            "model_type": base.base_model.model_type,
            "few_shot_k": 0,
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    return run(args.task, base_config=args.base_config, registry=args.registry)


if __name__ == "__main__":
    raise SystemExit(main())
