"""Task final evaluation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
from finetune.config import load_base_lora_config, load_dataset_config, resolve_experiment_config
from finetune.config.runtime_context import active_base_config, active_dataset_config, active_registry
from finetune.datasets.task_registry import get_task_dataset
from finetune.training.checkpoint_manager import atomic_write_json
from finetune.model_backends import get_model_backend
from finetune.workflows.evaluation_dispatch import evaluator_script_path
from finetune.workflows.artifact_paths import model_artifact_paths
from finetune.workflows.pipeline_catalog import base_config_for_task, dataset_config_for_task, get_pipeline_spec, implemented_dataset_names
from finetune.workflows.registry_gates import load_registry_gate
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / 'finetune/experiment_registry.yaml'
MODEL_PREFIX = 'Qwen3-VL-32B-Instruct-LoRA-'
TEST_COMPLETE = 'TEST_COMPLETED.json'
METRICS = 'metrics.json'
TASKS = implemented_dataset_names()

class TaskFinalTestError(RuntimeError):
    pass

def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return _json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value

def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskFinalTestError(f'invalid JSON artifact: {path}') from exc
    if not isinstance(value, dict):
        raise TaskFinalTestError(f'JSON artifact must be an object: {path}')
    return value

def _verify_adapter(adapter_dir: Path, task: str, expected_generation: Mapping[str, Any], base):
    required = ('adapter_config.json', 'adapter_model.safetensors', 'inference_config.json', 'selection.json', 'PROMOTION_MANIFEST.json', 'PROMOTION_COMPLETE.json')
    missing = [name for name in required if not (adapter_dir / name).is_file() or (adapter_dir / name).stat().st_size == 0]
    if missing:
        raise TaskFinalTestError(f'promoted adapter is incomplete: {missing}')
    promotion = _load_json(adapter_dir / 'PROMOTION_MANIFEST.json')
    marker = _load_json(adapter_dir / 'PROMOTION_COMPLETE.json')
    inference = _load_json(adapter_dir / 'inference_config.json')
    selection = _load_json(adapter_dir / 'selection.json')
    if promotion.get('task') != task or selection.get('task') != task:
        raise TaskFinalTestError('promoted task mismatch')
    if promotion.get('promotion_source') != 'canonical_validation_selection':
        raise TaskFinalTestError('adapter was not promoted from validation selection')
    if selection.get('selection_source') != 'canonical_validation_only':
        raise TaskFinalTestError('promoted selection is not validation-only')
    expected_marker = {'task': task, 'learning_rate': promotion.get('selected_learning_rate'), 'seed': promotion.get('selected_seed'), 'epoch': promotion.get('selected_epoch'), 'promotion_source': 'canonical_validation_selection'}
    if marker != expected_marker:
        raise TaskFinalTestError('promotion completion marker is inconsistent')
    expected_inference = {'base_model': 'Qwen3-VL-32B-Instruct', 'task': task, 'few_shot_k': 0, 'generation': dict(expected_generation)}
    for field, expected in expected_inference.items():
        if inference.get(field) != expected:
            raise TaskFinalTestError(f'inference configuration drift for {field}')
    get_model_backend(base).validate_adapter_directory(base, adapter_dir)
    return (promotion, inference)

def _command(task: str, base_path: Path, adapter_root: Path, result_dir: Path) -> list[str]:
    command = [sys.executable, '-u', str(evaluator_script_path(task)), '--model', f'{MODEL_PREFIX}{task}', '--few-shot-k', '0', '--output-dir', str(result_dir), '--cache-dir', './raw_dataset', '--qwen-model-path', str(base_path), '--qwen-finetuned-root', str(adapter_root)]
    worker_flag = '--sample-workers' if task in {'dior-rsvg', 'dior-rsvg-recognition', 'geobench-refdet'} else '--sample-parallel'
    command.extend((worker_flag, '1'))
    if task == 'geobench-refdet':
        command.extend(('--task', 'refdet'))
    if task == 'aerial-animal':
        command.extend(('--output', str(result_dir / 'predictions.jsonl')))
    return command

def _prediction_paths(task: str, result_dir: Path) -> list[Path]:
    direct = result_dir / 'predictions.jsonl'
    if direct.is_file():
        return [direct]
    all_paths = sorted((path for path in result_dir.glob(f'{MODEL_PREFIX}{task}*.jsonl') if path.is_file()))
    if task == 'cowc':
        combined = [path for path in all_paths if path.name.endswith('-combined-counting.jsonl')]
        if len(combined) != 1:
            raise TaskFinalTestError('COWC requires exactly one combined prediction JSONL')
        return combined
    if len(all_paths) != 1:
        raise TaskFinalTestError(f'{task} requires exactly one prediction JSONL, found {len(all_paths)}')
    return all_paths

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TaskFinalTestError(f'non-object prediction at {path}:{line_number}')
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskFinalTestError(f'invalid prediction JSONL: {path}') from exc
    return records

def _verify_predictions(task: str, records: list[Mapping[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    if len(records) != expected_count:
        raise TaskFinalTestError('Evaluation predictions are incomplete')
    identities: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for record in records:
        checks = {'canonical_split': 'test', 'model_kind': 'lora', 'adapter_name': task, 'prompt_mode': 'zero-shot', 'few_shot_k': 0}
        for field, expected in checks.items():
            if record.get(field) != expected:
                raise TaskFinalTestError(f'prediction has {field}={record.get(field)!r}, expected {expected!r}')
        if task in {'fpcd', 'rare-planes'}:
            identity = ('test', record.get('source_id'))
        else:
            identity = (record.get('city') if task == 'cowc' else record.get('raw_split'), record.get('index'))
        if identity[1] is None or identity in seen:
            raise TaskFinalTestError(f'invalid or duplicate test prediction identity: {identity}')
        seen.add(identity)
        identities.append({'source': identity[0], 'index': identity[1]})
    return identities

def _completed_result(*, task: str, result_dir: Path, expected_count: int, inference: Mapping[str, Any]) -> bool:
    marker_path = result_dir / TEST_COMPLETE
    if not marker_path.is_file():
        return False
    marker = _load_json(marker_path)
    paths = _prediction_paths(task, result_dir)
    records = [record for path in paths for record in _read_jsonl(path)]
    _verify_predictions(task, records, expected_count)
    metrics = _load_json(result_dir / METRICS)
    expected = {'task': task, 'few_shot_k': 0, 'test_split': 'canonical_test', 'resolution_candidate': inference['resolution_candidate'], 'generation': inference['generation']}
    for field, value in expected.items():
        if marker.get(field) != value:
            raise TaskFinalTestError(f'completed test marker drift for {field}')
    if metrics.get('task') != task:
        raise TaskFinalTestError('completed metrics are inconsistent')
    return True

def main(task: str, *, postprocess_existing: bool=False) -> int:
    if task not in TASKS:
        raise TaskFinalTestError(f'unsupported final task: {task!r}')
    spec = get_pipeline_spec(task)
    registry_path = active_registry(DEFAULT_REGISTRY)
    base_path = active_base_config(base_config_for_task(spec))
    base = load_base_lora_config(base_path, registry_path)
    get_model_backend(base).validate_config(base)
    blockers = load_registry_gate(registry_path, task).test_blockers
    if blockers:
        raise TaskFinalTestError(f'task {task!r} final test is blocked: {list(blockers)}')
    if base.base_model.model_type.casefold() == 'llava_onevision2':
        from finetune.workflows.llava_final_evaluation import run as run_llava_final
        return run_llava_final(task, base_config=base_path, registry=registry_path)
    dataset_path = active_dataset_config() or dataset_config_for_task(spec)
    dataset = load_dataset_config(dataset_path)
    resolved = resolve_experiment_config(base, dataset, registry_path)
    model_dataset = model_artifact_paths(base, resolved.dataset)
    adapter_dir = model_dataset.paths.final_adapter_dir.resolve()
    _, inference = _verify_adapter(adapter_dir, task, resolved.dataset.generation.model_dump(mode='json'), base)
    adapter = get_task_dataset(task).adapter_class(resolved.dataset)
    from finetune.datasets import CanonicalSplit
    expected_count = len(adapter.materialize_split(CanonicalSplit.TEST))
    result_dir = model_dataset.paths.test_result_dir.resolve()
    if result_dir.exists():
        unexpected_dirs = [path for path in result_dir.iterdir() if path.is_dir()]
        if unexpected_dirs:
            raise TaskFinalTestError(f'final results must be stored directly in {result_dir}; found {unexpected_dirs[0]}')
    if _completed_result(task=task, result_dir=result_dir, expected_count=expected_count, inference=inference):
        return 0
    result_dir.mkdir(parents=True, exist_ok=True)
    if not postprocess_existing:
        subprocess.run(_command(task, base.base_model.path, adapter_dir.parent, result_dir), cwd=PROJECT_ROOT, check=True)
    prediction_paths = _prediction_paths(task, result_dir)
    records = [record for path in prediction_paths for record in _read_jsonl(path)]
    _verify_predictions(task, records, expected_count)
    metrics: dict[str, Any] = {'task': task, 'prediction_files': []}
    for path in prediction_paths:
        authoritative = _json_compatible(adapter.semantics.module.evaluation(str(path)))
        metrics['prediction_files'].append({'path': str(path), 'metrics': authoritative})
    atomic_write_json(result_dir / METRICS, metrics)
    atomic_write_json(result_dir / TEST_COMPLETE, {'task': task, 'few_shot_k': 0, 'test_split': 'canonical_test', 'complete_canonical_test_split': True, 'resolution_candidate': inference['resolution_candidate'], 'generation': inference['generation']})
    print(json.dumps({'result_dir': str(result_dir), 'metrics': metrics}, indent=2), flush=True)
    return 0
