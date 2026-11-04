"""Loader."""
from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any, Mapping
import yaml
from pydantic import ValidationError
from .schema import BaseLoraConfig, DatasetConfig, ResolvedExperimentConfig

class ConfigValidationError(ValueError):
    pass

def _deep_merge(parent: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(parent))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def _read_yaml_mapping(path: Path, field_name: str, *, _seen: tuple[Path, ...]=()) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ConfigValidationError(f'{field_name}: missing YAML file: {path.resolve()}')
    try:
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f'{field_name}: invalid YAML at {path.resolve()}: {exc}') from exc
    if not isinstance(value, dict):
        raise ConfigValidationError(f'{field_name}: expected a YAML mapping at {path.resolve()}')
    if 'extends' in value:
        allowed = {'extends', 'overrides'}
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise ConfigValidationError(f'{field_name}: overlay has unexpected keys {unexpected} at {path.resolve()}')
        parent_value = value.get('extends')
        overrides = value.get('overrides')
        if not isinstance(parent_value, str) or not parent_value.strip():
            raise ConfigValidationError(f'{field_name}.extends: expected a non-empty path')
        if not isinstance(overrides, dict):
            raise ConfigValidationError(f'{field_name}.overrides: expected a mapping')
        parent_path = Path(parent_value).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        parent_path = parent_path.resolve()
        resolved_path = path.resolve()
        if resolved_path in _seen or parent_path in (*_seen, resolved_path):
            raise ConfigValidationError(f'{field_name}: cyclic overlay inheritance at {resolved_path}')
        if not parent_path.is_file():
            raise ConfigValidationError(f'{field_name}.extends: missing parent YAML: {parent_path}')
        parent = _read_yaml_mapping(parent_path, field_name, _seen=(*_seen, resolved_path))
        return _deep_merge(parent, overrides)
    return value

def load_registry_document(path: Path | str) -> dict[str, Any]:
    return _read_yaml_mapping(Path(path), 'registry')

def _format_validation_error(prefix: str, exc: ValidationError) -> ConfigValidationError:
    details = []
    for error in exc.errors(include_url=False):
        location = '.'.join((str(item) for item in error['loc']))
        details.append(f"{prefix}.{location}: {error['msg']}")
    return ConfigValidationError('\n'.join(details))

def _require_input(path: Path, field: str, *, directory: bool) -> None:
    path = Path(path).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    expected = 'directory' if directory else 'file'
    if not valid:
        raise ConfigValidationError(f'{field}: missing required {expected}: {path.resolve()}')

def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = Path(path).expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None

def _require_output_parent(path: Path, field: str) -> None:
    path = Path(path).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ConfigValidationError(f'{field}: output path exists but is not a directory: {path.resolve()}')
        if not os.access(path, os.W_OK):
            raise ConfigValidationError(f'{field}: output directory is not writable: {path.resolve()}')
        return
    parent = _nearest_existing_parent(path.parent)
    if parent is None:
        raise ConfigValidationError(f'{field}: no existing parent for output path: {path.resolve()}')
    if not parent.is_dir():
        raise ConfigValidationError(f'{field}: nearest existing output parent is not a directory: {parent.resolve()}')
    if not os.access(parent, os.W_OK):
        raise ConfigValidationError(f'{field}: nearest existing output parent is not writable: {parent.resolve()}')

def _template_probe(template: str) -> Path:
    try:
        rendered = template.format(task='config-check', learning_rate='1e-05', seed=0, adapter_name='adapter')
    except (KeyError, ValueError) as exc:
        raise ConfigValidationError(f'paths: invalid output template {template!r}: {exc}') from exc
    return Path(rendered)

def validate_paths(config: BaseLoraConfig | DatasetConfig) -> None:
    if isinstance(config, BaseLoraConfig):
        _require_input(config.registry.path, 'registry.path', directory=False)
        _require_input(config.base_model.path, 'base_model.path', directory=True)
        _require_input(config.paths.repository_root, 'paths.repository_root', directory=True)
        _require_input(config.paths.raw_dataset_root, 'paths.raw_dataset_root', directory=True)
        for name in config.base_model.identity_files:
            identity_path = config.base_model.path / name
            _require_input(identity_path, f'base_model.identity_files.{name}', directory=False)
        output_values = {'paths.prepared_data_template': _template_probe(config.paths.prepared_data_template), 'paths.temporary_trial_template': _template_probe(config.paths.temporary_trial_template), 'paths.final_adapter_template': _template_probe(config.paths.final_adapter_template), 'paths.permanent_result_root': config.paths.permanent_result_root, 'paths.validation_result_template': _template_probe(config.paths.validation_result_template), 'paths.test_result_template': _template_probe(config.paths.test_result_template), 'paths.manifest_template': _template_probe(config.paths.manifest_template)}
    elif isinstance(config, DatasetConfig):
        _require_input(config.paths.raw_dataset_root, 'paths.raw_dataset_root', directory=True)
        _require_input(config.prompt.authoritative_script, 'prompt.authoritative_script', directory=False)
        output_values = {'paths.prepared_dir': config.paths.prepared_dir, 'paths.final_adapter_dir': config.paths.final_adapter_dir, 'paths.validation_result_dir': config.paths.validation_result_dir, 'paths.test_result_dir': config.paths.test_result_dir, 'paths.manifest_dir': config.paths.manifest_dir}
    else:
        raise TypeError(f'Unsupported configuration type: {type(config).__name__}')
    for field, path in output_values.items():
        _require_output_parent(path, field)

def _compare(field: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, Path) and isinstance(expected, Path):
        matches = actual.expanduser().resolve(strict=False) == expected.expanduser().resolve(strict=False)
    else:
        matches = actual == expected
    if not matches:
        raise ConfigValidationError(f'registry mismatch at {field}: config={actual!r}, registry={expected!r}')

def load_base_lora_config(path: Path | str, registry_path: Path | str | None=None) -> BaseLoraConfig:
    data = _read_yaml_mapping(Path(path), 'base_config')
    import os
    try:
        config = BaseLoraConfig.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error('base_config', exc) from exc
    if registry_path is not None:
        _compare('registry.path', config.registry.path, Path(registry_path).expanduser())
    registry_file = Path(registry_path).expanduser() if registry_path is not None else config.registry.path
    _read_yaml_mapping(registry_file, 'registry')
    validate_paths(config)
    return config

def load_dataset_config(path: Path | str) -> DatasetConfig:
    data = _read_yaml_mapping(Path(path), 'dataset_config')
    try:
        config = DatasetConfig.model_validate(data)
    except ValidationError as exc:
        raise _format_validation_error('dataset_config', exc) from exc
    validate_paths(config)
    return config

def resolve_experiment_config(base: BaseLoraConfig, dataset: DatasetConfig, registry: Path | str | Mapping[str, Any] | None=None) -> ResolvedExperimentConfig:
    if registry is None:
        registry_data = _read_yaml_mapping(base.registry.path, 'registry')
    elif isinstance(registry, Mapping):
        registry_data = dict(registry)
    else:
        registry_path = Path(registry).expanduser()
        _compare('registry.path', registry_path, base.registry.path)
        registry_data = _read_yaml_mapping(registry_path, 'registry')
    if base.scope.allowed_tasks is not None and dataset.task not in base.scope.allowed_tasks:
        raise ConfigValidationError(f'dataset.task: {dataset.task!r} is outside the configured experiment scope')
    validate_paths(dataset)
    tasks = registry_data.get('tasks', {})
    if not isinstance(tasks, Mapping) or dataset.task not in tasks:
        raise ConfigValidationError(f'dataset.task: unknown registry task {dataset.task!r}')
    task = tasks[dataset.task]
    if not isinstance(task, Mapping):
        raise ConfigValidationError(f'dataset.task: invalid registry task {dataset.task!r}')
    if dataset.splits.seed != base.training.data_seed:
        raise ConfigValidationError('dataset split seed differs from training data seed')
    return ResolvedExperimentConfig(base=base, dataset=dataset, registry_task=dict(task))
