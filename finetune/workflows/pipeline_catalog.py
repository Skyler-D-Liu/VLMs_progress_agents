"""Pipeline catalog."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class PipelineCatalogError(ValueError):
    pass

@dataclass(frozen=True)
class DatasetPipelineSpec:
    task: str
    dataset_config: Path
    adapter_module: Path
    data_family: str
    image_count: int

    @property
    def job_name_prefix(self) -> str:
        return f'lora-{self.task}'
_ALIASES = {'aerial-animal': 'aerial-animal', 'aerial-animal-population-4tu': 'aerial-animal', 'cowc': 'cowc', 'cowc-m': 'cowc', 'dior-rsvg': 'dior-rsvg', 'dior-rsvg-recognition': 'dior-rsvg-recognition', 'dior-rsvg-recognition-local': 'dior-rsvg-recognition', 'geobench-refdet': 'geobench-refdet', 'geobench': 'geobench-refdet', 'farm-pond-change-detection': 'fpcd', 'fpcd': 'fpcd', 'landmarks': 'landmarks', 'neon-tree': 'neon-tree', 'neon-trees': 'neon-tree', 'pattern-net': 'pattern-net', 'patternnet': 'pattern-net', 'rare-planes': 'rare-planes', 'rareplanes': 'rare-planes', 'aircraft-counting': 'rare-planes', 'xview2': 'xview2'}

def _spec(task: str, config: str, adapter: str, family: str, image_count: int=1):
    return DatasetPipelineSpec(task=task, dataset_config=PROJECT_ROOT / 'finetune/experiment_specs/datasets' / config, adapter_module=PROJECT_ROOT / 'finetune/datasets' / adapter, data_family=family, image_count=image_count)
_PIPELINES = {'aerial-animal': _spec('aerial-animal', 'aerial_animal.yaml', 'aerial_animal_adapter.py', 'single_image_counting'), 'cowc': _spec('cowc', 'cowc.yaml', 'cowc_adapter.py', 'single_image_counting'), 'dior-rsvg': _spec('dior-rsvg', 'dior_rsvg.yaml', 'dior_rsvg_adapter.py', 'single_image_localization'), 'dior-rsvg-recognition': _spec('dior-rsvg-recognition', 'dior_rsvg_recognition.yaml', 'dior_rsvg_recognition_adapter.py', 'single_image_classification'), 'fpcd': _spec('fpcd', 'fpcd.yaml', 'fpcd_adapter.py', 'paired_change_detection', 2), 'geobench-refdet': _spec('geobench-refdet', 'geobench_refdet.yaml', 'geobench_refdet_adapter.py', 'single_image_localization'), 'landmarks': _spec('landmarks', 'landmarks_filtered.yaml', 'landmarks_adapter.py', 'single_image_classification'), 'neon-tree': _spec('neon-tree', 'neon_tree.yaml', 'neon_tree_adapter.py', 'single_image_counting'), 'pattern-net': _spec('pattern-net', 'pattern_net.yaml', 'pattern_net_adapter.py', 'single_image_classification'), 'rare-planes': _spec('rare-planes', 'rare_planes.yaml', 'rare_planes_adapter.py', 'single_image_counting'), 'xview2': _spec('xview2', 'xview2.yaml', 'xview2_adapter.py', 'paired_change_detection', 2)}

def normalize_dataset_name(value: str) -> str:
    normalized = '-'.join(value.strip().casefold().replace('_', '-').split())
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        raise PipelineCatalogError(f"unknown --trained-dataset {value!r}; known names: {', '.join(sorted(_ALIASES))}") from exc

def get_pipeline_spec(value: str) -> DatasetPipelineSpec:
    task = normalize_dataset_name(value)
    try:
        spec = _PIPELINES[task]
    except KeyError as exc:
        raise PipelineCatalogError(f'dataset {task!r} is not in the canonical task matrix') from exc
    missing = [str(path) for path in (spec.dataset_config, spec.adapter_module) if not path.is_file()]
    if missing:
        raise PipelineCatalogError(f'dataset {task!r} pipeline assets are missing: {missing}')
    return spec

def implemented_dataset_names() -> tuple[str, ...]:
    return tuple(sorted(_PIPELINES))

def dataset_config_for_task(spec: DatasetPipelineSpec) -> Path:
    if not spec.dataset_config.is_file():
        raise PipelineCatalogError(f'missing dataset config: {spec.dataset_config}')
    return spec.dataset_config

def base_config_for_task(spec: DatasetPipelineSpec, *, model_type: str='qwen3_vl') -> Path:
    filename = 'models/llava_ov2_8b.yaml' if model_type.casefold() == 'llava_onevision2' else 'base_lora.yaml'
    path = PROJECT_ROOT / 'finetune/experiment_specs' / filename
    if not path.is_file():
        raise PipelineCatalogError(f'missing base config: {path}')
    return path
