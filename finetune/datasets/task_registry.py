"""Task registry."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Type
from .adapter_contract import DatasetAdapter
from .aerial_animal_adapter import AerialAnimalAdapter
from .cowc_adapter import CowcAdapter
from .dior_rsvg_adapter import DiorRsvgAdapter
from .dior_rsvg_recognition_adapter import DiorRsvgRecognitionAdapter
from .fpcd_adapter import FPCDAdapter
from .geobench_refdet_adapter import GeobenchRefdetAdapter
from .landmarks_adapter import LandmarksAdapter
from .neon_tree_adapter import NeonTreeAdapter
from .pattern_net_adapter import PatternNetAdapter
from .rare_planes_adapter import RarePlanesAdapter
from .xview2_adapter import XView2Adapter
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class TaskDatasetRegistration:
    task: str
    config_filename: str
    adapter_class: Type[DatasetAdapter]

    @property
    def config_path(self) -> Path:
        return PROJECT_ROOT / 'finetune/experiment_specs/datasets' / self.config_filename
TASK_DATASETS = {registration.task: registration for registration in (TaskDatasetRegistration('aerial-animal', 'aerial_animal.yaml', AerialAnimalAdapter), TaskDatasetRegistration('cowc', 'cowc.yaml', CowcAdapter), TaskDatasetRegistration('dior-rsvg', 'dior_rsvg.yaml', DiorRsvgAdapter), TaskDatasetRegistration('dior-rsvg-recognition', 'dior_rsvg_recognition.yaml', DiorRsvgRecognitionAdapter), TaskDatasetRegistration('fpcd', 'fpcd.yaml', FPCDAdapter), TaskDatasetRegistration('geobench-refdet', 'geobench_refdet.yaml', GeobenchRefdetAdapter), TaskDatasetRegistration('landmarks', 'landmarks_filtered.yaml', LandmarksAdapter), TaskDatasetRegistration('neon-tree', 'neon_tree.yaml', NeonTreeAdapter), TaskDatasetRegistration('pattern-net', 'pattern_net.yaml', PatternNetAdapter), TaskDatasetRegistration('rare-planes', 'rare_planes.yaml', RarePlanesAdapter), TaskDatasetRegistration('xview2', 'xview2.yaml', XView2Adapter))}

def get_task_dataset(task: str) -> TaskDatasetRegistration:
    try:
        return TASK_DATASETS[task]
    except KeyError as exc:
        raise ValueError(f'unknown fine-tuning task: {task!r}') from exc
