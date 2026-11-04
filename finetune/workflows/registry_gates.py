"""Registry gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from finetune.config import load_registry_document


@dataclass(frozen=True)
class RegistryGate:

    task: str
    training_allowed: bool
    test_allowed: bool
    global_blockers: tuple[str, ...]
    task_blockers: tuple[str, ...]

    @property
    def training_blockers(self) -> tuple[str, ...]:
        blockers = list(self.task_blockers)
        if not self.training_allowed:
            blockers.insert(0, "registry.readiness.training_allowed=false")
        blockers.extend(self.global_blockers)
        return tuple(dict.fromkeys(blockers))

    @property
    def test_blockers(self) -> tuple[str, ...]:
        blockers = list(self.task_blockers)
        if not self.test_allowed:
            blockers.insert(0, "registry.readiness.test_allowed=false")
        blockers.extend(self.global_blockers)
        return tuple(dict.fromkeys(blockers))


def load_registry_gate(registry_path: str | Path, task: str) -> RegistryGate:
    path = Path(registry_path).expanduser().resolve()
    try:
        registry = load_registry_document(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load frozen experiment registry: {path}") from exc
    if not isinstance(registry, dict):
        raise RuntimeError(f"frozen experiment registry is not a mapping: {path}")
    tasks = registry.get("tasks")
    if not isinstance(tasks, dict) or task not in tasks:
        raise RuntimeError(f"task {task!r} is absent from frozen experiment registry")
    readiness = registry.get("readiness")
    if not isinstance(readiness, dict):
        raise RuntimeError("frozen registry readiness section is missing")
    task_record = tasks[task]
    if not isinstance(task_record, dict):
        raise RuntimeError(f"frozen registry task record is invalid: {task}")
    return RegistryGate(
        task=task,
        training_allowed=readiness.get("training_allowed") is True,
        test_allowed=readiness.get("test_allowed") is True,
        global_blockers=tuple(str(value) for value in readiness.get("blockers", ())),
        task_blockers=tuple(str(value) for value in task_record.get("blockers", ())),
    )
