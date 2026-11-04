"""Resolution statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping



class ResolutionStatisticsError(RuntimeError):
    pass


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResolutionStatisticsError(
            f"model-owned resolution statistics are missing: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolutionStatisticsError(
            f"invalid model-owned resolution statistics: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ResolutionStatisticsError(
            f"resolution statistics must be a JSON object: {path}"
        )
    return value


def validate_resolution_statistics(
    value: Mapping[str, Any],
    *,
    task: str,
    processor_path: str | Path,
) -> None:
    if value.get("task") != task:
        raise ResolutionStatisticsError(
            f"resolution statistics task mismatch: {value.get('task')!r} != {task!r}"
        )
    if value.get("statistics_splits") != ["train", "validation"]:
        raise ResolutionStatisticsError(
            "resolution statistics must use exactly train and validation"
        )
    for field in (
        "test_split_materialized",
        "test_split_inspected_for_resolution",
        "selection_performed",
    ):
        if value.get(field) is not False:
            raise ResolutionStatisticsError(
                f"resolution statistics require {field}=false"
            )
    processor = value.get("processor")
    if not isinstance(processor, Mapping):
        raise ResolutionStatisticsError("resolution statistics processor is missing")
    actual_path = processor.get("preprocessor_config")
    if not isinstance(actual_path, str) or not actual_path.strip():
        raise ResolutionStatisticsError(
            "resolution statistics preprocessor_config is missing"
        )
    expected = Path(processor_path).expanduser().resolve()
    actual = Path(actual_path).expanduser().resolve()
    if actual != expected:
        raise ResolutionStatisticsError(
            f"resolution statistics processor mismatch: {actual} != {expected}"
        )


def load_resolution_statistics(
    path: str | Path,
    *,
    task: str,
    processor_path: str | Path,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    value = _load_json_object(resolved)
    validate_resolution_statistics(
        value,
        task=task,
        processor_path=processor_path,
    )
    return value


