#!/usr/bin/env python3
"""Comparison summary utils."""

import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional


def shot_from_filename(filename: str) -> Dict[str, object]:
    if os.path.basename(filename).endswith("-CoT.jsonl"):
        return {
            "prompt_mode": "zero-shot",
            "few_shot_k": 0,
            "reasoning_method": "cot",
            "shot_label": "CoT",
        }
    match = re.search(r"-(?P<k>\d+)shot(?=\.jsonl$)", os.path.basename(filename))
    if not match:
        return {"prompt_mode": "zero-shot", "few_shot_k": 0, "shot_label": "zero-shot"}
    k = int(match.group("k"))
    return {"prompt_mode": "few-shot", "few_shot_k": k, "shot_label": f"{k}-shot"}


def shot_from_metrics(metrics: Dict, filename: Optional[str] = None) -> Dict[str, object]:
    if metrics.get("reasoning_method") == "cot":
        return {
            "prompt_mode": "zero-shot",
            "few_shot_k": 0,
            "reasoning_method": "cot",
            "shot_label": "CoT",
        }
    if filename:
        filename_shot = shot_from_filename(filename)
        if filename_shot["shot_label"] != "zero-shot":
            return filename_shot

    prompt_mode = metrics.get("prompt_mode")
    few_shot_k = metrics.get("few_shot_k")

    if prompt_mode is None and filename:
        return shot_from_filename(filename)

    if prompt_mode == "few-shot" or (few_shot_k not in (None, 0, "0")):
        try:
            k = int(few_shot_k)
        except (TypeError, ValueError):
            if filename:
                return shot_from_filename(filename)
            k = 0
        if k > 0:
            return {"prompt_mode": "few-shot", "few_shot_k": k, "shot_label": f"{k}-shot"}

    return {"prompt_mode": "zero-shot", "few_shot_k": 0, "shot_label": "zero-shot"}


def is_zero_shot_metrics(metrics: Dict, filename: Optional[str] = None) -> bool:
    return shot_from_metrics(metrics, filename)["shot_label"] == "zero-shot"


def json_safe(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_shot_summary_json(
    records: Iterable[Dict],
    output_path: str,
    dataset_name: str,
    summary_type: str = "model_comparison",
) -> None:
    safe_records: List[Dict] = []
    for record in records:
        cleaned = json_safe(dict(record))
        shot = shot_from_metrics(cleaned, cleaned.get("result_file") or cleaned.get("model_key"))
        cleaned.update(shot)
        safe_records.append(cleaned)

    safe_records.sort(
        key=lambda r: (
            str(r.get("shot_label") or ""),
            str(r.get("release_date") or ""),
            str(r.get("model_key") or ""),
        )
    )
    by_shot = {}
    for record in safe_records:
        by_shot.setdefault(record.get("shot_label", "unknown"), []).append(record)

    payload = {
        "dataset": dataset_name,
        "summary_type": summary_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": safe_records,
        "by_shot": by_shot,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Shot-aware JSON summary saved to: {output_path}")
