"""Cot prompting."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

REASONING_METHODS = ("baseline", "cot")
COT_ALLOWED_MODELS = frozenset({"gpt-5.5", "gemini-3.5-flash", "gpt-6.1-sol", "gemini-3.8-flash", "Qwen3-VL-32B-Instruct", "LLaVA-OV-2-8B"})
RAW_LOCAL_COT_MODEL_PREFIXES = ("Qwen", "LLaVA")
COT_INSTRUCTION = "Think step by step based on the image. Place your reasoning before the answer."
LOCAL_BASELINE_MAX_NEW_TOKENS = 128
LOCAL_COT_MAX_NEW_TOKENS = 1024

def prompt_sha256(prompt):
    return hashlib.sha256(str(prompt).encode()).hexdigest()

def effective_prompt(base_prompt, reasoning_method):
    validate_reasoning_method(reasoning_method)
    return base_prompt if reasoning_method == "baseline" else f"{base_prompt.rstrip()}\n\n{COT_INSTRUCTION}"

def effective_system_message(base_message, reasoning_method):
    validate_reasoning_method(reasoning_method)
    if reasoning_method == "baseline":
        return base_message
    return (str(base_message or "") + "\nYou may output a <think>...</think> reasoning block before the final answer. "
            "Any JSON-only instruction applies to the final answer, not the reasoning block. "
            "Keep the task and final answer format unchanged.")

def local_generation_max_new_tokens(user_prompt):
    return LOCAL_COT_MAX_NEW_TOKENS if COT_INSTRUCTION in str(user_prompt) else LOCAL_BASELINE_MAX_NEW_TOKENS

def extract_visible_content(response):
    if isinstance(response, Mapping):
        if response.get("choices"):
            return str(response["choices"][0].get("message", {}).get("content") or "")
        return str(response.get("content", response.get("answer_text", "")) or "")
    return "" if response is None else str(response)

def structured_answer_candidates(text):
    decoder = json.JSONDecoder()
    candidates = []
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                pass
            else:
                if isinstance(value, (dict, list)):
                    candidates.append(text[i:end])
                    i = end
                    continue
        i += 1
    return candidates

def strip_reasoning(text):
    text = str(text or "")
    text = re.sub(r"<(think|reasoning)>.*?</\1>", "", text, flags=re.S | re.I)
    text = re.sub(r"<(?:think|reasoning)>.*$", "", text, flags=re.S | re.I)
    return text.strip()


def parse_reasoning_response(response_text, reasoning_method):
    validate_reasoning_method(reasoning_method)
    raw = str(response_text or "")
    text = strip_reasoning(raw)
    candidates = structured_answer_candidates(text)
    if candidates:
        answer = candidates[-1]
    else:
        answer = text
    return {"answer_text": answer, "reasoning": raw[:raw.rfind(answer)].strip() or None
            if reasoning_method == "cot" and answer else None}


def call_with_reasoning(call, reasoning_method, *, task, context=None, prompt="", max_attempts=5):
    from evaluation.utils.response_protocol import validated_call
    return validated_call(call, reasoning_method, task=task, context=context, prompt=prompt, max_attempts=max_attempts)

def cot_row_metadata(*, base_prompt, effective_prompt_text, parsed_response,
                     base_system_message=None, effective_system_message_text=None):
    metadata = {"reasoning_method": "cot",
                "base_prompt_hash": prompt_sha256(base_prompt),
                "effective_prompt_hash": prompt_sha256(effective_prompt_text),
                "reasoning": parsed_response.get("reasoning")}
    if base_system_message is not None:
        metadata["base_system_prompt_hash"] = prompt_sha256(base_system_message)
    if effective_system_message_text is not None:
        metadata["effective_system_prompt_hash"] = prompt_sha256(effective_system_message_text)
    return metadata

def result_path_for_reasoning(path: str | Path, reasoning_method: str) -> str | Path:
    validate_reasoning_method(reasoning_method)
    if reasoning_method == "baseline":
        return path
    original = Path(path)
    if original.stem.endswith("-CoT"):
        updated = original
    else:
        updated = original.with_name(f"{original.stem}-CoT{original.suffix}")
    return updated if isinstance(path, Path) else str(updated)


def write_cot_metrics(result_path: str | Path, metrics: Mapping[str, Any]) -> Path | None:
    path = Path(result_path)
    if not path.name.endswith("-CoT.jsonl"):
        return None
    metrics_path = path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path


def evaluate_and_write_cot_metrics(
    evaluation_fn,
    result_path: str | Path,
    *args,
    **kwargs,
) -> Any:
    metrics = evaluation_fn(result_path, *args, **kwargs)
    if isinstance(metrics, Mapping):
        write_cot_metrics(result_path, metrics)
    return metrics


def validate_cot_run(
    *,
    reasoning_method: str,
    model_key: str,
    few_shot_k: int = 0,
    model_kind: str | None = None,
) -> None:
    validate_reasoning_method(reasoning_method)
    if reasoning_method == "baseline":
        return
    if model_key not in COT_ALLOWED_MODELS and not str(model_key).startswith(
        RAW_LOCAL_COT_MODEL_PREFIXES
    ):
        allowed = ", ".join(sorted(COT_ALLOWED_MODELS))
        raise ValueError(
            "CoT is supported for the benchmark API models "
            f"({allowed}) and raw local Qwen/LLaVA checkpoints; got {model_key!r}"
        )
    if few_shot_k:
        raise ValueError("CoT is zero-shot only and cannot be combined with few-shot prompting")
    if model_kind == "lora":
        raise ValueError("CoT cannot be combined with a fine-tuned/LoRA model")


def validate_reasoning_method(reasoning_method: str) -> None:
    if reasoning_method not in REASONING_METHODS:
        raise ValueError(
            f"reasoning_method must be one of {REASONING_METHODS}, got {reasoning_method!r}"
        )


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(value, Path):
        return str(value)
    return str(value)
