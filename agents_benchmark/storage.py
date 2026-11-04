"""Storage."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another client owns this output directory") from exc
        yield


def redact(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)


def clean_evidence(value, payload=False):
    if isinstance(value, list):
        return [clean_evidence(item, payload) for item in value]
    if isinstance(value, dict):
        image = value.get('type') in {'input_image', 'output_image', 'image', 'computer_screenshot'}
        tracking = {'id', 'session_id', 'agent_id', 'subagent_id', 'turn_id',
                    'call_id', 'event_id', 'item_id', 'environment_id', 'request_id', 'file_id',
                    'sender_agent_id', 'recipient_agent_id', 'parent_subagent_id', 'parent_agent_id', 'metadata', 'timestamp'}
        return {key: clean_evidence(item, payload or key in {'arguments', 'tool_arguments', 'output'})
                for key, item in value.items()
                if key != 'encrypted_content' and not (image and key in {'data', 'image_url', 'url'})
                and not (not payload and (key in tracking or key.endswith('_at')))}
    return redact(value) if isinstance(value, str) else value


def as_dict(value):
    return value if isinstance(value, dict) else value.model_dump(mode="json")


def api_error(exc) -> dict:
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    result = {"status_code": getattr(exc, "status_code", None)}
    if isinstance(body, dict):
        result.update({key: clean_evidence(body[key]) for key in ("type", "code", "param", "message") if key in body})
    if not result.get("message"):
        result["message"] = redact(str(exc))
    return result
