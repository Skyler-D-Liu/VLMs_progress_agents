"""Messages."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from finetune.datasets import AdapterExample
from finetune.training.input_errors import TrainingDataError


IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"


def _strip_leading_placeholders(text: str, image_count: int) -> str:
    count = text.count(IMAGE_PLACEHOLDER)
    if count == 0:
        return text
    if count != image_count:
        raise TrainingDataError(
            f"User message contains {count} <image> placeholders for {image_count} images"
        )

    remainder = text
    for _ in range(image_count):
        if not remainder.startswith(IMAGE_PLACEHOLDER):
            raise TrainingDataError("All <image> placeholders must form a prefix of the user message")
        remainder = remainder[len(IMAGE_PLACEHOLDER) :]
        remainder = re.sub(r"^[ \t]*(?:\r?\n)?", "", remainder, count=1)
    if IMAGE_PLACEHOLDER in remainder:
        raise TrainingDataError("Unexpected <image> placeholder inside user text")
    return remainder


def build_qwen_messages(example: AdapterExample) -> list[dict[str, Any]]:
    roles = tuple(message.role for message in example.messages)
    if roles not in {("system", "user", "assistant"), ("user", "assistant")}:
        raise TrainingDataError(
            "Qwen training requires [system, user, assistant] or [user, assistant]"
        )
    if roles.count("user") != 1 or roles.count("assistant") != 1:
        raise TrainingDataError("Qwen training supports exactly one user and one assistant turn")

    by_role = {message.role: message.content for message in example.messages}
    assistant = by_role["assistant"]
    if not assistant.strip():
        raise TrainingDataError("Assistant answer must be non-empty")
    user = by_role["user"]
    if VIDEO_PLACEHOLDER in user:
        raise TrainingDataError("Video input is not supported by the shared image training core")

    image_paths = []
    for index, raw_path in enumerate(example.image_paths):
        shown = str(raw_path)
        if shown.startswith(("http://", "https://", "data:")):
            raise TrainingDataError(f"image_paths.{index} must be a local file, got {shown!r}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise TrainingDataError(f"Missing image_paths.{index}: {path.resolve()}")
        image_paths.append(path.resolve())
    if not image_paths:
        raise TrainingDataError("At least one local image is required")

    prompt_text = _strip_leading_placeholders(user, len(image_paths))
    if not prompt_text.strip():
        raise TrainingDataError("User text is empty after removing image placeholders")

    messages: list[dict[str, Any]] = []
    if "system" in by_role:
        messages.append({"role": "system", "content": by_role["system"]})
    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": prompt_text})
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": assistant})
    return messages
