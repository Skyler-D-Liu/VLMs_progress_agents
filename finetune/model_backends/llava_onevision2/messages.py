"""Messages."""

from pathlib import Path
import re
from typing import Any

from finetune.datasets import AdapterExample
from finetune.training.input_errors import TrainingDataError


def build_llava_messages(example: AdapterExample) -> list[dict[str, Any]]:
    roles = tuple(message.role for message in example.messages)
    if roles not in {("system", "user", "assistant"), ("user", "assistant")}:
        raise TrainingDataError("LLaVA training requires [system, user, assistant] or [user, assistant]")
    if any(not isinstance(message.content, str) for message in example.messages):
        raise TrainingDataError("LLaVA message content must be text")
    by_role = {message.role: message.content for message in example.messages}
    assistant = by_role["assistant"]
    if not assistant.strip():
        raise TrainingDataError("Assistant answer must be non-empty")
    user = by_role["user"]
    if "<video>" in user.casefold():
        raise TrainingDataError("Video input is not supported by the shared image training core")
    if not example.image_paths:
        raise TrainingDataError("At least one local image is required")
    count = user.count("<image>")
    if count != len(example.image_paths):
        raise TrainingDataError(
            f"User message contains {count} <image> placeholders for {len(example.image_paths)} images"
        )
    prompt = user
    for _ in example.image_paths:
        prompt = prompt.replace("<image>", "", 1)
        prompt = re.sub(r"^[ \t]*(?:\r?\n)?", "", prompt, count=1)
    if not prompt.strip():
        raise TrainingDataError("User text is empty after removing image placeholders")
    images = []
    for index, raw_path in enumerate(example.image_paths):
        shown = str(raw_path)
        if shown.casefold().startswith(("http://", "https://", "data:")):
            raise TrainingDataError(f"image_paths.{index} must be a local file: {shown!r}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise TrainingDataError(f"image_paths.{index} must be a local file: {path.resolve()}")
        if path.is_symlink() and not path.resolve().is_file():
            raise TrainingDataError(
                f"image_paths.{index} symlink does not resolve to a local file: {path}"
            )
        if path.suffix.casefold() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}:
            raise TrainingDataError(
                f"image_paths.{index} points to a video; video training is disabled: {path}"
            )
        images.append({"type": "image"})
    content = images + [{"type": "text", "text": prompt}]
    messages = []
    if "system" in by_role:
        messages.append({"role": "system", "content": by_role["system"]})
    messages.extend([
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant},
    ])
    return messages
