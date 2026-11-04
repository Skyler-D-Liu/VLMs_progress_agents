"""Collator."""

from pathlib import Path
from typing import Any

from finetune.datasets import AdapterExample
from finetune.training.input_errors import AssistantMaskError, TrainingDataError

from .messages import build_llava_messages

IGNORE_INDEX = -100


def _token_id_values(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, int) and value >= 0:
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        values: list[int] = []
        for item in value:
            values.extend(_token_id_values(item))
        return tuple(values)
    return ()


def _known_token_id(tokenizer: Any, token: str) -> int | None:
    if tokenizer is None:
        return None
    vocabulary = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocabulary = get_vocab()
        except Exception:
            vocabulary = None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        value = convert(token)
    except (TypeError, ValueError):
        return None
    values = _token_id_values(value)
    if len(values) != 1:
        return None
    value = values[0]
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(unknown_id, int)
        and value == unknown_id
    ):
        if not isinstance(vocabulary, dict) or token not in vocabulary:
            return None
    return value


def _visual_token_ids(processor: Any) -> frozenset[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids: set[int] = set()
    for owner in (processor, tokenizer):
        if owner is None:
            continue
        for attribute in (
            "image_token_id",
            "image_token_index",
            "video_token_id",
            "video_token_index",
            "vision_start_token_id",
            "vision_end_token_id",
            "vision_pad_token_id",
        ):
            ids.update(_token_id_values(getattr(owner, attribute, None)))
    for token in (
        "<image>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_start|>",
        "<|video_end|>",
    ):
        value = _known_token_id(tokenizer, token)
        if value is not None:
            ids.add(value)
    return frozenset(ids)


def _assistant_termination_ids(tokenizer: Any) -> frozenset[int]:
    ids: set[int] = set()
    for attribute in ("eos_token_id", "im_end_token_id"):
        ids.update(_token_id_values(getattr(tokenizer, attribute, None)))
    for token in ("<|im_end|>", "<|end|>", "<|eos|>"):
        value = _known_token_id(tokenizer, token)
        if value is not None:
            ids.add(value)
    return frozenset(ids)


class LlavaOneVision2AssistantOnlyCollator:
    def __init__(self, processor: Any, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[AdapterExample]) -> Any:
        if len(features) != 1:
            raise TrainingDataError(f"Frozen micro-batch size is 1, received {len(features)} examples")
        example = features[0]
        if not isinstance(example, AdapterExample):
            raise TrainingDataError(
                f"Expected AdapterExample, got {type(example).__name__}"
            )
        messages = build_llava_messages(example)
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("LLaVA image collation requires Pillow") from exc
        images = []
        for path in example.image_paths:
            with Image.open(Path(path).expanduser().resolve()) as image:
                images.append(image.convert("RGB"))
        batch = self.processor(
            text=[rendered],
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        if "input_ids" not in batch:
            raise TrainingDataError(f"{example.sample_id}: processor returned no input_ids")
        missing_visual_fields = [
            field
            for field in ("pixel_values", "image_grid_thw", "patch_positions")
            if field not in batch or batch[field] is None
        ]
        if missing_visual_fields:
            raise TrainingDataError(
                f"{example.sample_id}: LLaVA processor omitted visual fields: "
                + ", ".join(missing_visual_fields)
            )
        if any(
            field in batch and batch[field] is not None
            for field in ("pixel_values_videos", "video_grid_thw", "second_per_grid_ts")
        ):
            raise TrainingDataError(
                f"{example.sample_id}: video tensors are forbidden in image training"
            )
        image_grid = batch["image_grid_thw"]
        if getattr(image_grid, "ndim", None) != 2 or image_grid.shape[-1] != 3:
            raise TrainingDataError(
                f"{example.sample_id}: image_grid_thw must have shape [images, 3]"
            )
        if int(image_grid.shape[0]) != len(images):
            raise TrainingDataError(
                f"{example.sample_id}: image_grid_thw image count does not match local images"
            )
        patch_positions = batch["patch_positions"]
        patch_shape = getattr(patch_positions, "shape", ())
        if len(patch_shape) not in {2, 3} or patch_shape[-1] != 3:
            raise TrainingDataError(
                f"{example.sample_id}: patch_positions must end in a [*, 3] axis"
            )
        input_ids = batch["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise TrainingDataError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")
        if "attention_mask" in batch and (
            batch["attention_mask"].shape != input_ids.shape
        ):
            raise TrainingDataError(
                f"{example.sample_id}: attention_mask shape does not match input_ids"
            )
        if input_ids.shape[1] > self.max_length:
            raise TrainingDataError(
                f"{example.sample_id}: sequence length exceeds max_length={self.max_length}"
            )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise TrainingDataError(f"{example.sample_id}: LLaVA processor has no tokenizer")


        prompt_messages = messages[:-1]
        prompt_rendered = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_batch = self.processor(
            text=[prompt_rendered],
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        if "input_ids" not in prompt_batch or prompt_batch["input_ids"] is None:
            raise AssistantMaskError(
                f"{example.sample_id}: prompt processor returned no input_ids"
            )
        prompt_ids = prompt_batch["input_ids"]
        answer_start = None
        if (
            prompt_ids.ndim == 2
            and prompt_ids.shape[0] == 1
            and prompt_ids.shape[1] < input_ids.shape[1]
            and input_ids[0, : prompt_ids.shape[1]].equal(prompt_ids[0])
        ):
            answer_start = int(prompt_ids.shape[1])
        end_ids = _assistant_termination_ids(tokenizer)
        ids = input_ids[0].tolist()
        if answer_start is None:


            encode = getattr(tokenizer, "encode", None)
            assistant_prefix = (
                encode("<|im_start|>assistant\n", add_special_tokens=False)
                if callable(encode)
                else []
            )
            if not isinstance(assistant_prefix, list) or not assistant_prefix:
                assistant_prefix = []
            starts = [
                i
                for i in range(len(ids) - len(assistant_prefix) + 1)
                if ids[i : i + len(assistant_prefix)] == assistant_prefix
            ]
            if len(starts) == 1:
                answer_start = starts[0] + len(assistant_prefix)
        if answer_start is None or not end_ids:
            raise AssistantMaskError(f"{example.sample_id}: assistant boundary is ambiguous")
        endings = [i for i in range(answer_start, len(ids)) if ids[i] in end_ids]
        if len(endings) != 1 or endings[0] <= answer_start:
            raise AssistantMaskError(f"{example.sample_id}: assistant answer span is empty or ambiguous")
        labels = input_ids.new_full(input_ids.shape, IGNORE_INDEX)
        labels[0, answer_start: endings[0] + 1] = input_ids[0, answer_start: endings[0] + 1]
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"] == 0, IGNORE_INDEX)
        visual_ids = _visual_token_ids(self.processor)
        if not visual_ids:
            raise AssistantMaskError(
                f"{example.sample_id}: visual token IDs cannot be identified safely"
            )
        visual_mask = input_ids.eq(-1)
        for token_id in visual_ids:
            visual_mask |= input_ids == token_id
        labels = labels.masked_fill(visual_mask, IGNORE_INDEX)
        if int((labels != IGNORE_INDEX).sum()) <= 1:
            raise AssistantMaskError(f"{example.sample_id}: assistant supervision is empty")
        batch["labels"] = labels
        return batch


LLaVAOneVision2AssistantOnlyCollator = LlavaOneVision2AssistantOnlyCollator

__all__ = [
    "IGNORE_INDEX",
    "LLaVAOneVision2AssistantOnlyCollator",
    "LlavaOneVision2AssistantOnlyCollator",
]
