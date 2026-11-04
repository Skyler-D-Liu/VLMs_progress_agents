"""Collator."""

from __future__ import annotations

from typing import Any, Sequence

from finetune.datasets import AdapterExample
from finetune.training.input_errors import AssistantMaskError, TrainingDataError

from .messages import build_qwen_messages


ASSISTANT_HEADER = "<|im_start|>assistant\n"
IGNORE_INDEX = -100


def _subsequence_starts(values: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        return []
    width = len(needle)
    return [
        index
        for index in range(len(values) - width + 1)
        if list(values[index : index + width]) == list(needle)
    ]


class QwenAssistantOnlyCollator:

    def __init__(self, processor: Any, max_length: int, *, template_options=None) -> None:
        self.template_options = template_options or {}
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.processor = processor
        self.max_length = max_length
        self._assistant_header_ids = tuple(
            processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        )
        self._im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not self._assistant_header_ids or self._im_end_id is None:
            raise AssistantMaskError("The Qwen tokenizer lacks required assistant boundary tokens")

    def __call__(self, features: list[AdapterExample]) -> Any:
        if len(features) != 1:
            raise TrainingDataError(
                f"Frozen micro-batch size is 1, but the collator received {len(features)} examples"
            )
        example = features[0]
        if not isinstance(example, AdapterExample):
            raise TrainingDataError(
                f"Expected AdapterExample, got {type(example).__name__}"
            )
        messages = build_qwen_messages(example)
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            processor_kwargs={"padding": False},
            return_dict=True,
            return_tensors="pt",
            **self.template_options,
        )
        input_ids = batch["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise TrainingDataError(f"Unexpected input_ids shape: {tuple(input_ids.shape)}")
        sequence_length = int(input_ids.shape[1])
        if sequence_length > self.max_length:
            raise TrainingDataError(
                f"{example.sample_id}: sequence length {sequence_length} exceeds max_length="
                f"{self.max_length}; gold answers are never truncated"
            )

        token_ids = input_ids[0].tolist()
        starts = _subsequence_starts(token_ids, self._assistant_header_ids)
        if len(starts) != 1:
            raise AssistantMaskError(
                f"{example.sample_id}: expected one assistant header, found {len(starts)}"
            )
        answer_start = starts[0] + len(self._assistant_header_ids)
        closing = [
            index
            for index in range(answer_start, len(token_ids))
            if token_ids[index] == self._im_end_id
        ]
        if len(closing) != 1 or closing[0] <= answer_start:
            raise AssistantMaskError(
                f"{example.sample_id}: expected one non-empty assistant span ending in <|im_end|>"
            )
        answer_end = closing[0]

        import torch

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[0, answer_start : answer_end + 1] = input_ids[0, answer_start : answer_end + 1]
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)
        if int((labels != IGNORE_INDEX).sum().item()) <= 1:
            raise AssistantMaskError(f"{example.sample_id}: assistant supervision is empty")

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            image_positions = input_ids == image_token_id
            if bool((labels[image_positions] != IGNORE_INDEX).any().item()):
                raise AssistantMaskError("Visual placeholder tokens must never be supervised")
        batch["labels"] = labels
        return batch
