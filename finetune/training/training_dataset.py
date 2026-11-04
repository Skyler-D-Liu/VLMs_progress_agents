"""Training dataset."""

from __future__ import annotations

from typing import Sequence

from finetune.datasets import AdapterExample


class AdapterExampleDataset:

    def __init__(self, examples: Sequence[AdapterExample]) -> None:
        if not examples:
            raise ValueError("At least one training example is required")
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> AdapterExample:
        return self._examples[index]
