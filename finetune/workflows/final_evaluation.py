"""Final evaluation."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from finetune.workflows.task_final_evaluation import MODEL_PREFIX, TaskFinalTestError, _verify_predictions, main as _task_main
MODEL_KEY = f'{MODEL_PREFIX}aerial-animal'
PREDICTIONS = 'predictions.jsonl'
FinalTestError = TaskFinalTestError


def main() -> int:
    return _task_main('aerial-animal')
if __name__ == '__main__':
    raise SystemExit(main())
