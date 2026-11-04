"""Dataset preparation entrypoint."""
from __future__ import annotations
import sys

def main() -> int:
    task = None
    for index, argument in enumerate(sys.argv[1:]):
        if argument == '--task' and index + 2 <= len(sys.argv[1:]):
            task = sys.argv[index + 2]
            break
        if argument.startswith('--task='):
            task = argument.split('=', 1)[1]
            break
    if task in {'cowc', 'dior-rsvg', 'dior-rsvg-recognition', 'fpcd', 'geobench-refdet', 'landmarks', 'neon-tree', 'pattern-net', 'xview2', 'rare-planes'}:
        from finetune.workflows.task_preparation import main as task_main
        return task_main()
    from finetune.workflows.dataset_preparation import main as aerial_main
    return aerial_main()
