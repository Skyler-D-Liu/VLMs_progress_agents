"""Evaluation dispatch."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from finetune.config import load_base_lora_config
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = REPO_ROOT / 'evaluation'
UTILS_DIR = EVALUATION_ROOT / 'utils'
TASKS_DIR = EVALUATION_ROOT / 'tasks'
sys.path.insert(0, str(UTILS_DIR))
from qwen_local_runtime import canonical_lora_model_name, resolve_lora_task
TASK_SCRIPTS: Dict[str, str] = {'aerial-animal': 'aerial_animal.py', 'cowc': 'cowc.py', 'dior-rsvg': 'dior_rsvg_localization.py', 'dior-rsvg-recognition': 'dior_rsvg_recognition.py', 'fpcd': 'fpcd.py', 'geobench-refdet': 'geobench_localization.py', 'landmarks': 'landmarks.py', 'neon-tree': 'neon_trees.py', 'pattern-net': 'pattern_net.py', 'rare-planes': 'rare_planes.py', 'xview2': 'xview2.py'}

def evaluator_script_path(task: str) -> Path:
    try:
        filename = TASK_SCRIPTS[task]
    except KeyError as exc:
        raise ValueError(f'Unknown Qwen LoRA task: {task}') from exc
    script_root = TASKS_DIR
    path = script_root / filename
    if not path.is_file():
        raise FileNotFoundError(f'Authoritative evaluator does not exist: {path}')
    return path

def build_command(args: argparse.Namespace) -> List[str]:
    task = resolve_lora_task(args.adapter)
    worker_flag = '--sample-workers' if task in {'dior-rsvg', 'dior-rsvg-recognition', 'geobench-refdet'} else '--sample-parallel'
    command = [sys.executable, str(evaluator_script_path(task)), '--model', canonical_lora_model_name(task), '--few-shot-k', '0', worker_flag, str(args.sample_parallel)]
    if task == 'geobench-refdet':
        command.extend(('--task', 'refdet'))
    optional_values = (('--output-dir', args.output_dir), ('--cache-dir', args.cache_dir), ('--qwen-model-path', args.qwen_model_path), ('--qwen-finetuned-root', args.qwen_finetuned_root))
    for flag, value in optional_values:
        if value is not None:
            command.extend((flag, str(value)))
    return command

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the matching task's existing evaluator with a Qwen3-VL-32B LoRA adapter.")
    parser.add_argument('--adapter', required=True, help='Task adapter alias or canonical Qwen3-VL-32B-Instruct-LoRA-* model name')
    parser.add_argument('--mode', choices=('standard', 'final'), default='standard', help='Use standard dispatch or the locked final-test workflow.')
    parser.add_argument('--output-dir')
    parser.add_argument('--cache-dir')
    parser.add_argument('--qwen-model-path')
    parser.add_argument('--qwen-finetuned-root')
    parser.add_argument('--base-config', type=Path, help='Optional active model config.')
    parser.add_argument('--backend', choices=('qwen3_vl', 'llava_onevision2'), default='qwen3_vl', help='Model backend for dispatch.')
    parser.add_argument('--sample-parallel', type=int, default=1, help='Local sample concurrency forwarded to scripts that expose it (default: 1)')
    parser.add_argument('--postprocess-existing', action='store_true', help='In final mode, require a complete existing prediction JSONL and only generate metrics and completion artifacts.')
    args = parser.parse_args()
    if args.sample_parallel < 1:
        parser.error('--sample-parallel must be at least 1')
    if args.postprocess_existing and args.mode != 'final':
        parser.error('--postprocess-existing is only valid with --mode final')
    return args

def main() -> int:
    args = parse_args()
    if args.base_config is not None:
        base = load_base_lora_config(args.base_config)
        if base.base_model.model_type.casefold() == 'llava_onevision2':
            args.backend = 'llava_onevision2'
    if args.backend == 'llava_onevision2' and args.mode != 'final':
        raise SystemExit('LLaVA-OneVision-2 evaluation requires --mode final and the canonical backend evaluator; refusing to invoke a Qwen task evaluator')
    if args.mode == 'final':
        try:
            task = resolve_lora_task(args.adapter)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        changed = {'output_dir': args.output_dir, 'cache_dir': args.cache_dir, 'qwen_model_path': args.qwen_model_path, 'qwen_finetuned_root': args.qwen_finetuned_root}
        supplied = [name for name, value in changed.items() if value is not None]
        if supplied or args.sample_parallel != 1:
            raise SystemExit('Final mode uses configured arguments and accepts no runtime overrides')
        if args.backend == 'llava_onevision2':
            from finetune.workflows.llava_final_evaluation import run as run_llava_final
            return run_llava_final(task, base_config=args.base_config)
        if task == 'aerial-animal':
            if args.postprocess_existing:
                raise SystemExit('Aerial Animal uses its dedicated final-evaluation recovery entrypoint')
            from finetune.workflows.final_evaluation import main as final_main
            return final_main()
        from finetune.workflows.task_final_evaluation import main as task_final_main
        return task_final_main(task, postprocess_existing=args.postprocess_existing)
    try:
        command = build_command(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    completed = subprocess.run(command, check=False)
    return completed.returncode
if __name__ == '__main__':
    raise SystemExit(main())
