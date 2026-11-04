"""Adapter promotion entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from finetune.config import load_base_lora_config
from finetune.config.runtime_context import active_base_config, active_registry
from finetune.model_backends import get_model_backend
from finetune.workflows.promotion import promote_selected_adapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=active_base_config(PROJECT_ROOT / "finetune/experiment_specs/base_lora.yaml"))
    parser.add_argument("--registry", type=Path, default=active_registry(PROJECT_ROOT / "finetune/experiment_registry.yaml"))
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    base = load_base_lora_config(args.base_config, args.registry)
    get_model_backend(base).validate_config(base)
    promoted = promote_selected_adapter(
        base_config=base,
        selection_path=args.selection,
        final_adapter_dir=args.destination,
    )
    print(promoted, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
