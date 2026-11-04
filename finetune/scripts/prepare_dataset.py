#!/usr/bin/env python3
"""Prepare dataset."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.workflows.dataset_preparation_entrypoint import main


if __name__ == "__main__":
    raise SystemExit(main())
