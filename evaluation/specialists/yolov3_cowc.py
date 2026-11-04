#!/usr/bin/env python3
"""Yolov3 cowc."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "specialist_models/repos/adversarial-yolov3-cowc"
WEIGHTS = PROJECT / "specialist_models/downloads/yolov3_cowc/wacv-code-files/weights/yolov3-cowc-256/yolov3-cowc_best_256.weights"
OUT = PROJECT / "results/analysis/specialist_reference/yolov3_cowc_counting.jsonl"
DATA_ROOT = Path("./raw_dataset/danielz01___cowc-m")
NAMES = ["Not Car", "Other", "Pickup", "Sedan", "Unknown"]
INPUT = 512

sys.path.insert(0, str(REPO))
darknet = types.ModuleType("darknet")
darknet.__file__ = str(REPO / "darknet.py")
source = (REPO / "darknet.py").read_text().split("\n# CHECKER")[0]
source = source.replace("self.seen = int(seen)", "self.seen = int(seen[0])")
exec(compile(source, darknet.__file__, "exec"), darknet.__dict__)
sys.modules["darknet"] = darknet
Darknet = darknet.Darknet
from utils import do_detect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    os.chdir(REPO)
    cfg = (REPO / "cfg/yolov3-cowc.cfg").read_text()
    cfg = cfg.replace("width=256", f"width={INPUT}").replace("height=256", f"height={INPUT}")
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as handle:
        handle.write(cfg)
    model = Darknet(handle.name)
    model.load_weights(str(WEIGHTS))
    stored = sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for name, b in model.named_buffers() if name.endswith(("running_mean", "running_var")))
    if stored != (WEIGHTS.stat().st_size - 20) // 4:
        raise ValueError(f"weights file holds {(WEIGHTS.stat().st_size - 20) // 4} floats, model {stored}")
    model = model.cuda().eval()
    if (model.width, model.height, model.num_classes) != (INPUT, INPUT, len(NAMES)):
        raise ValueError("unexpected network shape")
    from evaluation.specialists.inputs import load_inputs, input_image
    samples, inputs = load_inputs('cowc', args.data_root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as out:
        for i, sample in enumerate(samples):
            city = sample["city"]
            image = input_image(inputs, sample)
            if image.size != (INPUT, INPUT):
                raise ValueError('Unexpected COWC image dimensions')
            with torch.no_grad():
                boxes = do_detect(model, image, 0.05, 0.4, use_cuda=True)
            boxes = [[float(v) for v in box[:6]] + [int(box[6])] for box in boxes]
            out.write(json.dumps({
                **sample, "answer_text": json.dumps({"count": sum(b[4] >= 0.5 and b[6] != 0 for b in boxes)}), "index": sample["index"], "city": city, "label": sample["label"],
                "model": "yolov3-cowc-256", "boxes": boxes,
                "count@0.5": sum(b[4] >= 0.5 and b[6] != 0 for b in boxes)}) + "\n")
            if (i + 1) % 100 == 0:
                print("Evaluation state updated.", flush=True)


if __name__ == "__main__":
    main()
