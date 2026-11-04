#!/usr/bin/env python3
"""Ubdd xview2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from datasets import Dataset, concatenate_datasets
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
UBDD = PROJECT / "specialist_models/repos/ubdd"
OUT = PROJECT / "results/analysis/specialist_reference/ubdd_dino_swin_xview2.jsonl"
CONFIG = UBDD / "models/dino/config/DINO_5scale_UBDD_swin.py"
THRESHOLD = 0.15
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_model(weights: Path):

    sys.path.append(str(UBDD))
    from models.dino.models.registry import MODULE_BUILD_FUNCS
    from models.dino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(CONFIG))
    args.device = "cuda"
    model, _, postprocessors = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False)["model"])
    return model.eval().to("cuda"), postprocessors["bbox"]


def resize(image: Image.Image, size: int = 800, max_size: int = 1333) -> Image.Image:
    w, h = image.size
    small, large = float(min(w, h)), float(max(w, h))
    if large / small * size > max_size:
        size = int(round(max_size * small / large))
    if w < h:
        ow, oh = size, int(size * h / w)
    else:
        oh, ow = size, int(size * w / h)
    return F.resize(image, [oh, ow])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=PROJECT / "specialist_models/ckpt/ubdd/ubdd-dino-swin.pth")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    model, postprocess = load_model(args.weights)
    from evaluation.specialists.inputs import load_inputs, input_images
    samples, inputs = load_inputs('xview2', args.data_root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT
    with out.open("w") as handle, torch.no_grad():
        for i, sample in enumerate(samples):
            image = input_images(inputs, sample)[0]
            tensor = F.normalize(F.to_tensor(resize(image)), MEAN, STD).to("cuda")
            result = postprocess(model(tensor[None]), torch.tensor([[1.0, 1.0]], device="cuda"))[0]
            scores = result["scores"].cpu()
            boxes = result["boxes"].cpu() * torch.tensor([image.width, image.height] * 2)
            count = int((scores > THRESHOLD).sum())
            handle.write(json.dumps({**sample, "model": "ubdd-dino-swin", "answer_text": json.dumps({"count_before": count}), "pred": {"count_before": count},
                                     "scores": scores.tolist(), "labels": result["labels"].cpu().tolist(),
                                     "boxes": boxes.round().tolist()}) + "\n")
            if i % 100 == 0:
                print(f"pred={count} gt={sample['gt']['count_before']}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
