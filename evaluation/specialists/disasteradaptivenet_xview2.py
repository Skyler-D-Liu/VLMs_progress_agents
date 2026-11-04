#!/usr/bin/env python3
"""Disasteradaptivenet xview2."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from datasets import Dataset, concatenate_datasets
from PIL import Image
from skimage import measure

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "specialist_models/repos/DisasterAdaptiveNet"
OUT_DIR = PROJECT / "results/analysis/specialist_reference"
DAMAGE_KEYS = {1: "no_damage", 2: "minor_damage", 3: "major_damage", 4: "destroyed"}

EVENT_CONDITIONING = {
    "lower-puna-volcano": "volcano", "palu-tsunami": "earthquake", "mexico-earthquake": "earthquake",
    "socal-fire": "wildfire", "woolsey-fire": "wildfire", "portugal-wildfire": "wildfire",
    "pinery-bushfire": "wildfire", "nepal-flooding": "flood", "midwest-flooding": "flood",
    "moore-tornado": "storm", "joplin-tornado": "storm", "hurricane-florence": "flood",
    "hurricane-harvey": "flood", "hurricane-michael": "storm", "tuscaloosa-tornado": "storm",
    "guatemala-volcano": "volcano", "sunda-tsunami": "earthquake", "santa-rosa-wildfire": "wildfire",
    "hurricane-matthew": "storm",
}
CONDITIONING_KEY = {"wildfire": 0, "volcano": 1, "earthquake": 2, "storm": 3, "flood": 4}

HELD_OUT_EVENTS = {"nepal-flooding", "guatemala-volcano", "sunda-tsunami", "santa-rosa-wildfire",
                   "hurricane-matthew", "tuscaloosa-tornado"}
LOC_THRESHOLD = 0.38


def to_bgr(value) -> np.ndarray:

    image = value if isinstance(value, Image.Image) else Image.open(io.BytesIO(value["bytes"]))
    return np.asarray(image.convert("RGB"))[..., ::-1].astype(np.float32) / 127 - 1


def build_model(weights: Path, device: str) -> torch.nn.Module:
    sys.path.insert(0, str(REPO))

    sys.modules["utils.experiment_manager"] = SimpleNamespace(CfgNode=SimpleNamespace)
    from utils import models
    resnet34 = torchvision.models.resnet34
    models.torchvision.models.resnet34 = lambda weights=None: resnet34(weights=None)
    cfg = SimpleNamespace(MODEL=SimpleNamespace(OUT_CHANNELS=5),
                          DATASET=SimpleNamespace(CONDITIONING_KEY=CONDITIONING_KEY))
    net = models.DisasterAdaptiveNet(cfg)
    state = torch.load(weights, map_location="cpu", weights_only=False)["network"]
    net.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return net.to(device).eval()


@torch.no_grad()
def predict(model, pre: np.ndarray, post: np.ndarray, cond_id: int, device: str) -> dict:
    x = torch.from_numpy(np.concatenate([pre, post], axis=2).transpose(2, 0, 1).copy()).to(device)
    inp = torch.stack([x, torch.flip(x, dims=[1]), torch.flip(x, dims=[2]), torch.flip(x, dims=[1, 2])])
    lookup = torch.tensor([cond_id], device=device).repeat(4)
    y = torch.sigmoid(model(inp, lookup)).cpu()
    preds = torch.stack([y[0], torch.flip(y[1], dims=[1]), torch.flip(y[2], dims=[2]),
                         torch.flip(y[3], dims=[1, 2])]).numpy().mean(axis=0).transpose(1, 2, 0)
    loc = (preds[..., 0] > LOC_THRESHOLD).astype(np.uint8)
    dam = preds[..., 1:].argmax(axis=2) + 1
    labeled, count = measure.label(loc, connectivity=2, background=0, return_num=True)
    counts = {"count_before": int(count), **{key: 0 for key in DAMAGE_KEYS.values()}}
    for region in range(1, count + 1):
        counts[DAMAGE_KEYS[int(np.bincount(dam[labeled == region], minlength=5)[1:].argmax()) + 1]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT / (
        "specialist_models/downloads/disasteradaptivenet/model_weights/disasteradaptivenet_event.pt"))
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.weights, device)
    from evaluation.specialists.inputs import load_inputs, input_images
    samples, inputs = load_inputs('xview2', args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.weights.stem
    path = OUT_DIR / f"{name}_xview2.jsonl"
    with path.open("w") as handle:
        for i, sample in enumerate(samples):
            record = inputs.dataset[int(sample["index"])]
            before, after = input_images(inputs, sample)
            event = record["meta2"]["metadata"]["disaster"]
            counts = predict(model, to_bgr(before), to_bgr(after),
                             CONDITIONING_KEY[EVENT_CONDITIONING[event]], device)
            handle.write(json.dumps({**sample, "model": name, "event": event,
                                     "held_out_event": event in HELD_OUT_EVENTS, "answer_text": json.dumps(counts), "pred": counts}) + "\n")
            if i % 100 == 0:
                print(f"{event} {counts} gt={sample['gt']}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
