#!/usr/bin/env python3
"""Dino neon."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from torchvision.ops import nms

PROJECT = Path(__file__).resolve().parents[2]
CANOPYRS = PROJECT / "specialist_models/repos/CanopyRS"
OUT = PROJECT / "results/analysis/specialist_reference/dino_swinl_nqos_neon.jsonl"
ARCHITECTURE = "dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"
MIN_SIZE, MAX_SIZE = 1024, 1777
NMS_IOU = 0.7


def load_model(weights: Path):
    sys.path.insert(0, str(CANOPYRS / "detrex"))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate

    cfg = LazyConfig.load(str(CANOPYRS / "detrex/projects" / ARCHITECTURE))
    cfg.model.num_classes = 1
    cfg.model.criterion.num_classes = 1
    model = instantiate(cfg.model)

    import collections
    import typing

    import omegaconf
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    allowed = [omegaconf.ListConfig, omegaconf.DictConfig, ContainerMetadata, Metadata, AnyNode,
               typing.Any, list, dict, int, collections.defaultdict]
    with torch.serialization.safe_globals(allowed):
        DetectionCheckpointer(model).load(str(weights))
    return model.eval().to("cuda")


def resize(image: np.ndarray) -> np.ndarray:
    from detectron2.data.transforms import ResizeTransform

    h, w = image.shape[:2]
    scale = 1.0
    if min(h, w) < MIN_SIZE:
        scale = MIN_SIZE / min(h, w) if max(h, w) * MIN_SIZE / min(h, w) <= MAX_SIZE else MAX_SIZE / max(h, w)
    elif max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
    if scale == 1.0:
        return image
    return ResizeTransform(h, w, int(round(h * scale)), int(round(w * scale))).apply_image(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=PROJECT / "specialist_models/hf/dino-swin-l-384-multi-NQOS/model_best.pth")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    model = load_model(args.weights)
    from evaluation.specialists.inputs import load_inputs, input_image
    samples, inputs = load_inputs('neon-trees', args.data_root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle, torch.no_grad():
        for i, sample in enumerate(samples):
            label = sample['count']
            image = np.asarray(input_image(inputs, sample), dtype=np.uint8)
            resized = resize(image)
            h, w = resized.shape[:2]
            tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
            instances = model([{"image": tensor, "height": h, "width": w}])[0]["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.clone()
            boxes[:, [0, 2]] *= image.shape[1] / w
            boxes[:, [1, 3]] *= image.shape[0] / h
            scores = instances.scores
            keep = torch.zeros(len(scores), dtype=torch.bool)
            keep[nms(boxes, scores, NMS_IOU)] = True
            handle.write(json.dumps({**sample, "answer_text": json.dumps({"count": int(((scores >= 0.5) & keep).sum())}), "index": sample["index"], "path": sample["path"], "label": label,
                                     "model": "dino-swin-l-384-multi-NQOS",
                                     "scores": scores.tolist(), "nms_keep": keep.tolist(),
                                     "boxes": boxes.round().tolist()}) + "\n")
            if i % 50 == 0:
                print(f"gt={label} pred@0.5={int(((scores >= 0.5) & keep).sum())}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
