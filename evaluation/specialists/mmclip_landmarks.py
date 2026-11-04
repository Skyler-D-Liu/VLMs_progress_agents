#!/usr/bin/env python3
"""Mmclip landmarks."""

from __future__ import annotations

import argparse
import io
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer

PROJECT = Path(__file__).resolve().parents[2]


OUT_DIR = PROJECT / "results/analysis/specialist_reference"
CKPT_DIR = PROJECT / "specialist_models/MMLandmarks/pretrained_model"
BACKBONE = str(PROJECT / "specialist_models/hf/clip-vit-large-patch14-336")
IMG_SIZE = 336
TEMPLATE = "a satellite image of {}."
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def to_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    return Image.open(value["path"]).convert("RGB")


def preprocess(image: Image.Image) -> torch.Tensor:
    pixels = cv2.resize(np.asarray(image), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR_EXACT)
    pixels = (pixels.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(pixels.transpose(2, 0, 1))


def find_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("weights_best.pth")) or sorted(path.rglob("*.pth"))
    if not matches:
        raise FileNotFoundError(f"No .pth checkpoint under {path}")
    if len(matches) != 1:
        raise ValueError("Select a unique checkpoint explicitly with --checkpoint")
    return matches[0]


@torch.no_grad()
def run(checkpoint: Path, template: str, samples: list[dict], device: str, dataset) -> list[dict]:
    from evaluation.specialists.mmclip_model import MMCLIP
    model = MMCLIP(BACKBONE)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_checkpoint(state)
    with checkpoint.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        checkpoint_hash = digest.hexdigest()
    model = model.to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    rows = []
    for sample in samples:
        record = dataset[int(sample["index"])]
        if record["name"] != sample["name"]:
            raise ValueError(f"Index {sample['index']} resolves to {record['name']!r}")
        pixels = preprocess(to_pil(record["image"]))
        text = tokenizer([template.format(option) for option in sample["options"]],
                         padding=True, truncation=True, max_length=77, return_tensors="pt")
        _, image_features, _, _ = model(satellite=pixels.unsqueeze(0).to(device))
        _, _, text_features, _ = model(text={k: v.to(device) for k, v in text.items()})
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (image_features @ text_features.T).squeeze(0).float().cpu().tolist()
        choice = max(range(len(scores)), key=scores.__getitem__)
        rows.append({**sample, "model": "mmclip-vit-l-14-336", "template": template,
                     "checkpoint_sha256": checkpoint_hash, "scores": scores, "prediction": sample["options"][choice],
                     "is_correct": sample["options"][choice] == sample["name"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR,
                        help="weights_best.pth or a directory containing it")
    parser.add_argument("--template", default=TEMPLATE)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "mmclip_landmarks.jsonl")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from evaluation.utils.dataset_catalog import dataset_rows
    samples = dataset_rows("landmarks", args.data_root)
    import landmarks
    dataset, _ = landmarks.download_and_prepare_dataset("train")
    checkpoint = find_checkpoint(args.checkpoint)
    rows = run(checkpoint, args.template, samples, device, dataset)
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    accuracy = sum(row["is_correct"] for row in rows) / len(rows)
    print(f"MMCLIP ({checkpoint.name}): accuracy={accuracy:.4f} -> {path}")


if __name__ == "__main__":
    main()
