#!/usr/bin/env python3
"""Aitlas patternnet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import models, transforms

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results/analysis/specialist_reference/aitlas_resnet50_patternnet.jsonl"
LABELS = sorted([
    "airplane", "baseball_field", "basketball_court", "beach", "bridge", "cemetery", "chaparral",
    "christmas_tree_farm", "closed_road", "coastal_mansion", "crosswalk", "dense_residential",
    "ferry_terminal", "football_field", "forest", "freeway", "golf_course", "harbor", "intersection",
    "mobile_home_park", "nursing_home", "oil_gas_field", "oil_well", "overpass", "parking_lot",
    "parking_space", "railway", "river", "runway", "runway_marking", "shipping_yard", "solar_panel",
    "sparse_residential", "storage_tank", "swimming_pool", "tennis_court", "transformer_station",
    "wastewater_treatment_plant"])
TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


@torch.no_grad()
def classify(model, images, device: str) -> list[str]:
    batch = torch.stack([TRANSFORM(image.convert("RGB")) for image in images]).to(device)
    return [LABELS[i] for i in model(batch).argmax(dim=1).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=PROJECT / "specialist_models/ckpt/aitlas_patternnet_resnet50.pth.tar")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    parser.add_argument('--cohort', type=Path, required=True, help='Dataset manifest restricted to the checkpoint official held-out split')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=None, num_classes=len(LABELS))
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=False)["state_dict"])
    model = model.eval().to(device)

    from evaluation.specialists.inputs import load_inputs, input_image
    samples, inputs = load_inputs('pattern-net', args.data_root, args.cohort)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with OUT.open('w') as handle:
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            predictions = classify(model, [input_image(inputs, row) for row in batch], device)
            for row, predicted in zip(batch, predictions):
                category = inputs.module.CATEGORIES_REASSIGNED[predicted]
                label = inputs.module.CATEGORIES_REASSIGNED.get(row['label'], row['label'])
                correct += category == label
                handle.write(json.dumps({**row, 'model':'aitlas-resnet50', 'predicted':predicted,
                                         'answer_text':json.dumps({'category':category})}) + '\n')
    print(f"wrote {OUT}: top-1 accuracy {correct / len(samples):.5f}")


if __name__ == "__main__":
    main()
