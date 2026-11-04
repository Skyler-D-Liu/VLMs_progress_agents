#!/usr/bin/env python3
"""Yolov8x aerial animal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "python-targets/ultralytics-8.1.19"))

from datasets import Dataset, concatenate_datasets
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

OUT = PROJECT / "results/analysis/specialist_reference/yolov8x_p_aerial_animal.jsonl"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT / "specialist_models/ckpt/yolov8x_p_JE-TL19.pt")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    model = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(args.weights),
                                               confidence_threshold=0.25, device="cuda:0")
    from evaluation.specialists.inputs import load_inputs, input_image
    samples, inputs = load_inputs('aerial-animal', args.data_root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for i, sample in enumerate(samples):
            label = len(sample['objects']['bbox'])
            image = input_image(inputs, sample)
            result = get_sliced_prediction(
                image, model, slice_height=640, slice_width=640,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2, perform_standard_pred=True,
                postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.3,
                postprocess_class_agnostic=False, verbose=0)
            detections = [{"category": p.category.name, "score": float(p.score.value),
                           "bbox": [round(float(v), 1) for v in p.bbox.to_xyxy()]} for p in result.object_prediction_list]
            handle.write(json.dumps({**sample, "index": sample["index"], "path": sample.get("path", sample["sample_id"]), "label": label,
                                     "model": "yolov8x-p-je-tl19", "prediction_count": len(detections),
                                     "detections": detections}) + "\n")
            handle.flush()
            if i % 20 == 0:
                print(f"{sample['sample_id']} gt={label} pred={len(detections)}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
