#!/usr/bin/env python3
"""Falcon dior."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import dynamic_module_utils
from transformers.dynamic_module_utils import get_imports

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results/analysis/specialist_reference/falcon_large_dior_rsvg_classification.jsonl"
CHECKPOINT = PROJECT / "specialist_models/hf/Falcon-Single-Instruction-Large"
SYNONYMS = {"railwaystation": "trainstation"}


def norm(text: str) -> str:
    text = re.sub(r"[^a-z]", "", text.lower())
    return SYNONYMS.get(text, text)


def no_flash_attn(filename):
    return [name for name in get_imports(filename) if name != "flash_attn"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    dynamic_module_utils.get_imports = no_flash_attn
    try:
        model = AutoModelForCausalLM.from_pretrained(str(CHECKPOINT), trust_remote_code=True,
                                                     torch_dtype=torch.float16).cuda().eval()
        processor = AutoProcessor.from_pretrained(str(CHECKPOINT), trust_remote_code=True)
    finally:
        dynamic_module_utils.get_imports = get_imports
    from evaluation.specialists.inputs import load_inputs, input_image
    samples, inputs = load_inputs('dior-recognition', args.data_root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with OUT.open("w") as handle:
        for i, sample in enumerate(samples):
            import io
            from PIL import Image
            raw = inputs.module.dior.get_raw_image_bytes(inputs.dataset[sample['row_index']]['image'])
            image = Image.open(io.BytesIO(raw)).convert('RGB')
            x1, y1, x2, y2 = sample["bbox"]
            bins = [min(999, max(0, int(v // (size / 1000))))
                    for v, size in ((x1, image.width), (y1, image.height), (x2, image.width), (y2, image.height))]
            prompt = "Classify the region of " + "".join(f"<{b}>" for b in bins).join(("<box>", "</box>")) \
                + ".\nUse one or a few words."
            encoded = processor(text=prompt, images=image, return_tensors="pt")
            with torch.inference_mode():
                generated = model.generate(input_ids=encoded["input_ids"].cuda(),
                                           pixel_values=encoded["pixel_values"].cuda().half(),
                                           max_new_tokens=64, num_beams=3, do_sample=False)
            answer = processor.batch_decode(generated, skip_special_tokens=False)[0]
            answer = re.sub(r"</?s>|</?pad>", "", answer).strip()
            matches = [label for label in inputs.module.CATEGORIES if norm(label) == norm(answer)]
            answer_text = json.dumps({'category': matches[0]}) if len(matches) == 1 else answer
            hit = norm(answer) in (norm(sample["raw_category"]), norm(sample["label"]))
            correct += hit
            handle.write(json.dumps({
                **sample, "path": sample["path"], "bbox": sample["bbox"],
                "label": sample["label"], "raw_category": sample["raw_category"],
                "model": "falcon-single-instruction-large", "prompt": prompt,
                "raw_answer": answer, "answer_text": answer_text, "correct": hit}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"accuracy {correct / (i + 1):.4f}", flush=True)
    print(f"accuracy {correct / len(samples):.4f}")


if __name__ == "__main__":
    main()
