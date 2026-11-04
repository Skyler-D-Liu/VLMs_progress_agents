#!/usr/bin/env python3
"""Geochat."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from vleo_runtime.localization import localization_prompt
sys.path.insert(0, str(PROJECT / "specialist_models/repos/GeoChat"))

_mpt = types.ModuleType("geochat.model.language_model.geochat_mpt")
_mpt.GeoChatMPTForCausalLM = _mpt.GeoChatMPTConfig = None
sys.modules["geochat.model.language_model.geochat_mpt"] = _mpt

from geochat.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from geochat.conversation import SeparatorStyle, conv_templates
from geochat.mm_utils import tokenizer_image_token
from geochat.model import GeoChatLlamaForCausalLM
from geochat.model.builder import load_pretrained_model


_from_pretrained = GeoChatLlamaForCausalLM.from_pretrained.__func__
GeoChatLlamaForCausalLM.from_pretrained = classmethod(
    lambda cls, *args, **kwargs: _from_pretrained(cls, *args, ignore_mismatched_sizes=True, **kwargs))

SAMPLES = PROJECT / "results/analysis/specialist_reference/samples/refdet_nonoverlap.jsonl"
OUT_DIR = PROJECT / "results/analysis/specialist_reference"
CHECKPOINT = PROJECT / "specialist_models/hf/geochat-7B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refdet-samples", type=Path, required=True, help="Dataset-generated DOTA held-out input manifest")
    args = parser.parse_args()
    with args.refdet_samples.open() as handle:
        samples = [sample for line in handle if line.strip()
                   for sample in [json.loads(line)] if sample.get("source_dataset") == "DOTA"]
    if not samples:
        raise ValueError("Ref-Det manifest has no DOTA samples")
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(CHECKPOINT), None, "geochat-7B", device_map=None, device="cuda")
    model = model.cuda().eval()
    path = OUT_DIR / f"geochat_7b_{args.refdet_samples.stem}.jsonl"
    with path.open("w") as handle:
        for sample in samples:
            dataset = sample.get('source_dataset') or sample['task'].split()[0]
            question = '[refer] ' + localization_prompt(
                dataset, sample['image_width'], sample['image_height'], sample['official_prompt'])
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(sample["image"]).convert("RGB")
            pixels = image_processor.preprocess(
                [image], crop_size={"height": 504, "width": 504}, size={"shortest_edge": 504},
                return_tensors="pt")["pixel_values"].half().cuda()
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=pixels, do_sample=False, num_beams=1,
                                            max_new_tokens=256, use_cache=True)
            new_tokens = output_ids[:, input_ids.shape[1]:] if output_ids.shape[1] > input_ids.shape[1] \
                and torch.equal(output_ids[:, :input_ids.shape[1]], input_ids) else output_ids
            answer = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            if answer.endswith(stop):
                answer = answer[:-len(stop)].strip()
            handle.write(json.dumps({**sample, "model": "geochat-7b", "question": question,
                                     "raw_answer": answer}) + "\n")
            handle.flush()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
