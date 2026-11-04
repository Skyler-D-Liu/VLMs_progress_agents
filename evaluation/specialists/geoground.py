#!/usr/bin/env python3
"""Geoground."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from datasets import Dataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "specialist_models/repos/GeoChat"))
_mpt = types.ModuleType("geochat.model.language_model.geochat_mpt")
_mpt.GeoChatMPTForCausalLM = _mpt.GeoChatMPTConfig = None
sys.modules["geochat.model.language_model.geochat_mpt"] = _mpt

from geochat.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from geochat.conversation import SeparatorStyle, conv_templates
from geochat.mm_utils import tokenizer_image_token
from geochat.model.builder import load_pretrained_model
from geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower


CLIPVisionTower.clip_interpolate_embeddings = lambda self, **kwargs: None

OUT = PROJECT / "results/analysis/specialist_reference/geoground_dior_rsvg.jsonl"
CHECKPOINT = PROJECT / "specialist_models/hf/GeoGround/llava-v1.5-7b-task-lora-geoground"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    parser.add_argument("--cohort", type=Path, required=True, help="Dataset input manifest for the checkpoint-disjoint DIOR subset")
    args = parser.parse_args()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        str(CHECKPOINT), None, "geochat-geoground-7b", device_map=None, device="cuda")
    model = model.cuda().eval()
    if model.get_vision_tower().vision_tower.vision_model.embeddings.position_embedding.weight.shape[0] != 577:
        raise ValueError("CLIP tower is not at 336 px")
    from evaluation.specialists.inputs import load_inputs, input_image
    queries, inputs = load_inputs('dior-refdet', args.data_root, args.cohort)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for i, query in enumerate(queries):
            image = input_image(inputs, query)
            question = f"[refer] output the bounding box of the <ref>{query['object']['captions']}</ref> in the image."
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).cuda()
            pixels = image_processor.preprocess(
                [image.resize((336, 336))], crop_size={"height": 336, "width": 336},
                size={"shortest_edge": 336}, return_tensors="pt")["pixel_values"].half().cuda()
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=pixels, do_sample=False, num_beams=1,
                                            max_new_tokens=256, use_cache=True)
            new_tokens = output_ids[:, input_ids.shape[1]:] if output_ids.shape[1] > input_ids.shape[1] \
                and torch.equal(output_ids[:, :input_ids.shape[1]], input_ids) else output_ids
            answer = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            if answer.endswith(stop):
                answer = answer[:-len(stop)].strip()
            handle.write(json.dumps({
                **query,
                "image_width": image.width, "image_height": image.height,
                "model": "geoground-7b", "question": question, "raw_answer": answer}) + "\n")
            handle.flush()
            if i < 5 or i % 500 == 0:
                print(f"[{i}] {query['object']['captions']!r} -> {answer!r} gt={query['object']['bbox']}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
