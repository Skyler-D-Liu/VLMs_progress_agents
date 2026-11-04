#!/usr/bin/env python3
"""Earthdial."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from vleo_runtime.localization import localization_prompt
sys.path.insert(0, str(PROJECT / "specialist_models/repos/EarthDial/src"))
sys.path.insert(0, str(PROJECT))
from transformers import AutoTokenizer

from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.model.phi3.modeling_phi3 import Phi3ForCausalLM
from transformers.cache_utils import Cache

_prepare_inputs = Phi3ForCausalLM.prepare_inputs_for_generation


def _prepare_inputs_first_step(self, input_ids, past_key_values=None, attention_mask=None,
                               inputs_embeds=None, **kwargs):
    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
        inputs = _prepare_inputs(self, input_ids, None, attention_mask, inputs_embeds, **kwargs)
        inputs["past_key_values"] = past_key_values
        return inputs
    return _prepare_inputs(self, input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)


Phi3ForCausalLM.prepare_inputs_for_generation = _prepare_inputs_first_step

SAMPLE_DIR = PROJECT / "results/analysis/specialist_reference/samples"
OUT_DIR = PROJECT / "results/analysis/specialist_reference"
CHECKPOINT = PROJECT / "specialist_models/hf/EarthDial_4B_RGB"
FPCD_ROOT = Path("./raw_dataset")
REFDET_PREFIX = "[refer] [hr_rgb_0.5] <image> \n"
FPCD_PREFIX = "[changedet] [hr_rgb_temp_0.5] <image> <image> \n"
FPCD_QUESTION = (
    "Look for changes in the farm ponds between the pre- and post-images. Using the provided "
    "options, select the most accurate description of the detected change.\n"
    "A. Farm Pond Demolished\nB. Farm Pond Wetted\nC. Farm Pond Dried\nD. No Change\n"
    "E. Farm Pond Constructed\nAnswer with the option letter only."
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    target = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width, target_height = image_size * target[0], image_size * target[1]
    blocks = target[0] * target[1]
    resized = image.resize((target_width, target_height))
    columns = target_width // image_size
    tiles = [resized.crop(((i % columns) * image_size, (i // columns) * image_size,
                           (i % columns + 1) * image_size, (i // columns + 1) * image_size))
             for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class Runner:
    def __init__(self, max_num: int):
        self.tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=True, use_fast=False)
        self.model = InternVLChatModel.from_pretrained(
            str(CHECKPOINT), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).eval().cuda()
        config = self.model.config
        self.image_size = config.force_image_size or config.vision_config.image_size
        self.use_thumbnail = config.use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(self.image_size)

    def tiles(self, image: Image.Image) -> torch.Tensor:
        tiles = dynamic_preprocess(image.convert("RGB"), image_size=self.image_size,
                                   use_thumbnail=self.use_thumbnail, max_num=self.max_num)
        return torch.stack([self.transform(tile) for tile in tiles])

    @torch.no_grad()
    def ask(self, images: list[Image.Image], question: str, max_new_tokens: int) -> str:
        pixel_values = [self.tiles(image) for image in images]
        num_patches = [len(value) for value in pixel_values]
        pixel_values = torch.cat(pixel_values).to(torch.bfloat16).cuda()
        config = dict(num_beams=1, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=False)
        return self.model.chat(tokenizer=self.tokenizer, pixel_values=pixel_values, question=question,
                               generation_config=config, num_patches_list=num_patches, verbose=False)


def run_refdet(runner: Runner, refdet_samples: Path) -> Path:
    with refdet_samples.open() as handle:
        samples = [sample for line in handle if line.strip()
                   for sample in [json.loads(line)] if sample.get("source_dataset") == "FAIR1M"]
    if not samples:
        raise ValueError("Ref-Det manifest has no FAIR1M samples")
    path = OUT_DIR / f"earthdial_4b_rgb_{refdet_samples.stem}.jsonl"
    with path.open("w") as handle:
        for sample in samples:
            dataset = sample.get('source_dataset') or sample['task'].split()[0]
            question = REFDET_PREFIX + localization_prompt(
                dataset, sample['image_width'], sample['image_height'], sample['official_prompt'])
            answer = runner.ask([Image.open(sample["image"]).convert("RGB")], question, max_new_tokens=100)
            handle.write(json.dumps({**sample, "model": "earthdial-4b-rgb", "question": question,
                                     "raw_answer": answer}) + "\n")
    return path


def run_fpcd(runner: Runner, data_root: Path) -> Path:
    from finetune.utils.fpcd_data import build_fpcd_bundle
    bundle = build_fpcd_bundle(data_root, strict=True)
    records = list(bundle.records("test"))
    path = OUT_DIR / "earthdial_4b_rgb_fpcd.jsonl"
    question = FPCD_PREFIX + FPCD_QUESTION
    with path.open("w") as handle:
        for record in records:
            pre, post = (Image.open(io.BytesIO(raw)) for raw in bundle.read_image_bytes(record))
            answer = runner.ask([pre, post], question, max_new_tokens=20)
            handle.write(json.dumps({"sample_id": record.source_id, "source_id": record.source_id,
                                     "raw_split": record.raw_split, "raw_index": record.raw_index,
                                     "ground_truth": record.answer, "model": "earthdial-4b-rgb",
                                     "question": question, "raw_answer": answer}) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["refdet", "fpcd"], choices=["refdet", "fpcd"])
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--refdet-samples", type=Path, help="Dataset-generated FAIR1M held-out input manifest")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    if 'refdet' in args.tasks and args.refdet_samples is None:
        parser.error('--refdet-samples is required when evaluating Ref-Det')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.max_num)
    for task in args.tasks:
        path = (run_refdet(runner, args.refdet_samples) if task == "refdet"
                else run_fpcd(runner, args.data_root))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
