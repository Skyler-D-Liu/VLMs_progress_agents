#!/usr/bin/env python3
"""Promptadaptovd rareplanes."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT / "specialist_models/repos/PromptAdaptOVD/scripts/groundingdino_baseline"
WEIGHTS = PROJECT / "specialist_models/downloads/promptadaptovd/weights"

PROMPT = "airplane . aeroplane . airliner . aircraft ."

INFERENCE_DEFS = ["LowRankMultiheadAttention", "LowRankTransformerDecoderLayer",
                  "LowRankTransformerEncoderLayer", "load_model", "get_text_dict"]
FINETUNE_DEFS = ["PromptEnhancer", "model_forward_single_image"]


def official_namespace(groundingdino_dir: Path) -> dict:
    sys.path.insert(0, str(groundingdino_dir))
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Optional
    from torch import Tensor
    from groundingdino.models import build_model
    from groundingdino.models.GroundingDINO import ms_deform_attn
    from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
    from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    if not hasattr(ms_deform_attn, "_C"):


        warnings.warn("groundingdino._C missing: using multi_scale_deformable_attn_pytorch on GPU")

        class _TorchMSDA:
            @staticmethod
            def apply(value, shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                return ms_deform_attn.multi_scale_deformable_attn_pytorch(
                    value, shapes, sampling_locations, attention_weights)
        ms_deform_attn.MultiScaleDeformableAttnFunction = _TorchMSDA

    ns = dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Optional=Optional, os=__import__("os"),
              build_model=build_model, SLConfig=SLConfig, clean_state_dict=clean_state_dict,
              NestedTensor=NestedTensor, nested_tensor_from_tensor_list=nested_tensor_from_tensor_list,
              inverse_sigmoid=inverse_sigmoid, groundingdino_dir=str(groundingdino_dir),
              generate_masks_with_special_tokens_and_transfer_map=generate_masks_with_special_tokens_and_transfer_map)
    for filename, names in [("inference_groundingdino.py", INFERENCE_DEFS), ("finetune_groundingdino.py", FINETUNE_DEFS)]:
        tree = ast.parse((REPO / filename).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise RuntimeError(f"definitions missing from {filename}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REPO / filename), "exec"), ns)
    return ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundingdino-dir", type=Path, default=PROJECT / "specialist_models/repos/GroundingDINO")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=0.05, help="official eval threshold")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "raw_dataset")
    args = parser.parse_args()
    output = args.output or (PROJECT / "results/analysis/specialist_reference"
                             / f"promptadaptovd_r{args.rank}_rareplanes_detections.jsonl")

    from evaluation.specialists.inputs import load_inputs, input_image
    rows, inputs = load_inputs('rare-planes', args.data_root)

    ns = official_namespace(args.groundingdino_dir)
    import groundingdino.datasets.transforms as T_dino
    from PIL import Image


    image_transform = T_dino.Compose([
        T_dino.RandomResize([800], max_size=1333),
        T_dino.ToTensor(),
        T_dino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = ns["load_model"]("swint").cuda()
    model.eval()
    ckpt = WEIGHTS / f"enhancer_lr_same{args.rank}_both/rareplanes_enhancer_unseen_p3/gdinoswint_rareplanes_1shot_unseen_enhancer.pth"
    model.enhancer = ns["PromptEnhancer"](args.rank, args.rank, "both").cuda()
    model.enhancer.eval()
    model.enhancer.load_state_dict(torch.load(ckpt, map_location="cuda"))
    prompt = PROMPT
    print(f"checkpoint {ckpt}\nprompt {prompt!r}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), output.open("w") as handle:
        text_dict = ns["get_text_dict"](model, prompt)
        for i, row in enumerate(rows):
            im_pil = input_image(inputs, row)
            width, height = im_pil.size
            im_torch, _ = image_transform(im_pil, None)
            boxes, logits = ns["model_forward_single_image"](
                model, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in text_dict.items()},
                im_torch.cuda(), "enhancer")
            scores = logits.sigmoid()
            keep = scores > args.box_threshold
            boxes, scores = boxes[keep].cpu(), scores[keep].cpu()
            xyxy = [[(b[0] - b[2] / 2) * width, (b[1] - b[3] / 2) * height,
                     (b[0] + b[2] / 2) * width, (b[1] + b[3] / 2) * height] for b in boxes.tolist()]
            handle.write(json.dumps({
                **row, "answer_text": json.dumps({"Answer": len(scores)}), "box_threshold": args.box_threshold,
                "index": row["index"],
                "file_name": row["file_name"],
                "label": int(row["label"]),
                "scores": [round(float(s), 5) for s in scores],
                "boxes_xyxy": [[round(v, 2) for v in box] for box in xyxy],
            }) + "\n")
            if i % 100 == 0:
                print("Evaluation state updated.", flush=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
