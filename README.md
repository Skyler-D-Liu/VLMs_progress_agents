# Tracking Progress in General-Purpose VLMs and Agents for Earth Observation

This repository contains the official code for the paper *Tracking Progress in General-Purpose VLMs and Agents for Earth Observation*.

Most existing studies compare vision-language models (VLMs) at a single point in time. This project conducts a **longitudinal evaluation**: it tracks how the Earth observation (EO) capabilities of general-purpose VLMs evolve across successive model generations under a single, unified evaluation protocol. The repository covers:

1. **Longitudinal benchmarking (primary goal).** Track the zero-shot performance of **48 general-purpose VLMs** from the **GPT, Gemini, Qwen, and LLaVA** families, released between **2023 and 2026**, on **13 EO tasks** comprising **7,273 test samples**. Results are organized by public release date.
2. **Inference and adaptation strategies.** Assess four-shot prompting, chain-of-thought (CoT) prompting, varying reasoning effort, and task-specific QLoRA fine-tuning.
3. **Agent-based workflows.** Compare the standard API with autonomous and human-guided agents built on the OpenAI Agents API, holding the underlying model (GPT-6 Astra) fixed.
4. **Comparison with specialized EO models.** Use one specialized EO model per task as a reference point for zero-shot GPT-6 Astra.
5. **Statistical analysis.** Compute bootstrap confidence intervals, paired significance tests, and temporal trend analyses.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Benchmark Tasks](#benchmark-tasks)
- [Evaluation Metrics and Overall Score](#evaluation-metrics-and-overall-score)
- [Evaluation Subsets](#evaluation-subsets)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Reproducing the Experiments](#reproducing-the-experiments)
  - [1. Longitudinal Zero-Shot Evaluation](#1-longitudinal-zero-shot-evaluation)
  - [2. Few-Shot and Chain-of-Thought Prompting](#2-few-shot-and-chain-of-thought-prompting)
  - [3. Reasoning Effort](#3-reasoning-effort)
  - [4. Task-Specific Fine-Tuning](#4-task-specific-fine-tuning)
  - [5. Agent-Based Workflows](#5-agent-based-workflows)
  - [6. Comparison with Specialized EO Models](#6-comparison-with-specialized-eo-models)
  - [7. Statistical Analysis](#7-statistical-analysis)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [License](#license)
- [Citation](#citation)

---

## Repository Structure

```
.
├── evaluation/               # Benchmark evaluation
│   ├── tasks/                #   Per-task evaluation entry points
│   ├── experiments/          #   Open-weight VLM, prompting-strategy, and reasoning-effort experiments
│   ├── specialists/          #   Specialized EO model baselines
│   ├── analysis/             #   Prediction export, paired bootstrap, and trend analysis
│   └── utils/                #   Shared runtime, prompting, parsing, and scoring utilities
├── finetune/                 # QLoRA fine-tuning framework
│   ├── config/               #   Configuration schema and loader
│   ├── datasets/             #   Task-specific dataset adapters
│   ├── model_backends/       #   Qwen3-VL and LLaVA-OneVision-2 backends
│   ├── training/             #   Trainer, checkpointing, and validation callbacks
│   ├── workflows/            #   Preparation, successive halving, selection, and final evaluation
│   ├── experiment_specs/     #   Model and dataset YAML specifications
│   ├── experiment_registry.yaml
│   └── scripts/              #   Command-line entry points
├── agents_benchmark/         # Agent-based workflows (standard API / autonomous / human-guided)
├── vleo_runtime/             # Lightweight API clients and localization helpers
├── scripts/                  # Auxiliary scripts (e.g., specialist checkpoint download)
├── specialist_models/        # Location for specialist checkpoints (populated at setup time)
├── requirements/             # Dependency sets for each environment
├── tests/                    # Unit tests
└── pyproject.toml
```

---

## Benchmark Tasks

The benchmark builds on the RGB-based tasks of VLEO-Bench and GEOBench-VLM. It comprises **13 tasks on 11 public datasets**, organized by the spatial granularity of the required output: coarse-grained recognition, medium-grained counting, and fine-grained change detection and localization.

| Granularity | Category | Task | Metric | Entry point (task ID) |
|---|---|---|---|---|
| Coarse | Object classification | DIOR-RSVG Object Classification | Accuracy | `dior-recognition` |
| | Scene understanding | PatternNet Scene Recognition | Accuracy | `pattern-net` |
| | | Landmarks Recognition | Accuracy | `landmarks` |
| Medium | Object counting | RarePlanes Aircraft Counting | r², MAE, RMSE | `rare-planes` |
| | | COWC Car Counting | r², MAE, RMSE | `cowc` |
| | | xView2 Pre-disaster Building Counting | r², MAE, RMSE | `xview2` |
| | | NEON Tree Counting | r², MAE, RMSE | `neon-trees` |
| | | Aerial Animal Counting | r², MAE, RMSE | `aerial-animal` |
| Fine | Change detection | xView2 Change Detection (building damage) | Macro r², MAE, RMSE | `xview2` |
| | | FPCD Farm Pond Change Detection | Accuracy | `fpcd` |
| | Object localization | DIOR-RSVG Localization | mIoU | `dior-refdet` |
| | | DOTA Localization | mIoU | `geobench-refdet` |
| | | FAIR1M Localization | mIoU | `geobench-refdet` |

The 13 tasks are served by 11 entry points: the `xview2` entry point produces both the pre-disaster building counts and the per-category damage counts, and the `geobench-refdet` entry point evaluates the DOTA and FAIR1M localization tasks, which are scored separately by source dataset.

---

## Evaluation Metrics and Overall Score

Each task uses one primary metric suited to its output type. Model responses are parsed into labels, counts, or bounding boxes using task-specific rules before metrics are computed.

- **Accuracy** is used for object classification, scene understanding, and FPCD farm pond change detection.
- **mIoU** (mean Intersection over Union) is used for localization.
- **r²** is the primary metric for the five counting tasks. It is the **squared Pearson correlation** between predicted and ground-truth counts, $r^2 = \mathrm{corr}(y, \hat{y})^2$, which differs from the coefficient of determination (e.g., `sklearn.metrics.r2_score`). It is bounded in $[0, 1]$ and measures how well predictions track variation in object counts across images.
- **Macro r²** is the primary metric for xView2 change detection: r² is computed separately for each of the four damage categories (no damage, minor damage, major damage, destroyed) and then averaged with equal weights.
- **MAE and RMSE** are additionally reported for the five counting tasks and xView2 change detection, computed from the same predictions and parsing rules as r². For a task with $n$ test images, ground-truth counts $y_i$, and predicted counts $\hat{y}_i$:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert \hat{y}_i - y_i \rvert, \qquad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (\hat{y}_i - y_i)^2}
$$

  For xView2 change detection, MAE and RMSE are computed for each of the four damage categories and then averaged, as for macro r².

**Overall EO score.** Each task's primary metric is clipped to $[0, 1]$, and the overall score of model $m$ is the equally weighted average over its benchmark tasks $T_m$:

$$
S(m) = 100 \times \frac{1}{|T_m|} \sum_{t \in T_m} s(m, t)
$$

where $s(m, t)$ is the normalized score of model $m$ on task $t$. Equal weighting ensures that each task contributes equally to the overall score, regardless of its sample size. As robustness checks, the analysis code also supports category-balanced weighting and MAE- and RMSE-based variants of the overall score. In these variants, the six r² task scores are replaced by skill scores relative to the best constant prediction, $s_{\mathrm{MAE}} = \max(0, 1 - \mathrm{MAE}/\mathrm{MAE}_{\mathrm{ref}})$ and $s_{\mathrm{RMSE}} = \max(0, 1 - \mathrm{RMSE}/\mathrm{RMSE}_{\mathrm{ref}})$, where $\mathrm{MAE}_{\mathrm{ref}}$ and $\mathrm{RMSE}_{\mathrm{ref}}$ are the errors obtained by predicting the median and the mean test-set count, respectively, for every image.

---

## Evaluation Subsets

All experiments share the same task definitions, prompts, parsing rules, and canonical dataset partitions. Some experiments evaluate on predefined test subsets to control inference cost or to avoid overlap with specialist training data:

| Experiment | Evaluation data |
|---|---|
| Longitudinal zero-shot evaluation (48 models) | Full benchmark test sets (7,273 samples) |
| Few-shot / CoT / reasoning effort | Test samples shared across the settings compared for each model; four-shot demonstrations are drawn from each dataset's training partition and never use a test image |
| Task-specific fine-tuning | Training and validation partitions for learning-rate selection; one final evaluation on the test set per task |
| Agent-based workflows | Stratified random sample of up to 100 test samples per task, covering all 13 tasks; identical samples for all three conditions |
| Comparison with specialized EO models | Identical samples for GPT-6 Astra and the specialist; held-out subsets for PatternNet, DIOR-RSVG, DOTA, and FAIR1M to avoid training–test overlap |

---

## Installation

The code requires **Python ≥ 3.10**. Because the evaluated models have partially conflicting dependencies, we provide separate requirement sets and recommend one virtual environment per use case.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

| Use case | Requirements |
|---|---|
| API and open-weight VLM evaluation | `requirements/base.txt` |
| CUDA 12.8 PyTorch wheels (optional, install first) | `requirements/qwen_cu128.txt` |
| QLoRA fine-tuning | `requirements/finetune_qwen3.txt` |
| Agent-based workflows | `requirements/agents_benchmark.txt` |
| MMCLIP landmark specialist | `requirements/mmclip.txt` |
| Legacy Qwen-VL-Chat–based specialists | `requirements/qwen_vl_chat_legacy.txt` |

For example, the main evaluation environment can be created with:

```bash
pip install -r requirements/qwen_cu128.txt
pip install -r requirements/base.txt
```

**Hardware.** Open-weight inference was performed on NVIDIA L40S and H200 GPUs (the largest models use multiple H200 GPUs); fine-tuning uses one H200 GPU per trial.

---

## Data Preparation

All datasets are publicly available. By default, the code expects raw data under `./raw_dataset/`. Most task scripts download and cache their data automatically from Hugging Face on first use; datasets can also be pre-fetched explicitly, e.g.:

```bash
python evaluation/tasks/landmarks.py --download-only
python evaluation/tasks/dior_rsvg_localization.py --download-only --download-all-splits
python evaluation/tasks/geobench_localization.py --prepare-data
```

Canonical train/validation/test cohorts are generated deterministically with:

```bash
python finetune/scripts/create_canonical_cohorts.py --data-root raw_dataset
```

Please refer to the original dataset licenses and terms of use before downloading.

---

## Configuration

API credentials are read from a `.env` file at the repository root (this file is git-ignored):

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...        # GOOGLE_API_KEY is also accepted
HF_TOKEN=...              # required for gated Hugging Face models
```

Optional environment variables control cache locations (`HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TORCH_HOME`) and local model paths (`LLAVA_MODEL_PATH`, `QWEN_FINETUNED_ROOT`).

---

## Reproducing the Experiments

All outputs are written to `./results/` by default. Most task scripts support `--list-models` to show the available models and `--evaluate-only <file.jsonl>` to re-score saved predictions without issuing new requests.

### 1. Longitudinal Zero-Shot Evaluation

All 48 models are evaluated **zero-shot** (no in-context demonstrations) with **each model's default reasoning-effort setting**, using identical task definitions, prompts, and output requirements. These settings are the defaults of every task script.

Proprietary models (GPT and Gemini) are evaluated through the per-task entry points in `evaluation/tasks/`:

```bash
# Evaluate one model on one task
python evaluation/tasks/pattern_net.py --model gpt-5.5

# Evaluate all models of a family
python evaluation/tasks/xview2.py --model Gemini-all

# Localization on DOTA and FAIR1M
python evaluation/tasks/geobench_localization.py --model <model>

# Re-score saved predictions and generate comparison plots
python evaluation/tasks/cowc.py --plot-only
```

Open-weight models (Qwen and LLaVA) are downloaded and evaluated on all tasks with:

```bash
python evaluation/experiments/download_open_vlm_models.py --model Qwen3.5-27B
python evaluation/experiments/open_vlm.py --model Qwen3.5-27B
```

Per-model overall scores are then ordered by public release date to produce the longitudinal trends (see [Statistical Analysis](#7-statistical-analysis)).

### 2. Few-Shot and Chain-of-Thought Prompting

Four-shot and CoT prompting are evaluated on one representative model per family: **GPT-6.1 Sol, Gemini 3.8 Flash, Qwen3-VL-32B, and LLaVA-OneVision-2-8B**. Four-shot prompting prepends four image–question–answer demonstrations drawn from the dataset's training partition (excluding test images); CoT prompting asks the model for image-based reasoning before its final answer. Both are available in the task scripts via `--few-shot-k` and `--reasoning-method`, which are used for the open-weight models. A dedicated driver runs the study for the two API models:

```bash
python evaluation/experiments/inference_strategy.py --task <task> --model <model> --few-shot-k 4
python evaluation/experiments/inference_strategy.py --task <task> --model <model> --reasoning-method cot
```

### 3. Reasoning Effort

GPT-6.1 Sol (`low`, `medium`, `high`, `xhigh`, `max`) and Gemini 3.8 Flash (`low`, `medium`, `high`) are evaluated at each available reasoning-effort level, with the model and protocol otherwise fixed. Task scores, token usage (including reasoning tokens), and response time are recorded for every request.

```bash
python evaluation/experiments/reasoning_comparison.py --task <task> --model gpt-6.1-sol --reasoning-effort high
python evaluation/experiments/reasoning_comparison.py --task <task> --model gemini-3.8-flash --reasoning-effort low
```

### 4. Task-Specific Fine-Tuning

We fine-tune **Qwen3-VL-32B** and **LLaVA-OneVision-2-8B** with QLoRA (4-bit NF4 with double quantization, bfloat16 computation, one H200 GPU), training a separate adapter for each model–task pair. Hyperparameters are fixed in `finetune/experiment_specs/`:

- LoRA rank 8, alpha 32, dropout 0.05, applied only to the attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) of every language-model decoder layer; the vision encoder and multimodal projector stay frozen without LoRA adapters.
- Effective batch size 8 (microbatch 1, 8 gradient-accumulation steps), gradient checkpointing, maximum multimodal sequence length 16,384 tokens.
- Each training example is one conversation with the zero-shot system message and user prompt; the loss is computed only on the assistant answer tokens.

**Learning-rate selection** (the only hyperparameter selected on validation data) uses successive halving on a fixed subset of at most 256 validation samples (the first 256 in a deterministic ordering). The candidates 1e-5, 2e-5, and 5e-5 are trained for 32 optimizer steps; the top two continue to 64 steps and the top one to 128 steps. If the winner is 1e-5 or 5e-5, an additional 5e-6 or 1e-4 candidate is trained for 128 steps and compared with the incumbent. Candidates are ranked by the task's primary validation metric (accuracy, mIoU, or r² for counting tasks; the xView2 adapter, which serves both xView2 tasks, uses the equally weighted mean of building-counting r² and change-detection macro r²). Ties are broken by the lower parse-failure/refusal rate, earlier epoch, and lower learning rate.

**Final training** restarts from the pretrained model and trains a new adapter for one epoch with a cosine schedule spanning that epoch, on a deterministic stratified subset of at most 8,192 training samples (≤ 1,024 optimizer steps). The single end-of-epoch checkpoint is evaluated once on the full validation partition and used for the test evaluation without further checkpoint selection. Each task's test set is evaluated once.

```bash
# 1. Prepare task data
python finetune/scripts/prepare_dataset.py --task cowc

# 2. Select the input resolution (based on GPU memory only)
python -m finetune.workflows.resolution_selection --task cowc

# 3a. Successive-halving screening: rung 1 (32 steps) for 1e-5, 2e-5, 5e-5
python -m finetune.workflows.successive_halving_training --stage screening --task cowc --learning-rate 2e-5 --seed 0 --rung 1

# 3b. Rank the rungs; prints the next screening commands (rung 2/3 survivors and,
#     for a boundary winner, the 128-step expansion candidate) until selection completes
python -m finetune.workflows.successive_halving_training --stage select --task cowc --seed 0

# 3c. One-epoch final training with the selected learning rate (writes selection.json)
python -m finetune.workflows.successive_halving_training --stage final --task cowc --learning-rate <selected> --seed 0

# 4. Promote the final adapter
python finetune/scripts/promote_adapter.py --selection <final training dir>/selection.json --destination finetuned_qwen/<adapter>

# 5. Final test-set evaluation
python finetune/scripts/evaluate_lora.py --adapter <adapter> --mode final
```

### 5. Agent-Based Workflows

The `agents_benchmark` package compares three conditions that all use **GPT-6 Astra** as the underlying model:

| Condition | Description |
|---|---|
| Standard API (`direct`) | Direct inference through the standard API with the zero-shot prompts |
| Autonomous Agents API (`autonomous`) | The same prompts, run by an agent that plans its own steps, executes code, works with files, and verifies its answers |
| Human-guided Agents API (`guided`) | The autonomous setting plus predefined, task-specific guidance describing how a human would approach each task |

Experimental controls:

- Agents run on the **OpenAI Agents API** in an OpenAI-hosted environment.
- **Network access is disabled** in the hosted environment, and the instructions explicitly prohibit web searching and access to external data, reference answers, or annotation files.
- **All 13 tasks** are evaluated. Within each task, samples are grouped by their ground-truth annotation (e.g., class label or object count), and **up to 100 test samples per task** are drawn by proportional stratified random sampling. Tasks with fewer than 100 test samples use all of them.
- All three conditions are evaluated on **exactly the same samples**, enabling paired comparisons.

```bash
python -m agents_benchmark init                       # create the credential file
python -m agents_benchmark prepare --max-per-task 100 --seed 0
python -m agents_benchmark run --arm direct
python -m agents_benchmark run --arm autonomous
python -m agents_benchmark run --arm guided
python -m agents_benchmark score
python -m agents_benchmark.statistics --draws 2000    # bootstrap CIs and paired comparisons
```

### 6. Comparison with Specialized EO Models

Zero-shot GPT-6 Astra is compared with one specialized EO model per task, using the checkpoints released by the original authors. Both are evaluated on identical test samples; held-out subsets are used for PatternNet, DIOR-RSVG, DOTA, and FAIR1M to avoid overlap with specialist training data.

This comparison uses the specialists as **task-specific reference points** for zero-shot general-purpose performance. It is not a controlled architecture comparison: except for EarthDial, every specialist was trained on the training set of its evaluation dataset. No publicly released model has been trained on FPCD, so EarthDial, which was trained on related change-detection datasets (e.g., LEVIR-MCI and xBD), is used for that task.

| Task | Specialized EO model | Script |
|---|---|---|
| DIOR-RSVG Object Classification | Falcon | `evaluation/specialists/falcon_dior.py` |
| PatternNet Scene Recognition | AiTLAS ResNet50 | `evaluation/specialists/aitlas_patternnet.py` |
| Landmarks Recognition | MMCLIP | `evaluation/specialists/mmclip_landmarks.py` |
| RarePlanes Aircraft Counting | PromptAdaptOVD | `evaluation/specialists/promptadaptovd_rareplanes.py` |
| COWC Car Counting | YOLOv3 | `evaluation/specialists/yolov3_cowc.py` |
| xView2 Pre-disaster Building Counting | U-BDD++ | `evaluation/specialists/ubdd_xview2.py` |
| NEON Tree Counting | DINO-Swin-L | `evaluation/specialists/dino_neon.py` |
| Aerial Animal Counting | YOLOv8x | `evaluation/specialists/yolov8x_aerial_animal.py` |
| xView2 Change Detection | DisasterAdaptiveNet | `evaluation/specialists/disasteradaptivenet_xview2.py` |
| FPCD Farm Pond Change Detection | EarthDial | `evaluation/specialists/earthdial.py --tasks fpcd` |
| DIOR-RSVG Localization | GeoGround | `evaluation/specialists/geoground.py` |
| DOTA Localization | GeoChat | `evaluation/specialists/geochat.py` |
| FAIR1M Localization | EarthDial | `evaluation/specialists/earthdial.py --tasks refdet` |

Specialist input manifests are exported with `evaluation/specialists/samples.py`, and the training-disjoint localization subsets are built with `evaluation/specialists/refdet_nonoverlap.py`. The MMCLIP checkpoint can be downloaded with `bash scripts/download_specialist_models.sh mmclip`; the remaining checkpoints are available from their official releases and should be placed under `specialist_models/`.

### 7. Statistical Analysis

```bash
# Export aligned per-sample predictions for several models
python evaluation/analysis/export_predictions.py --task cowc \
    --prediction modelA=predA.jsonl --prediction modelB=predB.jsonl --output paired_input.json

# Bootstrap confidence intervals and paired comparisons
python evaluation/analysis/paired.py --input paired_input.json --output paired.json --draws 2000

# Longitudinal trends of the overall score across release dates
python evaluation/analysis/trends.py --score modelA=scoreA.json --score modelB=scoreB.json \
    --metadata models.csv --output trends.json
```

`models.csv` lists `model,family,release_date` for every evaluated model.

---

## Testing

```bash
python -m unittest discover -s tests
```

---

## Reproducibility

- **Unified protocol.** All models share the same task definitions, prompts, output requirements, and parsing rules.
- **Shared retry rule.** Every setting (zero-shot, four-shot, CoT, reasoning effort, fine-tuned, and Agents API) allows up to five attempts per sample (`evaluation/utils/response_protocol.py`; `agents_benchmark/cli.py` for the three agent-comparison arms). A transient request failure that returns no model response is retried with the same prompt; a response whose final answer fails format validation is followed by a repair prompt that specifies only the answer format, without ground truth or correctness feedback. The first valid response is accepted; otherwise the task's failure rules apply. For CoT, validation applies to the final answer, and a missing reasoning block alone does not trigger a retry.
- **DOTA and FAIR1M partitions.** GEOBench-VLM Ref-Det samples are divided 80:10:10 (image-disjoint, per source dataset, seed 0) into training, validation, and test partitions (`evaluation/tasks/create_geobench_refdet_manifest.py`). The four-shot demonstrations are one fixed set of four training samples with distinct images, and fine-tuning uses the same partitions.
- **Deterministic sampling.** Dataset subsets, few-shot demonstrations, agent-evaluation samples, and bootstrap resampling are controlled by explicit seeds (default `0`).
- **Leakage control.** Fine-tuning selects configurations without the test split, specialist comparisons use training-disjoint held-out subsets, and agents run with network access disabled.
- **Resumable runs.** Long-running evaluations write predictions incrementally to JSONL and can be resumed safely.
- **Auditable outputs.** Raw model responses are stored alongside parsed predictions, so all reported scores can be recomputed with `--evaluate-only`.

---

## License

The code in this repository will be released under an open-source license upon publication. Datasets and third-party model checkpoints remain subject to their original licenses.

## Citation

Citation information will be provided upon publication.
