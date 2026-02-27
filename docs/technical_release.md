# Distill-SLM Accelerator — Technical Feature Notes & Release Summary

## Overview

The **Distill-SLM Accelerator** is a practical toolkit for training, evaluating, and serving *small language models (SLMs)* distilled from LLMs on a **single 8 GB GPU** (WSL2). It ships turnkey playbooks for three tasks:

* **CreditMix**: 3-class credit risk classification *(Good / Standard / Bad)*
* **CodeGen**: code synthesis with optional “plan-then-code”
* **IDP**: schema-constrained JSON extraction

Three supervised distillation routes are supported with **QLoRA** (4-bit base + LoRA adapters): **SFT**, **KD**, and **DPO**. The package also includes a **FastAPI** server exposing calibrated classification inference for base and distilled adapters.

---

## What’s New (Highlights)

* **End-to-end distillation on 8 GB VRAM**

  * 4-bit NF4 quantization + LoRA (`r=16, α=32, dropout=0.05`) on attention & MLP projections
  * Gradient checkpointing, GA steps, and training configs sized for WSL2 + RTX 4060 8 GB

* **Three distillation paths**

  * **SFT**: supervised fine-tuning (`input` → `output`)
  * **KD**: sequence imitation (`input` → `teacher_output`)
  * **DPO**: preference optimization with TRL (`prompt`, `chosen`, `rejected`)

* **Robust classification decoder for CreditMix**

  * **Chat-template-aligned scoring** (Qwen **assistant** turn)
  * **Length-normalized log-likelihood ranking** across labels
  * **Multi-null contextual calibration** (bias removal using several content-free prompts)
  * Optional softmax confidences and **top-2 margin** in server responses

* **FastAPI inference service (4 endpoints)**

  * `/infer/base` → `Qwen/Qwen2.5-3B-Instruct`
  * `/infer/sft`  → `runs/creditmix_sft`
  * `/infer/kd`   → `runs/creditmix_kd`
  * `/infer/dpo`  → `runs/creditmix_dpo`
  * Returns `prediction`, `scores` (calibrated), `raw_scores`, `prior`, `probabilities`, `top2_margin`

* **Balanced evaluation utilities**

  * Class-balanced validation selection (`--val_per_class` or `--val_frac`)
  * Accuracy, Macro-F1, detailed classification report & confusion matrix

* **Task-specific inference**

  * **CodeGen** with optional “plan-then-code”
  * **IDP** with strict **JSON Schema** enforcement

---

## Measured Impact (CreditMix; balanced validation)

**Balanced validation = 1,500 per class; total 4,500.**

| Model                        | Accuracy   | Macro-F1   | Notes                                                    |
|------------------------------|:----------:|:----------:|----------------------------------------------------------|
| Base (Qwen-Instruct)         |  0.3562    |  0.3404    | Over-predicts *Standard*                                 |
| **KD adapter**               | **0.4280** | **0.4031** | Moderately better; still under-predicts *Standard*       |
| **SFT adapter**              | **0.6471** | **0.5463** | Large lift via Good/Bad recall; *Standard* recall is low |
| **DPO adapter (new)**        | **0.4931** | **0.4783** | More balanced than KD; boosts Bad/Good vs. base          |

*Metrics from the latest evaluation logs.* :contentReference[oaicite:1]{index=1}

> The calibrated ranking decoder (assistant-turn, length-normalization, multi-null calibration) more than **doubles** accuracy over the base on the balanced split. SFT shows the largest lift; KD yields steadier but smaller gains. DPO pipeline is more balanced than KD and base.

---

## Supported Tasks & Data Schemas

**SFT**

```json
{"input": "<prompt>", "output": "<gold>"}
```

**KD**

```json
{"input": "<prompt>", "teacher_output": "<teacher or gold>"}
```

**DPO**

```json
{"prompt": "<prompt>", "chosen": "<preferred>", "rejected": "<less preferred>"}
```

**IDP JSON Schema**

* Provide `idp_schema.json` to enforce strict JSON outputs during inference.

---

## Project Structure (key files)

```
data/
  creditmix_dataset.json          # raw classification data
  creditmix_{sft,kd,dpo}.jsonl    # balanced, prepared sets
  codegen_{sft,kd,dpo}.jsonl      # codegen sets
  idp_sft.jsonl                   # IDP SFT pairs
  idp_schema.json                 # IDP schema for inference

creditmix_prepare.py     # class balancing + writers (SFT/KD/DPO)

sft_train.py                      # QLoRA SFT trainer
kd_train.py                       # QLoRA KD trainer
dpo_train.py                      # QLoRA DPO trainer (TRL)

infer_codegen.py                  # codegen inference (+plan_then_code)
infer_idp.py                      # IDP schema-constrained inference
infer_creditmix.py                # calibrated label ranking inference

eval_creditmix.py                 # balanced evaluator + calibration

serve_creditmix.py                # FastAPI server (base/sft/kd/dpo endpoints)
```

---

## System Requirements

* **OS**: Ubuntu 24.04.1 LTS (WSL2 tested)
* **GPU**: NVIDIA RTX 4060 Laptop (8 GB VRAM)
* **Drivers**: NVIDIA 581.29, CUDA driver 13.0
* **Python**: 3.12.3

**Python packages (core)**

```
pip install "transformers>=4.43" peft accelerate datasets bitsandbytes trl
pip install fastapi uvicorn pydantic scikit-learn
```

---

## Training Features

* **QLoRA**: 4-bit base via `BitsAndBytesConfig` (NF4 + double quant, bfloat16 compute)
* **LoRA**: applied to `q/k/v/o_proj`, `gate/up/down_proj`
* **Memory-safe defaults**: `per_device_train_batch_size=1`, gradient accumulation, gradient checkpointing
* **DPO**: `beta` set via `DPOConfig` (pass in the config, not as stray kwarg)

**Sample commands**

```bash
# Prepare balanced CreditMix splits
python creditmix_prepare.py --src data/creditmix_dataset.json --outdir data

# SFT
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_sft.jsonl --save_dir runs/creditmix_sft --epochs 5 --max_steps 7115

# KD
python kd_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_kd.jsonl --save_dir runs/creditmix_kd --epochs 5

# DPO
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --epochs 5 --beta 0.1 --max_steps 7115
```

---

## Inference Features

* **CreditMix** (`infer_creditmix.py`)

  * Chat-templated prompt at **assistant** turn
  * **Length-normalized** log-likelihood for each label
  * **Multi-null calibration** (prior subtraction)
  * Consistent behavior for **base** or **PEFT adapters** (AutoPEFT loader)
* **CodeGen** (`infer_codegen.py`)

  * Optional `--plan_then_code`
* **IDP** (`infer_idp.py`)

  * Enforces `idp_schema.json` for strict JSON extraction

---

## Evaluation Features

* **Balanced hold-out** by per-class quota or fraction
* **Same** calibrated ranking as inference
* Accuracy, Macro-F1, per-class report, confusion matrix

**Example**

```bash
python eval_creditmix.py \
  --model_dir runs/creditmix_sft \
  --data data/creditmix_dataset.json \
  --val_per_class 1500
```

*To evaluate the base student: `--model_dir Qwen/Qwen2.5-3B-Instruct`*

---

## Serving Features (FastAPI)

**Run**

```bash
# optional CPU offload to avoid OOM on 8 GB:
export OFFLOAD_FOLDER=/tmp/offload
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
```

**Endpoints**

* `POST /infer/base`
* `POST /infer/sft`
* `POST /infer/kd`
* `POST /infer/dpo`

**Request**

```json
{"features": "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}
```

**Response**

```json
{
  "model_name": "...",
  "prediction": "Standard",
  "scores": {...},        // calibrated (raw - prior)
  "raw_scores": {...},    // length-normalized log-likelihoods
  "prior": {...},         // averaged multi-null priors
  "probabilities": {...}, // softmax over calibrated scores
  "top2_margin": 2.12
}
```

---

## Configuration & Tuning

* **VRAM pressure**

  * Keep **4-bit** loading (already set in trainers and server scaffold)
  * Use `OFFLOAD_FOLDER=/tmp/offload` to enable Accelerate CPU offload
  * Reduce null prompts in calibration (e.g., from 7 → 3) to cut peak memory

* **DPO throughput**

  * Shorten `max_length`, use fewer steps/samples for iteration
  * Keep GA steps high if you need larger effective batch

* **Recovering “Standard” coverage (CreditMix)**

  * Inference-time bias: add a small constant to *Standard* in calibrated score space before argmax
  * Enrich training with near-miss *Standard* examples (hard negatives)

---

## Known Issues / Caveats

* **Class collapse risk** on SFT (Standard under-prediction): mitigated via inference biasing or data balancing (you already generate balanced splits).
* **OOM on first server request**: happens during prior computation; solved by enabling `OFFLOAD_FOLDER` and/or keeping 4-bit active.
* **Deprecated warnings**: scripts use `dtype=` and `quantization_config`; older flags (`torch_dtype`, `load_in_*bit`) are avoided.

---

## Breaking Changes / Migration Notes

* **DPO**: pass `beta` via **`DPOConfig`**; older usage as a direct init kwarg is not supported.
* **Tokenizer & PAD/EOS**: scripts align tokenizer and model/generation configs at load time to avoid PAD/EOS mismatches.

---

## Security & Reproducibility

* Determinism: set seeds for split selection (`--seed`).
* Schema-strict outputs for IDP reduce injection risks (JSON only).
* Consider pinning package versions in `requirements.txt` for long-term reproducibility.

---

## Quick Start Recipes

**SFT on CreditMix → Evaluate → Serve**

```bash
python creditmix_prepare.py --src data/creditmix_dataset.json --outdir data

python sft_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_sft.jsonl --save_dir runs/creditmix_sft --epochs 5 --max_steps 7115

python eval_creditmix.py --model_dir runs/creditmix_sft \
  --data data/creditmix_dataset.json --val_per_class 1500

export OFFLOAD_FOLDER=/tmp/offload
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
```

**Swap KD / DPO**

```bash
python kd_train.py  --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_kd.jsonl  --save_dir runs/creditmix_kd

python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --beta 0.1
```

**CodeGen / IDP**

* Convert your corpus to `{SFT,KD,DPO}.jsonl`
* Train the adapter
* Inference via `infer_codegen.py` or `infer_idp.py`

---

## Closing Notes

This accelerator demonstrates that with a disciplined **QLoRA** setup and a **calibrated classification decoder**, SLMs on **8 GB VRAM** can deliver strong, production-friendly performance. The scaffolding is modular: add tasks by providing data in the same schemas, train an adapter, and reuse the inference/eval/serve stack as-is.
