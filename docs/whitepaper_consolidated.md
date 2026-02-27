# Distill-SLM Accelerator — End-to-End Whitepaper (Consolidated Edition)

**Scope.** This document consolidates the business use cases, technical release features, empirical results, and system designs for the Distill-SLM accelerator. It describes how to distill LLM capabilities into **Small Language Models (SLMs)** that train, evaluate, and serve on a **single 8 GB GPU (WSL2)** using **QLoRA** across three task families:

* **CreditMix**: 3-class credit risk classification (*Good / Standard / Bad*)
* **CodeGen**: code synthesis (optional plan-then-code prompting)
* **IDP**: schema-constrained JSON extraction

---

## 1) Executive Summary

* **Goal:** Make high-quality NLP/LLM capabilities feasible on commodity hardware with **8 GB VRAM**, preserving task quality via **adapter-level finetuning** (QLoRA) and robust **inference decoders**.
* **Methods:** Three supervised distillation routes

  * **SFT** – supervised finetuning on labeled I/O
  * **KD** – sequence-level imitation of a teacher (or gold)
  * **DPO** – preference optimization from (chosen, rejected) pairs
* **Results (CreditMix; balanced validation):**

  * **Base** (`Qwen/Qwen2.5-3B-Instruct`): Accuracy **0.3562**, Macro-F1 **0.3404**
  * **KD** adapter: Accuracy **0.4280**, Macro-F1 **0.4031**
  * **SFT** adapter: Accuracy **0.6471**, Macro-F1 **0.5463**
    The lift is driven by a **calibrated label-ranking decoder** (assistant-turn chat template + **length-normalized log-likelihood** + **multi-null prior subtraction**).
* **Deliverables:** Training scripts (SFT/KD/DPO), inference scripts (CodeGen/IDP/CreditMix), calibrated evaluator (balanced hold-out), and a **FastAPI** server with endpoints for base/SFT/KD/DPO models.

---

## 2) Business Value & Use Cases

### 2.1 Cross-industry “direct” applications

* **Classification (CreditMix-style):**
  Risk tiers, ticket severity, customer health, moderation levels, prioritization queues.
* **IDP (Schema-constrained):**
  Invoices/POs, receipts, claims, contracts/NDAs, HR forms, shipping docs, field reports.
* **CodeGen (Plan-then-Code optional):**
  Internal tools, migration helpers, boilerplate scaffolds, data/ops scripts.

### 2.2 “Indirect” and composable workflows

* Underwriting worklists, dispute routing, SLA event triage, catalog cleanup, ETL scaffolding, governance reports—composed by chaining **IDP → Classify → Act** or **Plan → Generate → Test**.

### 2.3 Value proposition

* **Lower TCO** (single 8 GB GPU), **privacy** (on-prem adapters), **fast iteration** (SFT/KD/DPO as data allows), **operational robustness** (calibrated classification), **API-ready** serving.

---

## 3) System Architecture (Narrative)

* **Data layer.**

  * Raw inputs (e.g., `creditmix_dataset.json`, code prompts/snippets, IDP documents & `idp_schema.json`).
  * Prepared JSONL per method: **SFT** `{input, output}`, **KD** `{input, teacher_output}`, **DPO** `{prompt, chosen, rejected}`.
  * A helper script **balances** CreditMix classes and emits SFT/KD/DPO JSONL.

* **Training layer (QLoRA on 8 GB).**

  * Base models: `Qwen/Qwen2.5-3B-Instruct` (classification/IDP) and `Qwen/Qwen2.5-Coder-3B` (CodeGen).
  * **4-bit** NF4 quantization + **LoRA** (`r=16, α=32, dropout=0.05`) on attention & MLP projections.
  * Gradient checkpointing, accumulation, per-device batch=1.

* **Inference layer.**

  * **CreditMix**: calibrated label-ranking decoder (assistant-turn prompt, length-normalized log-likelihood across labels, **multi-null prior** subtraction; argmax calibrated score).
  * **IDP**: strict **JSON Schema** output (schema-guided decoding & validation).
  * **CodeGen**: direct or **plan-then-code** prompting.

* **Evaluation layer.**

  * Balanced validation split (by per-class quota or fraction).
  * Uses the **same** calibrated decoder as inference; reports accuracy, macro-F1, per-class report, confusion matrix.

* **Serving layer (FastAPI).**

  * Endpoints: `/infer/base|sft|kd|dpo`.
  * Lazy model/tokenizer load and **prior cache**; optional **CPU offload** to tame first-call VRAM spikes; returns **prediction**, **calibrated scores**, **raw scores**, **priors**, **probabilities**, **top-2 margin**.

> **Note on diagrams:** Architecture, data-flow, and interaction Mermaid files are provided separately (parser-safe, ASCII): `ARCHITECTURE.mmd`, `DATA_FLOW.mmd`, `INTERACTION.mmd`.

---

## 4) Distillation Methods

### 4.1 Supervised Fine-Tuning (SFT)

* **Schema:** `{"input": "<prompt>", "output": "<gold>"}`
* **Loss:** CE on `output` tokens (teacher forcing).
* **When to use:** Reliable labels/targets; quickest strong baseline.
* **Observed effect (CreditMix):** Largest overall lift; risk of **class collapse** if data skewed—mitigated by balancing + calibrated inference.

### 4.2 Knowledge Distillation (KD)

* **Schema:** `{"input": "<prompt>", "teacher_output": "<teacher_or_gold>"}`
* **Loss:** CE on `teacher_output` tokens.
* **When to use:** You have a (larger/better) teacher to compress; or reuse gold as “teacher” to shape style.
* **Observed effect:** Moderate improvements; often more balanced than SFT but smaller gains.

### 4.3 Direct Preference Optimization (DPO)

* **Schema:** `{"prompt": "<p>", "chosen": "<better>", "rejected": "<worse>"}`
* **Training:** TRL `DPOTrainer`; **pass `beta` in `DPOConfig`**.
* **When to use:** You can rank outputs; good for style or safety preferences.
* **Throughput note:** Heavier than SFT/KD—tune `max_steps`, `max_length`, and dataset size on 8 GB.

---

## 5) Inference & Decoding Strategies

### 5.1 Calibrated label-ranking (CreditMix)

* **Chat template** at assistant turn to standardize conditional context.
* **Length-normalized log-likelihood** for each label (“Good”, “Standard”, “Bad”).
* **Multi-null calibration:** Collect several **content-free prompts**, score label likelihoods, average → **priors**; subtract priors from raw scores to remove template bias; take argmax.
* **Outputs (server):**

  * `prediction` (label), `scores` (calibrated), `raw_scores` (uncalibrated), `prior`, `probabilities` (softmax over calibrated), `top2_margin` (confidence proxy).

### 5.2 IDP schema-constrained generation

* Pass `idp_schema.json`.
* Template prompts solicit only JSON; decoder enforces validity (field completion, structure), then validates strictly against the schema.

### 5.3 CodeGen plan-then-code (optional)

* Stage 1: generate a plan/spec from the prompt.
* Stage 2: generate final code conditioned on the plan + prompt.

---

## 6) Evaluation Protocol

* **Balanced validation** (per-class quota or fraction) to avoid skewed metrics.
* **Same decoder** as inference for honest estimates.
* Reports: **Accuracy**, **Macro-F1**, per-class precision/recall/F1, confusion matrix.

**CreditMix results (balanced 1,500/class; total 4,500):**

| Model                |   Accuracy |   Macro-F1 | Comments                                            |
| -------------------- | ---------: | ---------: | --------------------------------------------------- |
| Base (Qwen-Instruct) | **0.3562** | **0.3404** | Over-predicts “Standard”                            |
| KD adapter           | **0.4280** | **0.4031** | Moderately better; still under-predicts Standard    |
| SFT adapter          | **0.6471** | **0.5463** | Large lift via Good/Bad recall; Standard recall low |
| DPO adapter          | **0.4931** | **0.4783** | DPO sits between KD and SFT, improving overall balance vs. the base and KD |

**Practical remedies (Standard under-prediction):**

1. keep training **balanced**, 2) add **hard “Standard” negatives**, 3) apply small **inference-time bias** to the “Standard” calibrated score (constant offset) before argmax.

---

## 7) Deployment & Operations

### 7.1 Environment

* **OS:** Ubuntu 24.04.1 LTS (WSL2)
* **GPU:** GeForce RTX 4060 Laptop (**8 GB**)
* **NVIDIA Driver:** 581.29, **CUDA Driver:** 13.0
* **Python:** 3.12.3
* **Core packages:** `transformers peft accelerate bitsandbytes trl datasets fastapi uvicorn scikit-learn`

### 7.2 Serving

* Start FastAPI: `uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000`
* Endpoints: `/infer/base`, `/infer/sft`, `/infer/kd`, `/infer/dpo`
* **8 GB tips:** set `OFFLOAD_FOLDER=/tmp/offload` to enable CPU offload; keep 4-bit at inference; reduce null prompt count if needed.

### 7.3 Ops & Monitoring

* Track **latency**, **VRAM**, **adapter load** time, **prior** warm-up timing.
* Log **top-2 margin** and per-class drift.
* Periodically re-run **eval** to detect drift (same balanced protocol).

---

## 8) Data Preparation

* **CreditMix:** `creditmix_prepare.py` caps each class to the **minimum class count**, then emits:

  * `creditmix_sft.jsonl` (`{input, output}`)
  * `creditmix_kd.jsonl` (`{input, teacher_output}`)
  * `creditmix_dpo.jsonl` (`{prompt, chosen, rejected}`)
* **CodeGen/IDP:** Format your corpora to the same SFT/KD/DPO schemas; for IDP provide `idp_schema.json`.

---

## 9) Reproducible Training on 8 GB

* **QLoRA config**

  * 4-bit NF4 (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`)
  * `bnb_4bit_compute_dtype=torch.bfloat16`
  * LoRA: `r=16, alpha=32, dropout=0.05`, targets: `q/k/v/o_proj`, `gate/up/down_proj`
  * Gradient checkpointing + accumulation; per-device batch = 1

* **Stability**

  * Align tokenizer PAD/EOS with model config at load (scripts handle this).
  * Training sets `use_cache=False`; inference leaves it `True`.

---

## 10) Security, Privacy, and Governance

* **Data locality:** Finetune adapters on-prem; only adapters are saved in `runs/…`.
* **IDP:** strict schema validation reduces injection and malformed-JSON risks.
* **Auditability:** Log model version, adapter hash, calibrated scores, priors used, and inputs (subject to data policy).
* **Versioning:** Treat each adapter directory under `runs/` as a versioned artifact.

---

## 11) Cost & Footprint

* **Single 8 GB GPU** suffices for training small adapters and for serving.
* Peak memory pressure occurs on **first request** (prior computation) → mitigate with **CPU offload** and fewer null prompts.
* Adapter re-use keeps **cloud costs** low—train once, serve many.

---

## 12) Risks & Mitigations

| Risk                 | Symptom                       | Mitigation                                                                                   |
| -------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| Class collapse       | Model over-predicts one label | Balanced training; **multi-null calibration**; inference-time score bias; add hard negatives |
| DPO slowness         | Very long epochs              | Reduce `max_steps`/`max_length`; sample fewer pairs; increase accumulation                   |
| OOM at serve warm-up | Crash on first call           | `OFFLOAD_FOLDER=/tmp/offload`; keep 4-bit; fewer null prompts                                |
| Schema drift (IDP)   | Invalid JSON                  | Strict schema enforcement; unit tests on real docs                                           |
| Tokenization quirks  | Empty/odd label tokens        | Dual scoring (with/without leading space); length-normalized scoring (already implemented)   |

---

## 13) KPIs

* **Classification:** Accuracy, **Macro-F1**, per-class recall/precision, **top-2 margin** distribution
* **IDP:** Schema validation rate, field-level precision/recall, manual correction time
* **CodeGen:** Build success rate, lint/test pass rate, time saved
* **Ops:** p50/p95 latency, VRAM, prior warm-up time, uptime

---

## 14) Runbooks (Quick Recipes)

**Prepare balanced CreditMix**

```bash
python creditmix_prepare.py --src data/creditmix_dataset.json --outdir data
```

**Train SFT**

```bash
python sft_train.py \
  --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_sft.jsonl \
  --save_dir runs/creditmix_sft \
  --epochs 5 --max_steps 7115
```

**Train KD**

```bash
python kd_train.py \
  --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_kd.jsonl \
  --save_dir runs/creditmix_kd \
  --epochs 5
```

**Train DPO**

```bash
python dpo_train.py \
  --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_dpo.jsonl \
  --save_dir runs/creditmix_dpo \
  --epochs 5 --beta 0.1 --max_steps 7115
```

**Evaluate (balanced split)**

```bash
python eval_creditmix.py \
  --model_dir runs/creditmix_sft \
  --data data/creditmix_dataset.json \
  --val_per_class 1500
# swap model_dir for KD/DPO or base ID: Qwen/Qwen2.5-3B-Instruct
```

**Serve**

```bash
export OFFLOAD_FOLDER=/tmp/offload
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
```

**Infer via API**

```bash
curl -s -X POST http://localhost:8000/infer/sft \
  -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}'
```

---

## 15) Roadmap

* **Adapter registry** & disk-cached priors per model for faster service warm-up
* **Label verbalizer sets** (optional synonyms per class) with max/mean aggregation in scoring and priors
* **Extended endpoints** for CodeGen and IDP in the FastAPI app
* **CI hooks** to auto-run eval on new data slices and detect drift
* **Pinned requirements** for long-term reproducibility

---

## 16) Conclusion

The Distill-SLM accelerator demonstrates that a **disciplined QLoRA recipe** plus **robust, calibrated decoders** can turn LLM capabilities into **small, dependable, API-deployable** solutions on **8 GB VRAM**—across **classification**, **structured extraction**, and **code generation**. The provided **training**, **inference**, **evaluation**, and **serving** assets give teams a fast path from prototype to production, with clear KPIs and operational safeguards.

> For extensions (e.g., serving CodeGen/IDP endpoints, adapter registries, verbalizers), build directly on the scripts and patterns summarized here.
