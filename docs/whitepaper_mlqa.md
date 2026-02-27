# Whitepaper — Distilling LLMs to SLMs for Credit Risk Classification, Codegen, and IDP (8 GB VRAM)

## Executive Summary

This whitepaper documents a practical, reproducible pipeline for distilling large language models (LLMs) into small language models (SLMs) on commodity hardware (single **8 GB** GPU under **WSL2**). We demonstrate three supervised distillation methods—**SFT**, **KD**, and **DPO**—across three use cases:

1. **CreditMix**: 3-class credit risk classification (**Good / Standard / Bad**)
2. **CodeGen**: code synthesis with optional planning
3. **IDP** (Intelligent Document Parsing): schema-constrained JSON extraction

On the **CreditMix** task, our SLM distilled with **SFT** improves balanced-set validation accuracy from **0.3562 → 0.6471** and macro-F1 from **0.3404 → 0.5463** over the base **Qwen/Qwen2.5-3B-Instruct** student. **KD** also beats the base (accuracy **0.4280**, macro-F1 **0.4031**), while **DPO** provides a preference-based alternative when chosen/rejected pairs are available.

---

## 1. Background & Motivation

Foundation-scale LLMs are often too heavy for edge or constrained environments. SLMs, equipped with low-rank adapters and quantization, can deliver strong quality with modest compute. We present a **QLoRA** training recipe and an **inference strategy** (likelihood ranking + contextual calibration) that preserve quality while fitting in **8 GB VRAM**.

**Hardware/OS used**

* Ubuntu **24.04.1** (WSL2), NVIDIA driver **581.29**, CUDA driver **13.0**
* **GeForce RTX 4060 Laptop (8 GB VRAM)**
* Python **3.12.3**

**Base (student) models**

* Classification/IDP: `Qwen/Qwen2.5-3B-Instruct`
* Code generation: `Qwen/Qwen2.5-Coder-3B` or `Qwen/Qwen2.5-3B-Instruct` (for instruction-following code tasks)

---

## 2. Distillation Methods

### 2.1 Supervised Fine-Tuning (SFT)

* **Data schema**: `{"input": "<prompt>", "output": "<gold>"}`
* Optimizes cross-entropy on gold outputs.
* Strong gains when gold labels/targets are reliable.

### 2.2 Knowledge Distillation (KD)

* **Data schema**: `{"input": "<prompt>", "teacher_output": "<teacher or gold>"}`
* Student imitates sequence-level teacher outputs (or gold if no teacher).
* Useful for compressing teacher behaviors or rationales.

### 2.3 Direct Preference Optimization (DPO)

* **Data schema**: `{"prompt": "<p>", "chosen": "<better>", "rejected": "<worse>"}`
* Preference-based learning without explicit reward modeling.
* Best when you have clear ranked pairs.

---

## 3. Training on 8 GB (QLoRA)

We use **4-bit NF4 quantization** for the base model plus **LoRA** adapters over attention and MLP projections. All trainers:

* set `use_cache=False` during training,
* enable **gradient checkpointing**,
* use `per_device_train_batch_size=1` with **gradient accumulation**.

**Key configs**

* **Quantization**: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=bfloat16`
* **LoRA**: `r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`

**Scripts**

* `sft_train.py`: SFT trainer (input → output)
* `kd_train.py`: KD trainer (input → teacher_output)
* `dpo_train.py`: DPO trainer (TRL’s `DPOTrainer`; pass **beta** in its config)

> Practical tip for DPO: keep sequence lengths realistic and consider smaller `max_steps`—DPO is compute-heavier than SFT/KD.

---

## 4. Inference Strategies

### 4.1 CodeGen (`infer_codegen.py`)

* Loads base + adapter.
* Supports optional **plan-then-code** prompting.
* Ensures tokenizer PAD/EOS alignment for Qwen.

### 4.2 IDP (`infer_idp.py`)

* Loads base + adapter.
* Consumes a **JSON Schema** (`idp_schema.json`) and a document string.
* Constrains output to **strict JSON** that validates against the schema.

### 4.3 CreditMix Classification (`infer_creditmix.py`)

**Why a special decoder?**
Short labels are prone to prior/length bias in generative models. To stabilize classification:

1. **Chat template alignment**: Build the prompt using Qwen’s chat format, score **at the assistant start**.
2. **Likelihood ranking**: For labels {Good, Standard, Bad}, compute **length-normalized** log-likelihood per label and choose the highest.
3. **Contextual calibration**: Subtract an averaged **null-prompt prior** (several content-free prompts) from each label score to remove template bias.

This simple switch (greedy → likelihood ranking + calibration) more than **doubled** validation accuracy in our CreditMix experiments on a balanced split.

---

## 5. Evaluation Protocol (CreditMix)

**Balanced validation**
`eval_creditmix.py` builds a balanced validation set (equal examples per class) using either:

* `--val_per_class N` (exact count per label), or
* `--val_frac f` (fraction of the whole set; internally balanced).

**Scoring**
Uses the same calibrated likelihood ranking as inference:

* chat template at assistant turn,
* length-normalized label log-likelihood,
* multi-null prior subtraction,
* reports **Accuracy**, **Macro-F1**, `classification_report`, and `confusion_matrix`.

---

## 6. Results on CreditMix (Balanced Validation: 1,500 per label; total 4,500)

| Model                                 |   Accuracy |   Macro-F1 | Notes                                                                          |
| ------------------------------------- | ---------: | ---------: | ------------------------------------------------------------------------------ |
| **Base** (`Qwen/Qwen2.5-3B-Instruct`) | **0.3562** | **0.3404** | Tends to over-predict **Standard**                                             |
| **KD** (`runs/creditmix_kd`)          | **0.4280** | **0.4031** | Moderately improves **Good** recall; still under-predicts **Standard**         |
| **SFT** (`runs/creditmix_sft`)        | **0.6471** | **0.5463** | Large gains via high recall on **Good**/**Bad**; collapses **Standard** recall |
| **DPO adapter** (`runs/creditmix_dpo`)| **0.4931** | **0.4783** | **DPO** sits between KD and SFT, improving overall balance vs. the base and KD |

### Class-wise deltas (highlights)

* **Base → SFT**
  * **Good**: F1 ↑ to **0.7085**, Recall ↑ to **0.9813**
  * **Bad**:  F1 ↑ to **0.8538**, Recall ↑ to **0.9187**
  * **Standard**: F1 ↓ to **0.0767**, Recall ↓ to **0.0413** (model rarely predicts Standard; only ~117/4500)

* **Base → KD**
  * **Good**: F1 ↑ to **0.5215**, Recall ↑ to **0.6847**
  * **Bad**: F1 ↑ to **0.4555**, Recall ↑ to **0.4233**
  * **Standard**: F1 ↓ to **0.2323**, Recall ↓ to **0.1760**

* **Base → DPO**
  * **Good**: F1 ↑ to **0.5502**, Recall ↑ to **0.5940**
  * **Bad**: F1 ↑ to **0.5965**, Recall ↑ to **0.6420**
  * **Standard**: F1 ↓ to **0.2883**, Recall ↓ to **0.2433**


### Confusion patterns

* **Base:** Standard-heavy bias; many **Good/Bad → Standard** mislabels.
* **SFT:** Flips the bias — predicts **Good/Bad** very often, **Standard** rarely (Standard recall ~**0.0413**).
* **KD:** More balanced than SFT but still under-predicts **Standard** (Standard recall ~**0.1760**).
* **DPO:** Most balanced of the three finetunes; clearly reduces the Base’s **→ Standard** collapse while keeping strong **Good/Bad** recognition. Still under-predicts **Standard** relative to its prevalence, but with **substantially higher Standard recall** than SFT/KD (Standard recall ~**0.2433**; Good recall ~**0.5940**; Bad recall ~**0.6420**).

**Takeaway**

* **SFT** delivers the largest overall lift (**+0.2909** accuracy, **+0.2059** macro-F1 vs Base) by aggressively recalling **Good/Bad**, at the expense of **Standard** coverage.
* **KD** yields smaller, steadier gains; it softens the Base bias but still under-predicts **Standard**.
* **DPO** sits between KD and SFT on headline metrics; it **improves macro-F1 vs Base/KD** and offers the **best Standard recall among the finetunes** (~**0.2433** vs KD **0.1760**, SFT **0.0413**) while maintaining solid **Good/Bad** recall.
* If **Standard** coverage matters, prefer **DPO** or apply a light **inference-time bias** in calibrated score space (add a small positive offset to the Standard score before argmax) and/or enrich training with **near-miss Standard** negatives. If maximum overall accuracy on Good/Bad is the priority, **SFT** remains the strongest choice.

---

## 7. Serving (FastAPI)

`serve_creditmix.py` exposes four endpoints using the same calibrated ranking logic:

* `POST /infer/base` → `Qwen/Qwen2.5-3B-Instruct`
* `POST /infer/sft`  → `runs/creditmix_sft`
* `POST /infer/kd`   → `runs/creditmix_kd`
* `POST /infer/dpo`  → `runs/creditmix_dpo`

**Request**

```json
{"features": "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}
```

**Response (example)**

```json
{
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "prediction": "Standard",
  "scores": { "Good": -1.34, "Standard": 1.22, "Bad": -0.90 },     // calibrated
  "raw_scores": { "Good": -3.78, "Standard": -0.28, "Bad": -1.52 }, // uncalibrated
  "prior": { "Good": -2.44, "Standard": -1.49, "Bad": -0.62 },
  "probabilities": { "Good": 0.09, "Standard": 0.84, "Bad": 0.07 },
  "top2_margin": 2.12
}
```

**Runtime considerations (8 GB)**

* Set `OFFLOAD_FOLDER=/tmp/offload` to enable CPU offload and avoid OOM on first request (prior computation).
* Keep **4-bit** quantization at inference.
* You can reduce the number of null prompts to shrink peak memory.

---

## 8. Project Structure & Script Roles

```
data/
  creditmix_dataset.json         # raw classification dataset
  creditmix_{sft,kd,dpo}.jsonl   # balanced, prepared datasets (per method)
  codegen_{sft,kd,dpo}.jsonl     # code generation datasets (per method)
  idp_sft.jsonl                  # IDP SFT pairs (document → JSON)
  idp_schema.json                # IDP JSON Schema used at inference

sft_train.py                     # QLoRA SFT trainer
kd_train.py                      # QLoRA KD trainer (sequence imitation)
dpo_train.py                     # QLoRA DPO trainer (TRL), pass beta via config
infer_codegen.py                 # codegen inference (+plan_then_code)
infer_idp.py                     # schema-constrained JSON extraction
infer_creditmix.py               # calibrated likelihood ranking for labels
eval_creditmix.py                # balanced split evaluator + calibration
creditmix_prepare.py    # class balancing + SFT/KD/DPO JSONL writers
serve_creditmix.py               # FastAPI server with 4 endpoints
```

**Inputs/Outputs (by script)**

* **Trainers**: consume `{SFT,KD,DPO}` JSONL; produce **LoRA adapter** folders under `runs/<name>`.
* **Inference**: take **model_dir** (base or adapter) and task inputs (prompt/document/features); print or return task outputs.
* **Eval**: takes model + dataset; prints Accuracy, Macro-F1, class report, confusion matrix.
* **Serve**: HTTP APIs; returns prediction, calibrated scores, probabilities, and margins.

---

## 9. Applying to CodeGen & IDP

* **CodeGen**

  * Convert your corpora into `{SFT,KD,DPO}.jsonl`.
  * Train `sft_train.py` / `kd_train.py` / `dpo_train.py` with a code-appropriate base (`Qwen2.5-Coder-3B` or Instruct for general tasks).
  * Inference: `infer_codegen.py` (enable `--plan_then_code` for two-stage prompts).

* **IDP**

  * Prepare `(document → JSON)` pairs for **SFT** (and KD if you have teacher JSON).
  * Train the corresponding method.
  * Inference: `infer_idp.py` with `--schema` enforcing strict JSON output.

* **CreditMix**

  * Run `creditmix_prepare.py` to generate balanced SFT/KD/DPO JSONL.
  * Train with your chosen method, then evaluate with `eval_creditmix.py`.
  * Serve with `serve_creditmix.py`.

---

## 10. Recommendations & Next Steps

* **Recover “Standard”** while keeping SFT gains

  * Add slight **inference-time bias** to Standard in calibrated score space (constant offset).
  * Augment training with **hard Standard** examples (near-miss Good/Bad).

* **Verbalizer ensembles (optional)**

  * For classification, allow synonyms per label (e.g., Standard/Average/Typical) and aggregate (max/mean) in both priors and scoring.

* **Operationalization**

  * Cache **priors** per model on disk so first request is fast.
  * Pin library versions and export `requirements.txt`.
  * Add basic telemetry (latency, top-2 margin distribution) to the FastAPI app.

---

## 11. Conclusion

With a disciplined **QLoRA** setup and a robust **classification decoder** (likelihood ranking + contextual calibration), an SLM on **8 GB VRAM** can outperform its base student significantly on a real-world classification task. The same distillation framework applies cleanly to **code generation** and **structured extraction**, enabling practical on-device or cost-efficient deployments without sacrificing quality where it matters.