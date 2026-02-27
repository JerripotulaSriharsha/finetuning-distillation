 ---

# distill_slm

Distillation playbooks for turning **LLMs → SLMs** on a single 8 GB GPU (WSL2).

Covers three use cases:
1. **CodeGen**: code synthesis with planning
2. **IDP** (Intelligent Document Parsing): JSON schema extraction
3. **CreditMix** classification: “Good / Standard / Bad” credit risk

Implements three supervised distillation methods with **QLoRA**:
* **SFT** (Supervised Fine-Tuning)
* **KD** (Knowledge Distillation) — sequence-level imitation
* **DPO** (Direct Preference Optimization) — preference distillation

Includes **training**, **inference**, **evaluation**, and a **FastAPI serving** stack with calibrated, label-ranking inference for classification.

---

## Contents

```
distill_slm/
├─ data/
│  ├─ creditmix_dataset.json            # raw credit risk dataset (question/answer pairs)
│  ├─ creditmix_sft.jsonl               # prepared SFT dataset (balanced)
│  ├─ creditmix_kd.jsonl                # prepared KD dataset  (balanced)
│  ├─ creditmix_dpo.jsonl               # prepared DPO dataset (balanced)
│  ├─ codegen_sft.jsonl                 # codegen SFT (prompt/output)
│  ├─ codegen_kd.jsonl                  # codegen KD (prompt/teacher_output)
│  ├─ codegen_dpo.jsonl                 # codegen DPO (prompt/chosen/rejected)
│  ├─ idp_sft.jsonl                     # IDP SFT (document → JSON)
│  └─ idp_schema.json                   # IDP JSON Schema for inference
├─ sft_train.py                         # SFT training (QLoRA)
├─ kd_train.py                          # KD training (QLoRA)
├─ dpo_train.py                         # DPO training (QLoRA, TRL)
├─ infer_codegen.py                     # codegen inference (+ optional plan-then-code)
├─ infer_idp.py                         # IDP inference (JSON schema-constrained)
├─ infer_creditmix.py                   # creditmix inference (likelihood ranking + calibration)
├─ eval_creditmix.py                    # creditmix evaluator (balanced split + calibration)
├─ creditmix_prepare.py                 # balances classes & writes SFT/KD/DPO jsonl
├─ serve_creditmix.py                   # FastAPI/uvicorn server with 4 endpoints (base/sft/kd/dpo)
└─ README.md                            # this file
```

> If your local names differ (e.g., `runs/<…>` folder names), adjust commands accordingly.

---

## Environment

**System** (as you used):

* Ubuntu **24.04.1** LTS (WSL2)
* NVIDIA driver **581.29**, CUDA driver **13.0**
* GPU: **GeForce RTX 4060 Laptop (8 GB VRAM)**
* Python **3.12.3**

**Python deps (core)**:

```
pip install "transformers>=4.43" peft accelerate datasets bitsandbytes trl
pip install fastapi uvicorn pydantic
pip install scikit-learn
```

> For IDP JSON-schema decoding, your final script uses a direct JSON-schema approach (no extra libs required).
> If you kept an earlier outlines attempt around, make sure the current `infer_idp.py` is the “schema-only” version that runs on your setup.

---

## Models

* **Base (student)**:

  * CodeGen: `Qwen/Qwen2.5-Coder-3B` or `Qwen/Qwen2.5-3B-Instruct`
  * IDP / CreditMix: `Qwen/Qwen2.5-3B-Instruct`
* **All fine-tunes use QLoRA (4-bit base + LoRA adapters)** to fit **8 GB VRAM**.

---

## Data schemas

### SFT (supervised)

```json
{"input":  "<prompt>", "output": "<gold output>"}
```

### KD (sequence imitation)

```json
{"input": "<prompt>", "teacher_output": "<teacher or gold output>"}
```

*(If you don’t have a teacher, point `teacher_output` to the gold label/output.)*

### DPO (preference)

```json
{"prompt": "<prompt>", "chosen": "<preferred>", "rejected": "<less preferred>"}
```

### IDP schema (example)

`data/idp_schema.json` – a standard **JSON Schema** describing the fields to extract; `infer_idp.py` uses it to enforce structured output.

---

## QLoRA config (8 GB VRAM)

All three trainers load the base in **4-bit** with NF4 and enable LoRA on attention/MLP modules:

```python
BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)

LoraConfig(
  r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
  task_type="CAUSAL_LM",
  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
```

> The trainers **set `use_cache=False`** during training and enable **gradient checkpointing** to stay within 8 GB.

---

## Prepare (CreditMix)

Create **balanced** SFT/KD/DPO JSONL files from the raw dataset (equal per class):

```bash
python creditmix_prepare.py \
  --src data/creditmix_dataset.json \
  --outdir data
```

**What it does**

* Reads `creditmix_dataset.json` (`{"question": "...","answer": "Good|Standard|Bad"}`)
* Derives a **per-class cap** (min of class counts)
* Writes:

  * `data/creditmix_sft.jsonl` → `{"input": prompt, "output": label}`
  * `data/creditmix_kd.jsonl`  → `{"input": prompt, "teacher_output": label}`
  * `data/creditmix_dpo.jsonl` → `{"prompt": prompt, "chosen": label, "rejected": negative_label}`
* Prompts are fixed-template classification asks (“Return only the label.”)

---

## Training

### 1) SFT

```bash
# For Codegen:
python sft_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_sft.jsonl --save_dir runs/sft_code --epochs 5
# For IDP:
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_sft.jsonl --save_dir runs/sft_idp --epochs 5
# Credit Mix
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_sft.jsonl --save_dir runs/creditmix_sft --epochs 5 --max_steps 7115
```

**What it does**

* Loads the student in **4-bit** (QLoRA), adds LoRA adapters
* Packs/feeds `input → output` pairs, teacher forcing cross-entropy
* Saves the **adapter** in `runs/creditmix_sft` (base weights are not duplicated)

> Notes:
>
> * `--max_steps` (if present in your trainer) overrides epochs; otherwise it uses `--epochs`.
> * On small VRAM, keep `--per_device_train_batch_size=1` and use `--gradient_accumulation_steps` to hit your effective batch.

### 2) KD (sequence imitation)

```bash
# For Codegen:
python kd_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_kd.jsonl --save_dir runs/kd_code --epochs 5
# For IDP:
python kd_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_kd.jsonl --idp_mode --save_dir runs/kd_idp --epochs 5
# Credit Mix
python kd_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_kd.jsonl --idp_mode --save_dir runs/creditmix_kd --epochs 5 --max_steps 7115
```

**What it does**

* Same QLoRA setup as SFT
* Minimizes LM loss on **teacher_output** given **input** (sequence imitation)
* Good when you have teacher-generated traces / rationales.

### 3) DPO (preference distillation)

```bash
# For Codegen:
python dpo_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_dpo.jsonl --save_dir runs/dpo_code --epochs 5
# IDP:
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_dpo.jsonl --idp_mode --save_dir runs/dpo_idp --epochs 5
# Credit Mix
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --epochs 5 --beta 0.1 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --max_len 256 --max_steps 7115
```

**What it does**

* Loads student in 4-bit + LoRA
* Uses **TRL’s `DPOTrainer`** with a `DPOConfig` (⚠️ pass **`beta` in the config**, not as a stray kwarg)
* Prefers **`chosen`** over **`rejected`** for each prompt

**Speed tips for DPO on 8 GB**

* Keep `per_device_train_batch_size=1`, use accumulation
* Reduce `--max_len` to your realistic prompt/answer length
* If it still crawls, narrow the dataset or run for fewer steps; DPO is compute-heavier than SFT/KD.

---

## Inference

### CodeGen

```bash
python infer_codegen.py --model_dir runs/kd_code --prompt "Write a function that sorts a list." --plan_then_code
python infer_codegen.py --model_dir runs/dpo_code --prompt "Write a function that sorts a list." --plan_then_code
python infer_codegen.py --model_dir runs/sft_code --prompt "Write a function that sorts a list." --plan_then_code
```

**What it does**

* Loads the **adapter** + base
* Applies a simple plan-then-code prompt (if flag set)
* Greedy or controlled generation; ensures PAD/EOS are set for Qwen

### IDP (JSON schema extraction)

```bash
python infer_idp.py --model_dir runs/kd_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
python infer_idp.py --model_dir runs/dpo_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
python infer_idp.py --model_dir runs/sft_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
```

**What it does**

* Loads adapter + base
* Uses the **JSON Schema** to constrain/validate decoding
* Returns **strict JSON** conforming to `idp_schema.json`

> We fixed early “`outlines.json` import” issues by removing that dependency path. The provided script uses only the schema and standard validation on your setup.

### CreditMix (classification)

```bash
python infer_creditmix.py --model_dir runs/creditmix_sft --features "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"
python infer_creditmix.py --model_dir runs/creditmix_kd --features "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"
python infer_creditmix.py --model_dir runs/creditmix_dpo --features "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"
```

**What it does**

* Loads adapter + base (or base only)
* Builds a **Qwen chat-template** prompt (assistant generation position)
* Scores candidate labels **{Good, Standard, Bad}** by **length-normalized** conditional log-likelihood
* Applies **multi-null contextual calibration** (subtract averaged priors measured on content-free prompts)
* Picks the highest calibrated score

This ranking-based inference aligned your validation accuracy jumps (31% → ~65%) on the balanced dataset.

---

## Evaluation (CreditMix)

```bash
python eval_creditmix.py --model_dir runs/creditmix_sft --data data/creditmix_dataset.json --val_per_class 1500
python eval_creditmix.py --model_dir runs/creditmix_kd --data data/creditmix_dataset.json --val_per_class 1500
python eval_creditmix.py --model_dir runs/creditmix_dpo --data data/creditmix_dataset.json --val_per_class 1500
```

**What it does**

* Loads model (adapter+base **or** base model ID)
* Builds a **balanced validation split** with equal examples per label (either `--val_per_class` or `--val_frac`)
* Uses the **same** ranking + calibration inference as above
* Prints accuracy, macro-F1, `classification_report`, and `confusion_matrix`.

**Tips**

* You can also run against the **base** model directly:

  ```
    python eval_creditmix.py --model_dir Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dataset.json --val_per_class 1500
  ```
* If VRAM is tight, set CPU offload:

  ```
  export OFFLOAD_FOLDER=/tmp/offload
  ```

---

## Serving (FastAPI)

Start a local inference API that supports **four endpoints**:

* `POST /infer/base` → `Qwen/Qwen2.5-3B-Instruct`
* `POST /infer/sft`  → `runs/creditmix_sft`
* `POST /infer/kd`   → `runs/creditmix_kd`
* `POST /infer/dpo`  → `runs/creditmix_dpo`

```bash
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
```

**Request**

```bash
sudo apt update && sudo apt install -y jq

curl -s -X POST http://localhost:8000/infer/base -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

curl -s -X POST http://localhost:8000/infer/sft  -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

curl -s -X POST http://localhost:8000/infer/kd   -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq
  
curl -s -X POST http://localhost:8000/infer/dpo  -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq
```

**Response**

```json
{
  "model_name": "runs/creditmix_sft",
  "prediction": "Standard",
  "scores": { "Good": -1.34, "Standard": 1.22, "Bad": -0.90 },
  "raw_scores": { "Good": -3.78, "Standard": -0.28, "Bad": -1.52 },
  "prior": { "Good": -2.44, "Standard": -1.49, "Bad": -0.62 },
  "probabilities": { "Good": 0.09, "Standard": 0.84, "Bad": 0.07 },
  "top2_margin": 2.12
}
```

**What it does**

* Lazily loads model + tokenizer per endpoint and **caches** them
* Computes/ caches **multi-null priors** per model
* For each request, builds the chat-template prompt, computes **raw** label scores (length-normalized), subtracts **priors** → **calibrated scores**, then returns the **prediction**, **scores**, **softmax probabilities** over calibrated scores, and **top-2 margin** (a quick confidence signal).

**8 GB VRAM tips for the server**

* Enable CPU offload (no code change):

  ```
  export OFFLOAD_FOLDER=/tmp/offload
  ```
* Optional: turn on 4-bit loading at inference (already enabled in your latest scaffold)
* If you still OOM on first call (when priors are computed), reduce the number of null prompts in `NULL_FEATURES`.

---

## Using the same stack for CodeGen, IDP, CreditMix

* **Training**
  * **CodeGen**: prepare `codegen_{sft,kd,dpo}.jsonl` with the schemas above.
    * For SFT/KD the outputs are code strings.
    * For DPO the `chosen/rejected` are alternative code snippets or styles.
  * **IDP**: SFT tuned on `(document → JSON)` pairs; for KD you can plug a teacher’s JSON. DPO pairs can reflect JSON quality.
  * **CreditMix**: use `creditmix_prepare.py` to create balanced splits for all three methods.

* **Inference**
  * **CodeGen**: `infer_codegen.py` (use `--plan_then_code` for 2-stage prompting).
  * **IDP**: `infer_idp.py` with `--schema` and `--document`.
  * **CreditMix**: `infer_creditmix.py` uses ranking + calibration for stable labels.

* **Evaluation**
  * For **CreditMix**, use `eval_creditmix.py` (balanced split, calibrated ranking).
  * For CodeGen/IDP you can adapt the label-ranking (e.g., multiple-choice tasks) or add task-specific metrics (compilation or schema-validation rates).

* **Serving**
  * The provided **FastAPI** app wraps the classification inference nicely.
  * You can extend it with CodeGen/IDP endpoints by following the same pattern:
    * Build the correct prompt (chat template)
    * Run constrained decoding (for IDP) or normal generation (for CodeGen)
    * Return structured JSON responses.

---

## Common warnings & notes

* **“`use_cache=True` is incompatible with gradient checkpointing`”**  
  The trainer sets `use_cache=False` while training. It’s normal.

* **Transformers deprecations**
  Prefer `dtype=` over `torch_dtype=`, and pass 4-bit settings via `quantization_config=BitsAndBytesConfig(...)` (already done in the scripts).

* **Tokenizer PAD/BOS/EOS updates**
  Qwen tokenizers sometimes add a PAD. The scripts **align model & generation config** to the tokenizer at load time; the warning is informational.

---

## FAQ

**Q: My DPO run is slow.**
A: Reduce `max_steps`, shorten `max_length`, keep `per_device_train_batch_size=1` with accumulation, and consider a smaller sample to iterate. DPO is compute-heavier.

**Q: I see OOM on serving first request.**
A: Set `OFFLOAD_FOLDER=/tmp/offload`. Optionally reduce the number of null prompts or keep 4-bit enabled for inference.

**Q: Why does CreditMix inference use “likelihood ranking” and “calibration”?**
A: Short labels are prone to prior/length bias. Ranking with **length-normalized** log-likelihood at the **assistant start** (chat template) + **multi-null calibration** significantly stabilizes predictions and improved accuracy in your runs.

---

## Quick recipes

**SFT on CreditMix, then evaluate & serve**

```bash
# prepare balanced jsonl
python creditmix_prepare.py --src data/creditmix_dataset.json --outdir data

# train
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct \
  --data data/creditmix_sft.jsonl --save_dir runs/creditmix_sft --epochs 5 --max_steps 7115

# eval (balanced)
python eval_creditmix.py --model_dir runs/creditmix_sft --data data/creditmix_dataset.json --val_per_class 1500

# serve
export OFFLOAD_FOLDER=/tmp/offload
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
```

**KD or DPO swap-in**

```bash
python kd_train.py  --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_kd.jsonl  --save_dir runs/creditmix_kd
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --beta 0.1
python eval_creditmix.py --model_dir runs/creditmix_kd  --data data/creditmix_dataset.json --val_per_class 1500
python eval_creditmix.py --model_dir runs/creditmix_dpo --data data/creditmix_dataset.json --val_per_class 1500
```

**CodeGen and IDP**

* Convert your data to `{SFT,KD,DPO}` JSONL schemas above.
* Train the corresponding adapters.
* Use `infer_codegen.py` or `infer_idp.py` to test outputs.

---