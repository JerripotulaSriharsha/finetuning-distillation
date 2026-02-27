# Distill SLM (8 GB VRAM) – KD / DPO / SFT

## Setup
```bash
python3 -m venv ~/venvs/llm && source ~/venvs/llm/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## KD (sequence-level)
```bash
# For Codegen:
python kd_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_kd.jsonl --save_dir runs/kd_code --epochs 5
# For IDP:
python kd_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_kd.jsonl --idp_mode --save_dir runs/kd_idp --epochs 5
```

## DPO
```bash
# For Codegen:
python dpo_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_dpo.jsonl --save_dir runs/dpo_code --epochs 5
# IDP:
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_dpo.jsonl --idp_mode --save_dir runs/dpo_idp --epochs 5
```

## SFT (Structure/Rationale)
```bash
# For Codegen:
python sft_train.py --student Qwen/Qwen2.5-Coder-3B --data data/codegen_sft.jsonl --save_dir runs/sft_code --epochs 5
# For IDP:
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/idp_sft.jsonl --save_dir runs/sft_idp --epochs 5
```

## Inference
```bash
# For Codegen:
python infer_codegen.py --model_dir runs/kd_code --prompt "Write a function that sorts a list." --plan_then_code
python infer_codegen.py --model_dir runs/dpo_code --prompt "Write a function that sorts a list." --plan_then_code
python infer_codegen.py --model_dir runs/sft_code --prompt "Write a function that sorts a list." --plan_then_code
# For IDP:
python infer_idp.py --model_dir runs/kd_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
python infer_idp.py --model_dir runs/dpo_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
python infer_idp.py --model_dir runs/sft_idp --schema data/idp_schema.json --document "Invoice #INV1 date 2025-10-01 from Bar LLC total USD 50."
```





## Credit Mix Dataset & Distilled Models & Evals
```bash
# Prep: make three JSONLs from your dataset
python3 creditmix_prepare.py --src data/creditmix_dataset.json --outdir data
# Wrote: data/creditmix_sft.jsonl, data/creditmix_kd.jsonl, data/creditmix_dpo.jsonl

# Train with SFT (fastest baseline)
python sft_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_sft.jsonl --save_dir runs/creditmix_sft --epochs 5 --max_steps 7115
# Uses QLoRA in 4-bit and masks the prompt tokens automatically (your existing sft_train.py).

# Train with KD (sequence-level imitation)
python kd_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_kd.jsonl --idp_mode --save_dir runs/creditmix_kd --epochs 5 --max_steps 7115
# We reuse --idp_mode so the script formats the prompt as “return only JSON” normally; here it still works because teacher_output is just a single label string. (If you want, you can remove --idp_mode; either way, the loss is over the gold label text.)
# If you later have a teacher model’s logits for label tokens, you can extend kd_train.py to load per-sample .pt logits and add logit-KD; for now sequence-KD is sufficient.

# Train with DPO (preference/ranking)
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --epochs 5 --beta 0.1 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --max_len 256 --max_steps 7115
# With your dataset (~22,764 pairs) and effective batch = 2 * 8 = 16, you get ~1,423 steps/epoch.
# --max_steps 9955 ≈ 7 epochs, not 5. If you truly want 5 epochs, use --max_steps 7115 (≈ 1423×5) or just drop --max_steps.
# Each example teaches the student to prefer the correct label (chosen) over a plausible wrong label (rejected). This aligns generation toward valid labels without PPO.
python dpo_train.py --student Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dpo.jsonl --save_dir runs/creditmix_dpo --epochs 5 --beta 0.1 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --max_len 256 --max_steps 7115 --resume_from_checkpoint runs/creditmix_dpo/checkpoint-3000

# Inference (return only a label)
python infer_creditmix.py --model_dir runs/creditmix_sft --features "Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"

## Evaluation on sample set

# Evaluate a finetuned adapter (SFT/KD/DPO):
CUDA_LAUNCH_BLOCKING=1 python eval_creditmix.py --model_dir runs/creditmix_sft --data data/creditmix_dataset.json --val_per_class 1500 # --val_frac 0.1

# Evaluate the base student (no adapter):
python eval_creditmix.py --model_dir Qwen/Qwen2.5-3B-Instruct --data data/creditmix_dataset.json --val_per_class 1500

```





## Credit Mix Inference Server
```bash
# (activate your venv)
pip install fastapi uvicorn "transformers>=4.43" peft accelerate

# optional: if you want CPU offload to save VRAM
export OFFLOAD_FOLDER=/tmp/offload

# Start Server
mkdir -p /home/aniket/distill_slm/offload
export OFFLOAD_FOLDER=/home/aniket/distill_slm/offload
uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000


# Request predictions from server
sudo apt update && sudo apt install -y jq

# The server caches the last-loaded model and priors for each key; first request for each endpoint will be slower (computes priors once).
# If 8 GB feels tight, set OFFLOAD_FOLDER to allow partial CPU offload.
# If you later rename your finetune directories, update MODEL_MAP.
# The scoring path is the same as your successful eval pipeline, so predictions should be consistent with your CLI evaluators.

curl -s -X POST http://localhost:8000/infer/base -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

curl -s -X POST http://localhost:8000/infer/sft  -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

curl -s -X POST http://localhost:8000/infer/kd   -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq
  
curl -s -X POST http://localhost:8000/infer/dpo  -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

```