#!/usr/bin/env python
"""
Preference Distillation via DPO on 8 GB (QLoRA).
Input dataset JSONL with triples:
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, os
from datasets import load_dataset
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # helps on Ada/Lovelace too
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from trl import DPOTrainer, DPOConfig
from transformers.trainer_utils import get_last_checkpoint
import mlflow
import mlflow.pytorch
from utils import sync_data_from_s3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--data", type=str, required=True, help="Path to dpo.jsonl")
    ap.add_argument("--save_dir", type=str, default="runs/dpo")
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--idp_mode", action="store_true")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--mlflow_experiment_name", type=str, default="dpo_experiment", help="MLflow experiment name")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.data.startswith("s3://"):
        local_data_path = sync_data_from_s3(args.data)
        data_path = local_data_path
    else:
        data_path = args.data

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow: Using tracking server at {mlflow.get_tracking_uri()}")
    else:
        # If no URI is set, default to a local 'mlruns' directory
        mlflow.set_tracking_uri("./mlruns")
        print("MLflow: No tracking URI specified. Defaulting to local './mlruns' directory.")

    # Initialize MLflow
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.start_run()


     # Log parameters
    mlflow.log_params({
        "student": args.student,
        "data": args.data,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_len": args.max_len,
        "idp_mode": args.idp_mode
    })
    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True, trust_remote_code=True)
    # tokenizer setup (add this one line)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # safer for causal LM batching

    # 4-bit base + LoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # torch dtype (not a string)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=data_path, split="train")

    def fmt(example):
        if args.idp_mode:
            prompt = f"Extract fields as strict JSON for the following document.\n\n<document>\n{example['prompt']}\n</document>\nReturn only JSON."
        else:
            prompt = f"You are a helpful coding assistant. Problem:\n{example['prompt']}\nRespond with correct, runnable code."
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    ds = ds.map(fmt, remove_columns=ds.column_names)

    training_config = DPOConfig(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,   # will be ignored if max_steps > 0
        max_steps=args.max_steps,       # <-- add this line
        learning_rate=args.learning_rate,
        max_length=args.max_len,
        beta=args.beta,                 # <-- move beta here
        logging_steps=200,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        reference_free=True,    # With reference_free=True, TRL uses a reference-free loss (it detaches the policy logits to serve as a baseline), so you only do one forward pass and save VRAM—ideal for your 8 GB setup.
        report_to="mlflow"                   
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # or a frozen copy if you use one
        args=training_config,           # <-- pass the config object, not .to_dict()
        train_dataset=ds,
        processing_class=tokenizer,
    )

    ckpt = args.resume_from_checkpoint
    if ckpt is None:
        # auto-pick the latest in save_dir if present
        ckpt = get_last_checkpoint(args.save_dir)


    try:    
        trainer.train(resume_from_checkpoint=ckpt)
        # trainer.train()
        trainer.save_model(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifacts(args.save_dir, "model_artifacts")
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()
