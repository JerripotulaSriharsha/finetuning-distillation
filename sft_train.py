#!/usr/bin/env python
"""
Structure/Rationale-Aware SFT for 8 GB (QLoRA).
Dataset JSONL (two flavors):
- Codegen plan->code:
  {"input": "...", "output": "<PLAN>...</PLAN>\n<CODE>...</CODE>"}
- IDP JSON + why:
  {"input": "...", "output": "{\"field\": \"...\", \"why\": \"...\"}"}
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, os, math
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from utils import sync_data_from_s3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", type=str, required=True, help="Path to sft.jsonl")
    ap.add_argument("--save_dir", type=str, default="runs/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps (post-accum).")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--mlflow_experiment_name", type=str, default="sft_experiment", help="MLflow experiment name")

    return ap.parse_args()

def main():
    args = parse_args()

    if args.data.startswith("s3://"):
        local_data_path = sync_data_from_s3(args.data)
        data_path = local_data_path
    else:
        data_path = args.data

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    print(f"Here is the mlflow uri : {tracking_uri}",flush=True)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow: Using tracking server at {mlflow.get_tracking_uri()}",flush=True)
    else:
        # If no URI is set, default to a local 'mlruns' directory
        mlflow.set_tracking_uri("./mlruns")
        print("MLflow: No tracking URI specified. Defaulting to local './mlruns' directory.",flush=True)

    # Initialize MLflow
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.start_run()

    mlflow.log_params({
        "student": args.student,
        "data": args.data,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_len": args.max_len
    })
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=data_path, split="train")

    def tok(example):
        prompt = example["input"]
        target = example["output"]
        text = prompt + "\n\n" + target
        toks = tokenizer(text, truncation=True, max_length=args.max_len)
        # Mask prompt tokens
        p_ids = tokenizer(prompt, truncation=True, max_length=args.max_len)["input_ids"]
        labels = [-100]*len(p_ids) + toks["input_ids"][len(p_ids):]
        return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}

    ds = ds.map(tok, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    dl = DataLoader(ds, batch_size=args.per_device_train_batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = max(1, math.ceil(len(dl) / args.gradient_accumulation_steps))
    if steps_per_epoch and steps_per_epoch > 0 and Accelerator().is_main_process:
        print(f"steps_per_epoch ≈ {steps_per_epoch}")
    sched = get_cosine_schedule_with_warmup(optim, int(0.03*steps_per_epoch*args.epochs), steps_per_epoch*args.epochs)

    model, optim, dl, sched = accelerator.prepare(model, optim, dl, sched)
    model.train()
    step=0
    try:
        for epoch in range(args.epochs):
            for batch in dl:
                with accelerator.accumulate(model):
                    out = model(**batch)
                    loss = out.loss
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                if accelerator.is_main_process and step%10==0:
                    accelerator.print(f"step {step} loss {loss.item():.4f}")
                step+=1
                if args.max_steps > 0 and step >= args.max_steps:
                    break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(args.save_dir); tokenizer.save_pretrained(args.save_dir)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            if tracking_uri:
                    mlflow.pytorch.log_model(unwrapped_model, "model", registered_model_name=args.student.split('/')[-1])
                    mlflow.log_artifacts(args.save_dir, "model_artifacts")
        accelerator.end_training()
    except Exception as e:
        print(f"An error occurred: {e}")
        if tracking_uri:
            mlflow.log_param("error", str(e))
        raise
    finally:
        if tracking_uri:
            mlflow.end_run()
            print("MLflow run ended.")


if __name__ == "__main__":
    main()
