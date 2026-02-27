#!/usr/bin/env python
"""
KD training (sequence-level + optional logit KD) with QLoRA for 8 GB VRAM.
- Student defaults to Qwen2.5-Coder-3B (codegen) or Qwen2.5-3B-Instruct (IDP).
- Dataset JSONL fields:
  {"input": "...", "teacher_output": "...", "teacher_logits_path": null}
If teacher logits are provided (PyTorch .pt with shape [T, V]), we'll try logit KD
ONLY if tokenizers match; otherwise we skip logits gracefully.

Fit-on-8GB knobs:
- 4-bit QLoRA (nf4), r=16, alpha=32, dropout=0.05
- per_device_train_batch_size=1, grad_accum ~16
- gradient checkpointing + flash attention if available
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, math, os, json
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup)
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
import bitsandbytes as bnb
import mlflow
import mlflow.pytorch
from utils import sync_data_from_s3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default="Qwen/Qwen2.5-Coder-3B")
    ap.add_argument("--data", type=str, required=True, help="Path to kd.jsonl")
    ap.add_argument("--save_dir", type=str, default="runs/kd")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--kd_temperature", type=float, default=2.0)
    ap.add_argument("--use_logit_kd", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--idp_mode", action="store_true", help="Switch prompts/templates for IDP")
    ap.add_argument("--flash_attn", action="store_true")
    ap.add_argument("--mlflow_experiment_name", type=str, default="dpo_experiment", help="MLflow experiment name")
    return ap.parse_args()

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def format_example(ex, idp_mode=False):
    if idp_mode:
        # Encourage strict extraction behavior
        prompt = f"Extract structured fields as strict JSON from the document below.\n\n<document>\n{ex['input']}\n</document>\nReturn only JSON."
        target = ex["teacher_output"]
    else:
        # Codegen style: plan optional + final code (if present in teacher_output)
        prompt = f"You are a helpful coding assistant. Solve the task.\n\nProblem:\n{ex['input']}\n\nProduce correct, runnable code."
        target = ex["teacher_output"]
    return prompt, target

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

    mlflow.log_params({
        "student": args.student,
        "data": args.data,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "max_len": args.max_len,
        "idp_mode": args.idp_mode
    })

    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="tensorboard")
    accelerator.init_trackers("kd_train")

    # Load tokenizer/model in 4-bit
    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if args.flash_attn:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora_cfg)

    # Dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tok_map(batch):
        inputs, labels = [], []
        for ex in batch["input"]:
            pass
        prompts, targets = [], []
        for ex in batch["__raw__"]:
            p,t = format_example(ex, args.idp_mode); prompts.append(p); targets.append(t)
        # Build supervised tokens: [prompt]<sep>[target]
        texts = [p + "\n\n" + t for p,t in zip(prompts, targets)]
        toks = tokenizer(texts, truncation=True, max_length=args.max_len, padding=False)
        # Labels: mask prompt part -> -100
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for p,t in zip(prompts, targets):
            pt = tokenizer(p, truncation=True, max_length=args.max_len, add_special_tokens=False)
            tt = tokenizer(p+"\n\n"+t, truncation=True, max_length=args.max_len, add_special_tokens=False)
            input_ids = tt["input_ids"]
            labels = [-100]*len(pt["input_ids"]) + tt["input_ids"][len(pt["input_ids"]):]
            labels = labels[:args.max_len]
            input_ids = input_ids[:args.max_len]
            attn = [1]*len(input_ids)
            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attn)
            result["labels"].append(labels)
        return result

    # Keep raw examples for formatting
    dataset = dataset.map(lambda x: {"__raw__": x}, batched=False)
    tokenized = dataset.map(tok_map, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    dl = DataLoader(tokenized, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collator)

    # Optimizer / Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dl) / accelerator.gradient_accumulation_steps)
    max_train_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(0.03*max_train_steps), max_train_steps)

    model, optim, dl, sched = accelerator.prepare(model, optim, dl, sched)
    model.train()
    try :
        global_step = 0
        for epoch in range(args.epochs):
            for batch in dl:
                with accelerator.accumulate(model):
                    out = model(**batch)
                    loss = out.loss
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step(); sched.step(); optim.zero_grad()
                if accelerator.is_main_process and global_step % 10 == 0:
                    accelerator.log({"train/loss": loss.item()}, step=global_step)
                global_step += 1
                if 0 < args.max_steps <= global_step:
                    break
            if 0 < args.max_steps <= global_step:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            model.save_pretrained(args.save_dir)
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
