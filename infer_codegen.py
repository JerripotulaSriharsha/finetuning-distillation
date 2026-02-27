#!/usr/bin/env python
"""
Codegen inference for trained adapters (QLoRA). 
- Works with outputs from kd_train.py / dpo_train.py / sft_train.py.
- Optional PLAN->CODE prompting.
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, os, torch, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="runs/kd or runs/dpo or runs/sft")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--plan_then_code", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()

    if args.plan_then_code:
        user = f"""You are a coding assistant. Solve the task.
Please first write a short <PLAN> with bullet steps, then provide the final <CODE>.
Problem:
{args.prompt}
"""
    else:
        user = f"You are a coding assistant. Provide correct, runnable code for:\n{args.prompt}"

    inputs = tokenizer(user, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, 
            top_p=args.top_p,
            do_sample=True, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
