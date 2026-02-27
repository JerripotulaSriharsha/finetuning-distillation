#!/usr/bin/env python
"""
IDP inference with strict JSON using Outlines.
- Provide a JSON Schema file with the expected fields and types.
- Example schema is in data/idp_schema.json.
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, json, torch
import outlines
from outlines.types import JsonSchema
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sync_data_from_s3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--document", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.schema.startswith("s3://"):
        local_data_path = sync_data_from_s3(args.schema)
        data_path = local_data_path
    else:
        data_path = args.schmea

    # Load HF model/tokenizer (your LoRA-adapted checkpoint directory)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Wrap as an Outlines model
    model = outlines.from_transformers(hf_model, hf_tokenizer)  # ← new API

    # Load JSON Schema and wrap it in JsonSchema (required for dict/string schemas)
    with open(data_path, "r") as f:
        schema_dict = json.load(f)
    output_type = JsonSchema(schema_dict)  # ← required wrapper

    prompt = (
        "Extract the required fields as strict JSON that matches the schema "
        "from the following document:\n\n" + args.document + "\n\nReturn only the JSON."
    )

    # Generate strictly-valid JSON
    result = model(prompt, output_type=output_type, max_new_tokens=args.max_new_tokens)
    print(result)

if __name__ == "__main__":
    main()
