#!/usr/bin/env python
"""
Unified evaluation script for DPO, KD, and SFT trained models.
Works with both local paths and PVC paths in KFP pipelines.
"""

import warnings
warnings.filterwarnings('ignore')

import argparse, os, json, torch, math, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import mlflow
import mlflow.pytorch
from utils import sync_data_from_s3
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate DPO, KD, or SFT trained models")
    
    # Required arguments
    ap.add_argument("--model_dir", type=str, required=True, help="Path to the trained model (PVC or local)")
    ap.add_argument("--data", type=str, required=True, help="Path to test data JSONL")
    ap.add_argument("--training_type", type=str, choices=["dpo", "kd", "sft"], required=True, 
                    help="Type of training approach used (dpo, kd, or sft)")
    
    # Model arguments
    ap.add_argument("--base_model", type=str, default=None, 
                    help="Base model used for training (inferred if not provided)")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--load_in_4bit", action="store_true", default=True,
                    help="Whether to load the model in 4-bit quantization")
    
    # Mode arguments
    ap.add_argument("--idp_mode", action="store_true", help="Whether the model was trained in IDP mode")
    
    # Output arguments
    ap.add_argument("--output_file", type=str, default=None, 
                    help="Path to save evaluation results (auto-generated if not provided)")
    ap.add_argument("--mlflow_experiment_name", type=str, default="model_evaluation", 
                    help="MLflow experiment name")
    
    # Additional arguments for KFP integration
    ap.add_argument("--val_per_class", type=int, default=0, help="Number of validation samples per class")
    ap.add_argument("--offload_folder", type=str, default=None, help="Folder to offload model weights")
    
    return ap.parse_args()

def get_default_base_model(training_type):
    """Get default base model based on training type."""
    if training_type == "dpo":
        return "Qwen/Qwen2.5-Coder-3B"
    elif training_type == "kd":
        return "Qwen/Qwen2.5-Coder-3B"
    elif training_type == "sft":
        return "Qwen/Qwen2.5-3B-Instruct"
    return None

def get_default_output_file(training_type, model_dir):
    """Get default output file based on training type."""
    # Use the model directory to save the results
    return os.path.join(model_dir, f"{training_type}_eval_results.jsonl")

def format_prompt(example, training_type, idp_mode=False):
    """Format prompt based on training type and mode."""
    if training_type == "dpo":
        if idp_mode:
            return f"Extract fields as strict JSON for the following document.\n\n<document>\n{example['prompt']}\n</document>\nReturn only JSON."
        else:
            return f"You are a helpful coding assistant. Problem:\n{example['prompt']}\nRespond with correct, runnable code."
    elif training_type == "kd":
        if idp_mode:
            return f"Extract structured fields as strict JSON from the document below.\n\n<document>\n{example['input']}\n</document>\nReturn only JSON."
        else:
            return f"You are a helpful coding assistant. Solve the task.\n\nProblem:\n{example['input']}\n\nProduce correct, runnable code."
    elif training_type == "sft":
        return example["input"]
    return ""

def calculate_perplexity(model, tokenizer, text, max_length=2048):
    """Calculate perplexity of a text."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    return math.exp(loss.item())

def extract_json(text):
    """Extract JSON from text, return None if not found."""
    try:
        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except:
        return None

def evaluate_json_output(generated, reference):
    """Simple evaluation for JSON outputs."""
    gen_json = extract_json(generated)
    ref_json = extract_json(reference)
    
    if gen_json is None or ref_json is None:
        return {"valid_json": False, "field_match": 0}
    
    # Check if generated JSON has the same fields as reference
    gen_fields = set(gen_json.keys())
    ref_fields = set(ref_json.keys())
    
    field_match = len(gen_fields.intersection(ref_fields)) / len(ref_fields)
    
    return {
        "valid_json": True,
        "field_match": field_match,
        "num_fields": len(ref_fields),
        "num_matched_fields": len(gen_fields.intersection(ref_fields))
    }

def evaluate_dpo(model, tokenizer, test_dataset, args):
    """Evaluate DPO-trained model."""
    results = []
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating DPO model"):
            prompt = format_prompt(example, "dpo", args.idp_mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            result = {
                "prompt": example["prompt"],
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
                "generated": response
            }
            results.append(result)
    
    # Calculate metrics (simplified example)
    metrics = {
        "num_examples": len(results)
    }
    
    return results, metrics

def evaluate_kd(model, tokenizer, test_dataset, args):
    """Evaluate KD-trained model."""
    results = []
    total_perplexity = 0
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating KD model"):
            prompt = format_prompt(example, "kd", args.idp_mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Calculate perplexity on the generated text
            full_text = prompt + "\n\n" + response
            perplexity = calculate_perplexity(model, tokenizer, full_text, args.max_len)
            total_perplexity += perplexity
            
            result = {
                "input": example["input"],
                "teacher_output": example.get("teacher_output", ""),
                "generated": response,
                "perplexity": perplexity
            }
            results.append(result)
    
    # Calculate metrics
    avg_perplexity = total_perplexity / len(results) if results else float('inf')
    metrics = {
        "num_examples": len(results),
        "avg_perplexity": avg_perplexity
    }
    
    return results, metrics

def evaluate_sft(model, tokenizer, test_dataset, args):
    """Evaluate SFT-trained model."""
    results = []
    total_perplexity = 0
    total_field_match = 0
    valid_json_count = 0
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating SFT model"):
            prompt = format_prompt(example, "sft", args.idp_mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            # Calculate perplexity on the generated text
            full_text = prompt + "\n\n" + response
            perplexity = calculate_perplexity(model, tokenizer, full_text, args.max_len)
            total_perplexity += perplexity
            
            # Evaluate JSON output if applicable
            json_eval = evaluate_json_output(response, example["output"])
            if json_eval["valid_json"]:
                valid_json_count += 1
                total_field_match += json_eval["field_match"]
            
            result = {
                "input": example["input"],
                "reference": example["output"],
                "generated": response,
                "perplexity": perplexity,
                "json_eval": json_eval
            }
            results.append(result)
    
    # Calculate metrics
    avg_perplexity = total_perplexity / len(results) if results else float('inf')
    valid_json_ratio = valid_json_count / len(results) if results else 0
    avg_field_match = total_field_match / valid_json_count if valid_json_count > 0 else 0
    
    metrics = {
        "num_examples": len(results),
        "avg_perplexity": avg_perplexity,
        "valid_json_ratio": valid_json_ratio,
        "avg_field_match": avg_field_match
    }
    
    return results, metrics

def load_model_and_tokenizer(args):
    """Load the trained model and tokenizer directly from PVC mount path."""
    # Set defaults if not provided
    if args.base_model is None:
        args.base_model = get_default_base_model(args.training_type)
    
    print(f"Loading model from PVC path: {args.model_dir}")
    
    # Load tokenizer from the trained model path if it exists, otherwise from base model
    if os.path.exists(os.path.join(args.model_dir, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
        print("Loaded tokenizer from model directory")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
        print("Loaded tokenizer from base model")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Set offload folder if provided
    offload_folder = args.offload_folder if args.offload_folder else None
    
    # Check if it's a PEFT model by looking for adapter_config.json
    is_peft_model = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))
    
    if is_peft_model:
        print("Detected PEFT/LoRA model, loading base model and applying adapters...")
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=offload_folder
        )
        
        # Apply PEFT adapters
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        print("Successfully loaded PEFT model")
    else:
        print("Loading full model directly...")
        # Load the full model directly
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map="auto",
            trust_remote_code=True,
            offload_folder=offload_folder
        )
        print("Successfully loaded full model")
    
    model.eval()
    return model, tokenizer

def main():
    args = parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = get_default_output_file(args.training_type, args.model_dir)
    
    # Handle S3 data path
    if args.data.startswith("s3://"):
        local_data_path = sync_data_from_s3(args.data)
        test_data_path = local_data_path
    else:
        test_data_path = args.data
    
    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow: Using tracking server at {mlflow.get_tracking_uri()}")
    else:
        mlflow.set_tracking_uri("./mlruns")
        print("MLflow: No tracking URI specified. Defaulting to local './mlruns' directory.")
    
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.start_run()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Log parameters
    mlflow.log_params({
        "model_dir": args.model_dir,
        "base_model": args.base_model,
        "data": args.data,
        "training_type": args.training_type,
        "max_len": args.max_len,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "idp_mode": args.idp_mode,
        "load_in_4bit": args.load_in_4bit,
        "val_per_class": args.val_per_class,
        "offload_folder": args.offload_folder
    })
    
    # Load test dataset
    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    
    # If val_per_class is specified and > 0, limit the dataset
    if args.val_per_class > 0:
        # This is a simple implementation - you might need to adjust based on your data structure
        test_dataset = test_dataset.select(range(min(args.val_per_class, len(test_dataset))))
    
    try:
        # Evaluate based on training type
        if args.training_type == "dpo":
            results, metrics = evaluate_dpo(model, tokenizer, test_dataset, args)
        elif args.training_type == "kd":
            results, metrics = evaluate_kd(model, tokenizer, test_dataset, args)
        elif args.training_type == "sft":
            results, metrics = evaluate_sft(model, tokenizer, test_dataset, args)
        else:
            raise ValueError(f"Unknown training type: {args.training_type}")
        
        # Save results
        with open(args.output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Log metrics and artifacts
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        mlflow.log_artifact(args.output_file, "results")
        
        print(f"Evaluation completed. Results saved to {args.output_file}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.log_param("error", str(e))
        raise
    
    finally:
        mlflow.end_run()

if __name__ == "__main__":
    main()