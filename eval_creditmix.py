#!/usr/bin/env python
# eval_creditmix.py  (balanced + length-normalized + multi-null calibrated)
import argparse, json, random, math, os, torch, torch.nn.functional as F
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from peft import AutoPeftModelForCausalLM

LABELS = ["Good", "Standard", "Bad"]
NULL_FEATURES = ["", "N/A", "None", "Unknown", " ", "---", "###"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/data/creditmix_dataset.json")
    ap.add_argument("--model_dir", required=True,
                    help="Either a PEFT adapter folder OR a base HF model id (e.g. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Ignored if --val_per_class is set")
    ap.add_argument("--val_per_class", type=int, default=None, help="Exact number of validation examples per label (balanced)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--offload_folder", default=None)
    return ap.parse_args()

def make_prompt(tok, features: str) -> str:
    sys = "You are a careful credit risk classifier."
    user = (
        "Classify the credit risk into exactly one of {Good, Standard, Bad}.\n\n"
        f"Features:\n{features}\n\nReturn only the label."
    )
    msgs = [{"role":"system","content":sys},
            {"role":"user","content":user}]
    # IMPORTANT: score at assistant start
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def label_logprob(model, tok, prompt_text: str, label: str) -> float:
    """
    Robust, length-normalized log P(label | prompt) at assistant start.
    - no special tokens; manual concat
    - try with/without leading space
    - skip zero-length variants; defensive alignment
    """
    p_ids = tok.encode(prompt_text, add_special_tokens=False) or []

    def score(label_text: str):
        lab_ids = tok.encode(label_text, add_special_tokens=False)
        L = len(lab_ids)
        if L == 0:
            return -math.inf, 0
        combo = p_ids + lab_ids
        inp = torch.tensor([combo], dtype=torch.long, device=model.device)
        out = model(input_ids=inp, use_cache=False)
        logits = out.logits[0, -L-1:-1, :]
        if logits.shape[0] != L:
            logits = out.logits[0, -L:, :]
            logits = torch.roll(logits, shifts=1, dims=0)[:-1, :]
        logprobs = F.log_softmax(logits, dim=-1)
        tgt = torch.tensor(lab_ids, dtype=torch.long, device=logprobs.device)
        token_lp = logprobs.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum()), L

    s1, L1 = score(" " + label)
    s2, L2 = score(label)
    cand = []
    if L1 > 0: cand.append(s1 / L1)
    if L2 > 0: cand.append(s2 / L2)
    return max(cand) if cand else float("-inf")

def build_balanced_val(rows, val_frac, val_per_class):
    buckets = defaultdict(list)
    for r in rows:
        y = r["answer"].strip()
        if y in LABELS:
            buckets[y].append(r)
    for lab in LABELS:
        random.shuffle(buckets[lab])

    if val_per_class is None:
        total = sum(len(buckets[l]) for l in LABELS)
        target_total = max(1, int(total * val_frac))
        per_class = max(1, target_total // len(LABELS))
        per_class = min(per_class, min(len(buckets[l]) for l in LABELS))
    else:
        per_class = min(val_per_class, min(len(buckets[l]) for l in LABELS))

    val = []
    for lab in LABELS:
        val.extend(buckets[lab][:per_class])
    random.shuffle(val)
    return val, per_class, {lab: len(buckets[lab]) for lab in LABELS}

def load_model_and_tokenizer(args):
    """
    If args.model_dir looks like a PEFT adapter (has adapter_config.json), load via AutoPEFT.
    Otherwise treat it as a base model id/folder and load via AutoModelForCausalLM.
    """
    is_adapter = os.path.isdir(args.model_dir) and os.path.exists(
        os.path.join(args.model_dir, "adapter_config.json")
    )

    # Tokenizer can always come from model_dir / HF id
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if is_adapter:
        # Finetuned adapter + base
        if args.offload_folder:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_dir, device_map="auto", offload_folder=args.offload_folder,
                dtype=torch.bfloat16, trust_remote_code=True
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model_dir, device_map={"": 0},
                dtype=torch.bfloat16, trust_remote_code=True
            )
    else:
        # Base student model (HF id or folder)
        if args.offload_folder:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, device_map="auto", offload_folder=args.offload_folder,
                dtype=torch.bfloat16, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, device_map={"": 0},
                dtype=torch.bfloat16, trust_remote_code=True
            )

    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id
    return model, tok

def main():
    args = parse_args()
    random.seed(args.seed)

    with open(args.data, "r", encoding="utf-8") as f:
        rows = json.load(f)

    val, per_class, orig_dist = build_balanced_val(rows, args.val_frac, args.val_per_class)

    # Load either PEFT adapter+base OR base student (auto-detected)
    model, tok = load_model_and_tokenizer(args)

    # ---- multi-null contextual calibration (averaged priors) ----
    null_prompts = [
        make_prompt(tok, nf if nf != "" else " ")
        for nf in NULL_FEATURES
    ]
    prior = {lab: 0.0 for lab in LABELS}
    for np in null_prompts:
        for lab in LABELS:
            prior[lab] += label_logprob(model, tok, np, lab)
    for lab in LABELS:
        prior[lab] /= len(null_prompts)

    # ---- evaluate ----
    y_true, y_pred = [], []
    for r in val:
        pt = make_prompt(tok, r["question"])
        scores = {lab: label_logprob(model, tok, pt, lab) - prior[lab] for lab in LABELS}
        pred = max(scores, key=scores.get)
        y_pred.append(pred)
        y_true.append(r["answer"].strip())

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    print("Original distribution:", orig_dist)
    print(f"Balanced validation per class: {per_class}  (total {per_class * len(LABELS)})")
    print("Label distribution (val true):", Counter(y_true))
    print("Label distribution (val pred):", Counter(y_pred))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")
    print("\nClassification Report: y_true, y_pred...")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix: Good, Standard, Bad...")
    print(confusion_matrix(y_true, y_pred, labels=["Good","Standard","Bad"]))

if __name__ == "__main__":
    main()
