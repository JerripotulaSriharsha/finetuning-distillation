#!/usr/bin/env python
import argparse, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

LABELS = ["Good","Standard","Bad"]
NULL_FEATURES = ["", "N/A", "None", "Unknown", " ", "---", "###"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--features", required=True)
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
    # ask model to generate as assistant next:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def label_logprob(model, tok, prompt_text: str, label: str) -> float:
    """
    Length-normalized log P(label | prompt) at assistant start.
    - no special tokens; manual concat
    - try with/without leading space
    - skip zero-length variants; defensive alignment
    """
    import math
    # tokenize chat-templated prompt once
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
    cands = []
    if L1 > 0: cands.append(s1 / L1)
    if L2 > 0: cands.append(s2 / L2)
    return max(cands) if cands else float("-inf")

def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load LoRA adapter + base
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
    model.eval(); model.config.use_cache = True
    model.config.pad_token_id = tok.pad_token_id

    # ---- multi-null calibration (average several priors) ----
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

    # ---- score current sample ----
    pt = make_prompt(tok, args.features)
    scores = {lab: label_logprob(model, tok, pt, lab) - prior[lab] for lab in LABELS}
    pred = max(scores, key=scores.get)
    print(pred)

if __name__ == "__main__":
    main()
