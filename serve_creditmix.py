#!/usr/bin/env python
# FastAPI inference server for CreditMix (base + SFT/KD/DPO adapters)
# Endpoints:
#   POST /infer/base   {"features": "..."}
#   POST /infer/sft    {"features": "..."}
#   POST /infer/kd     {"features": "..."}
#   POST /infer/dpo    {"features": "..."}
#
# Run: uvicorn serve_creditmix:app --host 0.0.0.0 --port 8000
# Optional: OFFLOAD_FOLDER=/tmp/offload

'''

Example Input:
--------------
curl -s -X POST http://localhost:8000/infer/base -H "Content-Type: application/json" \
  -d '{"features":"Age: 23.0, Occupation: Scientist, Annual Income: 19114.12, Outstanding Debt: 809.98, Credit Utilization Ratio: 35.03, Payment Behaviour: Low_spent_Small_value_payments"}' | jq

Example Output:
---------------
{
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "prediction": "Standard",
  "scores": {
    "Good": -1.3392857142857144,
    "Standard": 1.2190290178571428,
    "Bad": -0.9034598214285714
  },
  "raw_scores": {
    "Good": -3.78125,
    "Standard": -0.275390625,
    "Bad": -1.5234375
  },
  "prior": {
    "Good": -2.4419642857142856,
    "Standard": -1.4944196428571428,
    "Bad": -0.6199776785714286
  },
  "probabilities": {
    "Good": 0.06468190283292168,
    "Standard": 0.8353043801933099,
    "Bad": 0.10001371697376847
  },
  "top2_margin": 2.1224888392857144
}

Example Justification:
----------------------
Great — that response means your endpoint is working and here’s how to read each field:

- model_name: which model handled the request. Here it’s the base student (Qwen/Qwen2.5-3B-Instruct).

- prediction: the final label picked: "Standard".

- raw_scores: the model’s length-normalized log-likelihoods for each label given your prompt, before any calibration. Higher = better. They’re in natural-log units (often negative).
    Here: Standard (–0.275) > Bad (–1.523) > Good (–3.781), so even uncalibrated it favored Standard.

- prior: the model’s label priors measured on several “content-free” prompts (our multi-null contextual calibration). This captures bias from the prompt/template itself.
    E.g., the model tends to like “Good” a bit more than “Bad” in the abstract, etc.

- scores: calibrated scores = raw_scores – prior. This removes the template bias. We pick the label with the highest calibrated score.

Computation shown by example:
    Standard: −0.27539 − (−1.49442) ≈ +1.21903 (highest) → prediction = Standard.

- probabilities: softmax over calibrated scores (handy confidence, sums to 1)

- top2_margin: separation between the best and second-best label (larger = more confident)

'''

import os, math, torch, torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from threading import Lock

LABELS = ["Good", "Standard", "Bad"]
NULL_FEATURES = ["", "N/A", "None", "Unknown", " ", "---", "###"]

MODEL_MAP: Dict[str, str] = {
    "base": "Qwen/Qwen2.5-3B-Instruct",
    "sft" : "runs/creditmix_sft",
    "kd"  : "runs/creditmix_kd",
    "dpo" : "runs/creditmix_dpo",
}

DTYPE = torch.bfloat16
DEVICE_MAP_AUTO = "auto"
OFFLOAD_FOLDER = os.environ.get("OFFLOAD_FOLDER", None)

# ---------- Schemas ----------
class InferRequest(BaseModel):
    features: str

class InferResponse(BaseModel):
    model_name: str
    prediction: Literal["Good","Standard","Bad"]
    scores: Dict[str, float]            # calibrated (raw - prior)
    raw_scores: Dict[str, float]        # uncalibrated (length-normalized log-likelihood)
    prior: Dict[str, float]             # averaged null prior per label
    probabilities: Dict[str, float]     # softmax(scores) -- not calibrated probabilities, but useful confidence
    top2_margin: float                  # score(top1) - score(top2)
# -----------------------------

_load_lock = Lock()
_tokenizers: Dict[str, AutoTokenizer] = {}
_models: Dict[str, torch.nn.Module] = {}
_priors: Dict[str, Dict[str, float]] = {}

app = FastAPI(title="CreditMix Distilled Inference Server")

def _is_peft_adapter(path_or_id: str) -> bool:
    return os.path.isdir(path_or_id) and os.path.exists(os.path.join(path_or_id, "adapter_config.json"))

def _load_model_and_tokenizer(key: str):
    model_dir = MODEL_MAP[key]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if _is_peft_adapter(model_dir):
        if OFFLOAD_FOLDER:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir, device_map=DEVICE_MAP_AUTO, offload_folder=OFFLOAD_FOLDER,
                dtype=DTYPE, trust_remote_code=True
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_dir, device_map={"": 0},
                dtype=DTYPE, trust_remote_code=True
            )
    else:
        if OFFLOAD_FOLDER:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, device_map=DEVICE_MAP_AUTO, offload_folder=OFFLOAD_FOLDER,
                dtype=DTYPE, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, device_map={"": 0},
                dtype=DTYPE, trust_remote_code=True
            )
    model.eval(); model.config.use_cache = True; model.config.pad_token_id = tok.pad_token_id
    return model, tok

def _make_prompt(tok: AutoTokenizer, features: str) -> str:
    sys = "You are a careful credit risk classifier."
    user = (
        "Classify the credit risk into exactly one of {Good, Standard, Bad}.\n\n"
        f"Features:\n{features}\n\nReturn only the label."
    )
    msgs = [{"role":"system","content":sys},{"role":"user","content":user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # assistant start

@torch.no_grad()
def _label_logprob(model, tok, prompt_text: str, label: str) -> float:
    p_ids = tok.encode(prompt_text, add_special_tokens=False) or []
    def score(label_text: str):
        lab_ids = tok.encode(label_text, add_special_tokens=False)
        L = len(lab_ids)
        if L == 0: return -math.inf, 0
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
        val = float(token_lp.sum())
        # free temporaries ASAP
        del out, logits, logprobs, tgt, token_lp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val, L
    s1, L1 = score(" " + label)
    s2, L2 = score(label)
    cands = []
    if L1 > 0: cands.append(s1 / L1)
    if L2 > 0: cands.append(s2 / L2)
    return max(cands) if cands else float("-inf")

def _ensure_loaded(key: str):
    with _load_lock:
        if key not in _tokenizers:
            model, tok = _load_model_and_tokenizer(key)
            _models[key] = model
            _tokenizers[key] = tok
        if key not in _priors:
            tok = _tokenizers[key]; model = _models[key]
            null_prompts = [_make_prompt(tok, nf if nf != "" else " ") for nf in NULL_FEATURES]
            prior = {lab: 0.0 for lab in LABELS}
            for np in null_prompts:
                for lab in LABELS:
                    prior[lab] += _label_logprob(model, tok, np, lab)
            # free VRAM during prior computation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            for lab in LABELS:
                prior[lab] /= len(null_prompts)
            _priors[key] = prior

def _softmax(d: Dict[str, float]) -> Dict[str, float]:
    # numerically stable softmax over dict values
    mx = max(d.values())
    exps = {k: math.exp(v - mx) for k, v in d.items()}
    Z = sum(exps.values())
    return {k: exps[k] / Z for k in d}

def _predict_for_key(key: str, features: str):
    _ensure_loaded(key)
    tok, model, prior = _tokenizers[key], _models[key], _priors[key]
    pt = _make_prompt(tok, features)
    raw_scores = {lab: _label_logprob(model, tok, pt, lab) for lab in LABELS}
    scores = {lab: raw_scores[lab] - prior[lab] for lab in LABELS}
    pred = max(scores, key=scores.get)
    # extras
    probs = _softmax(scores)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2_margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else float('inf')
    return pred, scores, raw_scores, prior, probs, top2_margin

# -------- endpoints --------
app = FastAPI(title="CreditMix Distilled Inference Server")

def _endpoint(key: str):
    async def handler(req: InferRequest) -> InferResponse:
        try:
            pred, scores, raw_scores, prior, probs, margin = _predict_for_key(key, req.features)
            return InferResponse(
                model_name=MODEL_MAP[key], prediction=pred,
                scores=scores, raw_scores=raw_scores, prior=prior,
                probabilities=probs, top2_margin=margin
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return handler

@app.post("/infer/base", response_model=InferResponse)
async def infer_base(req: InferRequest): return await _endpoint("base")(req)

@app.post("/infer/sft",  response_model=InferResponse)
async def infer_sft(req: InferRequest):  return await _endpoint("sft")(req)

@app.post("/infer/kd",   response_model=InferResponse)
async def infer_kd(req: InferRequest):   return await _endpoint("kd")(req)

@app.post("/infer/dpo",  response_model=InferResponse)
async def infer_dpo(req: InferRequest):  return await _endpoint("dpo")(req)

@app.get("/healthz")
async def health(): return {"ok": True, "models": list(MODEL_MAP.keys())}