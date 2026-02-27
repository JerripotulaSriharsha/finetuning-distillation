# Distill-SLM Accelerator — Cross-Domain & Cross-Industry Business Use Cases

## Executive Summary

The Distill-SLM accelerator provides a **production-ready path to shrink LLM capabilities into SLMs** that run on **commodity 8 GB GPUs**—without giving up core task quality. It ships **three finetuning methods** (SFT, KD, DPO) and **turnkey pipelines** for:

* **Classification** (e.g., CreditMix: *Good / Standard / Bad*),
* **Code Generation** (with optional plan-then-code),
* **Intelligent Document Parsing (IDP)** (schema-constrained JSON extraction),
  plus a **FastAPI service** for API-based deployment.

This document maps those capabilities to concrete business applications across industries, highlighting **direct** (out-of-the-box) and **indirect** (composable) use cases, expected value, and adoption patterns.

---

## What the Accelerator Enables

* **Operate advanced NLP on small hardware**: QLoRA adapters over 4-bit quantized base models—reliable on an 8 GB GPU.
* **Choose a training path that fits your data**:

  * **SFT** for labeled examples,
  * **KD** for model imitation/teacher compression,
  * **DPO** for preference optimization when you can rank outputs.
* **Consistency at inference** for classification via **likelihood ranking + multi-null calibration** (reduces label bias).
* **Strict structure outputs** for IDP using **JSON Schema**.
* **Simple API integration** via FastAPI endpoints (base, sft, kd, dpo).

---

## Core Capabilities → Business Patterns

### 1) Structured Classification (CreditMix-style)

**What it does:** Given feature strings, predicts a discrete label with calibrated confidence.

**Business patterns:**

* **Risk triage & prioritization:** “approve / manual review / deny,” “escalate / normal / defer.”
* **Customer health & churn risk:** low / medium / high.
* **Ticket routing & severity:** P1 / P2 / P3.
* **Content moderation tiers:** safe / review / remove.

**Why now:** The calibrated decoder mitigates “always-predict-majority” behavior, producing more reliable, auditable classification under tight compute budgets.

---

### 2) Intelligent Document Parsing (IDP)

**What it does:** Converts semi-structured/unstructured documents into **strict JSON** defined by your schema.

**Business patterns:**

* **Finance & Ops:** invoice/header extraction, PO line-items, receipts, expense reports.
* **Procurement & Logistics:** packing lists, bills of lading, customs docs, delivery notes.
* **HR & Legal Ops:** CVs, contracts, NDAs, MSAs—extract key clauses/fields.
* **Healthcare Admin:** insurance cards, referrals, discharge summaries (non-diagnostic metadata).

**Why now:** Schema-constrained decoding reduces post-processing and validation costs; deployable on a single workstation.

---

### 3) Code Generation (Plan-then-Code optional)

**What it does:** Generates code and can separate **planning** (spec, outline) from **final code**.

**Business patterns:**

* **Internal tooling & scripts:** ETL glue, report jobs, data sanity checks.
* **Migration helpers:** boilerplate scaffolds, adapter layers, simple tests.
* **Dev velocity & onboarding:** starter examples, repetitive code patterns.

**Why now:** You can scope the model to your domain with SFT/KD/DPO and run locally for privacy/compliance.

---

## Vertical Use Cases

### Financial Services & Fintech

* **Direct**: credit risk tiers (CreditMix), collections prioritization, KYC document extraction (IDP), invoice automation.
* **Indirect**: underwriting worklists, dispute document triage, portfolio operations scripts (codegen).

### Insurance

* **Direct**: claim form extraction, coverage/limit fields (IDP), FNOL triage classification.
* **Indirect**: rating factor calculators, policy endorsement script generation (codegen).

### Healthcare Admin (non-clinical)

* **Direct**: eligibility/benefit EOB extraction (IDP), referral document parsing, revenue cycle classification.
* **Indirect**: facility ops scripts (scheduling/export), prior-auth packet assembly.

### Retail & E-commerce

* **Direct**: invoice & receipt extraction (IDP), return/review classification, seller compliance checks.
* **Indirect**: catalog enrichment scripts, A/B test helper code (codegen).

### Manufacturing & Supply Chain

* **Direct**: packing lists, CoA/CoC fields (IDP), shipment risk flags (classification).
* **Indirect**: BOM transform & checks (codegen), automated handoff scripts between systems.

### Energy & Utilities

* **Direct**: work order extraction, safety report structuring (IDP), event/ticket severity classification.
* **Indirect**: data ingestion and reconciliation scripts (codegen).

### Telecom

* **Direct**: field report IDP, outage severity classification.
* **Indirect**: provisioning/config scripts (codegen), SLA monitoring helpers.

### Public Sector / Education

* **Direct**: form extraction (IDP), request/appeal triage (classification).
* **Indirect**: data quality scripts, template code for agency portals.

### Software / SaaS / DevOps

* **Direct**: support ticket classification/routing, runbook extraction (IDP).
* **Indirect**: CI/CD helpers, boilerplate services (codegen).

---

## Value Proposition

* **Lower TCO**: Fine-tuned quality on **single 8 GB GPU** reduces cloud GPU needs.
* **Data privacy & control**: On-prem, adapter-based finetuning keeps IP and examples in-house.
* **Faster iteration**: SFT/KD/DPO let you start with labeled data, teacher outputs, or preferences—whichever you have first.
* **Operational robustness**: Calibrated classification improves **recall/precision trade-offs** without model bloat.
* **Composable**: Train once; reuse **serve** and **eval** stacks across domains.

---

## Adoption Playbook

1. **Pick the path that fits your data**

   * Have labeled inputs/outputs → **SFT**.
   * Have a stronger teacher → **KD** (imitate traces).
   * Have ranked outputs (A vs B) → **DPO**.

2. **Start with a narrow schema or label set**

   * Keep the JSON schema small in IDP v1.
   * Begin with 2–4 labels in classification; expand later.

3. **Train adapters on 8 GB**

   * Use default QLoRA settings; monitor steps vs. VRAM.
   * Log metrics; checkpoint early to compare.

4. **Evaluate with the provided evaluator**

   * Use **balanced validation** (per-class or fraction).
   * Inspect **confusion matrix** to guide data curation.

5. **Deploy with the API server**

   * Start with `/infer/base` to establish a baseline.
   * Swap to `/infer/sft` or `/infer/kd` when metrics justify.
   * Turn on **CPU offload** if you spike VRAM on first request.

6. **Iterate**

   * Add hard negatives for the *confused* label(s).
   * Calibrate further by adjusting null-prompt sets.
   * For classification, consider small **label bias offsets** at inference (kept in calibrated score space) if one class is under-predicted.

---

## KPIs & Measurement

* **Classification**: Accuracy, Macro-F1, class-wise recall/precision, top-2 margin distribution.
* **IDP**: Schema validation rate, field-level precision/recall, manual correction time.
* **Codegen**: Build success rate, lint/static check pass rate, time saved per task.
* **Ops**: Latency per request (p50/p95), VRAM utilization, adapter load time, API uptime.

---

## Risks & Mitigations

* **Class collapse (one label dominates)**
  *Mitigate*: balanced training sets, multi-null calibration, small inference-time bias for under-predicted labels, curate hard examples.

* **Schema drift / invalid JSON** (IDP)
  *Mitigate*: strict schema enforcement (already built-in), add unit tests for real docs.

* **VRAM spikes** (first API call)
  *Mitigate*: enable CPU offload, keep 4-bit quantization on, reduce null prompts.

* **Preference mis-specification** (DPO)
  *Mitigate*: pilot on small subsets, validate with evaluator before scaling.

---

## Implementation Patterns

* **Single-box pilots**: One workstation (WSL2 + 8 GB GPU) runs *train → eval → serve* to prove value quickly.
* **Department rollout**: A small node hosts the FastAPI service; teams send feature strings, documents, or prompts.
* **Adapter catalog**: Maintain `runs/` for domain-specific SFT/KD/DPO adapters; swap at API time per product/team.

---

## Example Business Scenarios

* **Finance Ops**: Reduce invoice key-in by 70–90% with IDP; auto-classify credit risk to prioritize manual reviews; serve both via internal API.
* **Support & Success**: Route tickets by calibrated severity; summarize attachments via IDP; generate triage scripts via codegen.
* **Supply Chain**: Extract shipment fields; classify exception risk; auto-generate reconciliation scripts for ERPs.

---

## Next Steps

* **Expand label sets** progressively once baseline is stable.
* **Build a small “adapter registry”** mapping business domains to adapters.
* **Automate evaluation runs** (cron or CI) to monitor drift.
* **Add governance hooks**: log feature inputs and outputs for audit, version adapters in `runs/`.

---

### Bottom Line

This accelerator turns **finetuned performance** into a **small-footprint, API-deployable** capability that fits **real constraints**—hardware, privacy, and iteration speed—unlocking practical NLP/ML value across industries without heavyweight infrastructure.
