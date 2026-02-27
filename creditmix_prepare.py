#!/usr/bin/env python
import json, random, os, argparse
random.seed(7)

LABELS = ["Good","Standard","Bad"]

def balanced_indices(rows, labels, per_class=None):
    buckets = {lab: [] for lab in labels}
    for i, r in enumerate(rows):
        y = r["answer"].strip()
        if y in buckets:
            buckets[y].append(i)
    if any(len(buckets[lab]) == 0 for lab in labels):
        raise ValueError("One or more labels have zero examples.")
    if per_class is None:
        per_class = min(len(buckets[lab]) for lab in labels)
    for lab in labels:
        if len(buckets[lab]) < per_class:
            raise ValueError(f"Label '{lab}' has only {len(buckets[lab])} examples, less than per_class={per_class}.")
    keep = []
    for lab in labels:
        keep.extend(random.sample(buckets[lab], per_class))
    random.shuffle(keep)
    return keep, per_class, {lab: len(buckets[lab]) for lab in labels}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data/creditmix_dataset.json")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--per_class", type=int, default=None, help="Optional cap per class; default=min class size")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.src, "r", encoding="utf-8") as f:
        rows = json.load(f)

    keep, per_class, dist = balanced_indices(rows, LABELS, args.per_class)
    print("Original distribution:", dist)
    print("Using per_class =", per_class, "=> total kept =", per_class*len(LABELS))

    def make_prompt(q):
        return (
            "Classify the credit risk into one of {Good, Standard, Bad}.\n\n"
            f"Features:\n{q}\n\nReturn only the label."
        )

    sft_path = os.path.join(args.outdir, "creditmix_sft.jsonl")
    with open(sft_path, "w", encoding="utf-8") as sft:
        for i in keep:
            r = rows[i]
            sft.write(json.dumps({"input": make_prompt(r["question"]),
                                  "output": r["answer"].strip()}, ensure_ascii=False) + "\n")

    kd_path = os.path.join(args.outdir, "creditmix_kd.jsonl")
    with open(kd_path, "w", encoding="utf-8") as kd:
        for i in keep:
            r = rows[i]
            kd.write(json.dumps({"input": make_prompt(r["question"]),
                                 "teacher_output": r["answer"].strip()}, ensure_ascii=False) + "\n")

    dpo_path = os.path.join(args.outdir, "creditmix_dpo.jsonl")
    with open(dpo_path, "w", encoding="utf-8") as dpo:
        for i in keep:
            r = rows[i]
            y = r["answer"].strip()
            negs = [l for l in LABELS if l != y]
            rej = random.choice(negs)
            dpo.write(json.dumps({"prompt": make_prompt(r["question"]),
                                  "chosen": y, "rejected": rej}, ensure_ascii=False) + "\n")

    print("Balanced per class written:", {lab: per_class for lab in LABELS})
    print("Wrote:", sft_path, kd_path, dpo_path)

if __name__ == "__main__":
    main()
