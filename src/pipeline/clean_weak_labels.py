# src/pipeline/clean_weak_labels.py

import json
import re
import os

SRC_TRAIN = "data/processed/weak_labels/train.jsonl"
SRC_VAL   = "data/processed/weak_labels/val.jsonl"
OUT_DIR   = "data/processed/weak_labels_clean"

BAD_PAT = re.compile(
    r"(thank you|please come again|bill#|gst summary|tax summary|subtotal|qty|u\.price|length\s*(of|af)?\s*stay|parking|invoice no|receipt no|table|cashier|change|rounding)",
    re.I
)

def keep(obj):
    cand = (obj.get("meta", {}).get("candidate_raw", "") or "").strip()
    tgt  = (obj.get("target", "") or "").strip()

    
    if BAD_PAT.search(cand):
        return False

    
    digit_ratio = sum(c.isdigit() for c in tgt) / max(1, len(tgt))
    if digit_ratio > 0.25:
        return False

    
    if len(tgt) < 4:
        return False

    return True


def clean(src, dst):
    kept = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if keep(obj):
                kept.append(obj)

    with open(dst, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return len(kept)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_train = clean(SRC_TRAIN, os.path.join(OUT_DIR, "train.jsonl"))
    n_val   = clean(SRC_VAL,   os.path.join(OUT_DIR, "val.jsonl"))

    print(f"[clean] wrote {n_train} train and {n_val} val -> {OUT_DIR}")


if __name__ == "__main__":
    main()
