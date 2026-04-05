

# src/evaluation/eval_metrics.py

import json
from collections import defaultdict
import os

LABELS_PATH = "data/eval/labels.jsonl"
PREDS_PATH = "data/eval/preds.jsonl"

# src/evaluation/eval_metrics.py
import json
from collections import defaultdict
import os

LABELS_PATH = "data/eval/labels.jsonl"
PREDS_PATH = "data/eval/preds.jsonl"


def load_labels(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                #  Skip blank lines directly
                continue
            if line.startswith("#"):
                # The comment lines are also skipped
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f" JSON parsing failed: {path}:{lineno}")
                print(f"   This line of content is：{repr(line)}")
                raise  
            gold[item["invoice_id"]] = item
    return gold



def load_preds(path):
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            pred = item.get("pred_fields") or {}
            preds[item["invoice_id"]] = pred
    return preds


def main():
    if not os.path.exists(LABELS_PATH):
        print(f"The standard answer file cannot be found: {LABELS_PATH}")
        return
    if not os.path.exists(PREDS_PATH):
        print(f"The standard answer file cannot be found: {PREDS_PATH}， first run run_eval_predictions.py")
        return

    gold = load_labels(LABELS_PATH)
    preds = load_preds(PREDS_PATH)

    fields = ["merchant", "date", "total_amount", "tax", "currency"]

    correct = defaultdict(int)
    total = defaultdict(int)
    invoice_all_correct = 0
    invoice_total = 0

    for inv_id, gold_item in gold.items():
        if inv_id not in preds:
            print(f"[WARN] No {inv_id} prediction found，skipping")
            continue

        pred_item = preds[inv_id]
        all_ok = True

        for f in fields:
            g = gold_item.get(f)
            p = pred_item.get(f)

            
            if f in ["total_amount", "tax"]:
                if g is None and p is None:
                    correct[f] += 1
                elif g is not None and p is not None and abs(float(g) - float(p)) < 0.01:
                    correct[f] += 1
                else:
                    all_ok = False
            else:
                if g == p:
                    correct[f] += 1
                else:
                    all_ok = False

            total[f] += 1

        if all_ok:
            invoice_all_correct += 1
        invoice_total += 1

    print("=== Field-level accuracy ===")
    for f in fields:
        acc = correct[f] / total[f] if total[f] > 0 else 0.0
        print(f"{f:13s} : {acc*100:5.1f}% ({correct[f]}/{total[f]})")

    print("\n=== Invoice-level exact match ===")
    if invoice_total > 0:
        print(f"{invoice_all_correct} / {invoice_total} invoices "
              f"({invoice_all_correct/invoice_total*100:.1f}%)")
    else:
        print("No invoices evaluated.")


if __name__ == "__main__":
    main()




