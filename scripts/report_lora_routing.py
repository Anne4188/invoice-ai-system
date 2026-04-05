#!/usr/bin/env python

#scripts/report_lora_routing.py

import argparse, json, os, re

BAD_CAND_PAT = re.compile(r"^(invoice|tax invoice|receipt|total|subtotal)$", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="jsonl from run_pipeline_on_ocr.py")
    ap.add_argument("--conf_th", type=float, default=0.75, help="LoRA routing threshold")
    args = ap.parse_args()

    n = 0
    routed = 0
    refined = 0
    blocked = 0

    # “changed”需要对比 base vs final，所以这里要求你在 preds 里有 base_merchant 字段
    # 如果你当前 run_pipeline_on_ocr.py 没写 base_merchant，也没关系：先输出触发率+refined_by
    changed = 0
    has_base = False

    with open(args.preds, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            n += 1

            conf = (obj.get("_conf") or {}).get("merchant", 1.0)
            meta = obj.get("_meta") or {}
            cand_raw = meta.get("merchant_candidate_raw") or (obj.get("merchant") or "")

            if conf < args.conf_th:
                routed += 1

            if BAD_CAND_PAT.match((cand_raw or "").strip()):
                blocked += 1

            if meta.get("merchant_refined_by") == "lora":
                refined += 1

            if "base_merchant" in obj:
                has_base = True
                if (obj.get("base_merchant") or "") != (obj.get("merchant") or ""):
                    changed += 1

    print(f"[n] {n}")
    print(f"[route] merchant_conf < {args.conf_th}: {routed}/{n} ({(routed/n*100 if n else 0):.1f}%)")
    print(f"[refined] refined_by=lora: {refined}/{n} ({(refined/n*100 if n else 0):.1f}%)")
    print(f"[blocked] BAD_CAND_PAT blocked: {blocked}/{n} ({(blocked/n*100 if n else 0):.1f}%)")

    if has_base:
        print(f"[changed] base!=final: {changed}/{n} ({(changed/n*100 if n else 0):.1f}%)")
    else:
        print("[changed] (skip) no base_merchant in preds jsonl")

if __name__ == "__main__":
    main()
