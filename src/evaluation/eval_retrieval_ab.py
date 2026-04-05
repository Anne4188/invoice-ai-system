

# src/evaluation/eval_retrieval_ab.py

from __future__ import annotations

import os, json, argparse
from typing import Dict, Any, List, Set

import numpy as np

from src.evaluation.metrics_retrieval import aggregate_metrics


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_to_txt_name(filename: str) -> str:
    """
    Convert invoice_01.jpg/png/pdf -> invoice_01.txt
    Convert invoice_01 (no ext) -> invoice_01.txt
    If already .txt -> keep
    """
    if not filename:
        return ""
    base = filename.strip()

    # strip directories
    base = base.split("/")[-1]

    lower = base.lower()
    if lower.endswith(".txt"):
        return base

    for ext in [".jpg", ".jpeg", ".png", ".pdf"]:
        if lower.endswith(ext):
            return base[: -len(ext)] + ".txt"

    return base + ".txt"


def load_ocr_text(ocr_dir: str, filename: str) -> str:
    """
    Try multiple locations for OCR txt:
      1) ocr_dir/<txt_name>
      2) ocr_dir/raw/<txt_name>
      3) ocr_dir/sroie2019_train/<txt_name>
    """
    txt_name = normalize_to_txt_name(filename)
    if not txt_name:
        return ""

    candidates = [
        os.path.join(ocr_dir, txt_name),
        os.path.join(ocr_dir, "raw", txt_name),
        os.path.join(ocr_dir, "sroie2019_train", txt_name),
    ]

    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return open(path, "r", encoding="utf-8", errors="ignore").read()

    return ""



def build_text_repr(ocr_text: str, fields: Dict[str, Any], mode: str) -> str:
    """
    mode:
      - ocr_only: just OCR
      - fields_only: structured fields only
      - hybrid: fields + OCR (recommended)
    """
    ocr_text = (ocr_text or "").strip()
    merchant = (fields.get("merchant") or "").strip()
    date = (fields.get("date") or "").strip()
    total = fields.get("total_amount")
    currency = (fields.get("currency") or "").strip()

    parts = []
    if mode in ("fields_only", "hybrid"):
        parts.append(f"merchant: {merchant}")
        parts.append(f"date: {date}")
        parts.append(f"total_amount: {total}")
        parts.append(f"currency: {currency}")

    if mode in ("ocr_only", "hybrid"):
        parts.append("ocr:")
        parts.append(ocr_text)

    return "\n".join([p for p in parts if p is not None])


def build_relevance_same_merchant(merchants: List[str]) -> List[Set[int]]:
    """
    relevant(i) = invoices with same (non-empty) merchant, excluding self.
    IMPORTANT: empty merchant should NOT be grouped together.
    """
    inv: Dict[str, List[int]] = {}
    for i, m in enumerate(merchants):
        m = (m or "").strip()
        if not m:
            continue
        inv.setdefault(m, []).append(i)

    rels: List[Set[int]] = []
    for i, m in enumerate(merchants):
        m = (m or "").strip()
        if not m:
            rels.append(set())
            continue
        cands = set(inv.get(m, []))
        cands.discard(i)
        rels.append(cands if cands else set())
    return rels


def run_tfidf(texts: List[str], relevant: List[Set[int]], k_list: List[int]) -> Dict[str, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)  # (N, D) sparse

    kmax = max(k_list)
    all_retrieved: List[List[int]] = []
    all_rel: List[Set[int]] = []

    for i in range(X.shape[0]):
        if not relevant[i]:
            continue
        sims = cosine_similarity(X[i], X).ravel()
        sims[i] = -1.0
        top = np.argsort(-sims)[:kmax].tolist()
        all_retrieved.append(top)
        all_rel.append(relevant[i])

    return aggregate_metrics(all_retrieved, all_rel, k_list)


def run_clip_text(texts: List[str], relevant: List[Set[int]], k_list: List[int]) -> Dict[str, float]:
   
    import faiss
    from src.embeddings.clip_encoder import ClipEncoder

    enc = ClipEncoder()
    embs = np.stack([enc.encode_text(t) for t in texts]).astype("float32")  # (N, 512)

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine if vectors are L2-normalized
    index.add(embs)

    kmax = max(k_list) + 1
    all_retrieved: List[List[int]] = []
    all_rel: List[Set[int]] = []

    for i in range(len(texts)):
        if not relevant[i]:
            continue
        D, I = index.search(embs[i:i + 1], kmax)
        ids = [int(x) for x in I[0].tolist() if int(x) != i]
        ids = ids[:max(k_list)]
        all_retrieved.append(ids)
        all_rel.append(relevant[i])

    return aggregate_metrics(all_retrieved, all_rel, k_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="jsonl with fields (file, merchant, ...)")
    ap.add_argument("--ocr_dir", required=True, help="dir with OCR .txt")
    ap.add_argument("--k", default="1,3,5,10")
    ap.add_argument("--repr", default="hybrid", choices=["ocr_only", "fields_only", "hybrid"])
    ap.add_argument("--out", default="data/eval/retrieval_metrics_ab.json")
    ap.add_argument("--methods", default="tfidf", help="comma-separated: tfidf,clip (default=tfidf)")

    args = ap.parse_args()
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}



    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    rows = load_jsonl(args.preds)

    texts: List[str] = []
    merchants: List[str] = []
    miss_ocr = 0

    for r in rows:
        # compatible with both formats:
        # 1) {"file": "...txt", "merchant": "...", ...}
        # 2) {"invoice_id": "...jpg", "pred_fields": {...}}
        fn = r.get("file") or r.get("invoice_id")
        fields = r if "merchant" in r else (r.get("pred_fields") or {})

        ocr = load_ocr_text(args.ocr_dir, fn) if fn else ""
        if not ocr.strip():
            miss_ocr += 1

        txt = build_text_repr(ocr_text=ocr, fields=fields, mode=args.repr)
        texts.append(txt)
        merchants.append((fields.get("merchant") or "").strip())

    relevant = build_relevance_same_merchant(merchants)

    n_total = len(texts)
    n_queries = sum(1 for rel in relevant if rel)
    print(f"[data] total={n_total}, queries_with_relevant={n_queries}")
    print(f"[ocr] missing_or_empty={miss_ocr}/{n_total} ({(miss_ocr/n_total*100 if n_total else 0):.1f}%)")


    tfidf_metrics = run_tfidf(texts, relevant, k_list) if "tfidf" in methods else {}
    clip_metrics = run_clip_text(texts, relevant, k_list) if "clip" in methods else {}


    result = {
        "preds": args.preds,
        "ocr_dir": args.ocr_dir,
        "repr": args.repr,
        "k_list": k_list,
        "n_total": n_total,
        "queries_with_relevant": n_queries,
        "ocr_missing_or_empty": miss_ocr,
        "tfidf": tfidf_metrics,
        "clip_text": clip_metrics,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== TF-IDF ===")
    for k in k_list:
        print(f"Recall@{k}: {tfidf_metrics.get(f'recall@{k}', 0):.4f}")
    print(f"MRR@{max(k_list)}: {tfidf_metrics.get(f'mrr@{max(k_list)}', 0):.4f}")

    if "clip" in methods:
        print("\n=== CLIP text encoder ===")
        for k in k_list:
            print(f"Recall@{k}: {clip_metrics.get(f'recall@{k}', 0):.4f}")
        print(f"MRR@{max(k_list)}: {clip_metrics.get(f'mrr@{max(k_list)}', 0):.4f}")
    else:
        print("\n=== CLIP skipped ===")


    print(f"\n[done] wrote -> {args.out}")


if __name__ == "__main__":
    main()
