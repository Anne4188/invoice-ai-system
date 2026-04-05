
# src/evaluation/ocr_eval_retrieval.py


from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Doc:
    doc_id: str
    merchant: str
    text: str


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_docs(preds_jsonl: str, ocr_dir: str) -> List[Doc]:
    rows = read_jsonl(preds_jsonl)

    docs: List[Doc] = []
    for r in rows:
        fn = r.get("file") or r.get("id") or r.get("name")
        if not fn:
            continue

        merchant = (r.get("merchant") or "").strip()
        if not merchant:
            continue

    
        ocr_path = r.get("ocr_path") or os.path.join(ocr_dir, fn)
        if not os.path.exists(ocr_path):
            
            base = os.path.splitext(fn)[0] + ".txt"
            ocr_path2 = os.path.join(ocr_dir, base)
            if os.path.exists(ocr_path2):
                ocr_path = ocr_path2
            else:
                continue

        with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        docs.append(Doc(doc_id=fn, merchant=merchant, text=text))

    return docs


def encode_texts_tfidf(texts: List[str]) -> np.ndarray:
    
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vec.fit_transform(texts)  # (n, vocab) sparse
    X = X.astype(np.float32)
    return X.toarray()


def build_faiss_index(emb: np.ndarray):
   
    import faiss

    emb = emb.astype(np.float32)
    # normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def make_relevant_sets(docs: List[Doc]) -> List[set]:
    
    merchant2idx: Dict[str, List[int]] = {}
    for i, d in enumerate(docs):
        merchant2idx.setdefault(d.merchant, []).append(i)

    rel: List[set] = []
    for i, d in enumerate(docs):
        same = set(merchant2idx.get(d.merchant, []))
        if i in same:
            same.remove(i)
        rel.append(same)
    return rel


def eval_retrieval(index, emb: np.ndarray, rel_sets: List[set], k_list: List[int]) -> Dict:
    import faiss

    n = emb.shape[0]
    max_k = max(k_list)

    D, I = index.search(emb, max_k + 1)  # I: (n, max_k+1)

    valid = 0
    hit_at_k = {k: 0 for k in k_list}
    rr_sum = 0.0

    for i in range(n):
        rel = rel_sets[i]
        if not rel:
            continue  
        valid += 1

        
        retrieved = [j for j in I[i].tolist() if j != i]

        # Recall@k：top-k 
        for k in k_list:
            topk = retrieved[:k]
            if any(j in rel for j in topk):
                hit_at_k[k] += 1

        # MRR
        rank = None
        for r, j in enumerate(retrieved[:max_k], start=1):
            if j in rel:
                rank = r
                break
        if rank is not None:
            rr_sum += 1.0 / rank

    metrics = {"n_total": n, "n_valid_queries": valid}
    for k in k_list:
        metrics[f"recall@{k}"] = (hit_at_k[k] / valid) if valid else 0.0
    metrics["mrr"] = (rr_sum / valid) if valid else 0.0
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="jsonl with file+merchant (e.g. preds_regex_lora.jsonl)")
    ap.add_argument("--ocr_dir", required=True, help="directory containing OCR .txt")
    ap.add_argument("--k", default="1,3,5,10", help="comma-separated k list")
    ap.add_argument("--out", default="", help="optional metrics json output")
    args = ap.parse_args()

    k_list = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    docs = load_docs(args.preds, args.ocr_dir)
    if len(docs) < 2:
        raise SystemExit(f"[eval] not enough docs loaded: {len(docs)}")

    texts = [d.text for d in docs]
    emb = encode_texts_tfidf(texts)

    index, emb_norm = build_faiss_index(emb)
    rel_sets = make_relevant_sets(docs)

    metrics = eval_retrieval(index, emb_norm, rel_sets, k_list)

    print("\n=== Retrieval Metrics (same merchant = relevant) ===")
    print(f"n_total        : {metrics['n_total']}")
    print(f"n_valid_queries: {metrics['n_valid_queries']}")
    for k in k_list:
        print(f"recall@{k:<2}     : {metrics[f'recall@{k}']:.4f}")
    print(f"mrr            : {metrics['mrr']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] wrote -> {args.out}")


if __name__ == "__main__":
    main()
