
#Purpose: compute Recall@K / MRR

# src/evaluation/metrics_retrieval.py


from __future__ import annotations
from typing import List, Set, Dict, Tuple

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    return 1.0 if any(i in relevant for i in topk) else 0.0

def mrr(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    for rank, idx in enumerate(retrieved[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0

def aggregate_metrics(
    all_retrieved: List[List[int]],
    all_relevant: List[Set[int]],
    k_list: List[int],
) -> Dict[str, float]:
    n = len(all_retrieved)
    out: Dict[str, float] = {}
    for k in k_list:
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)) / max(1, n)
    
    kmax = max(k_list) if k_list else 10
    out[f"mrr@{kmax}"] = sum(mrr(r, rel, kmax) for r, rel in zip(all_retrieved, all_relevant)) / max(1, n)
    out["n_queries"] = float(n)
    return out
