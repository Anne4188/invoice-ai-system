
# src/retrieval/representations.py 



from __future__ import annotations
from typing import Dict, Any

def build_text_repr(
    ocr_text: str,
    fields: Dict[str, Any],
    mode: str = "hybrid",
) -> str:
    ocr_text = (ocr_text or "").strip()
    fields = fields or {}

    merchant = (fields.get("merchant") or "").strip()
    date = (fields.get("date") or "").strip()
    total = fields.get("total_amount")
    currency = (fields.get("currency") or "").strip()

    
    parts = []
    if merchant: parts.append(f"merchant: {merchant}")
    if date: parts.append(f"date: {date}")
    if total is not None: parts.append(f"total: {total}")
    if currency: parts.append(f"currency: {currency}")

    fields_text = "\n".join(parts).strip()

    if mode == "ocr_only":
        return ocr_text
    if mode == "fields_only":
        return fields_text
    # default: hybrid
    if fields_text and ocr_text:
        return fields_text + "\n\n" + ocr_text
    return fields_text or ocr_text
