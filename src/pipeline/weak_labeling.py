
# src/pipeline/weak_labeling.py

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from difflib import SequenceMatcher

from src.ocr.ocr_utils import run_ocr
from src.extraction import regex_extract as rx

import re

def is_plausible_merchant_line(s: str) -> bool:
    if not s:
        return False
    low = s.lower()

    
    bad = [
        "thank you", "please come again", "bill#", "invoice no", "receipt no",
        "table", "cashier", "change", "subtotal", "sub total", "total", "amount due",
        "gst summary", "tax summary", "qty", "u.price", "unit price", "price", "item",
        "length of stay", "parking", "rounding",
    ]
    if any(k in low for k in bad):
        return False

    
    digits = sum(c.isdigit() for c in s)
    if digits / max(1, len(s)) > 0.35:
        return False

    
    if not re.search(r"[A-Za-z]", s):
        return False

    return True

def ocr_to_text(file_path: str) -> str:
    """
    run_ocr returns:
      - str for images
      - dict for PDFs (page_1, page_2, ...)
    Normalize to a single string.
    """
    ocr_result = run_ocr(file_path)

    if ocr_result is None or ocr_result is ...:
        return ""

    if isinstance(ocr_result, dict):
        pages: List[str] = []
        for _, v in sorted(ocr_result.items()):
            if isinstance(v, str) and v.strip():
                pages.append(v)
        return "\n\n".join(pages)

    if isinstance(ocr_result, str):
        return ocr_result

    return str(ocr_result)


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def merchant_confidence(ocr_text: str, candidate_raw: Optional[str], canonical: Optional[str]) -> float:
    """
    Confidence heuristic:
      0.95: hits known merchant patterns in text (high confidence)
      0.90: strong fuzzy match to known merchants
      0.85: medium fuzzy match
      else: 0.0 (discard)
    """
    if not ocr_text or not candidate_raw or not canonical:
        return 0.0

    text_lower = ocr_text.lower()

    # 1) Hard hit: if OCR text contains known merchant pattern keywords
    # rx.MERCHANT_PATTERNS is defined in your regex_extract.py
    patterns = getattr(rx, "MERCHANT_PATTERNS", {})
    for canon, kws in patterns.items():
        for kw in kws:
            if kw and kw.lower() in text_lower:
                # if we also map to same canonical, even better
                if canon.upper() == canonical.upper():
                    return 0.95
                # still high confidence because text clearly contains a known merchant keyword
                return 0.92

    # 2) Fuzzy match candidate to known merchant list
    known = getattr(rx, "KNOWN_MERCHANTS", [])
    cand_u = str(candidate_raw).upper().strip()

    best = 0.0
    for km in known:
        r = fuzzy_ratio(cand_u, str(km).upper())
        if r > best:
            best = r

    if best >= 0.90:
        return 0.90
    if best >= 0.80:
        return 0.85

    # 3) Fallback: fuzzy candidate to canonical itself
    canon_u = str(canonical).upper().strip()
    self_r = fuzzy_ratio(cand_u, canon_u)
    if self_r >= 0.90:
        return 0.85

    return 0.0


def build_merchant_sample(ocr_text: str) -> Optional[Dict[str, Any]]:
    """
    Build one weakly-labeled training sample for merchant normalization.
    input: OCR text + candidate merchant
    target: canonical merchant (normalized)
    """
    #candidate_raw = rx.extract_merchant(ocr_text)  # raw best line / keyword matched canonical
    candidate_raw = rx.extract_merchant(ocr_text)
    if candidate_raw and not is_plausible_merchant_line(candidate_raw):
        return None

    canonical = rx.normalize_merchant(candidate_raw)

    conf = merchant_confidence(ocr_text, candidate_raw, canonical)

    if not canonical:
        return None

    # Keep prompt very simple & consistent for seq2seq fine-tuning later
    inp = (
        "Normalize the merchant name from a receipt.\n"
        f"Candidate: {candidate_raw}\n"
        "Receipt OCR:\n"
        f"{ocr_text}\n"
        "Return ONLY the normalized merchant name."
    )

    return {
        "task": "merchant_norm",
        "input": inp,
        "target": canonical,
        "confidence": conf,
        "meta": {
            "candidate_raw": candidate_raw,
        },
    }


def iter_files(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    files: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate weak-supervision datasets (JSONL) from invoice OCR + rules.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory containing raw invoices")
    parser.add_argument("--output_dir", type=str, default="data/processed/weak_labels", help="Output directory")
    parser.add_argument("--min_conf", type=float, default=0.85, help="Minimum confidence to keep a sample")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = no limit)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = iter_files(input_dir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"[weak_labeling] input_dir={input_dir} files={len(files)}")
    print(f"[weak_labeling] output_dir={output_dir}")
    print(f"[weak_labeling] min_conf={args.min_conf} val_ratio={args.val_ratio}")

    samples: List[Dict[str, Any]] = []
    skipped_ocr = 0
    skipped_lowconf = 0
    skipped_exception = 0

    for fp in files:
        try:
            ocr_text = ocr_to_text(str(fp))
            if not isinstance(ocr_text, str) or not ocr_text.strip():
                skipped_ocr += 1
                continue

            sample = build_merchant_sample(ocr_text)
            if not sample:
                skipped_lowconf += 1
                continue

            if sample["confidence"] < args.min_conf:
                skipped_lowconf += 1
                continue

            sample["meta"]["file_path"] = str(fp)
            sample["meta"]["file_name"] = fp.name
            samples.append(sample)

        except Exception as e:
            skipped_exception += 1
            print(f"[WARN] failed on {fp}: {e}")

    print(f"[weak_labeling] kept={len(samples)} skipped_ocr={skipped_ocr} skipped_lowconf={skipped_lowconf} skipped_exception={skipped_exception}")

    random.seed(args.seed)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[weak_labeling] wrote train={len(train_samples)} -> {train_path}")
    print(f"[weak_labeling] wrote   val={len(val_samples)} -> {val_path}")


if __name__ == "__main__":
    main()
