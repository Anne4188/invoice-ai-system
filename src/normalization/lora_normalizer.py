





# src/normalization/lora_normalizer.py

from __future__ import annotations

import os
from typing import Optional, Tuple
from src.extraction.regex_extract import normalize_merchant



DEFAULT_TASK = os.getenv("LORA_TASK", "merchant_norm")

#The directory for training LoRA output (adapter weight directory)
DEFAULT_ADAPTER_DIR = os.getenv(
    "LORA_ADAPTER_DIR",
    "data/models/lora_merchant_norm",
)

# base model
DEFAULT_BASE_MODEL = os.getenv(
    "LORA_BASE_MODEL",
    "google/flan-t5-small",
)


DEFAULT_DEVICE = os.getenv("LORA_DEVICE", "cpu")


DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LORA_MAX_NEW_TOKENS", "16"))


# -----------------------
# Lazy-loaded singleton
# -----------------------
_MODEL = None
_TOKENIZER = None
_LOADED_KEY: Optional[Tuple[str, str, str]] = None  # (task, base_model, adapter_dir)


def _build_prompt(task: str, candidate: str, ocr_text: str) -> str:
    """
    Keep prompt consistent with weak-label data generation.
    """
    candidate = (candidate or "").strip()
    ocr_text = (ocr_text or "").strip()

    if task == "merchant_norm":
        return (
            "Normalize the merchant name from a receipt.\n"
            f"Candidate: {candidate}\n"
            "Receipt OCR:\n"
            f"{ocr_text}\n"
            "Return ONLY the normalized merchant name."
        )

    if task == "date_norm":
        return (
            "Normalize the invoice date to YYYY-MM-DD.\n"
            f"Candidate: {candidate}\n"
            "Receipt OCR:\n"
            f"{ocr_text}\n"
            "Return ONLY the normalized date."
        )

    # default fallback
    return (
        "Normalize the field from OCR.\n"
        f"Task: {task}\n"
        f"Candidate: {candidate}\n"
        "Receipt OCR:\n"
        f"{ocr_text}\n"
        "Return ONLY the normalized value."
    )


def _postprocess_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()

    
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    s = s.strip().strip('"').strip("'").strip()

    
    s = " ".join(s.split())
    return s


def _safe_load_model(task: str, base_model: str, adapter_dir: str, device: str):
    """
    Load base model + LoRA adapter using transformers + peft.

    If deps are missing OR adapter_dir not found, keep model as None.
    """
    global _MODEL, _TOKENIZER, _LOADED_KEY

    key = (task, base_model, adapter_dir)
    if _LOADED_KEY == key and _MODEL is not None and _TOKENIZER is not None:
        return

    # adapter_dir must exist
    if not os.path.isdir(adapter_dir):
        _MODEL, _TOKENIZER, _LOADED_KEY = None, None, None
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception:
        _MODEL, _TOKENIZER, _LOADED_KEY = None, None, None
        return

    try:
        from peft import PeftModel
    except Exception:
        _MODEL, _TOKENIZER, _LOADED_KEY = None, None, None
        return

    tok = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    #  LoRA adapter 
    model = PeftModel.from_pretrained(base, adapter_dir)

    # device
    try:
        model = model.to(device)
    except Exception:
        model = model.to("cpu")

    model.eval()

    _TOKENIZER = tok
    _MODEL = model
    _LOADED_KEY = key

#-----------------------------------------

def lora_normalize_merchant(
    candidate: str,
    ocr_text: str,
    *,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    device: str = DEFAULT_DEVICE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Optional[str]:
    """
    Return normalized merchant string, or None if:
      - model not available
      - output invalid/empty
    """
    task = "merchant_norm"
    _safe_load_model(task=task, base_model=base_model, adapter_dir=adapter_dir, device=device)

    if _MODEL is None or _TOKENIZER is None:
        return None

    prompt = _build_prompt(task=task, candidate=candidate, ocr_text=ocr_text)

    try:
        inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512)
        # move tensors to same device as model
        try:
            dev = next(_MODEL.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass

        out_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = _TOKENIZER.decode(out_ids[0], skip_special_tokens=True)
        text = _postprocess_text(text)

        

        if not text:
            return None

        # Then converge your existing rule base into canonical (for example, Wal-Mart -> WALMART)
        canon = normalize_merchant(text)
        return canon or text



    except Exception:
        return None
    
#-----------------------------------------

def lora_normalize_date(
    candidate: str,
    ocr_text: str,
    *,
    adapter_dir: str = os.getenv("LORA_ADAPTER_DIR_DATE", "data/models/lora_date_norm"),
    base_model: str = DEFAULT_BASE_MODEL,
    device: str = DEFAULT_DEVICE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Optional[str]:
    """
    Optional: normalized date string, or None if model not available.
    """
    task = "date_norm"
    _safe_load_model(task=task, base_model=base_model, adapter_dir=adapter_dir, device=device)

    if _MODEL is None or _TOKENIZER is None:
        return None

    prompt = _build_prompt(task=task, candidate=candidate, ocr_text=ocr_text)

    try:
        inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512)
        try:
            dev = next(_MODEL.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
        except Exception:
            pass

        out_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = _TOKENIZER.decode(out_ids[0], skip_special_tokens=True)
        text = _postprocess_text(text)

        if not text:
            return None

        return text

    except Exception:
        return None


def predict(
    candidate: str,
    ocr_text: str,
    task: str = "merchant_norm",
) -> Optional[str]:
    """
    Unified prediction entry for LoRA normalization.

    task:
      - "merchant_norm"
      - "date_norm"
    """
    if task == "merchant_norm":
        return lora_normalize_merchant(
            candidate=candidate,
            ocr_text=ocr_text,
        )

    if task == "date_norm":
        return lora_normalize_date(
            candidate=candidate,
            ocr_text=ocr_text,
        )

    return None
