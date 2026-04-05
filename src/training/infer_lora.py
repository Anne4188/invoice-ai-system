

"""
Docstring for training.infer_lora
src/training/infer_lora.py 

"""


# src/training/infer_lora.py
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--base_model", default="google/flan-t5-small")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--ocr_text", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"model_dir not found: {args.model_dir}")

    prompt = (
        "Normalize the merchant name from a receipt.\n"
        f"Candidate: {args.candidate}\n"
        "Receipt OCR:\n"
        f"{args.ocr_text}\n"
        "Return ONLY the normalized merchant name."
    )

    tok = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.model_dir).to(args.device)
    model.eval()

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)

    text = tok.decode(out[0], skip_special_tokens=True).strip()
    print(text)


if __name__ == "__main__":
    main()
