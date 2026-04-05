
#scripts/run_pipeline_on_ocr.py


import argparse
import json
import os
from glob import glob
from tqdm import tqdm
from openai import OpenAI


from src.extraction.field_extraction_pipeline import (
    extract_invoice_fields_pipeline,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr_dir", required=True, help="dir with .txt OCR files")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--use_llm", action="store_true", help="enable LLM refinement")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    txt_files = sorted(glob(os.path.join(args.ocr_dir, "*.txt")))
    if not txt_files:
        raise RuntimeError(f"No .txt found under: {args.ocr_dir}")
    
    client = OpenAI() if args.use_llm else None  #Decide whether to use an LLM based on the conditions

    print(f"[info] found {len(txt_files)} OCR files")
    print(f"[info] use_llm = {args.use_llm}")

    with open(args.out, "w", encoding="utf-8") as fout:
        for p in tqdm(txt_files):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                ocr_text = f.read()

            fields = extract_invoice_fields_pipeline(
                ocr_text,
                llm_client=client,     #llm_client=None
                use_llm=args.use_llm,  #llm_client=None
                debug=False,
            )

            rec = {
                "file": os.path.basename(p),
                **fields,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote -> {args.out}")


if __name__ == "__main__":
    main()
