

#---------------------------
# src/evaluation/run_eval_predictions.py

import os
import json
import sys
import argparse

# Correct the project root directory to sys.path to ensure that src.xxx can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.rag.rag_pipeline import InvoiceRAGSystem
from openai import OpenAI

DATA_DIR = "data/eval"
OUTPUT_PATH = "data/eval/preds.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_llm", type=int, default=0, help="0 = regex+LoRA only, 1 = regex+LoRA+LLM")
    args = parser.parse_args()

    use_llm = bool(args.use_llm)

    # The OpenAI client is initialized only when use_llm=True
    client = OpenAI() if use_llm else None

    system = InvoiceRAGSystem(
        use_llm=use_llm,
        llm_client=client,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for filename in os.listdir(DATA_DIR):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
                continue

            file_path = os.path.join(DATA_DIR, filename)
            invoice_id = filename

            print(f"Processing {invoice_id} ...")
            ctx = system.process_invoice(file_path, invoice_id)
            fields = ctx["fields"]

            record = {
                "invoice_id": invoice_id,
                "pred_fields": fields,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done, predictions saved to {OUTPUT_PATH}")
    print(f"Mode: {'regex + LoRA + LLM' if use_llm else 'regex + LoRA only'}")

#--------------------------------------------------------------


if __name__ == "__main__":
    main()



#---------------------------------------

