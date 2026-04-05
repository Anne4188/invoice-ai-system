# 新建一个「批量 OCR 脚本」

# scripts/run_ocr_batch.py

import os
import argparse
from pathlib import Path

from src.ocr.ocr_utils import run_ocr


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}


def run_batch(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXTS:
                files.append(Path(root) / fn)

    print(f"[OCR] Found {len(files)} files under {input_dir}")

    for i, img_path in enumerate(files, 1):
        rel = img_path.relative_to(input_dir)
        out_txt = output_dir / rel.with_suffix(".txt")
        out_txt.parent.mkdir(parents=True, exist_ok=True)

        if out_txt.exists():
            continue  # 已经 OCR 过的不重复跑

        print(f"[{i}/{len(files)}] OCR {rel}")
        text = run_ocr(str(img_path))

        if isinstance(text, dict):
            # PDF 多页情况
            text = "\n\n".join(text.values())

        out_txt.write_text(text or "", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    run_batch(args.input_dir, args.output_dir)
