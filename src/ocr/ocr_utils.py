
# src/ocr/ocr_utils.py

import pytesseract
from PIL import Image, ImageFilter, ImageOps
import pdf2image


# --------------------------------

def preprocess_image(img, bin_thresh: int = 175, upscale: bool = True) -> Image.Image:
 
    if upscale:
        w, h = img.size
        scale = 1.7
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    
    img = img.convert("L")

    img = ImageOps.autocontrast(img)

    img = img.filter(ImageFilter.MedianFilter())
    
    img = img.point(lambda x: 0 if x < bin_thresh else 255, "1")

    return img



TESS_CONFIG_RECEIPT = "--oem 3 --psm 6"


def _clean_ocr_text(text: str) -> str:
   
    if not text:
        return ""

    
    dirty_chars = ["`", "“", "”", "’", "‘", "«", "»", "™"]
    for ch in dirty_chars:
        text = text.replace(ch, "")

    
    for ch in ["—", "–", "‒", "―"]:
        text = text.replace(ch, "-")

    
    text = text.replace("\x0c", "\n").replace("\r", "\n")

    lines = [ln.rstrip() for ln in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# -------------------------------

def extract_text_from_image(image_path: str, lang: str = "eng") -> str:

    try:
        img = Image.open(image_path)

        img = preprocess_image(img, bin_thresh=175, upscale=True)

        raw_text = pytesseract.image_to_string(
            img,
            lang=lang,
            config=TESS_CONFIG_RECEIPT,
        )

        return _clean_ocr_text(raw_text)
    except Exception as e:
        print(f"OCR Error (image): {e}")
        return ""


# ------------------------------

def extract_text_from_pdf(pdf_path: str, lang: str = "eng", dpi: int = 300):

    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)

        page_texts = {}
        for idx, img in enumerate(images):
            img = preprocess_image(img, bin_thresh=175, upscale=True)
            raw_text = pytesseract.image_to_string(
                img,
                lang=lang,
                config=TESS_CONFIG_RECEIPT,
            )
            page_texts[f"page_{idx + 1}"] = _clean_ocr_text(raw_text)

        return page_texts
    except Exception as e:
        print(f"PDF OCR Error: {e}")
        return {}


# --------------------------------

def run_ocr(file_path: str, lang: str = "eng"):

    f = file_path.lower()
    if f.endswith(".pdf"):
        return extract_text_from_pdf(file_path, lang=lang)
    else:
        return extract_text_from_image(file_path, lang=lang)

#=============================================================================================






