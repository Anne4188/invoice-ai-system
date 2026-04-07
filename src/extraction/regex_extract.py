# src/extraction/regex_extract.py

import re
import difflib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import difflib  




# ---------- Digital cleaning----------

def clean_number(num_str: Optional[str]) -> Optional[str]:
    """
    Clean the numeric string and fix common numeric recognition errors in OCR.
    """
    if not num_str:
        return None

    s = num_str

    # Letters were mistakenly identified as numbers
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("l", "1")
    s = s.replace("I", "1")
    s = s.replace("S", "5")

    # Remove commas and Spaces
    s = s.replace(",", "").replace(" ", "")

    return s


# ----------  date processed----------

def _fix_year(y: int) -> int:
    """
    Apply a simple correction for abnormal years:
        -2000–2100 are considered valid
        -20–99 are interpreted as 2000 + yy
        -For clearly unreasonable years (e.g., 2618), attempt to normalize them to something plausible such as 2018
    """
    if 2000 <= y <= 2100:
        return y
    if 20 <= y <= 99:
        return 2000 + y
    # For instance, for 2618, take the last two digits of the year: 18 -> 2018
    if y > 2100:
        yy = y % 100
        return 2000 + yy
    return y


def normalize_date(date_str: str) -> Optional[str]:
    """
    Try to unify the various date formats into YYYY-MM-DD.
    Support two-digit vintages.
    """
    if not date_str:
        return None

    s = date_str.strip()
    s = s.replace(".", "-").replace("/", "-")

    
    patterns_4y = ["%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"]
    for fmt in patterns_4y:
        try:
            dt = datetime.strptime(s, fmt)
            y = _fix_year(dt.year)
            return datetime(y, dt.month, dt.day).strftime("%Y-%m-%d")
        except Exception:
            pass

    # years format：dd-mm-yy, mm-dd-yy, yy-mm-dd
    patterns_2y = ["%d-%m-%y", "%m-%d-%y", "%y-%m-%d"]
    for fmt in patterns_2y:
        try:
            dt = datetime.strptime(s, fmt)
            y = _fix_year(dt.year)
            return datetime(y, dt.month, dt.day).strftime("%Y-%m-%d")
        except Exception:
            pass

    return date_str  



MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "SEPT": 9, "OCT": 10,
    "NOV": 11, "DEC": 12,
}

def normalize_merchant(s: Optional[str]) -> Optional[str]:
    if not s:
        return None

    s = s.upper().strip()
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s)

    CANONICAL = {
        "TRADER JOE'S": ["TRADER JOES", "TRADER JOE"],
        "WALMART": ["WAL MART", "WAL-MART", "WMS"],
        "COSTCO": ["COSTCO WHOLESALE"],
        "WHOLE FOODS MARKET": ["WHOLEFOODS", "WHOLE FOODS MKT"],
        "MOMI & TOY'S": ["MOMI AND TOYS", "MOMI TOYS", "MOMI TOY"],
        "C W KHOO HARDWARE SDN BHD": ["KHOO HARDWARE"],
        "SYARIKAT PERNIAGAAN GIN KEE": ["GIN KEE"],
        "SANYU STATIONERY SHOP": ["SANYU STATIONERY", "SANYO STATIONERY"],
        "MR. D.I.Y. (M) SDN BHD": ["MR DIY", "MR D I Y", "MR DIY M SDN BHD"],
        "RESTORAN WAN SHENG": ["WAN SHENG"],
        "ADVANCO COMPANY": ["ADVANCO"],
        "YORKVILLE SOUND LTD": ["YORKVILLE SOUND"],
        "FAMILYMART": ["FAMILY MART"],
    }

    for canon, variants in CANONICAL.items():
        if s == canon:
            return canon
        if any(v in s for v in variants):
            return canon

    return s


def extract_date(text: str) -> Optional[str]:
    """
Extract the string that most likely represents the invoice date from the full text and normalize it.

    Priority:

    Standard numeric dates: dd/mm/YYYY, mm-dd-yy, etc.
    English month dates: APR 20 2016 / APR 20, 2016
    Year only (e.g., 2014-00-00 for the WHOLE FOODS example)
    """
  
    full = text.replace("\n", " ")

    # ---------- 1) numbers and date ----------
    date_regex = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    candidates = re.findall(date_regex, full)

    for cand in candidates:
        norm = normalize_date(cand)
        if norm:
            return norm

    # ---------- 2) English month and date----------
    # example: APR 20 2016 / APR 20, 2016 / 20 APR 2016
    # format 1: MON DD YYYY
    m1 = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{2,4})\b", full)
    if m1:
        mon_str, day_str, year_str = m1.groups()
        mon_key = mon_str[:3].upper()
        if mon_key in MONTH_MAP:
            month = MONTH_MAP[mon_key]
            day = int(day_str)
            year = int(year_str)
            year = _fix_year(year)
            try:
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

    # format 2: DD MON YYYY
    m2 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{2,4})\b", full)
    if m2:
        day_str, mon_str, year_str = m2.groups()
        mon_key = mon_str[:3].upper()
        if mon_key in MONTH_MAP:
            month = MONTH_MAP[mon_key]
            day = int(day_str)
            year = int(year_str)
            year = _fix_year(year)
            try:
                return f"{year:04d}-{month:02d}-{day:02d}"
            except ValueError:
                pass

    # --------------------
    year_matches = list(re.finditer(r"\b(20\d{2})\b", full))
    if year_matches:
        
        y = int(year_matches[0].group(1))
        return f"{y:04d}-00-00"

    return None



# --------------------

def _score_merchant_line(line: str) -> float:
    """
    Assign a score to a line of text based on how likely it is to be a merchant name:

    Contains letters
    Has a high proportion of uppercase letters
    Does not have too high a proportion of digits
    Must not be only registration-related information such as Reg No or GST Reg
    """
    s = line.strip()

    if not s:
        return 0.0
    if len(s) < 3 or len(s) > 60:
        return 0.0

    lower_s = s.lower()
    blacklist = [
        "invoice", "tax invoice", "invoice no", "inv no",
        "reg no", "gst reg", "tax invoice",
         "receipt", "gst", "vat", "cash", "tel",
        "phone", "manager", "change", "subtotal", "total", "amount due",
        "PP SR WAYS LOW FRICBS"
    ]
    if any(k in lower_s for k in blacklist):
        return 0.0

    letters = sum(c.isalpha() for c in s)
    uppers = sum(c.isupper() for c in s)
    digits = sum(c.isdigit() for c in s)

    if letters == 0:
        return 0.0

    upper_ratio = uppers / max(1, letters)
    digit_ratio = digits / max(1, len(s))

    score = upper_ratio - 0.5 * digit_ratio

    # SDN BHD / RESTORAN / TRADER / WALMART 
    bonus_keywords = [
        "sdn bhd", "sdn. bhd", "restoran", "trader",
        "walmart", "costco", "whole foods", "hardware",
        "stationery", "market", "advanco", "momi", "toy's",
        "yorkville", "familymart", "taylor"
    ]
    if any(k in lower_s for k in bonus_keywords):
        score += 0.5

    return score



KNOWN_MERCHANTS = [
    "TRADER JOE'S",
    "WALMART",
    "COSTCO",
    "WHOLE FOODS MARKET",
    "MOMI & TOY'S",
    "C W KHOO HARDWARE SDN BHD",
    "SYARIKAT PERNIAGAAN GIN KEE",
    "SANYU STATIONERY SHOP",
    "MR. D.I.Y. (M) SDN BHD",
    "RESTORAN WAN SHENG",
    "ADVANCO COMPANY",
]



MERCHANT_PATTERNS = {
    "TRADER JOE'S":        ["trader joe"],
    "WALMART":             ["walmart", "wal-mart", "wal mart", "walmart store", "walmart supercenter", "wms"],
    "COSTCO":              ["costco", "costco wholesale"],
    "WHOLE FOODS MARKET":  ["whole foods", "whole foods market"],
    "MOMI & TOY'S":        ["momi & toy", "momi & toy's", "momi toy", "momi to", "momi & toys", "creperie"],
    "C W KHOO HARDWARE SDN BHD": ["khoo hardware", "hardware sdn bhd"],
    "SYARIKAT PERNIAGAAN GIN KEE": ["gin kee"],
    "SANYU STATIONERY SHOP":      ["sanyu stationery", "sanyu stationary", "sanyu shop", "sanyo stationery"],
    "MR. D.I.Y. (M) SDN BHD":     ["mr. d.i.y", "mr diy", "mr. d.i.y.", "mr. d.i.y (m)", "mr. o.f.y", "mr. o.f.y."],
    "RESTORAN WAN SHENG":         ["restoran wan sheng", "wan sheng"],
    "ADVANCO COMPANY":            ["advanco company", "advanco co", "advanco"],
}



def extract_merchant_with_conf(text: str) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    return (merchant_candidate, confidence, meta)
    confidence rules：
    - HARD keyword：0.95
    - INVOICE window best_score ：0.85~0.92
    - best_score ：0.60~0.80
    - none：0.0
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, 0.0, {"source": "empty"}

    full_upper = "\n".join(lines).upper()

    HARD_MERCHANT_KEYWORDS = {
        "TRADER JOE'S":      ["TRADER JOE"],
        "WALMART": ["WALMART", "WAL-MART", "WMS "],
        "COSTCO":            ["COSTCO"],
        "WHOLE FOODS MARKET":["WHOLE FOODS"],
        "MOMI & TOY'S":      ["MOMI & TOY", "MOMI & TOY'S", "MOMI TOY"],
        "C W KHOO HARDWARE SDN BHD": ["KHOO HARDWARE"],
        "SYARIKAT PERNIAGAAN GIN KEE": ["GIN KEE"],
        "SANYU STATIONERY SHOP": ["SANYU STATIONERY", "SANYO STATIONERY"],
        "MR. D.I.Y. (M) SDN BHD": ["MR D.I.Y", "MR DIY", "MR. D.I.Y (M)"],
        "RESTORAN WAN SHENG": ["WAN SHENG", "RESTORAN WAN SHENG"],
        "ADVANCO COMPANY":   ["ADVANCO"],
        "YORKVILLE SOUND LTD": ["YORKVILLE SOUND"],
        "FAMILYMART":        ["FAMILYMART"],
    }

    
    for canon, kws in HARD_MERCHANT_KEYWORDS.items():
        if any(kw in full_upper for kw in kws):
            return canon, 0.95, {"source": "hard_keyword", "canon": canon}

    
    invoice_idx = None
    for i, ln in enumerate(lines):
        if "invoice" in ln.lower():
            invoice_idx = i
            break

    best_line_after = None
    best_score_after = 0.0
    if invoice_idx is not None:
        window = lines[invoice_idx + 1: invoice_idx + 6]
        for ln in window:
            sc = _score_merchant_line(ln)
            if sc > best_score_after:
                best_score_after = sc
                best_line_after = ln

    if best_line_after and best_score_after > 0.1:
        
        match = difflib.get_close_matches(best_line_after, KNOWN_MERCHANTS, n=1, cutoff=0.6)
        candidate = match[0] if match else best_line_after

        
        conf = 0.85
        if best_score_after > 0.6:
            conf = 0.92
        elif best_score_after > 0.3:
            conf = 0.88

        return candidate, conf, {
            "source": "invoice_window",
            "best_score": best_score_after,
            "raw_line": best_line_after,
        }

    
    best_line = None
    best_score = 0.0
    for ln in lines:
        sc = _score_merchant_line(ln)
        if sc > best_score:
            best_score = sc
            best_line = ln

    if not best_line:
        return None, 0.0, {"source": "no_candidate"}

    match = difflib.get_close_matches(best_line, KNOWN_MERCHANTS, n=1, cutoff=0.6)
    candidate = match[0] if match else best_line

    conf = 0.65
    if best_score > 0.6:
        conf = 0.85
    elif best_score > 0.3:
        conf = 0.75

    return candidate, conf, {
        "source": "global_scoring",
        "best_score": best_score,
        "raw_line": best_line,
    }


def extract_merchant(text: str) -> Optional[str]:
    merchant, conf, meta = extract_merchant_with_conf(text)
    return merchant

# --------------------

def detect_currency(text: str) -> str:
    """
    Infer the most likely currency from the full text:
    MYR / RM → MYR (Malaysian Ringgit)
    IDR / Rp → IDR (Indonesian Rupiah)
    Default: USD
    """
    t = text.upper()

    
    if "MYR" in t:
        return "MYR"
    
    if re.search(r"\bRM\s*\d", t):
        return "MYR"

    
    if "IDR" in t:
        return "IDR"
    if re.search(r"\bRP\s*\d", t):
        return "IDR"

    
    return "USD"



# -------------------

def extract_amounts_from_line(line: str) -> List[float]:
   
    raw_nums = re.findall(r"(\d+[.,]\d{2}|\d+)", line)
    values: List[float] = []

    for raw in raw_nums:
        cleaned = clean_number(raw)
        if not cleaned:
            continue

        
        if "." in cleaned:
            try:
                values.append(float(cleaned))
            except Exception:
                continue
        else:
            
            if len(cleaned) >= 7:
                continue
            try:
                values.append(float(cleaned))
            except Exception:
                continue

    return values



def extract_total_and_tax(text: str) -> Tuple[Optional[float], Optional[float]]:

    lines = text.splitlines()
    currency = detect_currency(text)  

    if currency == "IDR":
        max_reasonable = 1e9  
    elif currency == "MYR":
        max_reasonable = 1e6
    else:  
        max_reasonable = 1e5

    min_reasonable = 0.01

    all_amounts: List[float] = []

    
    total_candidates: List[tuple[int, float]] = []
    tax_amounts: List[float] = []

    for line in lines:
        lower = line.lower()
        vals = extract_amounts_from_line(line)
        if vals:
            all_amounts.extend(vals)

        # ---- ----
        if any(k in lower for k in [" tax", "gst", "vat", "sales tax"]):
            
            tax_amounts.extend(vals)

        
        if "subtotal" in lower or "sub total" in lower or "sub-total" in lower:
            continue

        priority = 0

        
        if any(k in lower for k in [
            "total inclusive gst",
            "total incl gst",
            "total incl. gst",
            "net total rounded",
            "net total",
            "final total",
            "grand total",
            "amount due",
            "balance due"
        ]):
            priority = 3
        
        elif any(k in lower for k in [
            "total", "net amount", "cash", "final amt", "final amount"
        ]):
            priority = 2

        if priority > 0 and vals:
            
            candidate = vals[-1]
            if min_reasonable <= candidate <= max_reasonable:
                total_candidates.append((priority, candidate))

    total_amount: Optional[float] = None
    tax: Optional[float] = None

    # ---- ----
    if total_candidates:
        
        total_candidates.sort(key=lambda x: (x[0], x[1]))
        
        best_pri = max(p for p, _ in total_candidates)
        best_vals = [v for p, v in total_candidates if p == best_pri]
        if best_vals:
            total_amount = max(best_vals)

    # --------
    if total_amount is None and all_amounts:
        candidates = [v for v in all_amounts
                      if min_reasonable <= v <= max_reasonable]
        if candidates:
            total_amount = max(candidates)

    # ---- ----
    if tax_amounts:
        
        if total_amount is not None:
            candidates = [v for v in tax_amounts
                          if min_reasonable <= v < total_amount]
        else:
            candidates = [v for v in tax_amounts
                          if min_reasonable <= v <= max_reasonable]

        if candidates:
            
            candidates.sort()
            
            if total_amount is not None:
                small_taxes = [v for v in candidates if v / max(total_amount, 1.0) <= 0.2]
            else:
                small_taxes = candidates

            if small_taxes:
                tax_sum = sum(small_taxes)
                
                if total_amount is not None:
                    ratio = tax_sum / max(total_amount, 1.0)
                    if ratio < 0.5:
                        tax = tax_sum
                    else:
                        
                        tax = 0.0
                else:
                    tax = tax_sum
            else:
                tax = min(candidates)

    
    if tax is None:
        full_lower = text.lower()
        if any(k in full_lower for k in [" tax", "gst", "vat", "sales tax"]):
            tax = 0.0

    return total_amount, tax



# -------------------

def extract_fields(text: str) -> Dict[str, Any]:
    
    print("[DEBUG] extract_fields input type =", type(text), "repr =", repr(text)[:200])
    assert isinstance(text, str), f"extract_fields got non-str: {type(text)} repr={repr(text)[:200]}"
    ...
    
    if not isinstance(text, str):
        text = "" if text is None or text is ... else str(text)

    fields: Dict[str, Any] = {
        "merchant": None,
        "date": None,
        "total_amount": None,
        "tax": None,
        "currency": "USD",
    }

    if not text:
        return fields

   

    raw_merchant, merchant_conf, merchant_meta = extract_merchant_with_conf(text)
    fields["merchant"] = normalize_merchant(raw_merchant)

    
    fields["_conf"] = {
        "merchant": float(merchant_conf),
        
    }
    fields["_meta"] = {
        "merchant_candidate_raw": raw_merchant,
        "merchant_source": merchant_meta.get("source"),
        "merchant_raw_line": merchant_meta.get("raw_line"),
        "merchant_best_score": merchant_meta.get("best_score"),
    }


   
    fields["date"] = extract_date(text)
    date_val = extract_date(text)
    fields["date"] = date_val

    date_conf = 0.0
    if date_val:
        date_conf = 0.85
        if date_val.endswith("-00-00"):
            date_conf = 0.50

    fields["_conf"]["date"] = float(date_conf)


    
    total_amount, tax = extract_total_and_tax(text)
    fields["total_amount"] = total_amount
    fields["tax"] = tax

    
    fields["currency"] = detect_currency(text)

    return fields
