
# src/extraction/field_extraction_pipeline.py


from typing import Dict, Any, Tuple     

# regex_extract_fields:
# First-stage rule-based extractor responsible for extracting merchant/date/total/tax/currency from OCR text
from .regex_extract import extract_fields as regex_extract_fields


# extract_fields_with_llm:
# Second-stage LLM corrector that performs semantic correction and completion based on the regex baseline

# invoice_fields_to_dict:
# Convert the LLM output in dataclass format into a regular dict
from .llm_extract import extract_fields_with_llm, invoice_fields_to_dict

import re

#======================================================================

# BAD_CAND_PAT:
# Used to filter out candidate text that does not look like a merchant name
# For example:
# - invoice
# - receipt
# - total
# - receipt no
# - invoice no
# - reg no
# - tel
# These should not be corrected by LoRA as merchant names
BAD_CAND_PAT = re.compile(
    r"^(invoice|tax invoice|receipt|total|subtotal|cash|change|rounding)$|"
    r"(receipt\s*no|invoice\s*no|bill\s*no|ref\s*no|reg\s*no|gst\s*id|tel|fax)\b|"
    r"\b(rec[-\s]?[a-z0-9]+)\b",
    re.I
)

#======================================================================
def _looks_like_merchant(s: str) -> bool:

    """
    LoRA output validator:
    Purpose: prevent LoRA from mistakenly treating invoice numbers, noisy text, or amount lines as the merchant.

    Why is this needed?
    Although LoRA can perform corrections, it may also produce incorrect outputs.
    Therefore, before the LoRA output overrides the regex result, an additional safety check is required.
    """

    s = (s or "").strip()

    # 1)Strings that are too short are unlikely to be merchant names
    if len(s) < 3:
        return False

    # 2) Too many digits (e.g., REC-123 / 029384 / INV2024) → unlikely to be a merchant name
    digit_ratio = sum(c.isdigit() for c in s) / max(1, len(s))
    if digit_ratio > 0.25:
        return False

    # 3) If it matches an obvious non-merchant pattern, return False directly
    if BAD_CAND_PAT.search(s):
        return False

    # Passed all checks → considered to look like a merchant name
    return True


#======================================================================


def extract_invoice_fields_pipeline(
    ocr_text: str,
    llm_client=None,
    use_llm: bool = True,
    debug: bool = False,
) -> Dict[str, Any] | Dict[str, Dict[str, Any]]:
    
    """
    High-level field extraction pipeline (main orchestrator / router)

    Overall flow:
    1. OCR text enters the pipeline
    2. Run the regex baseline first (low cost, interpretable)
    3. Decide whether to use LoRA merchant normalization based on merchant_conf
    4. If LLM is enabled, call llm_extract.py for second-stage semantic correction
    5. Return the final structured fields

    This is the orchestration layer of the entire extraction stack,
    not the extraction logic itself, but the control layer that decides
    what runs first, what runs next, and when each component is triggered.
        """

    # Step 0. Validate OCR input
    # Ensure OCR output is a string to avoid splitlines() errors downstream.
    
    if ocr_text is None or type(ocr_text) is type(...):  
        ocr_text = ""
    if not isinstance(ocr_text, str):
        ocr_text = str(ocr_text)


    # =========================
    # Step 1. regex baseline
    # =========================
    #
    # Run the rule-based extractor first.
    # Responsible for:
    # - Extracting merchant/date/total_amount/tax/currency
    # - Generating _conf (field confidence)
    # - Generating _meta (intermediate metadata)
    #
    # This step runs first because rule-based extraction is cheap,
    # fast, and interpretable, serving as the system baseline.
    regex_fields = regex_extract_fields(ocr_text)

    # =========================
    # Step 1.5 LoRA 
    # =========================
    #
    # Design idea:
    # Not all samples go through LoRA; only low-confidence merchant samples do.
    #
    # This is a typical confidence routing strategy:
    # high confidence -> trust regex directly
    # low confidence  -> try LoRA correction
    #
    # Benefits:
    # - Saves inference cost
    # - Preserves the interpretability of regex
    # - Prevents the model from unnecessarily modifying easy samples
    try:
        #from src.normalization.lora_normalizer import lora_normalize_merchant
        # Import the LoRA inference interface from the merchant normalizer module
        from src.normalization.lora_normalizer import predict
    except Exception:
        # If the model is not installed, dependencies are missing, or the path is incorrect,
        # do not let the entire pipeline crash.
        # Simply fall back to "skip LoRA".
        predict = None

    # Extract the field confidence dictionary from the regex output
    # If not found, default to 1.0 (very high confidence) so LoRA will not be triggered
    conf = (regex_fields.get("_conf") or {})
    m_conf = float(conf.get("merchant", 1.0))


    # =========================
    # Step 1.5.1 Low-confidence merchant triggers LoRA
    # =========================
    #
    # Current logic:
    # Only allow LoRA when merchant_conf < 0.90
    
    if predict is not None and m_conf < 0.90:  #here m_conf is merchant_conf < THRESHOLD

        # Prefer the original merchant candidate from regex
        # If not available in meta, fall back to regex_fields["merchant"]
        cand_raw = (regex_fields.get("_meta") or {}).get("merchant_candidate_raw") or regex_fields.get("merchant") or ""
        cand_raw = cand_raw.strip()

        # 1.If the regex candidate itself is clearly bad text (e.g., receipt no / total / invoice)
#       then skip LoRA entirely to avoid the model guessing on noisy input
        if BAD_CAND_PAT.search(cand_raw):
            fixed = None

        else:
            # 2. Otherwise call the LoRA normalizer
            # Input:
            # - candidate: current merchant candidate
            # - ocr_text: original OCR context
            # - task: merchant normalization
            fixed = predict(candidate=cand_raw, ocr_text=ocr_text, task="merchant_norm")


        # 3. After LoRA produces an output, run an additional safety check
        #    Only allow it to override the regex result if it looks like a merchant name
        if isinstance(fixed, str) and _looks_like_merchant(fixed):
            regex_fields["merchant"] = fixed.strip()

            # Record in _meta:
            # merchant has been corrected by LoRA
            regex_fields.setdefault("_meta", {})
            regex_fields["_meta"]["merchant_refined_by"] = "lora"


    # =========================
    # Step 2. Whether to continue with LLM
    # =========================
    #
    # This is the global LLM switch for the pipeline.
    #
    # If:
    # - use_llm = False
    # or
    # - llm_client is None
    #
    # Then the pipeline stops here and directly returns the regex/LoRA results.
    # If there is no LLM or LLM is explicitly disabled, return the regex results.
    if not use_llm or llm_client is None:
        if debug:
            return {
                "regex_fields": regex_fields,
                "final_fields": regex_fields,
            }
        return regex_fields

    
    
    # =========================
    # Step 3. LLM semantic correction
    # =========================
    #
    # This step runs only when:
    # - use_llm = True
    # - llm_client is not None
    #
    # It calls the logic in llm_extract.py:
    # - build prompt
    # - call the model
    # - parse JSON
    # - merge regex and LLM results

    print("[DEBUG] entering LLM stage...")  # Use this sentence to prove that the LLM was indeed invoked
    refined_fields_obj = extract_fields_with_llm(
        ocr_text=ocr_text,
        regex_fields=regex_fields,
        llm_client=llm_client,
    )

    # Convert dataclass to a regular dict to facilitate subsequent UI/JSON/storage
    final_fields = invoice_fields_to_dict(refined_fields_obj)



    if debug:
        return {
            "regex_fields": regex_fields,
            "final_fields": final_fields,
        }
    
    
    return final_fields





#-------------------------------------------------




