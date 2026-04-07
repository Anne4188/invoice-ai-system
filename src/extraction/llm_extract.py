# src/extraction/llm_extract.py

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

"""
llm_extract.py

Responsible for the second-stage field extraction and correction:
- Receives OCR text and the initial fields extracted by regex
- Calls the LLM to complete and correct the fields
- Outputs a unified InvoiceFields dataclass
"""

@dataclass
class InvoiceFields:
    """
    # Unified standardized invoice field structure.
    # - This dataclass can be used both for regex extraction and as the final output of LLM extraction.
    # - Benefits: clear typing, convenient for validation, serialization, and logging.
    """
    merchant: Optional[str] = None
    date: Optional[str] = None           # expection :  YYYY-MM-DD
    total_amount: Optional[float] = None
    tax: Optional[float] = None
    currency: str = "USD"


def build_extraction_prompt(ocr_text: str,
                            regex_fields: Dict[str, Any]) -> str:
    prompt = f"""

This is an invoice reimbursement system.

Below is the OCR text of an invoice (which may contain a small number of recognition errors):
--------------------
{ocr_text}
--------------------

Below are the fields preliminarily extracted using regular expressions (which may be incomplete or contain errors):
{regex_fields}

Based on the OCR text and the preliminary fields, infer the final structured information of this invoice as accurately as possible.

Pay special attention:
The merchant is usually the name of the store, shop, or company, and typically appears near the top of the receipt.
Do not treat items such as "MANAGER XXX / EFT DEBIT / CHANGE / SUBTOTAL / CASH / THANK YOU" as the merchant name.

-total_amount:

  -It is the final amount payable. Prefer the line labeled TOTAL / GRAND TOTAL / AMOUNT DUE / NET TOTAL / FINAL TOTAL.
  -If there are SUBTOTAL + TAX + TOTAL, choose TOTAL, not SUBTOTAL.
  -For a restaurant receipt with only subtotal + CASH / CHANGE, the amount before CASH is usually the total.

- tax：
  -It may appear as TAX / GST / VAT / SALES TAX / TAX1 / TAX2, etc.
  -If there are multiple tax items (Tax 1 + Tax 2), sum them to obtain the total tax.
  -If the receipt says “TOTAL INCLUSIVE GST” or “Total Inclusive Tax”, it means total_amount already includes tax; in this case, tax can be 0 or null.

- currency：
  -If MYR or RM appears, return "MYR".
  -If IDR or Rp appears, return "IDR".
  -If there is no obvious marker, you may infer from the merchant or amounts; using "USD" as the default is also acceptable.

Please note:
  -If a field cannot be found anywhere in the OCR text, set it to null.
  -For monetary amounts, convert them to numeric type (float) without currency symbols.
  -Convert the date to YYYY-MM-DD format whenever possible; if the exact format cannot be determined, use the original date string.
  -If i find an obvious inconsistency between total_amount and tax, make a reasonable correction based on the context.
    
Please output strictly according to the following JSON schema. Do not add any extra fields, and do not output any text outside the JSON.

{{
  "merchant": string or null,
  "date": string or null,
  "total_amount": number or null,
  "tax": number or null,
  "currency": string
}}
"""
    return prompt.strip()



# src/extraction/llm_extract.py

def call_llm_for_extraction(
    prompt: str,
    llm_client,
    model: str = "gpt-4.1-mini",  # or "gpt-4.1"
) -> Dict[str, Any]:
    """
    Call the LLM with OpenAI Chat Completions and return a dict.
    The llm_client must be a client created by openai.OpenAI().
    """
    import json

    # Invoke  OpenAI Chat Completion
    resp = llm_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise invoice information extraction engine. "
                           "Only output valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    response_text = resp.choices[0].message.content

    # JSON
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # If the output of the LLM is not valid JSON, return an empty dict and let the upper layer take the responsibility
        return {}

    if not isinstance(data, dict):
        return {}

    return data


def merge_fields(
    regex_fields: Optional[Dict[str, Any]],
    llm_fields: Optional[Dict[str, Any]],
) -> InvoiceFields:
    """
    Merge the regex extraction results and the LLM extraction results to generate the final InvoiceFields.
    Allow regex_fields / llm_fields to be None
    Prefer fields from the LLM; if the LLM is None or a field is missing, fall back to regex
    """
    
    regex_fields = regex_fields or {}
    llm_fields = llm_fields or {}

    def pick(key: str):
        v_llm = llm_fields.get(key, None)
        v_regex = regex_fields.get(key, None)
        return v_llm if v_llm not in (None, "", []) else v_regex

    merged = InvoiceFields(
        merchant=pick("merchant"),
        date=pick("date"),
        total_amount=pick("total_amount"),
        tax=pick("tax"),
        currency=llm_fields.get("currency")
                 or regex_fields.get("currency")
                 or "USD",
    )
    return merged




def extract_fields_with_llm(ocr_text: str,
                            regex_fields: Dict[str, Any],
                            llm_client,
                            model: str = "gpt-4.1") -> InvoiceFields:
    
    # If None is passed from the upper layer, it is treated as an empty dict
    regex_fields = regex_fields or {}

    prompt = build_extraction_prompt(ocr_text=ocr_text, regex_fields=regex_fields)
    llm_fields = call_llm_for_extraction(prompt=prompt, llm_client=llm_client, model=model)

    # When the llm_fields parsing fails, call_llm_for_extraction will return {}
    merged_fields = merge_fields(regex_fields, llm_fields)
    return merged_fields

def invoice_fields_to_dict(fields: InvoiceFields) -> Dict[str, Any]:
    """
    Convert the InvoiceFields dataclass into a regular dict for easier:
    Serialization to JSON
    Storage in a database
    Logging / returning via an API
    """
    return asdict(fields)

#==================================
