
#----------------------------
# src/rag/rag_pipeline.py


"""
High-level RAG pipeline for invoices.

This module glues together:
- OCR (image/PDF -> text)
- Field extraction (regex + optional LLM)
- Embeddings (CLIP)
- Vector store (for similar invoice search)
"""


import os
import inspect
from typing import Dict, Any, List


from src.ocr.ocr_utils import run_ocr
from src.extraction.field_extraction_pipeline import extract_invoice_fields_pipeline
from src.embeddings.clip_encoder import ClipEncoder
from src.embeddings.vector_store import VectorStore


class InvoiceRAGSystem:
    """
    Main orchestration class.

    Exposes a single main method:

        process_invoice(file_path, invoice_id) -> context dict

    which is what your Streamlit app calls.
    """

    def __init__(self, use_llm: bool = False, llm_client=None):
        
        self.encoder = ClipEncoder()  
        print(">>> embedding dimension =", self.encoder.dimension, type(self.encoder.dimension))
        self.vector_store = VectorStore(dimension=self.encoder.dimension)
    

        self.use_llm = use_llm
        self.llm_client = llm_client

   

    def _ocr_to_text(self, file_path: str) -> str:
        ocr_result = run_ocr(file_path)

        if isinstance(ocr_result, dict):
            pages = []
            for _, v in sorted(ocr_result.items()):
                if isinstance(v, str) and v.strip():
                    pages.append(v)
            return "\n\n".join(pages)
        elif isinstance(ocr_result, str):
            return ocr_result
        else:
            return ""


    def _get_image_path_for_embedding(self, file_path: str) -> str | None:
    
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            return file_path
        return None
    #-------------------------------------------------------

  
    def process_invoice(self, file_path: str, invoice_id: str):
        
        #-----------
    
        ocr_text = self._ocr_to_text(file_path)

       
        if ocr_text is ... or ocr_text is None:
            ocr_text = ""
        if not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)

        if not ocr_text.strip():
            return {
                "invoice_id": invoice_id,
                "raw_text": "",
                "ocr_text": "",
                "fields": {"merchant": None, "date": None, "total_amount": None, "tax": None, "currency": "USD"},
                "similar_invoices": [],
                "error": "OCR result is empty"
            }
        
        print("OCR type:", type(ocr_text), "raw ocr_result type:", type(run_ocr(file_path)))

        
        fields = extract_invoice_fields_pipeline(
            ocr_text=ocr_text,
            llm_client=self.llm_client,
            use_llm=self.use_llm,
        )

        # 3) embedding
        text_embedding = self.encoder.encode_text(ocr_text)

        image_path = self._get_image_path_for_embedding(file_path)
        image_embedding = self.encoder.encode_image(image_path) if image_path else None

       
        invoice_data = {"invoice_id": invoice_id, "fields": fields, "ocr_text": ocr_text}

        add_sig = inspect.signature(self.vector_store.add_invoice)
        param_names = [p.name for p in add_sig.parameters.values() if p.name != "self"]
        argc = len(param_names)

        if argc == 3:
            
            try:
                self.vector_store.add_invoice(text_embedding=text_embedding,
                                            image_embedding=image_embedding,
                                            invoice_data=invoice_data)
            except TypeError:
                self.vector_store.add_invoice(text_embedding, image_embedding, invoice_data)

        elif argc == 2:
           
            try:
                self.vector_store.add_invoice(embedding=text_embedding, invoice_data=invoice_data)
            except TypeError:
                
                try:
                    self.vector_store.add_invoice(text_embedding, invoice_data)
                except TypeError:
                    self.vector_store.add_invoice(invoice_data, text_embedding)

        else:
            
            print(f"[WARN] add_invoice signature unexpected: {add_sig}. Skip adding to vector store.")

        try:
            similar_invoices = self.vector_store.search(query_embedding=text_embedding, k=5)
        except TypeError:
            similar_invoices = self.vector_store.search(text_embedding, k=5)

        return {
            "invoice_id": invoice_id,
            "raw_text": ocr_text,   
            "ocr_text": ocr_text,
            "fields": fields,
            "similar_invoices": similar_invoices,
        }



if __name__ == "__main__":
    
    system = InvoiceRAGSystem()





