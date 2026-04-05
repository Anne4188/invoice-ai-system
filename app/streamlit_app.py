
# app/streamlit_app.py

import os
import sys
import json
import streamlit as st

# --- Key step: Add the project root directory to sys.path and fix No module named 'src' -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.rag.rag_pipeline import InvoiceRAGSystem
#from src.embeddings.vector_store import VectorStore


# Initialize the system (load only once)
# @st.cache_resource
def load_system():
    return InvoiceRAGSystem()
system = load_system()


# ---------------- UI start here ----------------

st.title("🤖 Multimodal RAG System for Invoice Reimbursement")


st.markdown("""
### Features
- Upload single or multiple invoices
- View OCR text
- See extracted fields
- Retrieve similar invoices
""")


uploaded_files = st.file_uploader(
    " Upload invoice files (multiple allowed) ",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data/raw", exist_ok=True)

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        st.markdown("---")
        st.subheader(f"invoices {idx}: {uploaded_file.name}")

        file_path = os.path.join("data", "raw", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        invoice_id = f"inv_{uploaded_file.name.rsplit('.', 1)[0]}"

        with st.spinner(f"Processing {uploaded_file.name} ..."):
            context = system.process_invoice(file_path, invoice_id)

        st.subheader("OCR text")
        with st.expander("Expand to view the OCR results"):
            st.text(context.get("raw_text", ""))

        st.subheader("Extract fields")
        fields = context.get("fields", {})
        st.json(fields)

        st.subheader("Similar historical invoices")
        similar_invoices = context.get("similar_invoices", [])

        if similar_invoices:
            for i, sim in enumerate(similar_invoices):
                with st.expander(f"similar invoices #{i+1} · Distance: {sim.get('distance', 0):.2f}"):
                    st.json(sim.get("invoice", {}))
        else:
            st.info("No similar historical invoices have been found yet.")

        st.subheader("Structured Reimbursement Report")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Table View")
            report_data = {
                "fields": ["merchant", "date", "total_amount", "tax", "currency"],
                "value": [
                    fields.get("merchant", "N/A"),
                    fields.get("date", "N/A"),
                    f"{fields.get('total_amount', 0):.2f}" if fields.get("total_amount") else "N/A",
                    f"{fields.get('tax', 0):.2f}" if fields.get("tax") else "N/A",
                    fields.get("currency", "USD"),
                ],
            }
            st.table(report_data)

        with col2:
            st.write("JSON View")
            st.json(fields)

        # Duplicate Reimbursement Test (for the current one)
        if similar_invoices:
            closest = similar_invoices[0].get("invoice", {})
            closest_fields = closest.get("fields", {})
            if (
                closest_fields.get("total_amount") == fields.get("total_amount")
                and closest_fields.get("merchant") == fields.get("merchant")
                and closest_fields.get("date") == fields.get("date")
            ):
                st.warning("Potential duplicate reimbursement detected")


# Sidebar: Text search
st.sidebar.header("Search invoices by text")
search_query = st.sidebar.text_input("Enter a keyword to search invoices (e.g.,Walmart, Hotel, Uber)")

if search_query:
    with st.spinner("searching..."):
       
        query_emb = system.encoder.encode_text(search_query)
        results = system.vector_store.search(query_emb, k=5)

        st.sidebar.subheader("search result")
        if not results:
            st.sidebar.write("No matching invoice was found")
        else:
            for r in results:
                inv = r["invoice"]
                flds = inv.get("fields", {})
                st.sidebar.write(f"**{flds.get('merchant', 'unknow merchant')}**")
                st.sidebar.write(f"Amount: ${flds.get('total_amount', 0):.2f}")
                st.sidebar.write(f"Date: {flds.get('date', 'unknow date')}")
                st.sidebar.write(f"Distance: {r.get('distance', 0):.2f}")
                st.sidebar.write("---")




"""
A lightweight Streamlit interface allows users to upload invoices, view OCR text, inspect extracted fields, 
and retrieve similar invoices, enabling an end-to-end demo of the multimodal RAG pipeline.
"""