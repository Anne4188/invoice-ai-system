
# FAISS + Invoice Domain Wrapper

# src/embeddings/vector_store.py
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any

class VectorStore:
    """
    A simple vector database wrapper:
    - Use FAISS IndexFlatL2 to store embeddings
    - A metadata list stores the invoice information (dictionary) corresponding to each vector
    - Provide interfaces such as add / search / save

    """
    def __init__(
        self,
        dimension: int,
        index_path: str = "data/vector_index.faiss",
        metadata_path: str = "data/metadata.json",
    ):
        self.dimension = int(dimension)       
        self.index_path = index_path
        self.metadata_path = metadata_path

       
        self.index = faiss.IndexFlatL2(self.dimension)   
        self.metadata: List[Dict[str, Any]] = []
        
        # If there is a ready-made index file, try loading it
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # If there are ready-made original data files, load them
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
    
    def add_invoice(self, embedding: np.ndarray, invoice_data: Dict[str, Any]) -> None:
        """
        Add an invoice vector and its metadata to the vector store.
        - embedding: a 1D vector (dim,) or 2D (1, dim)
        - invoice_data: a dictionary describing the invoice, for example { "path": ..., "fields": ... }

        """
        emb = np.array(embedding, dtype="float32").reshape(1, -1)
        if emb.shape[1] != self.dimension:
            raise ValueError(f"Embedding dim {emb.shape[1]} != index dim {self.dimension}")
        
        self.index.add(emb)
        self.metadata.append(invoice_data)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top k invoices most similar to query_embedding.
        Return:- A list where each element contains { "invoice": metadata, "distance": float }

        """
        if self.index.ntotal == 0:
            # If the vector library is empty, return an empty list directly
            return []
        
        q = np.array(query_embedding, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(q, k)
        
        results: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(
                    {
                        "invoice": self.metadata[idx],
                        "distance": float(distances[0][i]),
                    }
                )
        
        return results
    
    def save(self) -> None:
        """Save the current index and original data to the disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)








# Multimodal embedding & retrieval – uses CLIP and a vector database to retrieve similar invoices by image or text.

"""
Multimodal embeddings & retrieval – using CLIP to embed both images and text, and FAISS/Chroma as a vector store, so I can do image-to-image and text-to-invoice search.
"""