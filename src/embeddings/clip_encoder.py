
# src/embeddings/clip_encoder.py


#--------------------------------------
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cpu"  
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dimension = self.model.config.projection_dim  # 512
        print(">>> ClipEncoder embedding dimension =", self.dimension, type(self.dimension))

    def encode_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        with torch.no_grad():
            feats = self.model.get_text_features(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )

        vec = feats[0].cpu().numpy().astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-12)
        return vec

    def encode_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")

        with torch.no_grad():
            feats = self.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(self.device)
            )

        vec = feats[0].cpu().numpy().astype("float32")
        vec /= (np.linalg.norm(vec) + 1e-12)
        return vec




# Multimodal embedding & retrieval – uses CLIP and a vector database to retrieve similar invoices by image or text.""


