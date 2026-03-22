import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


class EmbeddingService:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Initialize CLIP model and processor from HuggingFace."""
        print(f"Loading CLIP model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"Model loaded on {self.device}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into a 512-dim L2-normalized vector.
        
        Args:
            text: Input text string
            
        Returns:
            np.ndarray: (512,) float32 L2-normalized vector
        """
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use the text model directly to get embeddings
            text_outputs = self.model.text_model(**inputs)
            text_features = text_outputs.pooler_output
            
            # Project to joint embedding space
            text_features = self.model.text_projection(text_features)
            
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten().astype(np.float32)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode image into a 512-dim L2-normalized vector.
        
        Args:
            image: PIL Image object
            
        Returns:
            np.ndarray: (512,) float32 L2-normalized vector
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use the vision model directly to get embeddings
            vision_outputs = self.model.vision_model(**inputs)
            image_features = vision_outputs.pooler_output
            
            # Project to joint embedding space
            image_features = self.model.visual_projection(image_features)
            
            # L2 normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().astype(np.float32)
