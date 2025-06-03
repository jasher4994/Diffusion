import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self, output_dim=128):  # Smaller output for simplicity
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = 77
        
        # Simple projection to reduce dimensions
        self.projection = nn.Linear(512, output_dim)
        
    def forward(self, text):
        if not isinstance(text, list):
            text = [text]
            
        # Tokenize
        text_inputs = self.tokenizer(
            text, 
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        attention_mask = text_inputs.attention_mask.to(self.text_encoder.device)
        
        # Get embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get pooled output and project
        text_embeddings = text_outputs.pooler_output
        text_embeddings = self.projection(text_embeddings)
        
        return text_embeddings