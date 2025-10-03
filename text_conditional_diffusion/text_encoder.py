"""CLIP Text Encoder for text-conditional diffusion."""
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """Wrapper around CLIP text encoder for diffusion conditioning.
    
    Clip effectively maps images and text to the same latent space.
    
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

        if freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False

        self.embedding_dim = self.text_model.config.hidden_size  # 512 for base model

    def forward(self, text_prompts):
        """
        Encode text prompts to embeddings.

        Args:
            text_prompts: List of strings or single string

        Returns:
            Text embeddings of shape [batch_size, embedding_dim]
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(self.text_model.device)

        with torch.set_grad_enabled(self.text_model.training):
            outputs = self.text_model(**tokens)
            embeddings = outputs.pooler_output  # [batch_size, 512]

        return embeddings

    def encode_batch(self, text_prompts):
        """Convenience method for batch encoding."""
        return self.forward(text_prompts)

    @property
    def device(self):
        return self.text_model.device

