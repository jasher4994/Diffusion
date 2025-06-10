import torch
import torch.nn as nn
import clip

class TextEncoder(nn.Module):
    """
    CLIP-based text encoder for converting text prompts to embeddings.
    """
    
    def __init__(self, clip_model_name="ViT-B/32", device='cuda'):
        super().__init__()
        self.device = device
        
        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        
        # Freeze CLIP parameters (we don't want to train CLIP)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # CLIP text embeddings are 512-dimensional for ViT-B/32
        self.text_dim = 512
        
    @torch.no_grad()
    def encode_text(self, text_prompts):
        """
        Convert text prompts to CLIP embeddings.
        
        Args:
            text_prompts: List of strings, e.g., ["a red car", "a blue sky"]
            
        Returns:
            Text embeddings, shape (batch_size, 512)
        """
        # Tokenize text
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Get text embeddings from CLIP
        text_embeddings = self.clip_model.encode_text(text_tokens)
        
        # Normalize embeddings (CLIP convention)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings.float()
    
    def forward(self, text_prompts):
        return self.encode_text(text_prompts)