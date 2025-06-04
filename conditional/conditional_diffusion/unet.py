import torch
import torch.nn as nn
import math
from typing import Optional

class UNet(nn.Module):
    """
    UNet architecture for diffusion models.
    
    This network takes a noisy image, timestep, and optional text conditioning
    and predicts the unit noise that was added to create the noisy image.
    
    The UNet has three main parts:
    1. Encoder (downsampling path) - reduces spatial resolution, increases channels
    2. Bottleneck - processes the lowest resolution features
    3. Decoder (upsampling path) - increases spatial resolution, reduces channels
    
    Skip connections connect corresponding encoder and decoder layers to preserve
    fine-grained details that might be lost during downsampling.
    """
    
    def __init__(
        self,
        in_channels: int = 3,      # RGB images
        out_channels: int = 3,     # Predict RGB noise
        time_emb_dim: int = 128,   # Timestep embedding dimension
        text_emb_dim: int = 512,   # Text embedding dimension (from CLIP)
        base_channels: int = 64,   # Base number of channels
    ):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        
        # Timestep embedding layers
        # We have the positional embeddings, but they are FIXED. 
        # We need the MLP to tranform them into a higher dimension and to be able to learn.
        # For example this gives the model the opportunity to learn what timesteps mean
        # For example, timestep 10 might mean "very little noise" and timestep 900 might mean "a lot of noise"
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),  # Swish activation
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        
    # Creates FIXED mathmatical functions to embeddings timeteps
    def get_timestep_embedding(self, timesteps, dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2 # half the dimension for sine and cosine
        emb = math.log(10000) / (half_dim - 1) #scaling factor
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb) # creates frequencies
        emb = timesteps[:, None] * emb[None, :] # multiply timesteps with frequencies
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # If timestep is 347 for example then
        # This gives us: [sin(347*1.0), sin(347*0.9), ..., cos(347*1.0), cos(347*0.9), ...]
        return emb
    
    def forward(self, x, timesteps, text_embeddings=None):
        """
        Forward pass of the UNet.
        
        Args:
            x: Noisy images, shape (batch, channels, height, width)
            timesteps: Timestep values, shape (batch,)
            text_embeddings: Optional text embeddings, shape (batch, text_emb_dim)
            
        Returns:
            Predicted unit noise, same shape as input x
        """
        pass