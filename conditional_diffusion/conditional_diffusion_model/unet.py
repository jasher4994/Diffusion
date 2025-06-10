import torch
import torch.nn as nn
import math
from typing import Optional

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for text conditioning."""
    
    def __init__(self, image_channels, text_emb_dim, num_heads=8):
        super().__init__()
        self.image_channels = image_channels
        self.text_emb_dim = text_emb_dim
        self.num_heads = num_heads
        self.head_dim = image_channels // num_heads
        
        assert image_channels % num_heads == 0, "image_channels must be divisible by num_heads"
        
        # Query from image features, Key/Value from text features
        self.to_q = nn.Linear(image_channels, image_channels, bias=False)
        self.to_k = nn.Linear(text_emb_dim, image_channels, bias=False)
        self.to_v = nn.Linear(text_emb_dim, image_channels, bias=False)
        
        self.to_out = nn.Linear(image_channels, image_channels)
        self.norm = nn.LayerNorm(image_channels)
        
    def forward(self, x, text_emb):
        """
        Args:
            x: Image features, shape (batch, channels, height, width)
            text_emb: Text embeddings, shape (batch, text_emb_dim)
        """
        batch, channels, height, width = x.shape
        
        # Reshape image features for attention: (batch, height*width, channels)
        x_reshaped = x.view(batch, channels, height * width).transpose(1, 2)
        
        # Add text embeddings as an extra "token"
        # text_emb shape: (batch, text_emb_dim) -> (batch, 1, text_emb_dim)
        text_emb = text_emb.unsqueeze(1)
        
        # Generate queries from image, keys/values from text
        q = self.to_q(x_reshaped)  # (batch, height*width, channels)
        k = self.to_k(text_emb)    # (batch, 1, channels)
        v = self.to_v(text_emb)    # (batch, 1, channels)
        
        # Reshape for multi-head attention
        q = q.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention weights: image features attend to text
        scale = self.head_dim ** -0.5
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, height * width, channels)
        attn_output = self.to_out(attn_output)
        
        # Residual connection and reshape back to image format
        x_reshaped = self.norm(x_reshaped + attn_output)
        x_out = x_reshaped.transpose(1, 2).view(batch, channels, height, width)
        
        return x_out

class ResBlock(nn.Module):
    """
    Residual block that processes images and incorporates time + text embeddings.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim=None, use_cross_attention=False):
        super().__init__()
        
        # Image processing layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Text conditioning (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention and text_emb_dim is not None:
            self.cross_attention = CrossAttentionBlock(out_channels, text_emb_dim)
        
        # Normalization and activation
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Residual connection (if input/output channels differ)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb, text_emb=None):
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time information
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_proj
        
        # Cross-attention with text (if enabled)
        if self.use_cross_attention and text_emb is not None:
            h = self.cross_attention(h, text_emb)
        
        # Second conv block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return h + residual

class UNet(nn.Module):
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
        self.text_emb_dim = text_emb_dim
        
        # Timestep embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        
        # Initial convolution to get to base_channels
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.encoder1 = nn.ModuleList([
            ResBlock(base_channels, base_channels, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels, base_channels, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        
        self.encoder2 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        self.encoder3 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 4, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels * 4, base_channels * 4, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.bottleneck = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels * 8, base_channels * 8, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ModuleList([
            ResBlock(base_channels * 8, base_channels * 4, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels * 4, base_channels * 4, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ModuleList([
            ResBlock(base_channels * 4, base_channels * 2, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels * 2, base_channels * 2, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1)
        self.decoder1 = nn.ModuleList([
            ResBlock(base_channels * 2, base_channels, time_emb_dim * 4, text_emb_dim, use_cross_attention=True),
            ResBlock(base_channels, base_channels, time_emb_dim * 4, text_emb_dim, use_cross_attention=True)
        ])

        # Final output layer to predict noise
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        
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
        """Forward pass with text conditioning."""
        
        # Step 1: Process timestep embeddings
        time_emb = self.get_timestep_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_mlp(time_emb)
        
        # Step 2: Initial convolution
        h = self.initial_conv(x)
        
        # Step 3: Encoder path (save skip connections)
        skip_connections = []
        
        # Encoder level 1
        for block in self.encoder1:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        skip_connections.append(h)
        h = self.down1(h)
        
        # Encoder level 2
        for block in self.encoder2:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        skip_connections.append(h)
        h = self.down2(h)
        
        # Encoder level 3
        for block in self.encoder3:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        skip_connections.append(h)
        h = self.down3(h)
        
        # Step 4: Bottleneck
        for block in self.bottleneck:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        
        # Step 5: Decoder path (use skip connections)
        # Decoder level 3
        h = self.up3(h)
        h = torch.cat([h, skip_connections.pop()], dim=1)
        for block in self.decoder3:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        
        # Decoder level 2
        h = self.up2(h)
        h = torch.cat([h, skip_connections.pop()], dim=1)
        for block in self.decoder2:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        
        # Decoder level 1
        h = self.up1(h)
        h = torch.cat([h, skip_connections.pop()], dim=1)
        for block in self.decoder1:
            h = block(h, time_emb, text_embeddings)  # Pass text embeddings
        
        # Step 6: Final prediction
        predicted_noise = self.final_conv(h)
        
        return predicted_noise