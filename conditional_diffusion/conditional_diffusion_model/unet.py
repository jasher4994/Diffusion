import torch
import torch.nn as nn
import math
from typing import Optional

class ResBlock(nn.Module):
    """
    Residual block that processes images and incorporates time + class embeddings.
    """
    
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim=None):
        super().__init__()
        
        # Image processing layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Class embedding projection (optional)
        self.class_proj = None
        if class_emb_dim is not None:
            self.class_proj = nn.Linear(class_emb_dim, out_channels)
        
        # Normalization and activation
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Residual connection (if input/output channels differ)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb, class_emb=None):
        residual = self.residual_conv(x)
        
        # First conv block
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time information
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_proj
        
        # Add class information (if provided)
        if self.class_proj is not None and class_emb is not None:
            class_proj = self.class_proj(class_emb)[:, :, None, None]
            h = h + class_proj
        
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
        num_classes: int = 10,     # Number of classes (CIFAR-10)
        class_emb_dim: int = 128,  # Class embedding dimension
        base_channels: int = 64,   # Base number of channels
        image_size: int = 128,     # Input image size (128x128)
    ):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.image_size = image_size
        
        # Calculate number of downsampling levels based on image size
        # For 128x128: 128 -> 64 -> 32 -> 16 -> 8 (4 levels)
        # For 32x32: 32 -> 16 -> 8 -> 4 (3 levels)
        self.num_levels = int(math.log2(image_size)) - 2  # Stop at 4x4 minimum
        print(f"🏗️  UNet: {self.num_levels} downsampling levels for {image_size}x{image_size} images")
        
        # Class embedding layer (learnable embeddings for each class)
        # +1 for unconditional class (-1)
        self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)
        
        # Timestep embedding layers
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        
        # Initial convolution to get to base_channels
        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder path - dynamic based on image size
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        current_channels = base_channels
        for level in range(self.num_levels):
            # Each encoder level has 2 ResBlocks
            encoder_blocks = nn.ModuleList([
                ResBlock(current_channels, current_channels, time_emb_dim * 4, class_emb_dim),
                ResBlock(current_channels, current_channels, time_emb_dim * 4, class_emb_dim)
            ])
            self.encoders.append(encoder_blocks)
            
            # Downsampler (except for last level)
            if level < self.num_levels - 1:
                next_channels = current_channels * 2
                self.downsamplers.append(
                    nn.Conv2d(current_channels, next_channels, 3, stride=2, padding=1)
                )
                current_channels = next_channels
            else:
                # No downsampler for last level
                self.downsamplers.append(None)
        
        # Bottleneck at the deepest level
        self.bottleneck = nn.ModuleList([
            ResBlock(current_channels, current_channels, time_emb_dim * 4, class_emb_dim),
            ResBlock(current_channels, current_channels, time_emb_dim * 4, class_emb_dim)
        ])
        
        # Decoder path - mirror of encoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for level in range(self.num_levels):
            # Upsampler (except for first decoder level)
            if level > 0:
                prev_channels = current_channels // 2
                self.upsamplers.append(
                    nn.ConvTranspose2d(current_channels, prev_channels, 3, stride=2, padding=1, output_padding=1)
                )
                # After upsampling + skip connection, input channels double
                decoder_input_channels = prev_channels * 2
                current_channels = prev_channels
            else:
                self.upsamplers.append(None)
                # First decoder level: no upsampling, just skip connection
                decoder_input_channels = current_channels * 2
            
            # Each decoder level has 2 ResBlocks
            decoder_blocks = nn.ModuleList([
                ResBlock(decoder_input_channels, current_channels, time_emb_dim * 4, class_emb_dim),
                ResBlock(current_channels, current_channels, time_emb_dim * 4, class_emb_dim)
            ])
            self.decoders.append(decoder_blocks)

        # Final output layer to predict noise
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # Initialize class embeddings
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        
        # Print architecture summary
        self._print_architecture_summary()
        
    def _print_architecture_summary(self):
        """Print a summary of the UNet architecture."""
        print(f"📐 UNet Architecture Summary:")
        print(f"   • Input: {self.image_size}x{self.image_size} -> Output: {self.image_size}x{self.image_size}")
        print(f"   • Downsampling levels: {self.num_levels}")
        
        current_size = self.image_size
        for level in range(self.num_levels):
            if level < self.num_levels - 1:
                next_size = current_size // 2
                print(f"   • Level {level}: {current_size}x{current_size}")
                current_size = next_size
            else:
                print(f"   • Bottom level: {current_size}x{current_size}")
        
    def get_timestep_embedding(self, timesteps, dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, timesteps, class_labels=None):
        """
        Forward pass with class conditioning.
        
        Args:
            x: Input images, shape (batch, 3, H, W)
            timesteps: Timestep for each image, shape (batch,)
            class_labels: Class labels, shape (batch,). Use -1 for unconditional
        """
        
        # Verify input size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            raise ValueError(f"Expected input size {self.image_size}x{self.image_size}, got {x.shape[-2]}x{x.shape[-1]}")
        
        # Step 1: Process timestep embeddings
        time_emb = self.get_timestep_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_mlp(time_emb)
        
        # Step 2: Process class embeddings
        class_emb = None
        if class_labels is not None:
            # Handle unconditional case (-1) by mapping to last embedding
            class_labels_mapped = torch.where(
                class_labels == -1, 
                torch.tensor(self.num_classes, device=class_labels.device), 
                class_labels
            )
            class_emb = self.class_embedding(class_labels_mapped)
        
        # Step 3: Initial convolution
        h = self.initial_conv(x)
        
        # Step 4: Encoder path (save skip connections)
        skip_connections = []
        
        for level in range(self.num_levels):
            # Apply encoder blocks
            for block in self.encoders[level]:
                h = block(h, time_emb, class_emb)
            
            # Save skip connection
            skip_connections.append(h)
            
            # Downsample (if not last level)
            if self.downsamplers[level] is not None:
                h = self.downsamplers[level](h)
        
        # Step 5: Bottleneck
        for block in self.bottleneck:
            h = block(h, time_emb, class_emb)
        
        # Step 6: Decoder path (use skip connections in reverse order)
        for level in range(self.num_levels):
            # Upsample (if not first decoder level)
            if self.upsamplers[level] is not None:
                h = self.upsamplers[level](h)
            
            # Concatenate with skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            
            # Apply decoder blocks
            for block in self.decoders[level]:
                h = block(h, time_emb, class_emb)
        
        # Step 7: Final prediction
        predicted_noise = self.final_conv(h)
        
        return predicted_noise