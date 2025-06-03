import torch
import torch.nn as nn
from .unet import Block, SinusoidalPositionEmbeddings

class UncondUNet(nn.Module):
    """
    Unconditional U-Net for diffusion models
    """
    def __init__(self, image_channels=3, model_channels=64, num_res_blocks=2, time_emb_dim=32):
        super().__init__()
        self.time_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Define channels at each resolution
        down_channels = (model_channels, model_channels*2, model_channels*4, model_channels*8, model_channels*16)
        up_channels = (model_channels*16, model_channels*8, model_channels*4, model_channels*2, model_channels)
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        
        # Downsample blocks
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_emb_dim)
            for i in range(len(down_channels)-1)
        ])
        
        # Upsample blocks
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
            for i in range(len(up_channels)-1)
        ])
        
        # Final output projection
        self.output = nn.Conv2d(model_channels, image_channels, 1)
        
    def forward(self, x, timestep):
        # Embed time
        t_emb = self.time_mlp(timestep)
        
        # Initial convolution
        x = self.conv0(x)
        
        # Apply U-Net and track skip connections
        residuals = []
        for down in self.downs:
            residuals.append(x)
            x = down(x, t_emb)
        
        # Reverse and apply upsampling with skip connections
        residuals = residuals[::-1]  # Reverse for easier pop
        for i, up in enumerate(self.ups):
            x = torch.cat([x, residuals[i]], dim=1)
            x = up(x, t_emb)
            
        # Final projection to image space
        return self.output(x)