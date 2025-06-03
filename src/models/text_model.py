import torch
import torch.nn as nn
from .unet import SinusoidalPositionEmbeddings, SimpleBlock

class SimpleTextConditionedUNet(nn.Module):
    def __init__(self, image_channels=3, model_channels=64, time_emb_dim=128, text_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Simple channel progression
        ch = model_channels
        self.ch1, self.ch2, self.ch3, self.ch4 = ch, ch*2, ch*4, ch*8
        
        # Encoder
        self.conv_in = nn.Conv2d(image_channels, self.ch1, 3, padding=1)
        self.down1 = SimpleBlock(self.ch1, self.ch2, time_emb_dim, text_emb_dim)
        self.down2 = SimpleBlock(self.ch2, self.ch3, time_emb_dim, text_emb_dim)
        self.down3 = SimpleBlock(self.ch3, self.ch4, time_emb_dim, text_emb_dim)
        
        # Middle
        self.mid = nn.Sequential(
            nn.Conv2d(self.ch4, self.ch4, 3, padding=1),
            nn.BatchNorm2d(self.ch4),
            nn.ReLU()
        )
        
        # Decoder (no skip connections for simplicity)
        self.up3 = SimpleBlock(self.ch4, self.ch3, time_emb_dim, text_emb_dim, up=True)
        self.up2 = SimpleBlock(self.ch3, self.ch2, time_emb_dim, text_emb_dim, up=True)
        self.up1 = SimpleBlock(self.ch2, self.ch1, time_emb_dim, text_emb_dim, up=True)
        
        # Output
        self.conv_out = nn.Conv2d(self.ch1, image_channels, 1)
        
    def forward(self, x, timestep, text_embeddings):
        # Get time embeddings
        t_emb = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Encoder
        x = self.down1(x, t_emb, text_embeddings)
        x = self.down2(x, t_emb, text_embeddings)
        x = self.down3(x, t_emb, text_embeddings)
        
        # Middle
        x = self.mid(x)
        
        # Decoder
        x = self.up3(x, t_emb, text_embeddings)
        x = self.up2(x, t_emb, text_embeddings)
        x = self.up1(x, t_emb, text_embeddings)
        
        # Output
        return self.conv_out(x)