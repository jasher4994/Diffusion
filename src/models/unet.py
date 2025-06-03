import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, text_emb_dim, up=False):
        super().__init__()
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        # Text embedding projection
        self.text_mlp = nn.Linear(text_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, time_emb, text_emb):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Add time and text embeddings
        time_emb = self.relu(self.time_mlp(time_emb))
        text_emb = self.relu(self.text_mlp(text_emb))
        
        # Combine embeddings and add to features
        combined_emb = time_emb + text_emb
        combined_emb = combined_emb[(..., ) + (None, ) * 2]  # Add spatial dimensions
        h = h + combined_emb
        
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Transform
        return self.transform(h)