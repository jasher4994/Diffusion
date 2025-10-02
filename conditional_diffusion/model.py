import torch
import torch.nn as nn
import math
import config

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, class_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.class_mlp = nn.Linear(class_dim, out_ch) if class_dim else None

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        # Add class embedding
        if self.class_mlp is not None and c_emb is not None:
            h = h + self.class_mlp(c_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)

class SimpleUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Embeddings
        self.time_emb = TimeEmbedding(config.TIME_DIM)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.TIME_DIM, config.TIME_DIM),
            nn.SiLU(),
            nn.Linear(config.TIME_DIM, config.TIME_DIM)
        )

        # Class embedding
        self.class_emb = nn.Embedding(num_classes + 1, config.CLASS_DIM)  # +1 for unconditional

        # Down path
        self.down1 = ResBlock(1, config.CHANNELS, config.TIME_DIM, config.CLASS_DIM)
        self.down2 = ResBlock(config.CHANNELS, config.CHANNELS * 2, config.TIME_DIM, config.CLASS_DIM)
        self.down3 = ResBlock(config.CHANNELS * 2, config.CHANNELS * 4, config.TIME_DIM, config.CLASS_DIM)

        # Middle
        self.mid = ResBlock(config.CHANNELS * 4, config.CHANNELS * 4, config.TIME_DIM, config.CLASS_DIM)

        # Up path
        self.up3 = ResBlock(config.CHANNELS * 8, config.CHANNELS * 2, config.TIME_DIM, config.CLASS_DIM)
        self.up2 = ResBlock(config.CHANNELS * 4, config.CHANNELS, config.TIME_DIM, config.CLASS_DIM)
        self.up1 = ResBlock(config.CHANNELS * 2, config.CHANNELS, config.TIME_DIM, config.CLASS_DIM)

        # Output
        self.out = nn.Conv2d(config.CHANNELS, 1, 1)

        # Pooling/Upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, c):
        # Handle unconditional generation
        c = torch.where(c == -1, self.num_classes, c)  # Map -1 to last embedding

        # Embeddings
        t_emb = self.time_mlp(self.time_emb(t))
        c_emb = self.class_emb(c)

        # Down
        h1 = self.down1(x, t_emb, c_emb)
        h2 = self.down2(self.pool(h1), t_emb, c_emb)
        h3 = self.down3(self.pool(h2), t_emb, c_emb)

        # Middle
        h = self.mid(self.pool(h3), t_emb, c_emb)

        # Up
        h = self.up3(torch.cat([self.upsample(h), h3], dim=1), t_emb, c_emb)
        h = self.up2(torch.cat([self.upsample(h), h2], dim=1), t_emb, c_emb)
        h = self.up1(torch.cat([self.upsample(h), h1], dim=1), t_emb, c_emb)

        return self.out(h)