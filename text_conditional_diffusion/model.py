"""Text-conditional U-Net for diffusion."""
import torch
import torch.nn as nn
import math
import config


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

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
    """Residual block with time and text conditioning."""

    def __init__(self, in_ch, out_ch, time_dim, text_dim=None):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.text_mlp = nn.Linear(text_dim, out_ch) if text_dim else None

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, text_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        # Add text embedding
        if self.text_mlp is not None and text_emb is not None:
            h = h + self.text_mlp(text_emb)[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


class TextConditionedUNet(nn.Module):
    """U-Net with CLIP text conditioning."""

    def __init__(self, text_dim=512):
        super().__init__()
        self.text_dim = text_dim

        self.time_emb = TimeEmbedding(config.TIME_DIM)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.TIME_DIM, config.TIME_DIM),
            nn.SiLU(),
            nn.Linear(config.TIME_DIM, config.TIME_DIM)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim)
        )

        # Down path
        self.down1 = ResBlock(1, config.CHANNELS, config.TIME_DIM, text_dim)
        self.down2 = ResBlock(config.CHANNELS, config.CHANNELS * 2, config.TIME_DIM, text_dim)
        self.down3 = ResBlock(config.CHANNELS * 2, config.CHANNELS * 4, config.TIME_DIM, text_dim)

        # Middle
        self.mid = ResBlock(config.CHANNELS * 4, config.CHANNELS * 4, config.TIME_DIM, text_dim)

        # Up path
        self.up3 = ResBlock(config.CHANNELS * 8, config.CHANNELS * 2, config.TIME_DIM, text_dim)
        self.up2 = ResBlock(config.CHANNELS * 4, config.CHANNELS, config.TIME_DIM, text_dim)
        self.up1 = ResBlock(config.CHANNELS * 2, config.CHANNELS, config.TIME_DIM, text_dim)

        # Output
        self.out = nn.Conv2d(config.CHANNELS, 1, 1)

        # Pooling/Upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, text_emb):
        """
        Args:
            x: [B, 1, H, W] noisy images
            t: [B] timesteps
            text_emb: [B, text_dim] CLIP text embeddings
        """
        # Embeddings
        t_emb = self.time_mlp(self.time_emb(t))
        text_emb = self.text_proj(text_emb)

        # Down
        h1 = self.down1(x, t_emb, text_emb)
        h2 = self.down2(self.pool(h1), t_emb, text_emb)
        h3 = self.down3(self.pool(h2), t_emb, text_emb)

        # Middle
        h = self.mid(self.pool(h3), t_emb, text_emb)

        # Up
        h = self.up3(torch.cat([self.upsample(h), h3], dim=1), t_emb, text_emb)
        h = self.up2(torch.cat([self.upsample(h), h2], dim=1), t_emb, text_emb)
        h = self.up1(torch.cat([self.upsample(h), h1], dim=1), t_emb, text_emb)

        return self.out(h)


if __name__ == "__main__":
    # Test model
    print("Testing Text-Conditioned U-Net...")
    model = TextConditionedUNet(text_dim=512)

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))
    text_emb = torch.randn(batch_size, 512)

    out = model(x, t, text_emb)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✅ Model test passed!")