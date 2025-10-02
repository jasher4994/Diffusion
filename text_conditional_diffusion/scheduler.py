import torch
import torch.nn.functional as F
import math


"""
  This scheduler has 3 main responsibilities:

  1. Setup (init) - Pre-compute noise schedule
  2. Training (q_sample) - Add noise to images
  3. Generation (p_sample_text + sample_text) - Remove noise
  step-by-step

"""


class SimpleDDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps

        # Linear beta schedule - can replace with cosine
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(
            self.alphas, dim=0
        )  # cumulative product - lets us jump to any timestep immediately.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others (pre-compute for efficiency)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # This tells us how much randomness is appropriate at this step.
        # Removing this would lead to mode-seeking behavior (and poor sample quality).
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Add noise to the clean images according to the noise schedule.

        So we can have examples at any timestep in the forward process."""
        # Generate original noise
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample_text(self, model, x, t, text_embeddings):
        """Sample x_{t-1} from x_t using the model with text conditioning."""
        # Get model prediction
        predicted_noise = model(x, t, text_embeddings)

        # Get coefficients
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(self.alphas), t, x.shape)

        # Compute x_{t-1}
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def sample_text(self, model, shape, text_embeddings, device="cuda"):
        """Generate samples using DDPM sampling with text conditioning."""
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample_text(model, img, t, text_embeddings)

            # Clamp to prevent explosion
            img = torch.clamp(img, -2.0, 2.0)

        return img


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to match x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
