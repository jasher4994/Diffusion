import torch
import torch.nn.functional as F
import math

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        """
        Base diffusion model that handles noise scheduling and diffusion process
        """
        self.timesteps = timesteps
        
        # Setup beta schedule
        if beta_schedule == 'linear':
            self.betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """Linear beta schedule"""
        return torch.linspace(start, end, timesteps)
    
    def _cosine_beta_schedule(self, T, s=0.008):
        """Cosine beta schedule from Nichol & Dhariwal (2021)"""
        steps = torch.arange(T + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    
    def _get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of vals, properly reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to the image
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t.to(x_0.device) * x_0 + sqrt_one_minus_alphas_cumprod_t.to(x_0.device) * noise, noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, cond=None):
        """
        Sample from p(x_{t-1} | x_t)
        """
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (with or without conditioning)
        if cond is not None:
            model_output = model(x, t, cond)
        else:
            model_output = model(x, t)
        
        # Calculate the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Add noise if not the final step
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x.shape)
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, device="cpu", cond=None):
        """
        Generate a sample using the model
        """
        # Start from pure noise
        img = torch.randn(*shape, device=device)
        
        # Iterate backward through timesteps
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond)
            
        return img