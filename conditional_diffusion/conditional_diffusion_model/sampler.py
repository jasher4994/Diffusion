import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

class DDPMSampler:
    """
    DDPM (Denoising Diffusion Probabilistic Model) Sampler.
    
    This class handles the reverse diffusion process - generating images
    from pure noise by iteratively removing predicted noise.
    """
    
    def __init__(self, noise_scheduler, unet, device='cuda'):
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.device = device
        
    @torch.no_grad()
    def sample(self, batch_size=1, image_size=(3, 64, 64), num_inference_steps=None, text_embeddings=None):
        """
        Generate images from pure noise.
        
        Args:
            batch_size: Number of images to generate
            image_size: (channels, height, width) of generated images
            num_inference_steps: Number of denoising steps (default: all timesteps)
            text_embeddings: Text embeddings for conditional generation (optional)
            
        Returns:
            Generated images, shape (batch_size, channels, height, width)
        """
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
            
        # Step 1: Start with proper noise (don't scale down!)
        x = torch.randn(batch_size, *image_size, device=self.device)
        
        # Step 2: Use proper DDPM timestep spacing
        # Instead of linear spacing, use the actual timesteps with proper spacing
        if num_inference_steps < self.noise_scheduler.num_timesteps:
            # Use every nth timestep for faster sampling
            step_size = self.noise_scheduler.num_timesteps // num_inference_steps
            timesteps = torch.arange(self.noise_scheduler.num_timesteps - 1, -1, -step_size, 
                                   dtype=torch.long, device=self.device)
            # Ensure we end at timestep 0
            if timesteps[-1] != 0:
                timesteps = torch.cat([timesteps, torch.tensor([0], device=self.device)])
        else:
            timesteps = torch.arange(self.noise_scheduler.num_timesteps - 1, -1, -1, 
                                   dtype=torch.long, device=self.device)
        
        self.unet.eval()
        
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            # Create timestep tensor for the batch
            t_batch = t.repeat(batch_size)
            
            # Predict noise using UNet (with optional text conditioning)
            if text_embeddings is not None:
                predicted_noise = self.unet(x, t_batch, text_embeddings)
            else:
                predicted_noise = self.unet(x, t_batch)
            
            # Remove predicted noise to get cleaner image
            x = self.denoise_step(x, predicted_noise, t)
            
            # More conservative clamping to preserve detail
            x = torch.clamp(x, -3.0, 3.0)
            
            # Check for NaN/Inf and break if found
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️  NaN/Inf detected at step {i}, stopping early")
                break
                
        return x
    
    def denoise_step(self, noisy_image, predicted_noise, timestep):
        """
        Single denoising step using DDPM sampling formula.
        
        At timestep 0: Only denoising, no stochastic noise added
        At timestep > 0: Denoising + controlled stochastic noise
        """
        # Get scheduler values for this timestep
        alpha_t = self.noise_scheduler.alphas[timestep]
        alpha_bar_t = self.noise_scheduler.alphas_cumprod[timestep]
        beta_t = self.noise_scheduler.betas[timestep]
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # Handle timestep 0 specially
        if timestep == 0:
            # At timestep 0, we don't add any noise - just return the denoised mean
            sqrt_alpha_t = torch.sqrt(alpha_t + eps)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t + eps)
            
            # Clamp predicted noise to prevent explosion
            predicted_noise = torch.clamp(predicted_noise, -5.0, 5.0)
            
            # Final denoising step (no variance, no added noise)
            x_prev = (1.0 / sqrt_alpha_t) * (
                noisy_image - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
            )
            
            return x_prev
        
        else:
            # Regular denoising step (timestep > 0)
            alpha_bar_prev = self.noise_scheduler.alphas_cumprod[timestep - 1]
            
            # Compute variance for this step (with clipping for stability)
            variance = torch.clamp(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t + eps) * beta_t,
                min=0.0, max=0.1  # Prevent variance explosion
            )
            
            # Sample random noise for stochastic sampling
            noise = torch.randn_like(noisy_image) * 0.1  # Scale down random noise
            
            # Apply the DDPM denoising formula with clamping
            sqrt_alpha_t = torch.sqrt(alpha_t + eps)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t + eps)
            
            # Clamp the predicted noise to reasonable values
            predicted_noise = torch.clamp(predicted_noise, -5.0, 5.0)
            
            mean = (1.0 / sqrt_alpha_t) * (
                noisy_image - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
            )
            
            # Add stochastic noise
            x_prev = mean + torch.sqrt(variance + eps) * noise
            
            return x_prev