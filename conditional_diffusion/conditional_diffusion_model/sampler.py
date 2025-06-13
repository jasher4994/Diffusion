import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

class DDPMSampler:
    """
    Enhanced DDPM Sampler with improved generation quality.
    
    This class handles the reverse diffusion process with:
    - Better noise initialization
    - Classifier-free guidance support
    - Dynamic thresholding for stability
    """
    
    def __init__(self, noise_scheduler, unet, device='cuda'):
        self.noise_scheduler = noise_scheduler
        self.unet = unet
        self.device = device
        
    @torch.no_grad()
    def sample(self, batch_size=1, image_size=(3, 64, 64), num_inference_steps=None, 
               text_embeddings=None, guidance_scale=1.0, dynamic_thresholding=True):
        """
        Generate images from pure noise with enhanced quality controls.
        
        Args:
            batch_size: Number of images to generate
            image_size: (channels, height, width) of generated images
            num_inference_steps: Number of denoising steps (default: all timesteps)
            text_embeddings: Text embeddings for conditional generation (optional)
            guidance_scale: Strength of conditional guidance (1.0 = no guidance)
            dynamic_thresholding: Apply dynamic thresholding for stability
            
        Returns:
            Generated images, shape (batch_size, channels, height, width)
        """
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
            
        # Improved noise initialization
        x = torch.randn(batch_size, *image_size, device=self.device)
        x = x * 0.8  # Scale down initial noise for better stability
        
        # Create denoising schedule
        timesteps = torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, 
                                 num_inference_steps, dtype=torch.long, device=self.device)
        
        self.unet.eval()
        
        for i, t in enumerate(tqdm(timesteps, desc="Generating", leave=False)):
            # Create timestep tensor for the batch
            t_batch = t.repeat(batch_size)
            
            # Predict noise with classifier-free guidance if text conditioning is used
            if text_embeddings is not None and guidance_scale > 1.0:
                # Conditional prediction
                predicted_noise_cond = self.unet(x, t_batch, text_embeddings)
                
                # Unconditional prediction (empty text embeddings)
                empty_embeddings = torch.zeros_like(text_embeddings)
                predicted_noise_uncond = self.unet(x, t_batch, empty_embeddings)
                
                # Apply classifier-free guidance
                predicted_noise = predicted_noise_uncond + guidance_scale * (
                    predicted_noise_cond - predicted_noise_uncond
                )
            elif text_embeddings is not None:
                # Standard conditional prediction without guidance
                predicted_noise = self.unet(x, t_batch, text_embeddings)
            else:
                # Unconditional generation
                predicted_noise = self.unet(x, t_batch)
            
            # Apply dynamic thresholding for stability
            if dynamic_thresholding:
                predicted_noise = self._dynamic_thresholding(predicted_noise)
            
            # Remove predicted noise to get cleaner image
            x = self.denoise_step(x, predicted_noise, t)
            
            # Clamp values to prevent explosion (with gradual relaxation)
            clamp_value = 1.0 + 0.5 * (1 - i / len(timesteps))  # Gradually increase clamp
            x = torch.clamp(x, -clamp_value, clamp_value)
        
        # Final clamping to valid range
        x = torch.clamp(x, -1, 1)
        return x
    
    def _dynamic_thresholding(self, predicted_noise, percentile=0.995):
        """Apply dynamic thresholding to predicted noise for stability."""
        batch_size = predicted_noise.shape[0]
        
        for i in range(batch_size):
            # Calculate dynamic threshold per sample
            sample = predicted_noise[i].flatten()
            dynamic_threshold = torch.quantile(torch.abs(sample), percentile)
            
            # Apply thresholding
            predicted_noise[i] = torch.clamp(
                predicted_noise[i], 
                -dynamic_threshold, 
                dynamic_threshold
            )
        
        return predicted_noise
    
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