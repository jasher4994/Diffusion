import torch
import math


class CosineNoiseScheduler:
    """
    Cosine noise scheduler that follows a cosine curve for noise addition.
    
    This scheduler adds noise more gradually at the beginning and end of the
    diffusion process, which often leads to better quality and training stability.

    At each timestep, a noise amount β(t) is defined by the schedule.

    The retention factor α(t) = 1 - β(t) represents how much signal is preserved in
    that single step (not how much of the original remains). 
    
    Alpha-bar (ᾱ) is the cumulative product of all individual alphas: ᾱ(t) = α(0) × α(1) × ... × α(t). 

    This cumulative alpha-bar tells us what percentage of the original image actually 
    remains at timestep t after all the noise additions up to that point. 

    An example of how the cumulative alpha-bar works:
    t=0: α(0)=0.99, ᾱ(0)=0.99 → 99% of original remains
    t=1: α(1)=0.98, ᾱ(1)=0.99×0.98=0.97 → 97% of original remains  
    t=2: α(2)=0.97, ᾱ(2)=0.97×0.97=0.94 → 94% of original remains
    ...
    and so on until t=1000 where ᾱ(1000)=0 (pure noise, no original content).

    """
    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        """
        Initialize the cosine noise scheduler and various instance attributes related to the noise schedule.
        Explanation of the parameters can be found in the class docstring.
        
        Args:
            num_timesteps: Total number of diffusion steps (T) - think of this as
                           how many timesteps from image to pure noise
            s: Small offset to prevent beta from being too small at t=0

        Returns:
            None
        """
        self.num_timesteps = num_timesteps
        self.s = s
        
        # Compute the cosine schedule
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Store alphas_cumprod (length: num_timesteps)
        self.alphas_cumprod = alphas_cumprod[:-1]  # Remove the last element to get length num_timesteps
        
        # Create alphas_cumprod_prev by padding with 1.0 at the beginning
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Compute individual alphas and betas
        self.alphas = self.alphas_cumprod / alphas_cumprod_prev
        self.betas = 1.0 - self.alphas
        
        # Pre-compute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def to(self, device):
        """Move scheduler tensors to device."""
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        return self
        
    def add_noise(self, original_images, timesteps):
        """
        Add noise to images according to the schedule.
        
        Args:
            original_images: Clean images, shape (batch, channels, height, width)
            timesteps: Timesteps for each image, shape (batch,)
            
        Returns:
            noisy_images: Images with noise added
            noise: The noise that was added
        """
        # Sample noise
        noise = torch.randn_like(original_images)
        
        # Get coefficients for the timesteps
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting (batch_size, 1, 1, 1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)
        
        # Add noise: x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        noisy_images = sqrt_alphas_cumprod * original_images + sqrt_one_minus_alphas_cumprod * noise
        
        return noisy_images, noise


