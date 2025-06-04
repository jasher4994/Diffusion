import torch
import math
from typing import Tuple
import pdb


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
        
        # Create timestep array - simple array from 0 to num_timesteps
        timesteps = torch.arange(0, num_timesteps + 1, dtype=torch.float32)

        # Compute alpha_bar values from the tensor for timesteps using the cosine schedule.
        # So each timestep has a corresponding alpha_bar value.
        # Results in a smooth curve from 1 (at t=0) to 0 (at t=num_timesteps) if you plot the tensor
        alphas_cumprod = torch.cos(((timesteps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2

        # Normalise the offsets after we added in the small offset s - get back to 1.0 at t=0
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        # Store alpha_bar values (remove the last one to have exactly num_timesteps)
        self.alphas_cumprod = alphas_cumprod[:-1]
        
        # Compute betas from alphas_cumprod - more efficient than computing alphas directly.
        # alpha(t) = alpha_bar(t) / alpha_bar(t-1)
        # For t=0, we use alpha_bar(0) / 1.0
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.alphas = self.alphas_cumprod / alphas_cumprod_prev
        self.betas = 1.0 - self.alphas
        
        # Clamp betas to prevent numerical issues
        self.betas = torch.clamp(self.betas, 0, 0.999)
    
    def add_noise(self, original_images: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to AT A SPECIFIC TIMESTEP using cosine schedule.

        The important point here is that we use the reparameterization trick to add noise
        at specific timesteps without iterating through each step. This allows us to
        directly compute the noisy image at any timestep t using the cumulative alpha values.

        This returns the noisy images and the CUMULATIVE noise that was added.
        The image is how the image looks at that timestep, and the noise is the
        random noise that was added to the original image to create the noisy image.

        
        Args:
            original_images: Clean images, shape (batch, channels, height, width)
            timesteps: Which timestep to noise to, shape (batch,)
        
        Returns:
            Tuple of (noisy_images, noise_that_was_added)

        """
        # Generate random noise with same shape as images
        noise = torch.randn_like(original_images)
        
        # Get the cumulative alpha values for the specific timesteps
        # both of these tensors are computed to be used in the reparameterization trick
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timesteps]) # to preserve the variance
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps]) 
        
        # Reshape for broadcasting (batch_size, 1, 1, 1)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)
        
        # Apply noise using the reparameterization trick
        # Only a portion of the noise is added based on the cumulative alpha values.
        noisy_images = sqrt_alphas_cumprod * original_images + sqrt_one_minus_alphas_cumprod * noise

        # This gives us the noisy image at a specific timestep and the amount of noise added.
        # INPUT to model = noisy_images + timestep of those noisy images
        # Target of model = predict the noise that was added to the original image.
        
        # But we actually return the UNSCALED NOISE 
        # Model actually learns to predict the UNSCALED noise 
        # Loss compares the UNSCALED predicted v original UNSCALED noise
        # HOWEVER we do not remove all of it, we scale it before we remove it in the denoising step.

        # The model learns to predict the "unit" noise, and the denoising formula handles
        # the appropriate scaling for each timestep.

        # This design makes the model's job simpler - it always predicts "standard" noise regardless
        # of timestep, and the math handles the rest!

        return noisy_images, noise
    

