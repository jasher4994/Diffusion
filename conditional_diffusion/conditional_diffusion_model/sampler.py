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
        
        # Get image size from UNet
        self.image_size = getattr(unet, 'image_size', 128)  # Default to 128 if not specified
        print(f"🎨 Sampler initialized for {self.image_size}x{self.image_size} images")
        
    @torch.no_grad()
    def sample(self, batch_size=1, image_size=None, num_inference_steps=None, 
               class_labels=None, guidance_scale=7.5):
        """
        Generate images from pure noise with class conditioning.
        
        Args:
            batch_size: Number of images to generate
            image_size: (channels, height, width) or int for square images. If None, uses UNet's default
            num_inference_steps: Number of denoising steps (default: all timesteps)
            class_labels: Class labels for conditional generation, shape (batch_size,)
            guidance_scale: Strength of classifier-free guidance (higher = more adherence to class)
            
        Returns:
            Generated images, shape (batch_size, channels, height, width)
        """
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps
        
        # Handle image_size parameter
        if image_size is None:
            # Use UNet's default size
            img_size = self.image_size
            image_shape = (3, img_size, img_size)
        elif isinstance(image_size, int):
            # Single int means square image
            image_shape = (3, image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 3:
            # Full shape provided
            image_shape = image_size
        else:
            raise ValueError("image_size must be None, int, or tuple of (channels, height, width)")
            
        print(f"🎯 Generating {batch_size} images of size {image_shape[1]}x{image_shape[2]}")
        
        # Step 1: Start with proper noise
        x = torch.randn(batch_size, *image_shape, device=self.device)
        
        # Step 2: Use proper DDPM timestep spacing
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
            
            # Predict noise using class conditioning
            if class_labels is not None and guidance_scale > 1.0:
                # Classifier-free guidance: predict both conditional and unconditional
                
                # Conditional prediction
                predicted_noise_cond = self.unet(x, t_batch, class_labels)
                
                # Unconditional prediction (use -1 for no class)
                unconditional_labels = torch.full_like(class_labels, -1)
                predicted_noise_uncond = self.unet(x, t_batch, unconditional_labels)
                
                # Apply classifier-free guidance
                predicted_noise = predicted_noise_uncond + guidance_scale * (
                    predicted_noise_cond - predicted_noise_uncond
                )
            elif class_labels is not None:
                # Simple conditional generation without guidance
                predicted_noise = self.unet(x, t_batch, class_labels)
            else:
                # Unconditional generation
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
    
    def sample_class_conditional(self, class_idx, num_samples=4, guidance_scale=7.5, 
                               image_size=None, num_inference_steps=None):
        """
        Generate images for a specific CIFAR-10 class.
        
        Args:
            class_idx: CIFAR-10 class index (0-9)
            num_samples: Number of images to generate
            guidance_scale: Classifier-free guidance strength
            image_size: Image size (uses UNet default if None)
            num_inference_steps: Number of denoising steps (uses all if None)
            
        Returns:
            Generated images
        """
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"Generating {num_samples} images of class: {class_names[class_idx]}")
        
        # Create class labels tensor
        class_labels = torch.full((num_samples,), class_idx, device=self.device)
        
        # Generate images
        images = self.sample(
            batch_size=num_samples,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            class_labels=class_labels,
            guidance_scale=guidance_scale
        )
        
        return images
    
    def sample_all_classes(self, samples_per_class=2, guidance_scale=7.5, 
                          image_size=None, num_inference_steps=None):
        """
        Generate images for all CIFAR-10 classes.
        
        Args:
            samples_per_class: Number of samples per class
            guidance_scale: Classifier-free guidance strength
            image_size: Image size (uses UNet default if None)
            num_inference_steps: Number of denoising steps (uses all if None)
            
        Returns:
            Dictionary with class names as keys and generated images as values
        """
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        results = {}
        
        for class_idx, class_name in enumerate(class_names):
            print(f"Generating {samples_per_class} images for class: {class_name}")
            images = self.sample_class_conditional(
                class_idx=class_idx,
                num_samples=samples_per_class,
                guidance_scale=guidance_scale,
                image_size=image_size,
                num_inference_steps=num_inference_steps
            )
            results[class_name] = images
            
        return results
    
    def sample_grid(self, grid_size=3, guidance_scale=7.5, image_size=None, 
                   num_inference_steps=50, classes=None):
        """
        Generate a grid of images for visualization.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size images)
            guidance_scale: Classifier-free guidance strength
            image_size: Image size (uses UNet default if None)
            num_inference_steps: Number of denoising steps (faster generation)
            classes: List of class indices to sample from (random if None)
            
        Returns:
            Grid of generated images
        """
        total_samples = grid_size * grid_size
        
        if classes is None:
            # Random classes
            class_labels = torch.randint(0, 10, (total_samples,), device=self.device)
        else:
            # Cycle through provided classes
            class_labels = torch.tensor([classes[i % len(classes)] for i in range(total_samples)], 
                                      device=self.device)
        
        print(f"🎨 Generating {grid_size}x{grid_size} grid of images...")
        
        # Generate all images
        images = self.sample(
            batch_size=total_samples,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            class_labels=class_labels,
            guidance_scale=guidance_scale
        )
        
        return images.reshape(grid_size, grid_size, *images.shape[1:])
    
    def sample_interpolation(self, class_start, class_end, num_steps=8, 
                           guidance_scale=7.5, image_size=None, num_inference_steps=50):
        """
        Generate images interpolating between two classes.
        
        Args:
            class_start: Starting class index
            class_end: Ending class index
            num_steps: Number of interpolation steps
            guidance_scale: Classifier-free guidance strength
            image_size: Image size (uses UNet default if None)
            num_inference_steps: Number of denoising steps
            
        Returns:
            Series of interpolated images
        """
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"🌈 Interpolating from {class_names[class_start]} to {class_names[class_end]}")
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        
        # Get class embeddings
        start_emb = self.unet.class_embedding(torch.tensor([class_start], device=self.device))
        end_emb = self.unet.class_embedding(torch.tensor([class_end], device=self.device))
        
        images = []
        
        for alpha in alphas:
            # Interpolate embeddings
            interp_emb = (1 - alpha) * start_emb + alpha * end_emb
            
            # This is a simplified version - full implementation would require
            # modifying the UNet to accept pre-computed embeddings
            # For now, we'll sample discrete classes
            if alpha < 0.5:
                class_label = torch.tensor([class_start], device=self.device)
            else:
                class_label = torch.tensor([class_end], device=self.device)
                
            image = self.sample(
                batch_size=1,
                image_size=image_size,
                num_inference_steps=num_inference_steps,
                class_labels=class_label,
                guidance_scale=guidance_scale
            )
            
            images.append(image)
        
        return torch.cat(images, dim=0)
    
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

# Utility functions for visualization
def tensor_to_image(tensor):
    """Convert tensor to PIL Image for saving/display."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.detach().cpu()
    tensor = (tensor.clamp(-1, 1) + 1) * 0.5
    
    import torchvision.transforms.functional as TF
    return TF.to_pil_image(tensor)

def save_image_grid(images, save_path, grid_size=None):
    """Save a grid of images."""
    import matplotlib.pyplot as plt
    
    if images.dim() == 4:
        batch_size = images.shape[0]
        if grid_size is None:
            grid_size = int(batch_size ** 0.5)
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten() if grid_size > 1 else [axes]
        
        for i in range(min(batch_size, grid_size * grid_size)):
            img = tensor_to_image(images[i])
            axes[i].imshow(img)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved image grid: {save_path}")
    else:
        # Single image
        img = tensor_to_image(images)
        img.save(save_path)
        print(f"💾 Saved image: {save_path}")