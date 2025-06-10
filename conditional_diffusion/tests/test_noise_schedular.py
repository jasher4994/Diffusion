"""
Test the Cosine Noise Scheduler with real images and visualizations.

This script demonstrates how the noise scheduler gradually corrupts images
and provides visual feedback to understand the diffusion process.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from typing import List


def create_output_dir() -> str:
    """
    Create output directory for saving test results.
    
    Returns:
        Path to the output directory
    """
    output_dir = os.path.join(os.path.dirname(__file__), '.', 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_test_image(image_path: str = None, size: int = 256) -> torch.Tensor:
    """
    Load and preprocess a test image for diffusion testing.
    
    Args:
        image_path: Path to image file (if None, creates a simple test pattern)
        size: Size to resize image to
        
    Returns:
        Preprocessed image tensor, shape (1, 3, size, size)
    """
    if image_path is None:
        # Create a simple test pattern if no image provided
        img = torch.zeros(3, size, size)
        
        # Create colorful test pattern
        square_size = size // 8
        for i in range(0, size, square_size * 2):
            for j in range(0, size, square_size * 2):
                img[0, i:i+square_size, j:j+square_size] = 1.0  # Red squares
                img[1, i+square_size:i+2*square_size, j+square_size:j+2*square_size] = 1.0  # Green squares
        
        # Add blue diagonal
        for i in range(size):
            if i < size - 10:
                img[2, i:i+10, i:i+10] = 1.0
        
        return img.unsqueeze(0)
    
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            return transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            print("Using test pattern instead...")
            return load_test_image(None, size)


def visualize_noise_progression(scheduler: CosineNoiseScheduler, 
                              original_image: torch.Tensor,
                              timesteps: List[int],
                              output_dir: str) -> None:
    """
    Visualize how noise is added to an image over different timesteps.
    
    Args:
        scheduler: The noise scheduler to use
        original_image: Original clean image
        timesteps: List of timesteps to visualize
        output_dir: Directory to save the visualization
    """
    fig, axes = plt.subplots(2, len(timesteps), figsize=(3*len(timesteps), 6))
    
    for i, t in enumerate(timesteps):
        timestep_tensor = torch.tensor([t])
        noisy_image, noise = scheduler.add_noise(original_image, timestep_tensor)
        
        # Convert to numpy for visualization
        img_np = torch.clamp(noisy_image[0], 0, 1).permute(1, 2, 0).numpy()
        noise_np = torch.clamp((noise[0] + 1) / 2, 0, 1).permute(1, 2, 0).numpy()
        
        # Plot noisy image
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'Noisy Image\nt={t}')
        axes[0, i].axis('off')
        
        # Plot noise
        axes[1, i].imshow(noise_np)
        axes[1, i].set_title(f'Added Noise\nt={t}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_dir, 'noise_progression.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Noise progression saved to: {save_path}")
    
    # Also save individual images for closer inspection
    for i, t in enumerate(timesteps):
        timestep_tensor = torch.tensor([t])
        noisy_image, noise = scheduler.add_noise(original_image, timestep_tensor)
        
        # Save noisy image
        img_np = torch.clamp(noisy_image[0], 0, 1).permute(1, 2, 0).numpy()
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.title(f'Noisy Image at t={t}')
        plt.axis('off')
        individual_path = os.path.join(output_dir, f'noisy_image_t{t:03d}.png')
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    plt.close('all')  # Close the main figure


def plot_schedule_curves(scheduler: CosineNoiseScheduler, output_dir: str) -> None:
    """
    Plot the alpha_bar and beta curves to understand the schedule.
    
    Args:
        scheduler: The noise scheduler
        output_dir: Directory to save the plot
    """
    timesteps = torch.arange(scheduler.num_timesteps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot alpha_bar (cumulative product)
    ax1.plot(timesteps, scheduler.alphas_cumprod)
    ax1.set_title('Alpha Bar (Cumulative Product)')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Alpha Bar')
    ax1.grid(True, alpha=0.3)
    
    # Plot betas
    ax2.plot(timesteps, scheduler.betas)
    ax2.set_title('Beta Values')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Beta')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, 'schedule_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Schedule curves saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Testing Cosine Noise Scheduler...")
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Create scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=1000)
    
    # Load test image
    print("Loading test image...")
    test_image = load_test_image(size=128)
    
    # Save the original test image for reference
    original_img_np = torch.clamp(test_image[0], 0, 1).permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(original_img_np)
    plt.title('Original Test Image')
    plt.axis('off')
    original_path = os.path.join(output_dir, 'original_image.png')
    plt.savefig(original_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Original image saved to: {original_path}")
    
    # Test different timesteps
    timesteps_to_show = [0, 100, 250, 500, 750, 999]
    
    print("Creating noise progression visualization...")
    visualize_noise_progression(scheduler, test_image, timesteps_to_show, output_dir)
    
    print("Creating schedule curves...")
    plot_schedule_curves(scheduler, output_dir)
    
    print("Test complete! Check the output directory for saved images.")