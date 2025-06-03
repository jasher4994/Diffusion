"""
Visualization utilities for diffusion model training and inference.

This module provides functions to visualize tensors, diffusion processes,
and save generated images for analysis and debugging.
"""

import os
from typing import List, Union, Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def show_tensor_image(image: torch.Tensor) -> None:
    """
    Convert a tensor image with values in [-1, 1] to a PIL image and display it.
    
    This function handles the conversion from PyTorch tensor format (CHW) with
    values in [-1, 1] range to a displayable PIL image format (HWC) with values
    in [0, 255] range.
    
    Args:
        image (torch.Tensor): Input tensor image with shape (C, H, W) or (B, C, H, W).
                             Values should be in range [-1, 1].
                             If batch dimension is present, only the first image is displayed.
    
    Note:
        This function displays the image using matplotlib but does not save it.
        The image is shown with axes turned off for cleaner visualization.
    
    Example:
        >>> tensor_img = torch.randn(3, 128, 128)  # Random tensor image
        >>> show_tensor_image(tensor_img)  # Displays the image
    """
    if len(image.shape) == 4: # Will be 3 if batch size is 1
        image = image[0]  # Take first image of batch
        
    # Convert tensor to displayable format
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # Scale from [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # Change from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # Scale from [0, 1] to [0, 255]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # Convert to uint8
        transforms.ToPILImage(),  # Convert to PIL Image
    ])
    
    plt.imshow(reverse_transforms(image))
    plt.axis('off')


def plot_diffusion_process(
    diffusion_model,
    original_image: torch.Tensor,
    timesteps: List[int] = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
) -> None:
    """
    Visualize the forward diffusion process on a given image at specific timesteps.
    
    This function shows how an image progressively becomes more noisy as it goes
    through the forward diffusion process. This is useful for understanding how
    the diffusion model adds noise during training.
    
    Args:
        diffusion_model: The diffusion model instance with q_sample method
        original_image (torch.Tensor): Original clean image tensor with shape (C, H, W)
                                      or (1, C, H, W). Values should be in [-1, 1].
        timesteps (List[int], optional): List of timestep values to visualize.
                                        Defaults to [0, 100, 200, 300, 400, 500, 600, 700, 800, 900].
    
    Note:
        The function creates a horizontal subplot showing the image at each timestep.
        Lower timesteps show less noise, higher timesteps show more noise.
        
    Example:
        >>> from models.diffusion import DiffusionModel
        >>> diffusion = DiffusionModel(timesteps=1000)
        >>> img = torch.randn(3, 128, 128)
        >>> plot_diffusion_process(diffusion, img, timesteps=[0, 250, 500, 750, 999])
    """
    fig, axs = plt.subplots(1, len(timesteps), figsize=(15, 3))
    
    # Ensure original_image has batch dimension and is on CPU for plotting
    if len(original_image.shape) == 3:
        original_image = original_image.unsqueeze(0)  # Add batch dimension
    original_image_cpu = original_image.cpu()
    
    for i, t in enumerate(timesteps):
        # Convert scalar timestep to tensor of appropriate shape
        t_tensor = torch.tensor([t], dtype=torch.long)
        
        # Apply forward diffusion to add noise
        noised_image, _ = diffusion_model.q_sample(original_image_cpu, t_tensor)
        
        # Plot the noised image
        axs[i].imshow(tensor_to_pil(noised_image[0]))
        axs[i].set_title(f"t={t}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor image to PIL Image format.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W) and values in [-1, 1]
    
    Returns:
        Image.Image: PIL Image ready for display or saving
    """
    # Scale from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and change format
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def save_generated_images(
    images: torch.Tensor,
    output_dir: str,
    base_filename: str = "generated",
    start_idx: int = 0
) -> List[str]:
    """
    Save a batch of generated images to disk.
    
    This function takes a batch of tensor images and saves them as individual
    PNG files with sequential numbering. Useful for saving model outputs during
    training or inference.
    
    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W) and values in [-1, 1]
        output_dir (str): Directory path where images will be saved
        base_filename (str, optional): Base name for saved files. Defaults to "generated".
        start_idx (int, optional): Starting index for file numbering. Defaults to 0.
    
    Returns:
        List[str]: List of saved file paths
        
    Note:
        Files are saved with format: "{base_filename}_{index:04d}.png"
        The output directory is created if it doesn't exist.
        
    Example:
        >>> generated_imgs = torch.randn(4, 3, 128, 128)  # 4 random images
        >>> saved_paths = save_generated_images(
        ...     generated_imgs, 
        ...     "outputs/samples", 
        ...     "sample", 
        ...     start_idx=10
        ... )
        >>> # Saves: sample_0010.png, sample_0011.png, sample_0012.png, sample_0013.png
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1] range
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    saved_paths: List[str] = []
    
    for i, img in enumerate(images):
        # Convert tensor to numpy array
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL and save
        img_pil = Image.fromarray(img_np)
        filename = f"{base_filename}_{start_idx + i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        img_pil.save(filepath)
        saved_paths.append(filepath)
    
    return saved_paths


def create_image_grid(
    images: torch.Tensor,
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Create a grid of images from a batch tensor.
    
    Args:
        images (torch.Tensor): Batch of images with shape (B, C, H, W)
        nrow (int, optional): Number of images per row. Defaults to 4.
        padding (int, optional): Padding between images. Defaults to 2.
        normalize (bool, optional): Whether to normalize values to [0, 1]. Defaults to True.
        value_range (Optional[Tuple[float, float]], optional): Input value range for normalization.
                                                              Defaults to None (auto-detect).
    
    Returns:
        torch.Tensor: Grid image tensor with shape (C, H_grid, W_grid)
        
    Example:
        >>> batch_imgs = torch.randn(16, 3, 64, 64)  # 16 images
        >>> grid = create_image_grid(batch_imgs, nrow=4)  # 4x4 grid
        >>> show_tensor_image(grid)
    """
    from torchvision.utils import make_grid
    
    if normalize and value_range is None:
        # Auto-detect value range
        if images.min() >= -1 and images.max() <= 1:
            value_range = (-1, 1)
        else:
            value_range = (images.min().item(), images.max().item())
    
    return make_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range
    )


def plot_training_progress(
    losses: List[float],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss progression over epochs.
    
    Args:
        losses (List[float]): List of loss values
        epochs (Optional[List[int]], optional): List of epoch numbers. 
                                              If None, uses sequential numbering.
        save_path (Optional[str], optional): Path to save the plot. If None, displays only.
        
    Example:
        >>> losses = [1.5, 1.2, 0.9, 0.7, 0.5]
        >>> plot_training_progress(losses, save_path="training_curve.png")
    """
    if epochs is None:
        epochs = list(range(1, len(losses) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    plt.show()