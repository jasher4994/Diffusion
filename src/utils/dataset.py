import os
import json
from typing import Tuple, List, Union, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Dataset class for loading images for unconditional diffusion training.
    
    This dataset loads images from a folder and applies standard preprocessing
    transformations including resizing, random horizontal flipping, and scaling
    to the [-1, 1] range expected by diffusion models.
    
    Args:
        folder_path (str): Path to directory containing image files
        img_size (int, optional): Target size for resizing images. Defaults to 128.
        
    Attributes:
        folder_path (str): Path to the image directory
        image_files (List[str]): List of valid image filenames
        transform (transforms.Compose): Image preprocessing pipeline
    """
    
    def __init__(self, folder_path: str, img_size: int = 128) -> None:
        """Initialize the ImageDataset. Extracts images on initialization."""
        self.folder_path: str = folder_path
        self.image_files: List[str] = [
            f for f in os.listdir(folder_path) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])
        
    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess an image by index.
        
        Args:
            idx (int): Index of the image to load
            
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (3, img_size, img_size)
                         and values in range [-1, 1]
        """
        img_path: str = os.path.join(self.folder_path, self.image_files[idx])
        image: Image.Image = Image.open(img_path).convert("RGB")
        return self.transform(image)


class TextImageDataset(Dataset):
    """
    Dataset for loading image-text pairs for text-conditional diffusion training.
    
    This dataset loads images and their corresponding text captions from a JSON file.
    Only images that have corresponding captions are included in the dataset.
    
    Args:
        folder_path (str): Path to directory containing image files
        captions_file (str): Path to JSON file mapping image filenames to captions
        img_size (int, optional): Target size for resizing images. Defaults to 128.
        
    Attributes:
        folder_path (str): Path to the image directory
        captions (dict): Dictionary mapping image filenames to caption strings
        image_files (List[str]): List of image filenames that have captions
        transform (transforms.Compose): Image preprocessing pipeline
        
    Note:
        The captions_file should be a JSON file with format:
        {"image_filename.jpg": "caption text", "another_image.png": "another caption", ...}
    """
    
    def __init__(self, folder_path: str, captions_file: str, img_size: int = 128) -> None:
        """Initialize the TextImageDataset."""
        self.folder_path: str = folder_path
        
        # Load captions from JSON file
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions: dict[str, str] = json.load(f)
            
        # Filter to keep only images that have captions and are valid image files
        self.image_files: List[str] = [
            f for f in os.listdir(folder_path) 
            if f in self.captions and f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])
        
    def __len__(self) -> int:
        """Return the total number of image-caption pairs in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load and preprocess an image-caption pair by index.
        
        Args:
            idx (int): Index of the image-caption pair to load
            
        Returns:
            Tuple[torch.Tensor, str]: A tuple containing:
                - Preprocessed image tensor with shape (3, img_size, img_size) 
                  and values in range [-1, 1]
                - Caption string describing the image
        """
        img_name: str = self.image_files[idx]
        img_path: str = os.path.join(self.folder_path, img_name)
        image: Image.Image = Image.open(img_path).convert("RGB")
        caption: str = self.captions[img_name]
        
        return self.transform(image), caption


def get_dataloader(
    dataset: Union[ImageDataset, TextImageDataset], 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4,
    pin_memory: Optional[bool] = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the given dataset.
    
    This is a convenience function that creates a DataLoader with commonly used
    settings for diffusion model training.
    
    Args:
        dataset (Union[ImageDataset, TextImageDataset]): The dataset to load from
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        pin_memory (Optional[bool], optional): Whether to pin memory for faster GPU transfer.
                                             Defaults to True if CUDA is available, False otherwise.
    
    Returns:
        DataLoader: Configured PyTorch DataLoader ready for training

    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )