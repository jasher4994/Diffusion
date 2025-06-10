import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, Callable, List
import re

class ImageTextDataset(Dataset):
    """
    Simple, focused dataset loader for image-text pairs.
    
    Expected format: JSON file with {"filename.jpg": "caption text"}
    """
    
    def __init__(
        self,
        captions_file: str,
        images_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        caption_max_length: int = 200
    ):
        """
        Args:
            captions_file: Path to JSON file with {"filename": "caption"} format
            images_dir: Directory containing images
            transform: Image transformations
            max_samples: Limit dataset size for testing
            caption_max_length: Maximum caption length (longer ones get truncated)
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.caption_max_length = caption_max_length
        
        # Load captions
        print(f"Loading captions from {captions_file}...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
        
        # Convert to list and filter existing images
        self.samples = self._prepare_samples(captions_data)
        
        # Limit dataset size if specified
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"✅ Loaded {len(self.samples)} image-caption pairs")
        self._print_stats()
    
    def _prepare_samples(self, captions_data):
        """Prepare and validate samples."""
        samples = []
        missing_count = 0
        
        for filename, caption in captions_data.items():
            image_path = self.images_dir / filename
            
            # Check if image exists
            if not image_path.exists():
                missing_count += 1
                continue
            
            # Clean caption
            caption = self._clean_caption(caption)
            
            # Skip if caption is too short
            if len(caption.strip()) < 5:
                continue
            
            samples.append({
                'filename': filename,
                'caption': caption,
                'path': image_path
            })
        
        if missing_count > 0:
            print(f"⚠️  {missing_count} images not found (skipped)")
        
        return samples
    
    def _clean_caption(self, caption: str) -> str:
        """Clean and standardize captions."""
        # Handle multi-line captions (take first line as main description)
        lines = caption.split('\n')
        main_description = lines[0].strip()
        
        # Remove citation markers [1], [2], etc.
        main_description = re.sub(r'\[\d+\]', '', main_description)
        
        # Remove language references
        if "reads in the original" in main_description.lower():
            parts = main_description.split("The caption reads")
            main_description = parts[0].strip()
        
        # Truncate if too long
        if len(main_description) > self.caption_max_length:
            truncated = main_description[:self.caption_max_length]
            # Try to end at a complete sentence
            last_period = truncated.rfind('.')
            if last_period > 50:  # Only if we have a reasonable amount of text
                main_description = truncated[:last_period + 1]
            else:
                main_description = truncated
        
        return main_description.strip()
    
    def _print_stats(self):
        """Print dataset statistics."""
        if not self.samples:
            return
        
        caption_lengths = [len(sample['caption']) for sample in self.samples]
        avg_length = sum(caption_lengths) / len(caption_lengths)
        
        print(f"📊 Dataset Stats:")
        print(f"   • Total samples: {len(self.samples)}")
        print(f"   • Caption length: {min(caption_lengths)} - {max(caption_lengths)} chars (avg: {avg_length:.1f})")
        
        # Show examples
        print(f"📝 Sample captions:")
        for i in range(min(2, len(self.samples))):
            sample = self.samples[i]
            preview = sample['caption'][:80] + "..." if len(sample['caption']) > 80 else sample['caption']
            print(f"   • {sample['filename']}: {preview}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading {sample['path']}: {e}")
            # Return gray fallback image
            image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['caption']

def create_dataloader(
    captions_file: str,
    images_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    max_samples: Optional[int] = None,
    num_workers: int = 2
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        captions_file: Path to JSON captions file
        images_dir: Path to images directory  
        batch_size: Batch size for training
        image_size: Size to resize images to
        max_samples: Limit dataset size (useful for testing)
        num_workers: Number of worker processes
    
    Returns:
        DataLoader ready for training
    """
    
    # Standard diffusion model transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] range
    ])
    
    # Create dataset
    dataset = ImageTextDataset(
        captions_file=captions_file,
        images_dir=images_dir,
        transform=transform,
        max_samples=max_samples
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Consistent batch sizes
    )
    
    return dataloader