import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import config

class CIFAR128Dataset(Dataset):
    """Dataset for 128x128 CIFAR-10 images in /data/cifar10-128x128/train/class0/ format."""
    
    def __init__(self, data_dir, transform=None, train=True):
        split_dir = "train" if train else "test"
        self.root_dir = os.path.join(data_dir, split_dir)
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        print(f"Loading CIFAR-10 128x128 from {self.root_dir}")
        
        for class_idx in range(10):
            class_folder = os.path.join(self.root_dir, f"class{class_idx}")
            if not os.path.exists(class_folder):
                print(f"Warning: {class_folder} does not exist")
                continue
                
            # Find all image files
            class_images = []
            for pattern in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                class_images.extend(glob.glob(os.path.join(class_folder, pattern)))
            
            print(f"   • class{class_idx}: {len(class_images)} images")
            
            # Add to dataset
            for img_path in class_images:
                self.images.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Total: {len(self.images)} images loaded")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            image = Image.new('RGB', (128, 128), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class OverfitDataset(Dataset):
    """Dataset with only N samples per class for overfitting test."""
    
    def __init__(self, data_dir, samples_per_class=1, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        class_names = config.CLASS_NAMES
        
        print(f"Creating overfit dataset with {samples_per_class} sample(s) per class")
        
        for class_idx in range(10):
            class_folder = os.path.join(data_dir, "train", f"class{class_idx}")
            if not os.path.exists(class_folder):
                continue
                
            class_images = []
            for pattern in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                class_images.extend(glob.glob(os.path.join(class_folder, pattern)))
            
            selected = class_images[:samples_per_class]
            print(f"   • {class_names[class_idx]}: {len(selected)} sample(s)")
            
            for img_path in selected:
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.samples.append(image)
                    self.labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        print(f"Overfit dataset created with {len(self.samples)} total samples")
        
        # Save original samples for reference
        self.save_originals()
    
    def save_originals(self):
        """Save the original samples for reference."""
        save_dir = "./checkpoints_overfit/original_samples"
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, (image, label) in enumerate(zip(self.samples, self.labels)):
            class_name = config.CLASS_NAMES[label]
            filename = f"{save_dir}/original_{class_name}_class{label}.png"
            image.save(filename)
        
        print(f"Original samples saved to: {save_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_transform(image_size=None):
    """Get the appropriate transform for the given image size."""
    if image_size is None:
        image_size = config.IMAGE_SIZE
    
    if image_size != 128:
        # Need to resize
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # No resize needed for 128x128
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def create_dataloader(overfit=False, samples_per_class=1, batch_size=None, num_workers=4, 
                     data_dir=None, image_size=None):
    """Create dataloader using your existing folder structure."""
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if data_dir is None:
        data_dir = config.DATA_DIR
    if image_size is None:
        image_size = config.IMAGE_SIZE
    
    transform = get_transform(image_size)
    
    if overfit:
        dataset = OverfitDataset(data_dir, samples_per_class, transform)
    else:
        dataset = CIFAR128Dataset(data_dir, transform, train=True)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def create_test_dataloader(batch_size=None, num_workers=4, data_dir=None, image_size=None):
    """Create test dataloader."""
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if data_dir is None:
        data_dir = config.DATA_DIR
    if image_size is None:
        image_size = config.IMAGE_SIZE
    
    transform = get_transform(image_size)
    dataset = CIFAR128Dataset(data_dir, transform, train=False)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def get_dataset_info(data_dir=None):
    """Get information about the dataset."""
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    info = {
        'train_samples': 0,
        'test_samples': 0,
        'classes': config.CLASS_NAMES,
        'class_counts_train': [0] * 10,
        'class_counts_test': [0] * 10
    }
    
    for split, counts_key in [('train', 'class_counts_train'), ('test', 'class_counts_test')]:
        split_dir = os.path.join(data_dir, split)
        total_samples = 0
        
        for class_idx in range(10):
            class_folder = os.path.join(split_dir, f"class{class_idx}")
            if not os.path.exists(class_folder):
                continue
                
            class_images = []
            for pattern in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                class_images.extend(glob.glob(os.path.join(class_folder, pattern)))
            
            count = len(class_images)
            info[counts_key][class_idx] = count
            total_samples += count
        
        info[f'{split}_samples'] = total_samples
    
    return info

def print_dataset_info(data_dir=None):
    """Print dataset information."""
    info = get_dataset_info(data_dir)
    
    print("📊 Dataset Information:")
    print(f"   • Train samples: {info['train_samples']:,}")
    print(f"   • Test samples: {info['test_samples']:,}")
    print(f"   • Classes: {len(info['classes'])}")
    
    print("\n📋 Class distribution (train):")
    for i, (class_name, count) in enumerate(zip(info['classes'], info['class_counts_train'])):
        print(f"   • {class_name}: {count:,} images")
    
    if info['test_samples'] > 0:
        print("\n📋 Class distribution (test):")
        for i, (class_name, count) in enumerate(zip(info['classes'], info['class_counts_test'])):
            print(f"   • {class_name}: {count:,} images")

if __name__ == "__main__":
    """Test the data loading functionality."""
    print("Testing data loading...")
    
    # Print dataset info
    print_dataset_info()
    
    # Test regular dataloader
    print("\nTesting regular dataloader...")
    dataloader = create_dataloader(overfit=False)
    print(f"Regular dataloader: {len(dataloader)} batches, {len(dataloader.dataset)} samples")
    
    # Test overfit dataloader
    print("\nTesting overfit dataloader...")
    overfit_dataloader = create_dataloader(overfit=True, samples_per_class=2)
    print(f"Overfit dataloader: {len(overfit_dataloader)} batches, {len(overfit_dataloader.dataset)} samples")
    
    # Test batch loading
    print("\nTesting batch loading...")
    batch = next(iter(dataloader))
    images, labels = batch
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")