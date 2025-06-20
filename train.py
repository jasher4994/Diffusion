import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
from collections import deque
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import glob

from conditional_diffusion.conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.conditional_diffusion_model.unet import UNet
from conditional_diffusion.conditional_diffusion_model.trainer import DiffusionTrainer
from conditional_diffusion.conditional_diffusion_model.sampler import DDPMSampler
import config

def tensor_to_image(tensor):
    """Convert tensor back to PIL Image for visualization."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.detach().cpu()
    tensor = (tensor.clamp(-1, 1) + 1) * 0.5
    
    import torchvision.transforms.functional as TF
    return TF.to_pil_image(tensor)

class CIFAR128Dataset(Dataset):
    """Dataset for 128x128 CIFAR-10 images organized in class folders."""
    
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        
        # Use train or test split
        split_dir = "train" if train else "test"
        self.root_dir = os.path.join(data_dir, split_dir)
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.images = []
        self.labels = []
        
        print(f"🔍 Loading CIFAR-10 128x128 dataset from {self.root_dir}")
        
        # Load images from each class folder
        for class_idx in range(10):
            class_folder = os.path.join(self.root_dir, f"class{class_idx}")
            
            if not os.path.exists(class_folder):
                print(f"⚠️  Warning: {class_folder} does not exist")
                continue
            
            # Find all image files in the class folder
            image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            class_images = []
            
            for pattern in image_patterns:
                class_images.extend(glob.glob(os.path.join(class_folder, pattern)))
            
            print(f"   • {self.class_names[class_idx]} (class{class_idx}): {len(class_images)} images")
            
            # Add to dataset
            for img_path in class_images:
                self.images.append(img_path)
                self.labels.append(class_idx)
        
        print(f"✅ Loaded {len(self.images)} total images")
        
        # Print class distribution
        class_counts = [0] * 10
        for label in self.labels:
            class_counts[label] += 1
        
        print("📊 Class distribution:")
        for i, count in enumerate(class_counts):
            print(f"   • {self.class_names[i]}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SingleSampleDataset(Dataset):
    """Dataset containing only one sample from each CIFAR-10 class."""
    
    def __init__(self, data_dir, samples_per_class=1, transform=None, train=True):
        self.samples_per_class = samples_per_class
        self.transform = transform
        
        # Use train or test split
        split_dir = "train" if train else "test"
        self.root_dir = os.path.join(data_dir, split_dir)
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.samples = []
        self.labels = []
        
        print(f"🔍 Extracting {samples_per_class} sample(s) per class from {self.root_dir}")
        
        for class_idx in range(10):
            class_folder = os.path.join(self.root_dir, f"class{class_idx}")
            
            if not os.path.exists(class_folder):
                print(f"⚠️  Warning: {class_folder} does not exist")
                continue
            
            # Find all image files in the class folder
            image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
            class_images = []
            
            for pattern in image_patterns:
                class_images.extend(glob.glob(os.path.join(class_folder, pattern)))
            
            # Take only the first N samples
            selected_images = class_images[:samples_per_class]
            
            print(f"   ✅ {self.class_names[class_idx]}: selected {len(selected_images)} of {len(class_images)} images")
            
            for img_path in selected_images:
                # Load image now to store in memory
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.samples.append(image)
                    self.labels.append(class_idx)
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
        
        print(f"📊 Created dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def save_samples(self, save_dir):
        """Save the original samples for reference."""
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, (image, label) in enumerate(zip(self.samples, self.labels)):
            filename = f"{save_dir}/original_{self.class_names[label]}_class{label}.png"
            image.save(filename)
            print(f"💾 Saved original sample: {filename}")

class TrainingMonitor:
    """Monitor and visualize training progress."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.loss_history = []
        self.epoch_times = []
        self.start_time = time.time()
        
        # Create monitoring directory
        self.monitor_dir = f"{checkpoint_dir}/monitoring"
        os.makedirs(self.monitor_dir, exist_ok=True)
        
    def log_epoch(self, epoch, avg_loss, epoch_time):
        """Log epoch results."""
        self.loss_history.append(avg_loss)
        self.epoch_times.append(epoch_time)
        
        # Print progress
        total_time = time.time() - self.start_time
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   • Average loss: {avg_loss:.6f}")
        print(f"   • Epoch time: {epoch_time:.1f}s")
        print(f"   • Total time: {total_time/60:.1f}min")
        
        if len(self.loss_history) > 1:
            loss_change = avg_loss - self.loss_history[-2]
            trend = "📈" if loss_change > 0 else "📉"
            print(f"   • Loss change: {trend} {loss_change:+.6f}")
        
        # Save loss plot every epoch
        self.plot_loss_curve(epoch + 1)
        
    def plot_loss_curve(self, current_epoch):
        """Plot and save loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title(f'Training Loss Progress (Epoch {current_epoch})')
        plt.grid(True, alpha=0.3)
        
        # Add trend line for last 10 epochs if we have enough data
        if len(self.loss_history) >= 10:
            recent_epochs = list(range(len(self.loss_history)-9, len(self.loss_history)+1))
            recent_losses = self.loss_history[-10:]
            z = np.polyfit(recent_epochs, recent_losses, 1)
            p = np.poly1d(z)
            plt.plot(recent_epochs, p(recent_epochs), "r--", alpha=0.8, label='Trend (last 10 epochs)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.monitor_dir}/loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def estimate_completion(self, current_epoch, total_epochs):
        """Estimate training completion time."""
        if len(self.epoch_times) > 0:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = total_epochs - current_epoch - 1
            estimated_time = remaining_epochs * avg_epoch_time
            
            hours = int(estimated_time // 3600)
            minutes = int((estimated_time % 3600) // 60)
            
            print(f"⏱️  Estimated completion: {hours}h {minutes}m")

def generate_sample_during_training(unet, noise_scheduler, device, epoch, monitor_dir, test_classes=None, image_size=128):
    """Generate sample images during training to track progress."""
    
    print("🎨 Generating training samples...")
    
    # Create a simple sampler for quick generation
    sampler = DDPMSampler(noise_scheduler, unet, device=device)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Use provided test classes or default ones
    if test_classes is None:
        test_classes = [0, 2, 5, 7]  # airplane, bird, dog, horse
    
    # Generate samples with fewer inference steps for speed
    with torch.no_grad():
        for i, class_idx in enumerate(test_classes):
            class_name = class_names[class_idx]
            filename = f'{monitor_dir}/epoch_{epoch:03d}_sample_{i+1}_{class_name}.png'
            
            try:
                # Quick generation (fewer steps)
                generated_image = sampler.sample_class_conditional(
                    class_idx=class_idx,
                    num_samples=1,
                    guidance_scale=3.0  # Lower guidance for training samples
                )
                
                if not torch.isnan(generated_image).any():
                    # Convert and save
                    img = tensor_to_image(generated_image[0])
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.title(f"Epoch {epoch+1}: {class_name}")
                    plt.axis('off')
                    plt.savefig(filename, dpi=100, bbox_inches='tight')
                    plt.close()
                else:
                    print(f"⚠️  NaN detected in generated image for {class_name}")
                    
            except Exception as e:
                print(f"⚠️  Error generating sample for {class_name}: {e}")

def create_cifar128_dataloader(data_dir, batch_size=32, train=True, num_workers=4, image_size=128):
    """Create a dataloader for 128x128 CIFAR-10 dataset."""
    
    # Transform for 128x128 images
    if image_size != 128:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
    
    # Create dataset
    dataset = CIFAR128Dataset(
        data_dir=data_dir,
        transform=transform,
        train=train
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader

def create_overfit_dataloader(data_dir, samples_per_class=1, batch_size=10, num_workers=2, image_size=128):
    """Create a dataloader for overfitting with limited samples."""
    
    # Transform for the specified image size
    if image_size != 128:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
    
    # Create single sample dataset
    dataset = SingleSampleDataset(
        data_dir=data_dir,
        samples_per_class=samples_per_class,
        transform=transform,
        train=True
    )
    
    # Save original samples for reference
    dataset.save_samples("./checkpoints/original_samples")
    
    # Create dataloader - use smaller batch size for overfitting
    dataloader = DataLoader(
        dataset, 
        batch_size=min(batch_size, len(dataset)), 
        shuffle=True, 
        num_workers=num_workers
    )
    
    return dataloader

def train_diffusion_model(overfit_mode=False, samples_per_class=1, image_size=128):
    """Train class-conditional diffusion model with optional overfitting mode."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    print(f"💾 CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if device == 'cuda' else "")
    
    if overfit_mode:
        print(f"\n🎯 OVERFITTING MODE: Training on {samples_per_class} sample(s) per class")
        print("This is perfect for testing if your model architecture works!")
    
    print("\n🔧 Configuration:")
    print(f"  • Mode: {'Overfit' if overfit_mode else 'Full'} training")
    print(f"  • Image size: {image_size}x{image_size}")
    print(f"  • Epochs: {config.NUM_EPOCHS}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Classes: 10 (CIFAR-10)")
    print(f"  • Timesteps: {config.TIMESTEPS}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    print(f"  • Classifier-free guidance: {config.USE_CLASSIFIER_FREE}")
    
    # Initialize components
    print("\n🏗️  Initializing model components...")
    noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
    
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config.TIME_EMB_DIM,
        num_classes=10,
        class_emb_dim=config.CLASS_EMB_DIM,
        base_channels=config.BASE_CHANNELS
    )
    
    trainer = DiffusionTrainer(
        unet=unet,
        noise_scheduler=noise_scheduler,
        device=device,
        lr=config.LEARNING_RATE,
        num_classes=10
    )
    
    # Print model info
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Set data directory
    data_dir = "/home/azureuser/Diffusion/data/cifar10-128x128"
    
    # Load dataset
    print(f"\n📁 Loading {'overfit' if overfit_mode else 'full'} dataset...")
    
    if overfit_mode:
        dataloader = create_overfit_dataloader(
            data_dir=data_dir,
            samples_per_class=samples_per_class,
            batch_size=config.BATCH_SIZE,
            image_size=image_size
        )
        # For overfitting, test all classes we have samples for
        test_classes = list(range(10))  # All 10 classes
    else:
        dataloader = create_cifar128_dataloader(
            data_dir=data_dir,
            batch_size=config.BATCH_SIZE,
            train=True,
            image_size=image_size
        )
        test_classes = [0, 2, 5, 7]  # Subset for full training
    
    print(f"✅ Dataset loaded: {len(dataloader.dataset):,} samples")
    print(f"📊 Batches per epoch: {len(dataloader)}")
    
    # Create directories
    checkpoint_suffix = "_overfit" if overfit_mode else ""
    checkpoint_dir = f"{config.CHECKPOINT_DIR}{checkpoint_suffix}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize monitoring
    monitor = TrainingMonitor(checkpoint_dir)
    
    print(f"\n🏃 Starting class-conditional training...")
    print(f"📁 Monitoring data saved to: {monitor.monitor_dir}/")
    print(f"💾 Checkpoints saved to: {checkpoint_dir}/")
    
    if overfit_mode:
        print(f"\n💡 Overfitting Tips:")
        print(f"   • Loss should decrease rapidly and reach very low values")
        print(f"   • Generated images should closely match the original samples")
        print(f"   • This verifies your model architecture works correctly")
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        epoch_losses = []
        batch_times = deque(maxlen=10)
        
        # Set model to training mode
        unet.train()
        
        # For overfitting, repeat the small dataset multiple times per epoch
        if overfit_mode:
            # Repeat data multiple times to simulate more training steps
            repeats = max(1, 100 // len(dataloader))  # At least 100 steps per epoch
            print(f"🔄 Repeating dataset {repeats} times per epoch for overfitting")
            
            pbar = tqdm(range(repeats * len(dataloader)), desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
            
            for step in pbar:
                # Cycle through the data
                batch_idx = step % len(dataloader)
                if batch_idx == 0:
                    data_iter = iter(dataloader)
                
                images, class_labels = next(data_iter)
                batch_start_time = time.time()
                
                try:
                    # Training step
                    if config.USE_CLASSIFIER_FREE:
                        loss = trainer.train_step_classifier_free(images, class_labels)
                    else:
                        loss = trainer.train_step(images, class_labels)
                    
                    epoch_losses.append(loss)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    
                    # Update progress bar
                    avg_loss_so_far = sum(epoch_losses) / len(epoch_losses)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{avg_loss_so_far:.4f}'
                    })
                    
                except Exception as e:
                    print(f"❌ Error in step {step}: {e}")
                    continue
        else:
            # Normal training loop
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
            
            for batch_idx, (images, class_labels) in enumerate(pbar):
                batch_start_time = time.time()
                
                try:
                    # Training step
                    if config.USE_CLASSIFIER_FREE:
                        loss = trainer.train_step_classifier_free(images, class_labels)
                    else:
                        loss = trainer.train_step(images, class_labels)
                    
                    epoch_losses.append(loss)
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    
                    # Update progress bar
                    avg_loss_so_far = sum(epoch_losses) / len(epoch_losses)
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'avg_loss': f'{avg_loss_so_far:.4f}',
                        'batch_time': f'{batch_time:.1f}s'
                    })
                    
                    # Detailed logging every N batches
                    if batch_idx % config.LOG_EVERY == 0:
                        print(f"\n  📦 Batch {batch_idx + 1}/{len(dataloader)}")
                        print(f"     • Current loss: {loss:.6f}")
                        print(f"     • Average loss: {avg_loss_so_far:.6f}")
                        print(f"     • Batch time: {batch_time:.1f}s")
                        
                        # Memory usage if CUDA
                        if device == 'cuda':
                            memory_used = torch.cuda.memory_allocated() / 1e9
                            memory_cached = torch.cuda.memory_reserved() / 1e9
                            print(f"     • GPU memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
                    
                except Exception as e:
                    print(f"❌ Error in batch {batch_idx}: {e}")
                    continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        
        # Log epoch results
        monitor.log_epoch(epoch, avg_loss, epoch_time)
        if not overfit_mode:  # Skip time estimation for overfitting
            monitor.estimate_completion(epoch, config.NUM_EPOCHS)
        
        # Generate sample images every few epochs (or every epoch for overfitting)
        sample_frequency = 1 if overfit_mode else max(1, config.CHECKPOINT_EVERY // 2)
        if (epoch + 1) % sample_frequency == 0:
            try:
                generate_sample_during_training(unet, noise_scheduler, device, 
                                              epoch, monitor.monitor_dir, test_classes, image_size)
            except Exception as e:
                print(f"⚠️  Error generating samples: {e}")
        
        # Save checkpoint
        checkpoint_frequency = max(1, config.CHECKPOINT_EVERY // 2) if overfit_mode else config.CHECKPOINT_EVERY
        if (epoch + 1) % checkpoint_frequency == 0 or (epoch + 1) == config.NUM_EPOCHS:
            try:
                checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pt"
                trainer.save_checkpoint(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Error saving checkpoint: {e}")
        
        # Clear cache periodically
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Early stopping suggestion for overfitting
        if overfit_mode and avg_loss < 0.01:
            print(f"\n🎉 Great! Loss is very low ({avg_loss:.6f}). Your model is working!")
            print("Consider stopping here and checking the generated samples.")
    
    # Save final model
    try:
        final_path = f"{checkpoint_dir}/model_final.pt"
        trainer.save_checkpoint(final_path)
        print(f"💾 Final model saved: {final_path}")
    except Exception as e:
        print(f"⚠️  Error saving final model: {e}")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final average loss: {monitor.loss_history[-1]:.6f}")
    print(f"📈 Loss curve saved to: {monitor.monitor_dir}/loss_curve.png")
    print(f"🖼️  Sample images saved to: {monitor.monitor_dir}/")
    
    if overfit_mode:
        print(f"\n💡 Next steps:")
        print(f"   • Check generated samples - they should look like the originals")
        print(f"   • If loss decreased rapidly and samples look good, your model works!")
        print(f"   • Now you can run full training: python train.py --full")

def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--overfit', action='store_true', 
                       help='Train on single sample per class to test model')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of samples per class for overfitting (default: 1)')
    parser.add_argument('--full', action='store_true',
                       help='Run full training on entire dataset')
    parser.add_argument('--size', type=int, default=128,
                       help='Image size for training (default: 128)')
    
    args = parser.parse_args()
    
    try:
        if args.overfit:
            train_diffusion_model(overfit_mode=True, samples_per_class=args.samples, image_size=args.size)
        else:
            train_diffusion_model(overfit_mode=False, image_size=args.size)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()