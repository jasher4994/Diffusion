import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import config
from torch.cuda.amp import autocast, GradScaler

class DiffusionTrainer:
    """
    Training loop for the diffusion model.
    
    This class handles:
    - Loading and preprocessing data
    - Training the UNet to predict noise
    - Saving/loading model checkpoints
    """
    
    def __init__(self, unet, noise_scheduler, device='cuda', lr=config.LEARNING_RATE):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.lr = lr
        
        # Move model to device
        self.unet.to(device)
        
        # Optimizer with weight decay for better generalization
        self.optimizer = optim.AdamW(
            self.unet.parameters(), 
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period each restart
            eta_min=self.lr * 0.01  # Min LR is 1% of initial
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision training for better performance
        self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        if self.use_amp:
            self.scaler = GradScaler()
            print("🚀 Mixed precision training enabled")
        
        # Training configuration
        self.grad_clip_norm = 1.0  # Gradient clipping for stability
        
    def train_step(self, clean_images):
        """
        Single training step with improved stability and performance.
        
        Args:
            clean_images: Batch of clean images, shape (batch, 3, H, W)
            
        Returns:
            loss: Training loss value
        """
        batch_size = clean_images.shape[0]
        clean_images = clean_images.to(self.device)
        
        # Step 1: Sample random timesteps for each image in the batch
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, 
                                (batch_size,), device=self.device)
        
        # Step 2: Add noise to images using noise scheduler
        noisy_images, noise = self.noise_scheduler.add_noise(clean_images, timesteps)
        
        # Step 3: Forward pass with mixed precision if available
        if self.use_amp:
            with autocast():
                predicted_noise = self.unet(noisy_images, timesteps)
                loss = self.criterion(predicted_noise, noise)
        else:
            predicted_noise = self.unet(noisy_images, timesteps)
            loss = self.criterion(predicted_noise, noise)
        
        # Step 4: Backpropagation with improved stability
        self.optimizer.zero_grad()
        
        if self.use_amp:
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping before step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs, save_every=1000):
        """
        Full training loop.
        
        Args:
            dataloader: PyTorch DataLoader with training images
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N steps
        """
        self.unet.train()
        step = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Progress bar for this epoch
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (clean_images, _) in enumerate(pbar):
                # Single training step
                loss = self.train_step(clean_images)
                epoch_losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                # Save checkpoint
                if step % save_every == 0 and step > 0:
                    self.save_checkpoint(f'checkpoint_step_{step}.pt')
                
                step += 1
            
            # Print epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
            
    def save_checkpoint(self, filepath):
        """Save model checkpoint with scheduler state."""
        checkpoint = {
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
        
    def load_checkpoint(self, filepath):
        """Load model checkpoint with scheduler state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f'Checkpoint loaded: {filepath}')
    
    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()
        
    def get_current_lr(self):
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]