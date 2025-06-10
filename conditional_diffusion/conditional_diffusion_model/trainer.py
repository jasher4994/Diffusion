import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import config

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
        self.lr =lr
        
        # Move model to device
        self.unet.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.unet.parameters(), self.lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_step(self, clean_images):
        """
        Single training step.
        
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
        
        # Step 3: Predict noise using UNet
        predicted_noise = self.unet(noisy_images, timesteps)
        
        # Step 4: Compute loss (predicted vs actual noise)
        loss = self.criterion(predicted_noise, noise)
        
        # Step 5: Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
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
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f'Checkpoint saved: {filepath}')
        
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Checkpoint loaded: {filepath}')