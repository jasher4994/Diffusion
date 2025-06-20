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
    Training loop for the class-conditional diffusion model.
    
    This class handles:
    - Loading and preprocessing CIFAR-10 data with class labels
    - Training the UNet to predict noise conditioned on class
    - Saving/loading model checkpoints
    """
    
    def __init__(self, unet, noise_scheduler, device='cuda', lr=config.LEARNING_RATE, num_classes=10):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.lr = lr
        self.num_classes = num_classes
        
        # Move model to device
        self.unet.to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=self.lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # CIFAR-10 class names for reference
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def train_step(self, clean_images, class_labels):
        """
        Single training step with class conditioning.
        
        Args:
            clean_images: Batch of clean images, shape (batch, 3, H, W)
            class_labels: Batch of class labels, shape (batch,)
            
        Returns:
            loss: Training loss value
        """
        batch_size = clean_images.shape[0]
        clean_images = clean_images.to(self.device)
        class_labels = class_labels.to(self.device)
        
        # Step 1: Sample random timesteps for each image in the batch
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, 
                                (batch_size,), device=self.device)
        
        # Step 2: Add noise to images using noise scheduler
        noisy_images, noise = self.noise_scheduler.add_noise(clean_images, timesteps)
        
        # Step 3: Predict noise using UNet with class conditioning
        predicted_noise = self.unet(noisy_images, timesteps, class_labels)
        
        # Step 4: Compute loss (predicted vs actual noise)
        loss = self.criterion(predicted_noise, noise)
        
        # Step 5: Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_classifier_free(self, clean_images, class_labels, unconditional_prob=0.1):
        """
        Training step with classifier-free guidance preparation.
        Randomly drops class conditioning to enable unconditional generation.
        
        Args:
            clean_images: Batch of clean images
            class_labels: Batch of class labels
            unconditional_prob: Probability of dropping class conditioning
        """
        batch_size = clean_images.shape[0]
        clean_images = clean_images.to(self.device)
        class_labels = class_labels.to(self.device)
        
        # Randomly mask some class labels for classifier-free guidance
        mask = torch.rand(batch_size, device=self.device) < unconditional_prob
        class_labels_masked = class_labels.clone()
        class_labels_masked[mask] = -1  # Use -1 to indicate no class conditioning
        
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, 
                                (batch_size,), device=self.device)
        
        noisy_images, noise = self.noise_scheduler.add_noise(clean_images, timesteps)
        predicted_noise = self.unet(noisy_images, timesteps, class_labels_masked)
        
        loss = self.criterion(predicted_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_epochs, save_every=1000, use_classifier_free=True):
        """
        Full training loop with class conditioning.
        
        Args:
            dataloader: PyTorch DataLoader with training images and labels
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N steps
            use_classifier_free: Whether to use classifier-free guidance training
        """
        self.unet.train()
        step = 0
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Progress bar for this epoch
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, (clean_images, class_labels) in enumerate(pbar):
                # Single training step
                if use_classifier_free:
                    loss = self.train_step_classifier_free(clean_images, class_labels)
                else:
                    loss = self.train_step(clean_images, class_labels)
                    
                epoch_losses.append(loss)
                
                # Update progress bar with class distribution info
                unique_classes = torch.unique(class_labels)
                class_info = f"Classes: {unique_classes.tolist()}"
                pbar.set_postfix({'loss': f'{loss:.4f}', 'batch_classes': class_info})
                
                # Save checkpoint
                if step % save_every == 0 and step > 0:
                    self.save_checkpoint(f'checkpoint_step_{step}.pt')
                
                step += 1
            
            # Print epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
            
    def create_cifar10_dataloader(self, batch_size=32, train=True):
        """
        Create CIFAR-10 dataloader with proper transforms.
        
        Args:
            batch_size: Batch size
            train: Whether to load training or test set
            
        Returns:
            DataLoader for CIFAR-10
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Scale to [-1, 1]
        ])
        
        dataset = CIFAR10(root='./data', train=train, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }, filepath)
        print(f'Checkpoint saved: {filepath}')
        
    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
            
        print(f'Checkpoint loaded: {filepath}')