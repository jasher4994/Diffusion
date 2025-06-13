import torch
import os
import matplotlib.pyplot as plt
from conditional_diffusion.conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.conditional_diffusion_model.unet import UNet
from conditional_diffusion.conditional_diffusion_model.trainer import DiffusionTrainer
from conditional_diffusion.conditional_diffusion_model.text_encoder import TextEncoder
from conditional_diffusion.conditional_diffusion_model.data_loader import create_dataloader
from conditional_diffusion.conditional_diffusion_model.sampler import DDPMSampler
import config
import time
from collections import deque

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

def generate_sample_during_training(unet, noise_scheduler, text_encoder, device, epoch, monitor_dir):
    """Generate sample images during training to track progress with enhanced quality."""
    
    print("🎨 Generating training samples...")
    
    # Create enhanced sampler
    sampler = DDPMSampler(noise_scheduler, unet, device=device)
    
    # Quick test prompts from config, fallback to defaults if not set
    test_prompts = getattr(config, "TEST_PROMPTS", [None, "A dog", "A truck"])
    
    # Generate samples with fewer inference steps for speed but better quality
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            if prompt:
                text_embeddings = text_encoder([prompt])
                filename = f'{monitor_dir}/epoch_{epoch:03d}_sample_{i+1}_{prompt.replace(" ", "_")}.png'
                guidance_scale = 1.5  # Light guidance for better conditioning
            else:
                text_embeddings = None
                filename = f'{monitor_dir}/epoch_{epoch:03d}_sample_{i+1}_unconditional.png'
                guidance_scale = 1.0  # No guidance for unconditional
            
            # Enhanced generation with better quality controls
            generated_image = sampler.sample(
                batch_size=1,
                image_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                num_inference_steps=15,  # Slightly more steps for better quality
                text_embeddings=text_embeddings,
                guidance_scale=guidance_scale,
                dynamic_thresholding=True
            )
            
            if not torch.isnan(generated_image).any():
                # Convert and save with improved processing
                img = (generated_image[0] + 1) / 2
                img = torch.clamp(img, 0, 1)
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.title(f"Epoch {epoch+1}: {prompt or 'Unconditional'}")
                plt.axis('off')
                plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
            else:
                print(f"⚠️  NaN detected in generated image for prompt: {prompt}")
    
    print("✅ Sample generation completed")

def train_diffusion_model():
    """Train conditional diffusion model with enhanced monitoring."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    print("🔧 Configuration:")
    print(f"  • Epochs: {config.NUM_EPOCHS}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  • Max samples: {config.MAX_SAMPLES or 'All'}")
    print(f"  • Timesteps: {config.TIMESTEPS}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    
    # Initialize components
    noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=config.BASE_CHANNELS,
        time_emb_dim=config.TIME_EMB_DIM,
        text_emb_dim=config.TEXT_EMB_DIM
    )
    text_encoder = TextEncoder(device=device)
    trainer = DiffusionTrainer(unet, noise_scheduler, device=device, lr=config.LEARNING_RATE)
    
    # Load dataset with enhanced augmentation
    dataloader = create_dataloader(
        captions_file=config.CAPTIONS_FILE,
        images_dir=config.IMAGES_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        max_samples=config.MAX_SAMPLES,
        augment_data=getattr(config, 'USE_DATA_AUGMENTATION', True)
    )
    
    print(f"✅ Dataset loaded: {len(dataloader.dataset)} samples")
    print(f"📊 Batches per epoch: {len(dataloader)}")
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize monitoring
    monitor = TrainingMonitor(config.CHECKPOINT_DIR)
    
    print(f"\n🏃 Starting training with enhanced monitoring...")
    print(f"📁 Monitoring data saved to: {monitor.monitor_dir}/")
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        epoch_losses = []
        batch_times = deque(maxlen=10)  # Track recent batch times
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Training step
            with torch.no_grad():
                text_embeddings = text_encoder(captions)
            
            batch_size_actual = images.shape[0]
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            timesteps_batch = torch.randint(0, noise_scheduler.num_timesteps, 
                                          (batch_size_actual,), device=device)
            
            noisy_images, noise = noise_scheduler.add_noise(images, timesteps_batch)
            
            # Use mixed precision if available
            if trainer.use_amp:
                with torch.cuda.amp.autocast():
                    predicted_noise = unet(noisy_images, timesteps_batch, text_embeddings)
                    loss = trainer.criterion(predicted_noise, noise)
            else:
                predicted_noise = unet(noisy_images, timesteps_batch, text_embeddings)
                loss = trainer.criterion(predicted_noise, noise)
            
            # Improved backpropagation with gradient clipping
            trainer.optimizer.zero_grad()
            
            if trainer.use_amp:
                trainer.scaler.scale(loss).backward()
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), trainer.grad_clip_norm)
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), trainer.grad_clip_norm)
                trainer.optimizer.step()
            
            epoch_losses.append(loss.item())
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Enhanced logging
            if batch_idx % config.LOG_EVERY == 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(dataloader) - batch_idx - 1
                eta_epoch = remaining_batches * avg_batch_time
                
                print(f"  📦 Batch {batch_idx + 1}/{len(dataloader)}")
                print(f"     • Loss: {loss.item():.6f}")
                print(f"     • Batch time: {batch_time:.1f}s")
                print(f"     • ETA this epoch: {eta_epoch/60:.1f}min")
                print(f"     • Caption: {captions[0][:50]}...")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Log epoch results
        monitor.log_epoch(epoch, avg_loss, epoch_time)
        monitor.estimate_completion(epoch, config.NUM_EPOCHS)
        
        # Step learning rate scheduler
        trainer.step_scheduler()
        current_lr = trainer.get_current_lr()
        print(f"📈 Learning rate: {current_lr:.2e}")
        
        # Generate sample images every few epochs
        if (epoch + 1) % max(1, config.CHECKPOINT_EVERY // 2) == 0:
            generate_sample_during_training(unet, noise_scheduler, text_encoder, 
                                          device, epoch, monitor.monitor_dir)
        
        # Save checkpoint
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            checkpoint_path = f"{config.CHECKPOINT_DIR}/model_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = f"{config.CHECKPOINT_DIR}/model_final.pt"
    trainer.save_checkpoint(final_path)
    print(f"💾 Final model saved: {final_path}")
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Final average loss: {monitor.loss_history[-1]:.6f}")
    print(f"📈 Loss curve saved to: {monitor.monitor_dir}/loss_curve.png")

if __name__ == "__main__":
    import numpy as np  # For trend line
    train_diffusion_model()