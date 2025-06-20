import torch
import os
import argparse
from tqdm import tqdm

from conditional_diffusion.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.unet import UNet
from conditional_diffusion.trainer import DiffusionTrainer
from data import create_dataloader, print_dataset_info
from generate import generate_training_samples
import config

def train(overfit=False, samples_per_class=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Print dataset information
    print_dataset_info()
    
    # Initialize model
    noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config.TIME_EMB_DIM,
        num_classes=10,
        class_emb_dim=config.CLASS_EMB_DIM,
        base_channels=config.BASE_CHANNELS,
        image_size=config.IMAGE_SIZE
    )
    
    trainer = DiffusionTrainer(unet, noise_scheduler, device, lr=config.LEARNING_RATE)
    dataloader = create_dataloader(overfit=overfit, samples_per_class=samples_per_class)
    
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Training on {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")
    
    if overfit:
        print("🎯 OVERFITTING MODE - Training should reach very low loss quickly")
    
    # Create sample directory
    suffix = "_overfit" if overfit else ""
    sample_dir = f"{config.CHECKPOINT_DIR}{suffix}/samples"
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        for images, labels in pbar:
            loss = trainer.train_step(images, labels)
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.6f}")
        
        # Generate sample images
        sample_frequency = 1 if overfit else max(1, config.CHECKPOINT_EVERY // 2)
        if (epoch + 1) % sample_frequency == 0:
            try:
                generate_training_samples(unet, noise_scheduler, device, epoch, sample_dir)
            except Exception as e:
                print(f"Warning: Could not generate samples: {e}")
        
        # Save checkpoint
        if (epoch + 1) % config.CHECKPOINT_EVERY == 0:
            suffix = "_overfit" if overfit else ""
            checkpoint_dir = f"{config.CHECKPOINT_DIR}{suffix}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    suffix = "_overfit" if overfit else ""
    checkpoint_dir = f"{config.CHECKPOINT_DIR}{suffix}"
    final_path = f"{checkpoint_dir}/model_final.pt"
    trainer.save_checkpoint(final_path)
    print(f"Final model saved: {final_path}")
    print(f"Sample images saved in: {sample_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Conditional Diffusion Model')
    parser.add_argument('--overfit', action='store_true', help='Overfit on few samples to test model')
    parser.add_argument('--samples', type=int, default=1, help='Samples per class for overfitting')
    args = parser.parse_args()
    
    train(args.overfit, args.samples)

if __name__ == "__main__":
    main()