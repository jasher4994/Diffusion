import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

import config
from model import SimpleUNet
from data import QuickDrawDataset
from scheduler import SimpleDDPMScheduler

def train_quickdraw(epochs=None, save_every=1000, device='cuda'):
    """Train the Quick Draw diffusion model."""
    if epochs is None:
        epochs = config.NUM_EPOCHS

    print(f"🚀 Starting Quick Draw diffusion training for {epochs} epochs")

    # Create dataset and dataloader
    print("📂 Loading dataset...")
    dataset = QuickDrawDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Create model
    print("🏗️ Creating model...")
    model = SimpleUNet(dataset.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)
    criterion = nn.MSELoss()

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    model.train()
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Sample random timesteps
            timesteps = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=device)

            # Add noise to images
            noise = torch.randn_like(images)
            noisy_images = scheduler.q_sample(images, timesteps, noise)

            # Random unconditional training (10% of the time)
            if torch.rand(1) < 0.1:
                labels = torch.full_like(labels, -1)  # Use -1 for unconditional

            # Predict noise
            predicted_noise = model(noisy_images, timesteps, labels)

            # Compute loss
            loss = criterion(predicted_noise, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Save checkpoint
            if step % save_every == 0:
                checkpoint_path = f"checkpoints/quickdraw_step_{step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'loss': loss.item(),
                    'num_classes': dataset.num_classes
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Save final model
    final_path = f"checkpoints/quickdraw_final_epoch_{epochs}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epochs,
        'loss': avg_loss,
        'num_classes': dataset.num_classes
    }, final_path)
    print(f"✅ Training complete! Final model saved: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Quick Draw Diffusion Model')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs to train (default: {config.NUM_EPOCHS})')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N steps (default: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'

    train_quickdraw(args.epochs, args.save_every, args.device)

if __name__ == "__main__":
    main()