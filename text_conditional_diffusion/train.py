"""Training script for text-conditional diffusion model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

import config
from model import TextConditionedUNet
from data import QuickDrawTextDataset
from scheduler import SimpleDDPMScheduler
from text_encoder import CLIPTextEncoder


def train_text_conditional(epochs=None, save_every_epochs=5, max_samples=None, device='cuda'):
    """Train the text-conditional diffusion model."""
    if epochs is None:
        epochs = config.NUM_EPOCHS

    print(f"🚀 Starting text-conditional diffusion training for {epochs} epochs")

    # Create dataset and dataloader
    print("📂 Loading dataset...")
    dataset = QuickDrawTextDataset(
        split="train",
        max_samples=max_samples,
        num_classes=config.NUM_CLASSES_FILTER
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Create models
    print("🏗️ Creating models...")
    model = TextConditionedUNet(text_dim=config.TEXT_DIM).to(device)
    text_encoder = CLIPTextEncoder(
        model_name=config.CLIP_MODEL,
        freeze=config.FREEZE_CLIP
    ).to(device)

    # Optimizer (only model parameters, CLIP is frozen)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Scheduler and loss
    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)
    criterion = nn.MSELoss()

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 CLIP frozen: {config.FREEZE_CLIP}")

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    model.train()
    text_encoder.eval()  # Keep CLIP in eval mode even if not frozen
    step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, text_prompts) in enumerate(pbar):
            images = images.to(device)

            # Encode text prompts with CLIP
            with torch.no_grad():  # CLIP is frozen
                text_embeddings = text_encoder(text_prompts)

            # Sample random timesteps
            timesteps = torch.randint(0, config.TIMESTEPS, (images.shape[0],), device=device)

            # Add noise to images
            noise = torch.randn_like(images)
            noisy_images = scheduler.q_sample(images, timesteps, noise)

            # Predict noise
            predicted_noise = model(noisy_images, timesteps, text_embeddings)

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

        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every_epochs == 0:
            checkpoint_path = f"checkpoints/text_diffusion_epoch_{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'config': {
                    'text_dim': config.TEXT_DIM,
                    'clip_model': config.CLIP_MODEL,
                }
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = f"checkpoints/text_diffusion_final_epoch_{epochs}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epochs,
        'loss': avg_loss,
        'config': {
            'text_dim': config.TEXT_DIM,
            'clip_model': config.CLIP_MODEL,
        }
    }, final_path)
    print(f"✅ Training complete! Final model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Text-Conditional Diffusion Model')
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of epochs to train (default: {config.NUM_EPOCHS})')
    parser.add_argument('--save-every-epochs', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'

    train_text_conditional(args.epochs, args.save_every_epochs, args.max_samples, args.device)


if __name__ == "__main__":
    main()