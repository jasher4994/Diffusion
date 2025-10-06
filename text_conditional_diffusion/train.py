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

    # Print configuration summary
    print("\n" + "="*60)
    print("🚀 TEXT-CONDITIONAL DIFFUSION TRAINING")
    print("="*60)

    # GPU Information
    print("\n📍 GPU Configuration:")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"   Device: CPU (CUDA not available)")

    # Training Configuration
    print("\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {config.BATCH_SIZE}")
    print(f"   Learning rate: {config.LEARNING_RATE}")
    print(f"   Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   Timesteps: {config.TIMESTEPS}")

    # Model Architecture
    print("\n🏗️  Model Architecture:")
    print(f"   Channels: {config.CHANNELS}")
    print(f"   Time dim: {config.TIME_DIM}")
    print(f"   Text dim: {config.TEXT_DIM}")
    print(f"   CLIP model: {config.CLIP_MODEL}")
    print(f"   CLIP frozen: {config.FREEZE_CLIP}")
    print(f"   CFG drop prob: {config.CFG_DROP_PROB}")
    print(f"   CFG guidance scale: {config.CFG_GUIDANCE_SCALE}")

    # Dataset Information
    print("\n📂 Dataset:")
    print(f"   Dataset: {config.DATASET_NAME}")
    print(f"   Number of classes: {config.NUM_CLASSES_FILTER}")
    print(f"   Max samples: {max_samples if max_samples else 'All'}")

    print("\n" + "="*60 + "\n")

    # Create dataset and dataloader
    print("📂 Loading dataset...")
    dataset = QuickDrawTextDataset(
        split="train",
        max_samples=max_samples,
        num_classes=config.NUM_CLASSES_FILTER
    )

    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Classes: {dataset.class_names}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Create models
    print("\n🏗️ Creating models...")
    model = TextConditionedUNet(text_dim=config.TEXT_DIM).to(device)
    text_encoder = CLIPTextEncoder(
        model_name=config.CLIP_MODEL,
        freeze=config.FREEZE_CLIP
    ).to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        # Don't wrap text_encoder - it processes strings, not tensors

    # Optimizer (only model parameters, CLIP is frozen)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Scheduler and loss
    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)
    criterion = nn.MSELoss()

    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Batches per epoch: {len(dataloader)}")
    print(f"✓ Total training steps: {epochs * len(dataloader):,}")
    print("\n" + "="*60 + "\n")

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

            # Classifier-free guidance: randomly drop text conditioning
            # This teaches the model to work both with and without text
            cfg_mask = torch.rand(images.shape[0], device=device) < config.CFG_DROP_PROB
            text_embeddings[cfg_mask] = 0.0  # Zero out text embeddings for unconditional

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
            # Handle DataParallel wrapper
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
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
    # Handle DataParallel wrapper
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
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