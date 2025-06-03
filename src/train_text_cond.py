import torch
import torch.nn.functional as F
from torch.optim import Adam
import os
from tqdm import tqdm
import argparse
import json
import time
from datetime import datetime

# Import our modules
from models.diffusion import DiffusionModel
from models.text_model import SimpleTextConditionedUNet
from models.text_encoder import TextEncoder
from utils.dataset import TextImageDataset, get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditioned diffusion model")
    parser.add_argument("--data_dir", type=str, default="/home/azureuser/Diffusion/data/conditional/images")
    parser.add_argument("--captions_file", type=str, default="/home/azureuser/Diffusion/data/conditional/captions.json")
    parser.add_argument("--output_dir", type=str, default="/home/azureuser/Diffusion/outputs/conditional")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--keep_checkpoints", type=int, default=3, help="Number of checkpoints to keep")
    parser.add_argument("--save_optimizer", action="store_true", help="Save optimizer state (larger files)")
    return parser.parse_args()

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def save_checkpoint_efficient(model, text_encoder, optimizer, epoch, loss, output_dir, 
                            save_optimizer=False, keep_checkpoints=3):
    """Efficiently save checkpoint with automatic cleanup"""
    
    # Create checkpoint data
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        'model_config': {
            'image_channels': 3,
            'model_channels': 64,
            'time_emb_dim': 128,
            'text_emb_dim': 128
        }
    }
    
    # Only save optimizer if requested (saves ~50% space)
    if save_optimizer:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    
    try:
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file size
        file_size = os.path.getsize(checkpoint_path) / (1024**2)  # MB
        print(f"✓ Saved checkpoint: {checkpoint_path} ({file_size:.1f}MB)")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(output_dir, keep_checkpoints)
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to save checkpoint: {e}")
        return False

def cleanup_old_checkpoints(output_dir, keep_checkpoints):
    """Remove old checkpoints, keeping only the most recent ones"""
    try:
        # Find all checkpoint files
        checkpoint_files = []
        for file in os.listdir(output_dir):
            if file.startswith("checkpoint_epoch_") and file.endswith(".pt"):
                file_path = os.path.join(output_dir, file)
                checkpoint_files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for file_path, _ in checkpoint_files[keep_checkpoints:]:
            try:
                os.remove(file_path)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(file_path)}")
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not cleanup old checkpoints: {e}")

def save_training_log(output_dir, epoch, loss, learning_rate, batch_size):
    """Save training metrics to JSON log"""
    log_path = os.path.join(output_dir, "training_log.json")
    
    log_entry = {
        'epoch': epoch,
        'loss': float(loss),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat()
    }
    
    # Load existing log or create new
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        except:
            log_data = []
    else:
        log_data = []
    
    log_data.append(log_entry)
    
    # Save updated log
    try:
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save training log: {e}")

def train_step(model, text_encoder, diffusion, batch, optimizer, device):
    images, captions = batch
    optimizer.zero_grad()
    
    # Move to device
    x0 = images.to(device)
    batch_size = x0.shape[0]
    
    # Encode text
    text_embeddings = text_encoder(captions)
    
    # Sample timesteps
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
    
    # Add noise
    x_t, noise = diffusion.q_sample(x0, t)
    
    # Predict noise
    pred_noise = model(x_t, t, text_embeddings)
    
    # Loss
    loss = F.mse_loss(noise, pred_noise)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    model = SimpleTextConditionedUNet().to(device)
    text_encoder = TextEncoder().to(device)
    diffusion = DiffusionModel(timesteps=args.timesteps)
    
    # Print model info
    model_size = get_model_size(model)
    text_encoder_size = get_model_size(text_encoder)
    print(f"📊 Model size: {model_size:.1f}MB")
    print(f"📊 Text encoder size: {text_encoder_size:.1f}MB")
    print(f"📊 Total model size: {model_size + text_encoder_size:.1f}MB")
    
    # Dataset and dataloader
    dataset = TextImageDataset(args.data_dir, args.captions_file, img_size=args.img_size)
    dataloader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    print(f"🚀 Starting training with {len(dataset)} images...")
    print(f"📈 Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"💾 Saving checkpoints every {args.save_every} epochs")
    print(f"🗂️  Keeping {args.keep_checkpoints} most recent checkpoints")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            loss = train_step(model, text_encoder, diffusion, batch, optimizer, device)
            epoch_loss += loss
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  📉 Average Loss: {avg_loss:.4f}")
        print(f"  ⏱️  Epoch Time: {epoch_time:.1f}s")
        
        # Save training log
        save_training_log(args.output_dir, epoch + 1, avg_loss, args.lr, args.batch_size)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            print(f"💾 Saving checkpoint...")
            save_checkpoint_efficient(
                model, text_encoder, optimizer, epoch + 1, avg_loss, 
                args.output_dir, args.save_optimizer, args.keep_checkpoints
            )
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final model (always include optimizer for final model)
    print(f"🏁 Training completed! Saving final model...")
    final_path = os.path.join(args.output_dir, "final_model.pt")
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'text_encoder_state_dict': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'final_loss': avg_loss,
        'training_args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        torch.save(final_checkpoint, final_path)
        final_size = os.path.getsize(final_path) / (1024**2)
        print(f"✅ Final model saved: {final_path} ({final_size:.1f}MB)")
    except Exception as e:
        print(f"❌ Failed to save final model: {e}")
    
    total_time = time.time() - start_time
    print(f"🎉 Training completed in {total_time/3600:.1f} hours")

if __name__ == "__main__":
    main()