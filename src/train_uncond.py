import torch
import torch.nn.functional as F
from torch.optim import Adam
import os
from tqdm import tqdm
import argparse

# Import our modules
from models.diffusion import DiffusionModel
from models.uncond_model import UncondUNet
from utils.dataset import ImageDataset, get_dataloader
from utils.visualization import show_tensor_image
from utils.sampling import sample_and_show

def parse_args():
    parser = argparse.ArgumentParser(description="Train an unconditional diffusion model")
    parser.add_argument("--data_dir", type=str, default="/home/azureuser/Diffusion/data/unconditional/reshaped")
    parser.add_argument("--output_dir", type=str, default="/home/azureuser/Diffusion/unconditional/outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sample_interval", type=int, default=10)
    return parser.parse_args()

def train_step(model, diffusion, batch, optimizer, device):
    optimizer.zero_grad()
    
    # Move batch to device
    x0 = batch.to(device)
    batch_size = x0.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
    
    # Add noise to the clean images
    x_t, noise = diffusion.q_sample(x0, t)
    
    # Predict the noise
    pred_noise = model(x_t, t)
    
    # Calculate loss (noise prediction)
    loss = F.mse_loss(noise, pred_noise)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model and diffusion
    model = UncondUNet(image_channels=3).to(device)
    diffusion = DiffusionModel(timesteps=args.timesteps)
    
    # Setup dataset and dataloader
    dataset = ImageDataset(args.data_dir, img_size=args.img_size)
    dataloader = get_dataloader(dataset, batch_size=args.batch_size)
    
    # Setup optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            loss = train_step(model, diffusion, batch, optimizer, device)
            epoch_loss += loss
            progress_bar.set_postfix({"loss": loss})
        
        # Calculate average loss over epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate and visualize samples
        if (epoch + 1) % args.sample_interval == 0:
            print("Generating samples...")
            samples = sample_and_show(model, diffusion, num_images=4, 
                                    image_size=args.img_size, device=device)
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()