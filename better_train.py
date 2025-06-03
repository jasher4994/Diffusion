# Create better_train.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from src.models.diffusion import DiffusionModel
from src.models.text_model import SimpleTextConditionedUNet
from src.models.text_encoder import TextEncoder
from utils.dataset import TextImageDataset, get_dataloader

# Better training settings
def train_better():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Models
    model = SimpleTextConditionedUNet().to(device)
    text_encoder = TextEncoder().to(device)
    diffusion = DiffusionModel(timesteps=1000)
    
    # Better optimizer settings
    optimizer = Adam(
        list(model.parameters()) + list(text_encoder.parameters()),
        lr=2e-4,  # Higher learning rate
        betas=(0.9, 0.999),
        weight_decay=1e-4  # Add weight decay
    )
    
    # Dataset
    dataset = TextImageDataset(
        "/home/azureuser/Diffusion/data/conditional/images",
        "/home/azureuser/Diffusion/data/conditional/captions.json"
    )
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=True)  # Smaller batch
    
    print(f"Training with {len(dataset)} images")
    
    for epoch in range(20):  # Shorter epochs first
        epoch_loss = 0
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # Text embeddings
            text_embeddings = text_encoder(captions)
            
            # Random timestep
            batch_size = images.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
            
            # Add noise
            x_t, noise = diffusion.q_sample(images, t)
            
            # Predict noise
            optimizer.zero_grad()
            pred_noise = model(x_t, t, text_embeddings)
            
            # Loss
            loss = F.mse_loss(noise, pred_noise)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, f"better_checkpoint_epoch_{epoch+1:03d}.pt")

if __name__ == "__main__":
    train_better()