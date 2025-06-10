import torch
from conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion_model.unet import UNet
from conditional_diffusion_model.trainer import DiffusionTrainer

def test_pipeline():
    """Test that all components work together."""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize components
    print("Initializing components...")
    noise_scheduler = CosineNoiseScheduler(num_timesteps=1000).to(device)
    unet = UNet(in_channels=3, out_channels=3, base_channels=64)
    trainer = DiffusionTrainer(unet, noise_scheduler, device=device)
    
    # Test with fake data
    print("Testing with fake batch...")
    batch_size = 4
    fake_images = torch.randn(batch_size, 3, 64, 64)  # Fake RGB images
    
    # Test single training step
    loss = trainer.train_step(fake_images)
    print(f"Training step successful! Loss: {loss:.4f}")
    
    # Test forward pass
    print("Testing UNet forward pass...")
    with torch.no_grad():
        # Move inputs to the correct device
        fake_images_gpu = fake_images.to(device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)  # Create on GPU directly
        
        noisy_images, noise = noise_scheduler.add_noise(fake_images_gpu, timesteps)
        predicted_noise = unet(noisy_images, timesteps)
        print(f"Forward pass successful! Output shape: {predicted_noise.shape}")
    
    print("✅ All tests passed! Ready for training.")

if __name__ == "__main__":
    test_pipeline()