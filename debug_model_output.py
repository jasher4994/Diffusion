# debug_model_output.py
import torch
import matplotlib.pyplot as plt
from src.models.diffusion import DiffusionModel
from src.models.text_model import SimpleTextConditionedUNet
from src.models.text_encoder import TextEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleTextConditionedUNet().to(device)
text_encoder = TextEncoder().to(device)
diffusion = DiffusionModel(timesteps=1000)

checkpoint = torch.load("outputs/conditional/final_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

model.eval()

print("🔍 DEBUGGING MODEL OUTPUT")
print("=" * 40)

with torch.no_grad():
    # Test inputs
    text_emb = text_encoder(["a simple test"])
    x = torch.randn(1, 3, 128, 128, device=device)
    t = torch.tensor([500], device=device)
    
    print(f"Input image range: {x.min():.4f} to {x.max():.4f}")
    print(f"Text embedding range: {text_emb.min():.4f} to {text_emb.max():.4f}")
    
    # Check model output
    output = model(x, t, text_emb)
    
    print(f"Model output range: {output.min():.4f} to {output.max():.4f}")
    print(f"Model output mean: {output.mean():.4f}")
    print(f"Model output std: {output.std():.4f}")
    
    # Check if output is all zeros or very close to zero
    if torch.abs(output).max() < 1e-6:
        print("❌ MODEL OUTPUT IS ESSENTIALLY ZERO!")
    elif output.std() < 1e-3:
        print("❌ MODEL OUTPUT HAS VERY LOW VARIANCE!")
    else:
        print("✅ Model output looks reasonable")
    
    # Test sampling process
    print("\n🔄 Testing sampling process...")
    try:
        shape = (1, 3, 128, 128)
        sample = diffusion.sample(model, shape, device=device, cond=text_emb)
        print(f"Sample output range: {sample.min():.4f} to {sample.max():.4f}")
        
        # Save debug visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original input
        img1 = x[0].cpu().permute(1, 2, 0)
        img1 = (img1 + 1) / 2  # [-1,1] to [0,1]
        axes[0].imshow(img1)
        axes[0].set_title('Input (Random Noise)')
        
        # Model prediction
        img2 = output[0].cpu().permute(1, 2, 0)
        img2 = torch.clamp((img2 + 1) / 2, 0, 1)
        axes[1].imshow(img2)
        axes[1].set_title('Model Prediction')
        
        # Final sample
        img3 = sample[0].cpu().permute(1, 2, 0)
        img3 = torch.clamp((img3 + 1) / 2, 0, 1)
        axes[2].imshow(img3)
        axes[2].set_title('Final Sample')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('debug_model_output.png', dpi=150)
        print("✅ Debug visualization saved as debug_model_output.png")
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")