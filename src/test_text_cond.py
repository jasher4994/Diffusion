import torch
import torch.nn.functional as F
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Import your modules
from models.diffusion import DiffusionModel
from models.text_model import SimpleTextConditionedUNet
from models.text_encoder import TextEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Test text-conditioned diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="test_outputs")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    return parser.parse_args()

@torch.no_grad()
def sample_from_model(model, text_encoder, diffusion, prompt, num_samples, img_size, device):
    """Generate images from a text prompt using your DiffusionModel"""
    model.eval()
    
    # Encode the text prompt
    text_embeddings = text_encoder([prompt] * num_samples)
    
    print(f"Generating images for: '{prompt}'")
    
    # Use your diffusion model's built-in sample method
    shape = (num_samples, 3, img_size, img_size)
    images_tensor = diffusion.sample(model, shape, device=device, cond=text_embeddings)
    
    # Convert tensor to PIL images
    images_tensor = torch.clamp(images_tensor, -1, 1)
    images_tensor = (images_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    images = []
    for i in range(num_samples):
        img_tensor = images_tensor[i].cpu()
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype('uint8')
        images.append(Image.fromarray(img_array))
    
    return images

@torch.no_grad()
def sample_from_model_manual(model, text_encoder, diffusion, prompt, num_samples, img_size, device):
    """Alternative manual sampling if you want more control"""
    model.eval()
    
    # Encode the text prompt
    text_embeddings = text_encoder([prompt] * num_samples)
    
    # Start with random noise
    x = torch.randn(num_samples, 3, img_size, img_size, device=device)
    
    print(f"Generating images for: '{prompt}'")
    
    # Reverse diffusion process using your model's p_sample method
    for i in reversed(range(0, diffusion.timesteps)):
        if i % 100 == 0:  # Progress indicator
            print(f"  Denoising step {i}/{diffusion.timesteps}")
            
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        # Use your diffusion model's p_sample method
        x = diffusion.p_sample(model, x, t, cond=text_embeddings)
    
    # Convert to PIL images
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    images = []
    for i in range(num_samples):
        img_tensor = x[i].cpu()
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype('uint8')
        images.append(Image.fromarray(img_array))
    
    return images

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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Final loss: {checkpoint.get('loss', 'unknown')}")
    
    # Test prompts relevant to your historical book illustrations
    test_prompts = [
        "an illustration of a forest scene",
        "a historical engraving of a castle",
        "a victorian illustration of flowers", 
        "a detailed drawing of animals",
        "an ornate book illustration",
        "a medieval manuscript illustration"
    ]
    
    print("🎨 Generating images...")
    
    # Generate images for each prompt
    for i, prompt in enumerate(test_prompts):
        try:
            # Use the built-in sample method (recommended)
            images = sample_from_model(
                model, text_encoder, diffusion, prompt, 
                args.num_samples, args.img_size, device
            )
            
            # Save individual images
            for j, img in enumerate(images):
                filename = f"prompt_{i+1:02d}_sample_{j+1:02d}.png"
                img.save(os.path.join(args.output_dir, filename))
            
            # Create a grid image for this prompt
            fig, axes = plt.subplots(1, args.num_samples, figsize=(args.num_samples * 3, 3))
            if args.num_samples == 1:
                axes = [axes]
                
            for j, img in enumerate(images):
                axes[j].imshow(img)
                axes[j].axis('off')
                axes[j].set_title(f"Sample {j+1}")
            
            plt.suptitle(f"'{prompt}'", fontsize=12)
            plt.tight_layout()
            grid_filename = f"grid_prompt_{i+1:02d}.png"
            plt.savefig(os.path.join(args.output_dir, grid_filename), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Generated images for prompt {i+1}: '{prompt}'")
            
        except Exception as e:
            print(f"❌ Error generating images for prompt {i+1}: '{prompt}'")
            print(f"    Error: {e}")
            continue
    
    print(f"🎉 Generated images saved to {args.output_dir}")
    print(f"📁 Check the following files:")
    print(f"   - Individual images: prompt_XX_sample_XX.png")
    print(f"   - Grid images: grid_prompt_XX.png")

if __name__ == "__main__":
    main()