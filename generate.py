import torch
import matplotlib.pyplot as plt
import os
import argparse
from conditional_diffusion.conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.conditional_diffusion_model.unet import UNet
from conditional_diffusion.conditional_diffusion_model.text_encoder import TextEncoder
from conditional_diffusion.conditional_diffusion_model.sampler import DDPMSampler
import config

def generate_images(checkpoint_name="model_final.pt"):
    """Generate images using config settings and specified checkpoint."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    print("🔧 Using config settings:")
    print(f"  • Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"  • Timesteps: {config.TIMESTEPS}")
    print(f"  • Inference steps: {config.NUM_INFERENCE_STEPS}")
    print(f"  • Base channels: {config.BASE_CHANNELS}")
    
    # Initialize components from config
    noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=config.BASE_CHANNELS,
        time_emb_dim=config.TIME_EMB_DIM,
        text_emb_dim=config.TEXT_EMB_DIM
    )
    text_encoder = TextEncoder(device=device)
    
    # Load checkpoint
    checkpoint_path = f"{config.CHECKPOINT_DIR}/{checkpoint_name}"
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        unet.to(device)
        unet.eval()
        print("✅ Checkpoint loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("💡 Available checkpoints:")
        if os.path.exists(config.CHECKPOINT_DIR):
            for f in os.listdir(config.CHECKPOINT_DIR):
                if f.endswith('.pt'):
                    print(f"   • {f}")
        return
    
    # Create sampler
    sampler = DDPMSampler(noise_scheduler, unet, device=device)
    
    # Get prompts from config - error if not found
    if not hasattr(config, 'DEFAULT_PROMPTS'):
        raise ValueError("DEFAULT_PROMPTS not found in config.py! Please add DEFAULT_PROMPTS list to config.")
    
    test_prompts = config.DEFAULT_PROMPTS
    print(f"🎨 Generating {len(test_prompts)} images from config prompts...")
    
    # Create output directory
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Generating image {i+1}/{len(test_prompts)} ---")
            
            if prompt:
                print(f"Prompt: '{prompt}'")
                text_embeddings = text_encoder([prompt])
                safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
                filename = f'{output_dir}/generated_{i+1}_{safe_prompt}.png'
            else:
                print("No prompt (unconditional)")
                text_embeddings = None
                filename = f'{output_dir}/generated_{i+1}_unconditional.png'
            
            # Generate image using config settings
            generated_image = sampler.sample(
                batch_size=1,
                image_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                text_embeddings=text_embeddings
            )
            
            print(f"Generated range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
            
            if torch.isnan(generated_image).any():
                print("❌ NaN detected in generation!")
                continue
            
            # Convert to displayable format
            img = (generated_image[0] + 1) / 2  # Convert from [-1,1] to [0,1]
            img = torch.clamp(img, 0, 1)
            img = img.detach().cpu().permute(1, 2, 0).numpy()
            
            # Save image
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(prompt if prompt else "Unconditional Generation", fontsize=10)
            plt.axis('off')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved: {filename}")
    
    print(f"\n🎉 Generation completed!")
    print(f"🔍 Check the generated images in: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images with diffusion model')
    parser.add_argument('--checkpoint', type=str, default='model_final.pt', 
                       help='Checkpoint filename (e.g., model_epoch_150.pt)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt for generation')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Output path for generated image')
    
    args = parser.parse_args()
    
    # If single image requested, just generate one
    if args.prompt is not None:
        # Generate single image
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {device}")
        
        # Initialize components
        noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
        unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=config.BASE_CHANNELS,
            time_emb_dim=config.TIME_EMB_DIM,
            text_emb_dim=config.TEXT_EMB_DIM
        )
        text_encoder = TextEncoder(device=device)
        
        # Load checkpoint
        checkpoint_path = f"{config.CHECKPOINT_DIR}/{args.checkpoint}"
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unet.load_state_dict(checkpoint['model_state_dict'])
            unet.to(device)
            unet.eval()
            print("✅ Checkpoint loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("💡 Available checkpoints:")
            if os.path.exists(config.CHECKPOINT_DIR):
                for f in os.listdir(config.CHECKPOINT_DIR):
                    if f.endswith('.pt'):
                        print(f"   • {f}")
            exit(1)
        
        # Initialize sampler
        sampler = DDPMSampler(noise_scheduler, unet, device)
        
        # Generate single image
        print(f"🎨 Generating image for prompt: '{args.prompt}'")
        with torch.no_grad():
            # Encode text prompt
            text_emb = text_encoder.encode_text([args.prompt])
            
            # Generate image
            image = sampler.sample(
                batch_size=1,
                image_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                text_embeddings=text_emb
            )[0]
        
        # Save image
        save_path = args.save_path or f"generated_{args.prompt.replace(' ', '_')}.png"
        
        # Convert to numpy and normalize for display
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image_np)
        plt.title(args.prompt)
        plt.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {save_path}")
    else:
        # Generate default set
        generate_images(args.checkpoint)

    