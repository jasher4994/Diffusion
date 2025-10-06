"""Generation script for text-conditional diffusion model."""
import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms

import config
from model import TextConditionedUNet
from scheduler import SimpleDDPMScheduler
from text_encoder import CLIPTextEncoder


def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # tensor is in range [-1, 1], convert to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    transform = transforms.ToPILImage()
    return transform(tensor.squeeze(0))


def generate_samples(checkpoint_path, prompt="a drawing of a cat", num_samples=4, guidance_scale=3.0, device='cuda'):
    """Generate samples using text prompts with classifier-free guidance.

    Args:
        checkpoint_path: Path to model checkpoint
        prompt: Text prompt for generation
        num_samples: Number of samples to generate
        guidance_scale: CFG scale (1.0 = no guidance, 3.0-7.0 typical, higher = stronger)
        device: Device to use
    """
    print(f"🎨 Generating {num_samples} samples with prompt: '{prompt}'")
    print(f"📊 Guidance scale: {guidance_scale}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    ckpt_config = checkpoint.get('config', {})
    text_dim = ckpt_config.get('text_dim', config.TEXT_DIM)
    clip_model = ckpt_config.get('clip_model', config.CLIP_MODEL)

    # Create model
    model = TextConditionedUNet(text_dim=text_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create text encoder
    text_encoder = CLIPTextEncoder(model_name=clip_model, freeze=True).to(device)
    text_encoder.eval()

    # Create scheduler
    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)

    print(f"📊 Model loaded (text_dim={text_dim})")
    print(f"📊 CLIP model: {clip_model}")

    # Encode the text prompt once
    with torch.no_grad():
        text_embedding = text_encoder(prompt)
        # Repeat for batch generation
        text_embeddings = text_embedding.repeat(num_samples, 1)

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Generate samples
    print(f"🎨 Generating {num_samples} samples...")
    with torch.no_grad():
        # Generate all samples in a batch
        shape = (num_samples, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        samples = scheduler.sample_text(model, shape, text_embeddings, device, guidance_scale)

        # Save each sample
        for i in range(num_samples):
            # Create safe filename from prompt
            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
            safe_prompt = safe_prompt.replace(" ", "_")[:50]  # Limit length
            sample_name = f"text_sample_{i+1}_{safe_prompt}"

            # Convert to image and save
            img = tensor_to_image(samples[i])
            img_path = f"outputs/{sample_name}.png"
            img.save(img_path)
            print(f"💾 Saved: {img_path}")

    print("✅ Generation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate samples from text-conditional diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--prompt', type=str, default="a drawing of a cat and dog",
                       help='Text prompt for generation')
    parser.add_argument('--num-samples', type=int, default=4,
                       help='Number of samples to generate (default: 4)')
    parser.add_argument('--guidance-scale', type=float, default=config.CFG_GUIDANCE_SCALE,
                       help=f'Classifier-free guidance scale (1.0 = no guidance, 3.0-7.0 typical, default: {config.CFG_GUIDANCE_SCALE})')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'

    generate_samples(args.checkpoint, args.prompt, args.num_samples, args.guidance_scale, args.device)


if __name__ == "__main__":
    main()