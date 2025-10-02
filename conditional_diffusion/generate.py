import torch
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms

import config
from model import SimpleUNet
from data import QuickDrawDataset
from scheduler import SimpleDDPMScheduler

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # tensor is in range [-1, 1], convert to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    transform = transforms.ToPILImage()
    return transform(tensor.squeeze(0))

def generate_samples(checkpoint_path, num_samples=8, class_id=None, guidance_scale=1.0, device='cuda'):
    """Generate samples using the trained Quick Draw model."""
    print(f"🎨 Generating {num_samples} Quick Draw samples...")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    num_classes = checkpoint.get('num_classes', 10)  # fallback
    model = SimpleUNet(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create scheduler
    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)

    print(f"📊 Model loaded with {num_classes} classes")
    print(f"📊 Using guidance scale: {guidance_scale}")

    # Generate samples
    with torch.no_grad():
        for i in range(num_samples):
            print(f"🎨 Generating sample {i+1}/{num_samples}")

            # Prepare class labels
            if class_id is not None:
                # Generate specific class
                labels = torch.tensor([class_id], device=device)
                sample_name = f"quickdraw_sample_{i+1}_class_{class_id}"
            else:
                # Generate random class
                labels = torch.randint(0, num_classes, (1,), device=device)
                sample_name = f"quickdraw_sample_{i+1}_class_{labels.item()}"

            # Generate image
            shape = (1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)

            if guidance_scale != 1.0:
                # Classifier-free guidance
                # Generate unconditional
                uncond_labels = torch.tensor([-1], device=device)
                uncond_sample = scheduler.sample(model, shape, uncond_labels, device)

                # Generate conditional
                cond_sample = scheduler.sample(model, shape, labels, device)

                # Apply guidance
                sample = uncond_sample + guidance_scale * (cond_sample - uncond_sample)
            else:
                # Simple conditional generation
                sample = scheduler.sample(model, shape, labels, device)

            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)

            # Convert to image and save
            img = tensor_to_image(sample[0])
            img_path = f"outputs/{sample_name}.png"
            img.save(img_path)
            print(f"💾 Saved: {img_path}")

    print("✅ Generation complete!")

def list_classes():
    """List available classes from the dataset."""
    print("📋 Loading dataset to show available classes...")
    try:
        dataset = QuickDrawDataset()
        print(f"📊 Dataset has {dataset.num_classes} classes")

        # Show class distribution
        class_counts = dataset.df['label'].value_counts().sort_index()
        print("\n📊 Class distribution:")
        for class_id, count in class_counts.items():
            print(f"  Class {class_id}: {count} samples")

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate Quick Draw samples')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples to generate (default: 8)')
    parser.add_argument('--class-id', type=int, default=None,
                       help='Specific class to generate (default: random)')
    parser.add_argument('--guidance', type=float, default=1.0,
                       help='Guidance scale for classifier-free guidance (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--list-classes', action='store_true',
                       help='List available classes and exit')

    args = parser.parse_args()

    # List classes if requested
    if args.list_classes:
        list_classes()
        return

    # Check if checkpoint is required for other operations
    if not args.checkpoint:
        parser.error('--checkpoint is required for generation')

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'

    # Generate samples
    generate_samples(
        args.checkpoint,
        args.num_samples,
        args.class_id,
        args.guidance,
        args.device
    )

if __name__ == "__main__":
    main()