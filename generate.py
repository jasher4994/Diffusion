import torch
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

from conditional_diffusion.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.unet import UNet
from conditional_diffusion.sampler import DDPMSampler
import config

def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.detach().cpu()
    tensor = (tensor.clamp(-1, 1) + 1) * 0.5
    import torchvision.transforms.functional as TF
    return TF.to_pil_image(tensor)

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    unet = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=config.TIME_EMB_DIM,
        num_classes=10,
        class_emb_dim=config.CLASS_EMB_DIM,
        base_channels=config.BASE_CHANNELS,
        image_size=config.IMAGE_SIZE
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        unet.load_state_dict(checkpoint['model_state_dict'])
    else:
        unet.load_state_dict(checkpoint)
    
    unet.to(device).eval()
    
    noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(device)
    sampler = DDPMSampler(noise_scheduler, unet, device)
    
    return unet, sampler

def generate_class_samples(sampler, class_idx, num_samples=4, num_steps=50, save_dir=None, prefix=""):
    """Generate samples for a specific class."""
    class_name = config.CLASS_NAMES[class_idx]
    
    print(f"Generating {num_samples} {class_name} images...")
    
    with torch.no_grad():
        images = sampler.sample_class_conditional(
            class_idx=class_idx,
            num_samples=num_samples,
            num_inference_steps=num_steps
        )
    
    results = []
    for i, img_tensor in enumerate(images):
        if torch.isnan(img_tensor).any():
            print(f"Warning: NaN detected in image {i+1}")
            continue
            
        img = tensor_to_image(img_tensor)
        results.append(img)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{save_dir}/{prefix}{class_name}_{i+1}.png"
            img.save(filename)
            print(f"  Saved: {filename}")
    
    return results

def generate_all_classes(sampler, samples_per_class=2, num_steps=50, save_dir=None, prefix=""):
    """Generate samples for all classes."""
    print(f"Generating {samples_per_class} samples for each class...")
    
    all_results = {}
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        try:
            images = generate_class_samples(
                sampler, class_idx, samples_per_class, num_steps, save_dir, prefix
            )
            all_results[class_name] = images
        except Exception as e:
            print(f"Error generating {class_name}: {e}")
            all_results[class_name] = []
    
    return all_results

def generate_grid(sampler, grid_size=3, num_steps=50, save_path=None, classes=None):
    """Generate a grid of images for visualization."""
    total_samples = grid_size * grid_size
    
    if classes is None:
        # Random classes
        class_indices = torch.randint(0, 10, (total_samples,))
    else:
        # Cycle through provided classes
        class_indices = torch.tensor([classes[i % len(classes)] for i in range(total_samples)])
    
    print(f"Generating {grid_size}x{grid_size} grid...")
    
    with torch.no_grad():
        images = sampler.sample(
            batch_size=total_samples,
            class_labels=class_indices.to(sampler.device),
            num_inference_steps=num_steps
        )
    
    # Create grid visualization
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i, (img_tensor, class_idx) in enumerate(zip(images, class_indices)):
        if i >= len(axes):
            break
            
        if torch.isnan(img_tensor).any():
            # Create black image for NaN
            img = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))
        else:
            img = tensor_to_image(img_tensor)
        
        axes[i].imshow(img)
        axes[i].set_title(config.CLASS_NAMES[class_idx.item()])
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grid saved: {save_path}")
    
    plt.close()
    return images

def generate_training_samples(unet, noise_scheduler, device, epoch, save_dir, num_classes=5, num_steps=50):
    """Generate samples during training (optimized for speed)."""
    print(f"Generating training samples for epoch {epoch+1}...")
    
    sampler = DDPMSampler(noise_scheduler, unet, device)
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate samples for first few classes to keep it quick
    for class_idx in range(min(num_classes, len(config.CLASS_NAMES))):
        try:
            images = generate_class_samples(
                sampler, 
                class_idx, 
                num_samples=1, 
                num_steps=num_steps,
                save_dir=save_dir,
                prefix=f"epoch_{epoch+1:03d}_"
            )
        except Exception as e:
            print(f"  Error generating {config.CLASS_NAMES[class_idx]}: {e}")

def main():
    """CLI for generating images from trained models."""
    parser = argparse.ArgumentParser(description='Generate images from trained diffusion model')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--class', type=int, dest='class_idx', help='Class to generate (0-9)', default=None)
    parser.add_argument('--samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--steps', type=int, default=100, help='Number of inference steps')
    parser.add_argument('--output', default='./generated', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Generate samples for all classes')
    parser.add_argument('--grid', type=int, help='Generate NxN grid of random samples')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    try:
        unet, sampler = load_model(args.checkpoint, device)
        print(f"Model loaded from: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate based on arguments
    if args.grid:
        grid_path = f"{args.output}/grid_{args.grid}x{args.grid}.png"
        generate_grid(sampler, args.grid, args.steps, grid_path)
        
    elif args.all:
        generate_all_classes(sampler, args.samples, args.steps, args.output)
        
    elif args.class_idx is not None:
        if 0 <= args.class_idx <= 9:
            generate_class_samples(sampler, args.class_idx, args.samples, args.steps, args.output)
        else:
            print("Class index must be between 0 and 9")
            
    else:
        print("Please specify --class, --all, or --grid")
        print("Examples:")
        print(f"  python generate.py {args.checkpoint} --class 0 --samples 4")
        print(f"  python generate.py {args.checkpoint} --all --samples 2")
        print(f"  python generate.py {args.checkpoint} --grid 3")

if __name__ == "__main__":
    main()