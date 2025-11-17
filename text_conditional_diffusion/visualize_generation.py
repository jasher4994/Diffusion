"""Visualize the diffusion generation process - capture images at each timestep."""
import torch
import argparse
import os
import matplotlib.pyplot as plt

import config
from model import TextConditionedUNet
from scheduler import SimpleDDPMScheduler
from text_encoder import CLIPTextEncoder
from generate import tensor_to_image


def sample_with_snapshots(scheduler, model, shape, text_embeddings, device='cuda',
                          guidance_scale=1.0, snapshot_steps=None):
    """Modified sampling that captures snapshots at specific timesteps."""
    b = shape[0]
    img = torch.randn(shape, device=device)

    # Default: capture 10 evenly spaced steps
    if snapshot_steps is None:
        interval = scheduler.num_timesteps // 10
        snapshot_steps = list(range(scheduler.num_timesteps - 1, -1, -interval))
        if 0 not in snapshot_steps:
            snapshot_steps.append(0)

    snapshots = {}

    for i in reversed(range(0, scheduler.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = scheduler.p_sample_text(model, img, t, text_embeddings, guidance_scale)
        img = torch.clamp(img, -2.0, 2.0)

        if i in snapshot_steps:
            snapshots[i] = img.clone().detach()

    return img, snapshots


def plot_denoising_process(snapshots, prompt, output_path, sample_idx=0):
    """Plot snapshots side by side showing noise -> final image."""
    timesteps = sorted(snapshots.keys(), reverse=True)  # noise to clean
    num_steps = len(timesteps)

    fig, axes = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 3))
    if num_steps == 1:
        axes = [axes]

    fig.suptitle(f'Denoising Process: "{prompt}"', fontsize=12, fontweight='bold')

    for idx, t in enumerate(timesteps):
        img_tensor = snapshots[t][sample_idx]
        img = tensor_to_image(img_tensor)

        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f't={t}' if t > 0 else 'Final', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize denoising process')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default="a drawing of a cat")
    parser.add_argument('--guidance-scale', type=float, default=config.CFG_GUIDANCE_SCALE)
    parser.add_argument('--num-steps', type=int, default=10,
                       help='Number of snapshots to capture')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    ckpt_config = checkpoint.get('config', {})

    model = TextConditionedUNet(text_dim=ckpt_config.get('text_dim', config.TEXT_DIM)).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    text_encoder = CLIPTextEncoder(
        model_name=ckpt_config.get('clip_model', config.CLIP_MODEL), freeze=True
    ).to(args.device)
    text_encoder.eval()

    scheduler = SimpleDDPMScheduler(config.TIMESTEPS)

    # Generate with snapshots
    with torch.no_grad():
        text_embedding = text_encoder(args.prompt)
        shape = (1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)

        _, snapshots = sample_with_snapshots(
            scheduler, model, shape, text_embedding, args.device, args.guidance_scale
        )

    # Save visualization
    os.makedirs("outputs", exist_ok=True)
    safe_prompt = "".join(c if c.isalnum() or c in " _" else "" for c in args.prompt)[:50]
    output_path = f"outputs/denoising_{safe_prompt}.png"

    plot_denoising_process(snapshots, args.prompt, output_path)
    print(f"✅ Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
