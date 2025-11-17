---
license: mit
tags:
- diffusion
- text-to-image
- quickdraw
- pytorch
- clip
- ddpm
language:
- en
datasets:
- Xenova/quickdraw-small
---

# Text-Conditional QuickDraw Diffusion Model

A text-conditional diffusion model for generating Google QuickDraw-style sketches from text prompts. This model uses DDPM (Denoising Diffusion Probabilistic Models) with CLIP text encoding and classifier-free guidance to generate 64x64 grayscale sketches.

## Model Description

This is a U-Net based diffusion model that generates sketches conditioned on text prompts. It uses:
- **CLIP text encoder** (`openai/clip-vit-base-patch32`) for text conditioning
- **DDPM** for the diffusion process (1000 timesteps)
- **Classifier-free guidance** for improved text-image alignment
- Trained on **Google QuickDraw** dataset

## Model Details

- **Model Type**: Text-conditional DDPM diffusion model
- **Architecture**: U-Net with cross-attention for text conditioning
- **Image Size**: 64x64 grayscale
- **Base Channels**: 256
- **Text Encoder**: CLIP ViT-B/32 (frozen)
- **Training Steps**: 100 epochs
- **Diffusion Timesteps**: 1000
- **Guidance Scale**: 5.0 (default)

### Training Configuration

- **Dataset**: Xenova/quickdraw-small (5 classes)
- **Batch Size**: 128 (32 per GPU × 4 GPUs)
- **Learning Rate**: 1e-4
- **CFG Drop Probability**: 0.15
- **Optimizer**: Adam

## Usage

### Installation

```bash
pip install torch torchvision transformers diffusers datasets matplotlib pillow tqdm
```

### Generate Images

```python
import torch
from model import TextConditionedUNet
from scheduler import SimpleDDPMScheduler
from text_encoder import CLIPTextEncoder
from generate import generate_samples

# Load checkpoint
checkpoint_path = "text_diffusion_final_epoch_100.pt"
checkpoint = torch.load(checkpoint_path)

# Initialize model
model = TextConditionedUNet(text_dim=512).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Initialize text encoder
text_encoder = CLIPTextEncoder(model_name="openai/clip-vit-base-patch32", freeze=True).cuda()
text_encoder.eval()

# Generate samples
scheduler = SimpleDDPMScheduler(1000)
prompt = "a drawing of a cat"
num_samples = 4
guidance_scale = 5.0

with torch.no_grad():
    text_embedding = text_encoder(prompt)
    text_embeddings = text_embedding.repeat(num_samples, 1)

    shape = (num_samples, 1, 64, 64)
    samples = scheduler.sample_text(model, shape, text_embeddings, 'cuda', guidance_scale)
```

### Command Line Usage

```bash
# Generate samples
python generate.py --checkpoint text_diffusion_final_epoch_100.pt \
                   --prompt "a drawing of a fire truck" \
                   --num-samples 4 \
                   --guidance-scale 5.0

# Visualize denoising process
python visualize_generation.py --checkpoint text_diffusion_final_epoch_100.pt \
                               --prompt "a drawing of a cat" \
                               --num-steps 10
```

## Example Prompts

Try these prompts for best results:
- "a drawing of a cat"
- "a drawing of a fire truck"
- "a drawing of an airplane"
- "a drawing of a house"
- "a drawing of a tree"

**Note**: The model is trained on a limited set of QuickDraw classes, so it works best with simple object descriptions in the format "a drawing of a [object]".

## Classifier-Free Guidance

The model supports classifier-free guidance to improve text-image alignment:
- `guidance_scale = 1.0`: No guidance (pure conditional generation)
- `guidance_scale = 3.0-7.0`: Recommended range (default: 5.0)
- Higher values: Stronger adherence to text prompt (may reduce diversity)

## Model Architecture

### U-Net Structure
```
Input: (batch, 1, 64, 64)
├── Down Block 1: 1 → 256 channels
├── Down Block 2: 256 → 512 channels
├── Down Block 3: 512 → 512 channels
├── Middle Block: 512 channels
├── Up Block 3: 1024 → 512 channels (with skip connections)
├── Up Block 2: 768 → 256 channels (with skip connections)
└── Up Block 1: 512 → 1 channel (with skip connections)
Output: (batch, 1, 64, 64) - predicted noise
```

### Text Conditioning
- Text prompts encoded via CLIP ViT-B/32
- 512-dimensional text embeddings
- Injected into U-Net via cross-attention
- Classifier-free guidance with 15% dropout during training

## Training Details

- **Framework**: PyTorch 2.0+
- **Hardware**: 4x NVIDIA GPUs
- **Training Time**: ~100 epochs
- **Dataset**: Google QuickDraw sketches (5 classes)
- **Noise Schedule**: Linear (β from 0.0001 to 0.02)

## Limitations

- Limited to 64x64 resolution
- Grayscale output only
- Best performance on simple objects from QuickDraw classes
- May not generalize well to complex or out-of-distribution prompts

## Citation

If you use this model, please cite:

```bibtex
@misc{quickdraw-text-diffusion,
  title={Text-Conditional QuickDraw Diffusion Model},
  author={Your Name},
  year={2024},
  howpublished={\url{https://huggingface.co/YOUR_USERNAME/quickdraw-text-diffusion}}
}
```

## License

MIT License

## Acknowledgments

- Google QuickDraw dataset
- OpenAI CLIP
- DDPM paper: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- Classifier-free guidance: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
