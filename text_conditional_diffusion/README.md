# Quick Draw Diffusion

A simple diffusion model for generating Google Quick Draw sketches. This is a completely self-contained implementation that doesn't depend on the main conditional_diffusion folder.

## Features

- Simple U-Net architecture optimized for grayscale 64x64 images
- Conditional generation with class embeddings
- Classifier-free guidance support
- Clean, minimal codebase with no external dependencies beyond PyTorch

## Quick Start

### 1. Train the model

```bash
cd quickdraw_diffusion
python train.py --epochs 50
```

### 2. Generate samples

```bash
# Generate random samples
python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt

# Generate specific class
python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --class-id 0

# Use classifier-free guidance
python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --guidance 2.0
```

### 3. List available classes

```bash
python generate.py --list-classes
```

## Files

- `config.py` - Simple configuration
- `model.py` - U-Net model with time and class embeddings
- `data.py` - Quick Draw dataset loading
- `scheduler.py` - DDPM noise scheduler
- `train.py` - Training script
- `generate.py` - Generation script

## Configuration

Edit `config.py` to adjust:
- Image size (default: 64x64)
- Batch size (default: 16)
- Learning rate (default: 1e-4)
- Number of timesteps (default: 1000)

## Model Architecture

- Input: Grayscale 64x64 images
- Time embedding: Sinusoidal positional encoding
- Class embedding: Learned embeddings for each Quick Draw class
- U-Net: Down-Middle-Up structure with residual blocks
- Skip connections between encoder and decoder

## Training Details

- Uses DDPM (Denoising Diffusion Probabilistic Models)
- 10% unconditional training for classifier-free guidance
- Gradient clipping and checkpointing
- Balanced class sampling from Quick Draw dataset