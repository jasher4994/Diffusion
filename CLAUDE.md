# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two independent PyTorch-based diffusion model implementations for generating Google QuickDraw sketches:

1. **Simple Class-Conditional Model** (`conditional_diffusion/`) - U-Net with class embeddings for generating 64x64 grayscale sketches
2. **Text-Conditional Model** (`text_conditional_diffusion/`) - U-Net with CLIP text encoder for text-to-sketch generation

Both models use DDPM (Denoising Diffusion Probabilistic Models) with classifier-free guidance support.

## Development Commands

The project uses a comprehensive Makefile for all development tasks:

### Simple Class-Conditional Model
- `make train-simple` (or `make train`) - Train simple model (default: 50 epochs)
- `make overfit-simple` (or `make overfit`) - Quick test training (default: 5 epochs)
- `make generate-simple` (or `make generate`) - Generate samples (auto-detects latest checkpoint)
- `make clean-simple` - Clean simple model checkpoints and outputs

### Text-Conditional Model (CLIP)
- `make train-text` - Train text-conditional model (default: 5 epochs)
- `make overfit-text` - Quick test training for text model
- `make generate-text` - Generate samples from text prompts
- `make clean-text` - Clean text model checkpoints and outputs

### General Commands
- `make clean` - Clean both models
- `make clean-all` - Deep clean including cache and temporary files
- `make clean-cuda` - Clear CUDA memory cache
- `make gpu-status` - Check GPU status and CUDA availability
- `make status` - Show training status and existing checkpoints for both models
- `make monitor` - Monitor GPU usage during training

### Custom Parameters
- `make train-simple EPOCHS=100` - Train with custom epochs
- `make generate-simple SAMPLES=16` - Generate custom number of samples
- `make generate-text PROMPT="a cat" SAMPLES=8` - Generate with custom prompt

### Direct Python Commands

**Simple Model:**
- `cd conditional_diffusion && python train.py --epochs 50`
- `cd conditional_diffusion && python generate.py --checkpoint checkpoints/quickdraw_final_epoch_50.pt --num-samples 8`
- `cd conditional_diffusion && python generate.py --checkpoint <path> --class-id 0 --guidance 2.0`

**Text Model:**
- `cd text_conditional_diffusion && python train.py --epochs 5`
- `cd text_conditional_diffusion && python generate.py --checkpoint <path> --prompt "a cat" --num-samples 4`

## Architecture

### Simple Class-Conditional Model (`conditional_diffusion/`)

**Files:**
- `model.py` - U-Net architecture with time and class embeddings
  - `TimeEmbedding` - Sinusoidal positional encoding for timesteps
  - `ResBlock` - Residual blocks with time and class conditioning
  - `SimpleUNet` - Main U-Net model with down-middle-up structure
- `scheduler.py` - DDPM noise scheduler for forward/reverse diffusion
- `data.py` - QuickDraw dataset loading and preprocessing
- `train.py` - Training script with classifier-free guidance
- `generate.py` - Inference script with guidance support
- `config.py` - Configuration (image size, batch size, learning rate, etc.)

**Key Features:**
- Image size: 64x64 grayscale
- Model channels: 64 base
- Time embedding: 128 dimensions
- Class embedding: 32 dimensions
- Diffusion timesteps: 1000
- Classifier-free guidance: 10% unconditional training
- Dataset: Google QuickDraw (5000 samples max)

### Text-Conditional Model (`text_conditional_diffusion/`)

**Files:**
- `model.py` - U-Net with text conditioning via cross-attention
- `text_encoder.py` - CLIP text encoder wrapper
  - Uses `openai/clip-vit-base-patch32`
  - Frozen CLIP weights for efficiency
  - Maps text to 512-dimensional embeddings
- `scheduler.py` - DDPM noise scheduler
- `data.py` - QuickDraw dataset with text prompts
- `train.py` - Training script with text conditioning
- `generate.py` - Text-to-image generation script
- `config.py` - Configuration specific to text model

**Key Features:**
- Image size: 64x64 grayscale
- Model channels: 128 base (larger than simple model)
- Time embedding: 128 dimensions
- Text embedding: 512 dimensions (CLIP output)
- Diffusion timesteps: 1000
- CLIP model: Frozen `clip-vit-base-patch32`
- Dataset: `Xenova/quickdraw-small` (filtered to 10 classes)

## Data Structure

- Uses Google QuickDraw dataset (64x64 grayscale sketches)
- Simple model: Downloads from HuggingFace datasets
- Text model: Uses `Xenova/quickdraw-small` dataset
- Classes are embedded either as class IDs (simple) or text prompts (text model)
- Checkpoints saved to `*/checkpoints/` directories
- Generated images saved to `*/outputs/` directories

## Testing and Debugging

- Use `make overfit-simple` or `make overfit-text` for quick sanity checks (5 epochs)
- Sample images are generated periodically during training for visual inspection
- GPU status can be monitored with `make gpu-status` and `make monitor`
- Use `make status` to check existing checkpoints and outputs
- For debugging, edit `config.py` in respective directories to adjust hyperparameters

## Common Workflows

**Quick test of simple model:**
```bash
make overfit-simple
make generate-simple SAMPLES=4
```

**Train text model and generate with custom prompt:**
```bash
make train-text EPOCHS=10
make generate-text PROMPT="a cute dog" SAMPLES=8
```

**Clean everything and start fresh:**
```bash
make clean-all
make clean-cuda
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0 (for CLIP)
- diffusers >= 0.18.0
- datasets >= 2.14.0 (for HuggingFace QuickDraw)
- matplotlib, pillow, tqdm

Install with: `pip install -r requirements.txt`