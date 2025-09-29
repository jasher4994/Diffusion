# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based conditional diffusion model project for generating CIFAR-10 class-conditioned images. The model uses a U-Net architecture with cosine noise scheduling to learn the reverse diffusion process, allowing generation of 128x128 images conditioned on specific CIFAR-10 classes.

## Development Commands

The project uses a comprehensive Makefile for all development tasks:

- `make install` - Install dependencies from requirements.txt
- `make train` - Train the model with full dataset
- `make overfit` - Train with 1 sample per class (overfitting test for debugging)
- `make test` - Alias for overfit training
- `make quick-test` - Quick 2-epoch overfit test
- `make generate` - Generate samples from trained model (auto-detects checkpoint)
- `make clean` - Clean checkpoints and generated images
- `make clean-all` - Deep clean including cache and temporary files
- `make clean-cuda` - Clear CUDA memory cache
- `make gpu-status` - Check GPU status and CUDA availability
- `make status` - Show training status and existing checkpoints
- `make monitor` - Monitor GPU usage during training

### Direct Python Commands

- `python train_model.py --full` - Full training
- `python train_model.py --overfit --samples 1` - Overfit training with 1 sample per class
- `python generate_samples.py <checkpoint_path> --samples 4` - Generate samples from checkpoint

## Architecture

### Core Components

- **`conditional_diffusion/`** - Main diffusion model package
  - `unet.py` - U-Net architecture with time and class conditioning
  - `noise_scheduler.py` - Cosine noise scheduler for forward/reverse diffusion
  - `sampler.py` - DDPM sampling for image generation
  - `training_engine.py` - Training loop with learning rate scheduling and checkpointing

- **`config.py`** - Central configuration file for all hyperparameters
  - Training settings (learning rate, batch size, epochs)
  - Model architecture parameters (embedding dimensions, channels)
  - Diffusion parameters (timesteps)
  - Dataset paths and class definitions

- **`train_model.py`** - Main training script with overfit/full training modes
- **`generate_samples.py`** - Inference script for generating samples from checkpoints
- **`data.py`** - CIFAR-10 data loading and preprocessing utilities
- **`training_monitor.py`** - Training progress logging and visualization

### Training Modes

The project supports two training modes:
1. **Full training** - Normal training on complete dataset
2. **Overfit training** - Training on limited samples per class for debugging

Checkpoints are saved to different directories (`checkpoints/` vs `checkpoints_overfit/`) based on training mode.

### Key Configuration

- Default image size: 128x128 pixels
- CIFAR-10 classes: 10 categories (airplane, automobile, bird, etc.)
- Diffusion timesteps: 1000
- Base model channels: 64
- Time and class embedding dimensions: 128 each

## Data Structure

- Uses upscaled CIFAR-10 dataset at 128x128 resolution
- Data directory: `/home/azureuser/Diffusion/data/cifar10-128x128`
- Classes are embedded as conditional information for generation

## Testing and Debugging

- Use `make overfit` or `make quick-test` to verify model can overfit on small datasets
- Training monitor creates plots and logs in checkpoint directories
- Sample images are generated periodically during training for visual inspection
- GPU status can be monitored with `make gpu-status` and `make monitor`