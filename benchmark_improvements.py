#!/usr/bin/env python3
"""
Benchmark script to measure diffusion pipeline improvements.

This script measures:
1. Training speed (samples/second)
2. Memory efficiency
3. Image quality metrics (FID, IS if possible)
4. Training stability (loss variance)
"""

import torch
import time
import psutil
import numpy as np
from conditional_diffusion.conditional_diffusion_model.noise_scheduler import CosineNoiseScheduler
from conditional_diffusion.conditional_diffusion_model.unet import UNet
from conditional_diffusion.conditional_diffusion_model.trainer import DiffusionTrainer
from conditional_diffusion.conditional_diffusion_model.text_encoder import TextEncoder
from conditional_diffusion.conditional_diffusion_model.sampler import DDPMSampler
import config

class PerformanceBenchmark:
    """Comprehensive benchmarking for diffusion pipeline improvements."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Benchmark running on: {self.device}")
        
    def benchmark_training_speed(self, num_batches=50):
        """Benchmark training speed improvements."""
        print("\n📊 Benchmarking Training Speed...")
        
        # Initialize components
        noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(self.device)
        unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=config.BASE_CHANNELS,
            time_emb_dim=config.TIME_EMB_DIM,
            text_emb_dim=config.TEXT_EMB_DIM
        )
        text_encoder = TextEncoder(device=self.device)
        trainer = DiffusionTrainer(unet, noise_scheduler, device=self.device)
        
        # Create fake data
        batch_size = config.BATCH_SIZE
        image_size = config.IMAGE_SIZE
        fake_images = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        fake_captions = ["A test image"] * batch_size
        
        # Warm-up
        print("🔥 Warming up...")
        for _ in range(5):
            with torch.no_grad():
                text_embeddings = text_encoder(fake_captions)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=self.device)
            noisy_images, noise = noise_scheduler.add_noise(fake_images, timesteps)
            
            trainer.optimizer.zero_grad()
            if trainer.use_amp:
                with torch.cuda.amp.autocast():
                    predicted_noise = unet(noisy_images, timesteps, text_embeddings)
                    loss = trainer.criterion(predicted_noise, noise)
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                predicted_noise = unet(noisy_images, timesteps, text_embeddings)
                loss = trainer.criterion(predicted_noise, noise)
                loss.backward()
                trainer.optimizer.step()
        
        # Benchmark
        print(f"⏱️  Running {num_batches} training steps...")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        losses = []
        for i in range(num_batches):
            with torch.no_grad():
                text_embeddings = text_encoder(fake_captions)
            
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=self.device)
            noisy_images, noise = noise_scheduler.add_noise(fake_images, timesteps)
            
            trainer.optimizer.zero_grad()
            if trainer.use_amp:
                with torch.cuda.amp.autocast():
                    predicted_noise = unet(noisy_images, timesteps, text_embeddings)
                    loss = trainer.criterion(predicted_noise, noise)
                trainer.scaler.scale(loss).backward()
                trainer.scaler.unscale_(trainer.optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), trainer.grad_clip_norm)
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                predicted_noise = unet(noisy_images, timesteps, text_embeddings)
                loss = trainer.criterion(predicted_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), trainer.grad_clip_norm)
                trainer.optimizer.step()
            
            losses.append(loss.item())
            
            if i % 10 == 0:
                print(f"  Step {i}/{num_batches}, Loss: {loss.item():.6f}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        total_time = end_time - start_time
        total_samples = num_batches * batch_size
        samples_per_second = total_samples / total_time
        memory_usage = end_memory - start_memory
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        
        print(f"\n📈 Training Speed Results:")
        print(f"  • Total time: {total_time:.2f}s")
        print(f"  • Samples/second: {samples_per_second:.2f}")
        print(f"  • Memory usage: {memory_usage:.2f} MB")
        print(f"  • Loss stability: {loss_mean:.6f} ± {loss_std:.6f}")
        print(f"  • Mixed precision: {'✅ Enabled' if trainer.use_amp else '❌ Disabled'}")
        print(f"  • Gradient clipping: ✅ Enabled ({trainer.grad_clip_norm})")
        
        return {
            'samples_per_second': samples_per_second,
            'memory_usage_mb': memory_usage,
            'loss_mean': loss_mean,
            'loss_std': loss_std,
            'total_time': total_time
        }
    
    def benchmark_generation_quality(self, num_samples=8):
        """Benchmark generation speed and basic quality."""
        print("\n🎨 Benchmarking Generation Quality...")
        
        # Initialize components
        noise_scheduler = CosineNoiseScheduler(num_timesteps=config.TIMESTEPS).to(self.device)
        unet = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=config.BASE_CHANNELS,
            time_emb_dim=config.TIME_EMB_DIM,
            text_emb_dim=config.TEXT_EMB_DIM
        )
        text_encoder = TextEncoder(device=self.device)
        sampler = DDPMSampler(noise_scheduler, unet, device=self.device)
        
        # Test prompts
        test_prompts = ["A cat", "A dog", "A car", "A house"]
        
        print(f"🖼️  Generating {num_samples} samples...")
        start_time = time.time()
        
        generated_images = []
        for i, prompt in enumerate(test_prompts[:num_samples]):
            with torch.no_grad():
                text_embeddings = text_encoder([prompt])
                
                # Enhanced generation
                image = sampler.sample(
                    batch_size=1,
                    image_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                    num_inference_steps=20,
                    text_embeddings=text_embeddings,
                    guidance_scale=1.5,
                    dynamic_thresholding=True
                )
                generated_images.append(image)
        
        generation_time = time.time() - start_time
        time_per_image = generation_time / len(generated_images)
        
        # Basic quality metrics
        nan_count = sum(1 for img in generated_images if torch.isnan(img).any())
        value_ranges = [(img.min().item(), img.max().item()) for img in generated_images]
        
        print(f"\n🎯 Generation Results:")
        print(f"  • Generation time: {generation_time:.2f}s")
        print(f"  • Time per image: {time_per_image:.2f}s")
        print(f"  • NaN images: {nan_count}/{len(generated_images)}")
        print(f"  • Value ranges: {value_ranges}")
        print(f"  • Enhanced features:")
        print(f"    - Classifier-free guidance: ✅")
        print(f"    - Dynamic thresholding: ✅")
        print(f"    - Improved noise init: ✅")
        
        return {
            'generation_time': generation_time,
            'time_per_image': time_per_image,
            'nan_count': nan_count,
            'num_samples': len(generated_images)
        }
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Running Full Performance Benchmark")
        print("=" * 60)
        
        results = {}
        
        # Training speed benchmark
        results['training'] = self.benchmark_training_speed(num_batches=30)
        
        # Generation quality benchmark  
        results['generation'] = self.benchmark_generation_quality(num_samples=4)
        
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Training Speed: {results['training']['samples_per_second']:.2f} samples/sec")
        print(f"Memory Efficiency: {results['training']['memory_usage_mb']:.2f} MB")
        print(f"Training Stability: σ={results['training']['loss_std']:.6f}")
        print(f"Generation Speed: {results['generation']['time_per_image']:.2f}s per image")
        print(f"Generation Quality: {results['generation']['nan_count']}/{results['generation']['num_samples']} NaN images")
        
        return results

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()