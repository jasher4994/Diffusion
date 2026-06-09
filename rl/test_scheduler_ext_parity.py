"""Parity test for rl/scheduler_ext.py.

Confirms that p_step_with_logprob, run at the full 1000-step schedule with a fixed
RNG seed, produces images bit-identical (within 1e-5) to the existing
SimpleDDPMScheduler.p_sample_text loop.

If this passes, we know the new code preserves the exact DDPM math while adding
log-prob bookkeeping on top.
"""

from __future__ import annotations

import sys
import torch

from rl.model_loader import load_pretrained, pick_device
from rl.config import DEFAULT_BASE_CKPT, TRAINED_PROMPTS
from rl.scheduler_ext import RLScheduler


def run_baseline(model, scheduler, text_emb, shape, device, seed):
    """Replicate sample_text but with deterministic RNG via torch.Generator."""
    gen = torch.Generator(device=device).manual_seed(seed)
    img = torch.randn(shape, generator=gen, device=device)
    for i in reversed(range(scheduler.num_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        # Replicate p_sample_text logic but use our generator for the noise.
        predicted_noise = model(img, t, text_emb)
        from text_conditional_diffusion.scheduler import extract  # local import
        betas_t = extract(scheduler.betas, t, img.shape)
        sqrt_one_minus = extract(scheduler.sqrt_one_minus_alphas_cumprod, t, img.shape)
        sqrt_recip_alphas = extract(1.0 / torch.sqrt(scheduler.alphas), t, img.shape)
        mean = sqrt_recip_alphas * (img - betas_t * predicted_noise / sqrt_one_minus)
        if i == 0:
            img = mean
        else:
            posterior_var = extract(scheduler.posterior_variance, t, img.shape)
            noise = torch.randn(img.shape, generator=gen, device=device)
            img = mean + torch.sqrt(posterior_var) * noise
        img = torch.clamp(img, -2.0, 2.0)
    return img


def run_new(model, base_scheduler, text_emb, shape, device, seed):
    """Same sampling using RLScheduler.p_step_with_logprob, ignoring log_prob."""
    rls = RLScheduler(base_scheduler, num_inference_steps=None)
    gen = torch.Generator(device=device).manual_seed(seed)
    img = torch.randn(shape, generator=gen, device=device)
    for step_idx in range(rls.num_inference_steps):
        img, _logp, _mean, _std = rls.p_step_with_logprob(
            model, img, step_idx, text_emb, cfg_scale=1.0, generator=gen,
        )
        img = torch.clamp(img, -2.0, 2.0)
    return img


def main():
    device = pick_device("cuda")
    print(f"Device: {device}")

    model, text_encoder, scheduler, meta = load_pretrained(DEFAULT_BASE_CKPT, device)
    model.eval()

    # Use 2 prompts × 1 sample to keep it fast (still ~110s on MPS for two 1000-step rollouts)
    prompts = list(TRAINED_PROMPTS[:2])
    text_emb = text_encoder(prompts)
    shape = (len(prompts), 1, 64, 64)
    seed = 12345

    print("Running baseline (p_sample_text loop)...")
    with torch.no_grad():
        img_base = run_baseline(model, scheduler, text_emb, shape, device, seed)

    print("Running new (RLScheduler.p_step_with_logprob loop)...")
    with torch.no_grad():
        img_new = run_new(model, scheduler, text_emb, shape, device, seed)

    diff = (img_base - img_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max |diff| = {max_diff:.3e}")
    print(f"mean |diff| = {mean_diff:.3e}")

    tol = 1e-5
    if max_diff < tol:
        print(f"PASS  (max diff < {tol})")
        return 0
    else:
        print(f"FAIL  (max diff >= {tol})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
