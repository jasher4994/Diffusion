"""Scheduler extension for GRPO.

Wraps the existing SimpleDDPMScheduler to add:
  - posterior_mean_variance / predict_x0_from_eps: math helpers
  - p_step_with_logprob: one denoising step that also returns log π_θ(x_prev | x_t)
  - make_respaced_timesteps + RLScheduler(num_inference_steps=...): faster sampling
    by using a subset of the original 1000 timesteps with recomputed coefficients

The original scheduler is not modified.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def make_respaced_timesteps(num_inference_steps: int, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Pick `num_inference_steps` timesteps from the original [0, num_train_timesteps)
    schedule, evenly spaced, descending (high noise → low noise).

    Returns a LongTensor of shape [num_inference_steps] with the chosen original-schedule
    timestep indices (these are what the UNet receives as `t`).
    """
    if num_inference_steps > num_train_timesteps:
        raise ValueError(f"num_inference_steps={num_inference_steps} > num_train_timesteps={num_train_timesteps}")
    step = num_train_timesteps / num_inference_steps
    # Centered samples: e.g. for 50 steps in 1000, picks indices [999, 979, ..., 19] approximately.
    ts = (torch.arange(num_inference_steps).float() * step).round().long()
    return torch.flip(ts, dims=[0])


class RLScheduler:
    """Wraps SimpleDDPMScheduler with log-prob-returning sampling.

    Two modes:
      * Full schedule (num_inference_steps=None): identical to base scheduler, used for
        parity testing.
      * Respaced (num_inference_steps=K): subset of K timesteps with coefficients
        recomputed so the per-step math is still a valid DDPM posterior.
    """

    def __init__(self, base_scheduler, num_inference_steps: Optional[int] = None):
        self.base = base_scheduler
        N = base_scheduler.num_timesteps

        if num_inference_steps is None:
            # Full schedule: timesteps go [N-1, N-2, ..., 0] (descending = noisy → clean).
            self.timesteps = torch.arange(N - 1, -1, -1, dtype=torch.long)
            # Re-index base scheduler's t-indexed tensors into inference-step order so
            # self.<tensor>[step_idx] always returns the coefficient at self.timesteps[step_idx].
            self.alphas_cumprod = base_scheduler.alphas_cumprod[self.timesteps].clone()
            self.alphas = base_scheduler.alphas[self.timesteps].clone()
            self.betas = base_scheduler.betas[self.timesteps].clone()
            self.posterior_variance = base_scheduler.posterior_variance[self.timesteps].clone()
        else:
            # Respaced: pick a subset and recompute per-step coefficients.
            self.timesteps = make_respaced_timesteps(num_inference_steps, N)
            ac = base_scheduler.alphas_cumprod  # shape [N], indexed by t
            self.alphas_cumprod = ac[self.timesteps]  # shape [K], in inference order
            # "previous" alphas_cumprod: shift by one in inference order. For the final
            # step the "next" state is x_0, which has ᾱ = 1.
            ac_prev = torch.cat([self.alphas_cumprod[1:], torch.tensor([1.0])])
            self.alphas = self.alphas_cumprod / ac_prev
            self.betas = 1.0 - self.alphas
            self.posterior_variance = self.betas * (1.0 - ac_prev) / (1.0 - self.alphas_cumprod)

        # Derived tensors used in the step formula (ε-form, matching p_sample_text):
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = 1.0 / torch.sqrt(self.alphas)

        self.num_inference_steps = len(self.timesteps)

    # ------------------------------------------------------------------ helpers

    def _coef(self, name: str, step_idx: int, device, dtype, shape):
        """Get a per-step coefficient as a tensor broadcastable to `shape`."""
        v = getattr(self, name)[step_idx].to(device=device, dtype=dtype)
        return v.view(1, *([1] * (len(shape) - 1))).expand(shape[0], *([1] * (len(shape) - 1)))

    def predict_x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Invert the forward identity: x̂_0 = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t."""
        ac = self.alphas_cumprod[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        return (x_t - torch.sqrt(1.0 - ac) * eps) / torch.sqrt(ac)

    def posterior_mean_variance(self, x_t: torch.Tensor, eps: torch.Tensor, step_idx: int):
        """Posterior mean μ̃ and variance β̃ for x_{t-1} | x_t, x̂_0(eps).

        Uses the ε-form to match SimpleDDPMScheduler.p_sample_text exactly:
            μ = (1/√α_t) · (x_t - β_t · ε̂ / √(1-ᾱ_t))
        Returns (mean [B,...], variance [scalar tensor]).
        """
        beta = self.betas[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        sqrt_recip_alpha = self.sqrt_recip_alphas[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        mean = sqrt_recip_alpha * (x_t - beta * eps / sqrt_one_minus_ac)
        variance = self.posterior_variance[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        return mean, variance

    # ------------------------------------------------------------------ main API

    def p_step_with_logprob(
        self,
        model,
        x_t: torch.Tensor,
        step_idx: int,
        text_embeddings: torch.Tensor,
        cfg_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        prev_sample: Optional[torch.Tensor] = None,
    ):
        """One reverse-process step. Mirrors SimpleDDPMScheduler.p_sample_text exactly.

        Args:
            model: the UNet (TextConditionedUNet)
            x_t: current noisy image [B, C, H, W]
            step_idx: index into self.timesteps (0 = noisiest, num_inference_steps-1 = cleanest)
            text_embeddings: [B, text_dim]
            cfg_scale: classifier-free-guidance scale (1.0 = off)
            generator: torch.Generator for reproducible noise (optional)
            prev_sample: if provided, use this as x_{t-1} instead of sampling fresh;
                         use during PPO update phase to score a stored action under the new policy.

        Returns:
            (x_prev, log_prob, mean, std) where
              x_prev    : [B, C, H, W]   the sampled (or provided) next state
              log_prob  : [B]            log π_θ(x_prev | x_t), summed over pixels
              mean      : [B, C, H, W]   the predicted Gaussian mean μ̃
              std       : scalar tensor  √β̃_t (or 0 at the last step)
        """
        # Look up the original-schedule timestep to pass to the model (its time embedding
        # was trained for these values).
        t_model = torch.full(
            (x_t.shape[0],), int(self.timesteps[step_idx]),
            device=x_t.device, dtype=torch.long,
        )

        # UNet forward — predict noise. With CFG, blend conditional & unconditional.
        eps_pred = model(x_t, t_model, text_embeddings)
        if cfg_scale > 1.0:
            uncond = torch.zeros_like(text_embeddings)
            eps_uncond = model(x_t, t_model, uncond)
            eps_pred = eps_uncond + cfg_scale * (eps_pred - eps_uncond)

        # Posterior mean (ε-form, exactly matches SimpleDDPMScheduler.p_sample_text)
        mean, variance = self.posterior_mean_variance(x_t, eps_pred, step_idx)
        std = torch.sqrt(variance)

        # Final step is deterministic (variance == 0 by construction).
        is_last = (step_idx == self.num_inference_steps - 1)

        if prev_sample is None:
            if is_last:
                x_prev = mean
            else:
                noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
                x_prev = mean + std * noise
        else:
            x_prev = prev_sample

        # Log-prob of the action under the current Gaussian policy.
        # For an isotropic Gaussian 𝒩(μ, σ²·I) at point x:
        #   log p(x) = -‖x-μ‖² / (2σ²) - (n/2)·log(2πσ²)
        # We sum over all non-batch dimensions.
        if is_last:
            # Deterministic move — no policy-gradient contribution from this step.
            log_prob = torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        else:
            n = x_t[0].numel()
            sq = ((x_prev - mean) ** 2).flatten(1).sum(dim=1)
            log_prob = -sq / (2.0 * variance) - 0.5 * n * math.log(2.0 * math.pi * float(variance))

        return x_prev, log_prob, mean, std
