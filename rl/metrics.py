"""Evaluation metrics for RL-finetuned diffusion models.

All functions are pure: given tensors / lists, return numbers / dicts. No file
IO, no model state. The trainer and `eval.py` are responsible for calling
these and writing results.

Convention: images `x` are tensors of shape `[B, C, H, W]` in the range
`[-1, 1]` (the same normalisation the diffusion model trains on). Symmetry
metrics return *non-positive* values where 0 is perfect symmetry, matching
the reward convention used in `rewards.py`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Symmetry scores
# ---------------------------------------------------------------------------


def _hflip(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (axis=W). Mirrors left↔right => tests vertical symmetry."""
    return torch.flip(x, dims=[-1])


def symmetry_l2(x: torch.Tensor) -> torch.Tensor:
    """Negated per-pixel L2 distance between x and its horizontal flip.

    Returns `[B]` tensor; 0 == perfectly symmetric, more negative == less so.
    """
    diff = x - _hflip(x)
    # mean over (C, H, W) so the scale is independent of image size
    return -diff.pow(2).mean(dim=(-3, -2, -1)).sqrt()


def symmetry_l1(x: torch.Tensor) -> torch.Tensor:
    """Negated per-pixel L1 distance to horizontal flip. Robust variant of L2."""
    diff = x - _hflip(x)
    return -diff.abs().mean(dim=(-3, -2, -1))


def _gaussian_kernel(
    window_size: int = 11, sigma: float = 1.5, channels: int = 1, device=None
) -> torch.Tensor:
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    )
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def symmetry_ssim(x: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 − SSIM(x, hflip(x)), negated so higher is more symmetric.

    Pure-torch SSIM (Wang et al. 2004) on a single channel. Expects inputs in
    [-1, 1]; remapped to [0, 1] internally so the L1/L2 dynamic range
    constants behave as published.

    Returns `[B]`; 0 == perfectly symmetric.
    """
    x01 = (x.clamp(-1, 1) + 1) * 0.5
    y01 = _hflip(x01)

    c, _, _ = x01.shape[-3:]
    kernel = _gaussian_kernel(window_size, channels=c, device=x.device)
    pad = window_size // 2

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=pad, groups=c)

    mu_x, mu_y = filt(x01), filt(y01)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = filt(x01 * x01) - mu_xx
    sigma_yy = filt(y01 * y01) - mu_yy
    sigma_xy = filt(x01 * y01) - mu_xy

    C1, C2 = 0.01**2, 0.03**2
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    ssim_map = num / den.clamp(min=1e-12)
    ssim_per = ssim_map.mean(dim=(-3, -2, -1))
    return -(1.0 - ssim_per)  # 0 when ssim==1; more negative for less symmetric


# ---------------------------------------------------------------------------
# Diversity / collapse detection
# ---------------------------------------------------------------------------


def diversity(x: torch.Tensor) -> torch.Tensor:
    """Mean pairwise pixel-L2 distance across the batch.

    Low values flag mode collapse (all samples look alike). Returns a scalar
    tensor on the same device as `x`.
    """
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device)
    flat = x.flatten(start_dim=1)  # [B, D]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b ; numerically OK at our scales.
    sq = (flat * flat).sum(dim=1, keepdim=True)
    d2 = sq + sq.T - 2 * flat @ flat.T
    d2 = d2.clamp(min=0)
    d = d2.sqrt()
    n = x.shape[0]
    # Upper-triangle mean (exclude diagonal).
    iu = torch.triu_indices(n, n, offset=1, device=x.device)
    return d[iu[0], iu[1]].mean()


def per_pixel_std(x: torch.Tensor) -> torch.Tensor:
    """Mean per-pixel std across the batch. Complement to `diversity`."""
    return x.std(dim=0).mean()


# ---------------------------------------------------------------------------
# KL between policy and reference diffusion steps
# ---------------------------------------------------------------------------


def gaussian_kl(
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Closed-form KL( N(mean_p, std_p^2) || N(mean_q, std_q^2) ) per dim.

    Inputs broadcast; reduction is over all dims by default.
    """
    var_p = std_p.pow(2)
    var_q = std_q.pow(2)
    kl = (
        torch.log(std_q / std_p.clamp(min=1e-12))
        + (var_p + (mean_p - mean_q).pow(2)) / (2 * var_q.clamp(min=1e-12))
        - 0.5
    )
    if reduce == "sum":
        return kl.sum()
    if reduce == "mean":
        return kl.mean()
    return kl


# ---------------------------------------------------------------------------
# Group-wise aggregation utility
# ---------------------------------------------------------------------------


def aggregate_by_group(values: torch.Tensor, group_keys: Sequence) -> dict:
    """Return `{key: mean(values[mask])}` for each unique key in `group_keys`.

    `values` is `[N]`, `group_keys` is a length-N list of hashables.
    """
    out: dict = {}
    keys = list(group_keys)
    for k in sorted(set(keys)):
        mask = torch.tensor(
            [gk == k for gk in keys], dtype=torch.bool, device=values.device
        )
        out[str(k)] = float(values[mask].mean().item())
    return out


# ---------------------------------------------------------------------------
# One-shot eval bundle
# ---------------------------------------------------------------------------


def all_image_metrics(x: torch.Tensor, prompts: Iterable[str] | None = None) -> dict:
    """Compute every per-image metric we currently care about, plus per-prompt
    breakdowns when `prompts` is provided.

    Returns a plain dict of Python floats so it serialises straight to JSON.
    """
    s_l2 = symmetry_l2(x)
    s_l1 = symmetry_l1(x)
    s_ss = symmetry_ssim(x)
    div = diversity(x)
    pp_std = per_pixel_std(x)

    out = {
        "symmetry_l2_mean": float(s_l2.mean().item()),
        "symmetry_l1_mean": float(s_l1.mean().item()),
        "symmetry_ssim_mean": float(s_ss.mean().item()),
        "diversity": float(div.item()),
        "per_pixel_std": float(pp_std.item()),
        "n_samples": int(x.shape[0]),
    }
    if prompts is not None:
        plist = list(prompts)
        out["symmetry_l2_by_prompt"] = aggregate_by_group(s_l2, plist)
        out["symmetry_ssim_by_prompt"] = aggregate_by_group(s_ss, plist)
    return out
