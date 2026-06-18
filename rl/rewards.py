"""Reward function registry for GRPO.

A reward takes a batch of final images and (optionally) prompts/meta, and returns
a scalar tensor per sample in [B]. Larger = better. (We follow the convention that
all rewards are non-positive with 0 = perfect, mirroring the symmetry metrics in
rl/metrics.py.)
"""

from __future__ import annotations

from typing import Callable

import torch

from rl.metrics import symmetry_mse


def vsym_l2(
    x0: torch.Tensor, prompts: list[str] | None = None, meta: dict | None = None
) -> torch.Tensor:
    """Vertical-symmetry reward, per-sample. Returns shape [B].

    r_i = -mean_{pixels}((x_i - flip_horizontal(x_i))^2)

    Identical to `metrics.symmetry_mse` by construction, so the eval metric
    `symmetry_mse_mean` is directly comparable to this training reward.

    Caveat: the global optimum is *any* symmetric image, and low-contrast /
    blank blobs reach it easily — this reward can reward-hack via collapse
    (see Experiment 1). Use `vsym_scale_inv` or `vsym_plus_clip` to counter that.
    """
    return symmetry_mse(x0)


def vsym_scale_inv(
    x0: torch.Tensor,
    prompts: list[str] | None = None,
    meta: dict | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Scale-invariant vertical-symmetry reward, per-sample. Returns shape [B].

    r_i = - ||x_i - flip(x_i)||^2 / (||x_i - mean(x_i)||^2 + eps)

    The denominator is the per-image content variance. Dividing by it removes
    the trivial *contrast-reduction* exploit that `vsym_l2` suffers from: fading
    the whole image toward a constant scales numerator and denominator together,
    so it no longer lowers the (normalized) symmetry error. The model can only
    improve by making structure genuinely more symmetric at fixed contrast.

    Note: this does not by itself enforce *recognizability* — a perfectly
    constant image is regularized to reward 0 via `eps`. Pair with an on-prompt
    term (`vsym_plus_clip`) when recognizability matters.
    """
    flipped = torch.flip(x0, dims=[-1])
    num = (x0 - flipped).pow(2).flatten(1).mean(dim=1)
    mean = x0.flatten(1).mean(dim=1, keepdim=True)
    var = (x0.flatten(1) - mean).pow(2).mean(dim=1)
    return -num / (var + eps)


REWARDS: dict[str, Callable] = {
    "vsym_l2": vsym_l2,
    "vsym_scale_inv": vsym_scale_inv,
}
