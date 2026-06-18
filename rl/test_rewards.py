"""Unit tests for rl/rewards.py.

Runnable as `python -m rl.test_rewards`. Verifies:
  1. vsym_l2 is bit-identical to metrics.symmetry_mse (the reconcile guarantee).
  2. Symmetry rewards order inputs correctly (symmetric > asymmetric).
  3. vsym_scale_inv is invariant to uniform contrast scaling — the property
     that removes vsym_l2's contrast-reduction collapse exploit.
"""

from __future__ import annotations

import sys

import torch

from rl.metrics import symmetry_mse
from rl.rewards import vsym_l2, vsym_scale_inv


def _symmetric(b=4, h=16, w=16, seed=0):
    """A horizontally-symmetric batch: left half mirrored onto the right."""
    g = torch.Generator().manual_seed(seed)
    half = torch.randn(b, 1, h, w // 2, generator=g)
    return torch.cat([half, torch.flip(half, dims=[-1])], dim=-1)


def _asymmetric(b=4, h=16, w=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 1, h, w, generator=g)


def main() -> int:
    ok = True

    # 1. Reconcile: reward == metric (no sqrt), exactly.
    x = _asymmetric()
    diff = (vsym_l2(x) - symmetry_mse(x)).abs().max().item()
    print(f"[1] vsym_l2 vs symmetry_mse max|diff| = {diff:.3e}")
    ok &= diff == 0.0

    # 2. Symmetric input scores ~0 (perfect) and beats asymmetric, for both rewards.
    sym, asym = _symmetric(), _asymmetric()
    for name, fn in [("vsym_l2", vsym_l2), ("vsym_scale_inv", vsym_scale_inv)]:
        r_sym = fn(sym).mean().item()
        r_asym = fn(asym).mean().item()
        print(f"[2] {name}: symmetric={r_sym:.4f}  asymmetric={r_asym:.4f}")
        ok &= r_sym > r_asym
        ok &= r_sym <= 1e-5  # non-positive, ~0 for perfect symmetry

    # 3. Contrast invariance: halving contrast of an asymmetric image leaves
    #    vsym_scale_inv (nearly) unchanged but inflates vsym_l2's reward (the
    #    exploit). Use a zero-mean image so scaling is pure contrast change.
    a = _asymmetric(seed=2)
    a = a - a.flatten(1).mean(dim=1).view(-1, 1, 1, 1)  # zero per-image mean
    faded = a * 0.5
    l2_full, l2_fade = vsym_l2(a).mean().item(), vsym_l2(faded).mean().item()
    si_full, si_fade = vsym_scale_inv(a).mean().item(), vsym_scale_inv(faded).mean().item()
    print(f"[3] vsym_l2:       full={l2_full:.4f}  faded={l2_fade:.4f}  (Δ={l2_fade - l2_full:+.4f})")
    print(f"[3] vsym_scale_inv:full={si_full:.4f}  faded={si_fade:.4f}  (Δ={si_fade - si_full:+.4f})")
    # Fading inflates vsym_l2 toward 0 (exploit); scale_inv barely moves.
    ok &= l2_fade > l2_full + 1e-4
    ok &= abs(si_fade - si_full) < 1e-3

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
