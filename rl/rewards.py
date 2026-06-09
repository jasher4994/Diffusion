"""Reward function registry for GRPO.

A reward takes a batch of final images and (optionally) prompts/meta, and returns
a scalar tensor per sample in [B]. Larger = better. (We follow the convention that
all rewards are non-positive with 0 = perfect, mirroring the symmetry metrics in
rl/metrics.py.)
"""

from __future__ import annotations

from typing import Callable

import torch

from rl.metrics import symmetry_l2


def vsym_l2(x0: torch.Tensor, prompts: list[str] | None = None, meta: dict | None = None) -> torch.Tensor:
    """Vertical-symmetry reward, per-sample. Returns shape [B].

    r_i = -mean_{pixels}((x_i - flip_horizontal(x_i))^2)
    """
    flipped = torch.flip(x0, dims=[-1])
    sq = (x0 - flipped) ** 2
    # Mean over channels, height, width — keep batch.
    return -sq.flatten(1).mean(dim=1)


REWARDS: dict[str, Callable] = {
    "vsym_l2": vsym_l2,
}
