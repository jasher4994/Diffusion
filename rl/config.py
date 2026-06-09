"""Default hyperparameters for RL fine-tuning runs.

A run-config dict (loaded from JSON/YAML by sweep.py) overrides these.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


# ---- Paths ----
BASE_DIR = "."  # repo root assumed
RUNS_DIR = "runs"

# ---- Base model source ----
# Either a local checkpoint path, or "hf:<repo_id>:<filename>" for HF Hub.
DEFAULT_BASE_CKPT = (
    "hf:jamesaasher/quickdraw-text-diffusion:text_diffusion_final_epoch_100.pt"
)

# ---- Image / model dims (inherited from text_conditional_diffusion/config.py) ----
IMAGE_SIZE = 64
TIMESTEPS = 1000
TEXT_DIM = 512
CLIP_MODEL = "openai/clip-vit-base-patch32"


# The 5 QuickDraw classes the base model was trained on
# (first 5 alphabetically; NUM_CLASSES_FILTER=5 in text_conditional_diffusion/config.py).
# Underscored names match the training prompt format `f"a drawing of a {label_name}"`.
TRAINED_PROMPTS = [
    "a drawing of a aircraft_carrier",
    "a drawing of a airplane",
    "a drawing of a alarm_clock",
    "a drawing of a ambulance",
    "a drawing of a angel",
]


@dataclass
class RunConfig:
    """All hyperparameters for a single GRPO run."""

    # Identification
    run_id: str
    base_ckpt: str = DEFAULT_BASE_CKPT
    seed: int = 0

    # GRPO core
    beta: float = 0.04  # KL-to-ref coefficient
    group_size: int = 8  # K rollouts per prompt
    eps_clip: float = 0.2  # PPO ratio clip
    n_steps: int = 200  # outer training steps
    ppo_inner_epochs: int = 1  # passes over each rollout batch

    # Rollout
    t_inf: int = 50  # respaced DDPM steps for rollouts
    cfg_scale: float = 5.0
    prompts_per_step: int = 2  # B_prompts; total batch = B_prompts * group_size

    # Optimiser
    lr: float = 1e-5
    grad_clip: float = 1.0

    # Reward
    reward_name: str = "vsym_l2"

    # Eval / checkpointing cadence
    eval_every: int = 25  # outer steps between samples + checkpoint dump
    eval_prompts: list = field(default_factory=lambda: list(TRAINED_PROMPTS))
    eval_n_seeds: int = 8  # fixed seeds per prompt in eval grid

    # Hardware
    device: str = "cuda"  # falls back to cpu / mps in code if unavailable

    def to_dict(self) -> dict:
        return asdict(self)
