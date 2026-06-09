"""Load the pretrained text-conditional diffusion model.

Bridges the existing `text_conditional_diffusion/` package (which uses bare
`import config`) into the `rl/` package without modifying it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
TCD_DIR = REPO_ROOT / "text_conditional_diffusion"


def _ensure_tcd_on_path() -> None:
    """Make `text_conditional_diffusion/` importable as top-level modules."""
    p = str(TCD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_tcd():
    """Import model / scheduler / text_encoder from the original module."""
    _ensure_tcd_on_path()
    import config as tcd_config  # noqa: F401  (registers module)
    from model import TextConditionedUNet
    from scheduler import SimpleDDPMScheduler
    from text_encoder import CLIPTextEncoder

    return tcd_config, TextConditionedUNet, SimpleDDPMScheduler, CLIPTextEncoder


def resolve_checkpoint(spec: str) -> str:
    """Resolve a checkpoint spec to a local file path.

    Accepts:
      - "/abs/path/to.pt"             -> returned as-is
      - "relative/path.pt"            -> resolved against repo root
      - "hf:<repo_id>:<filename>"     -> downloaded via huggingface_hub
    """
    if spec.startswith("hf:"):
        _, repo_id, filename = spec.split(":", 2)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename)

    path = Path(spec)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def load_pretrained(
    base_ckpt: str,
    device: str = "cuda",
) -> tuple[torch.nn.Module, torch.nn.Module, object, dict]:
    """Load (policy_model, text_encoder, scheduler, ckpt_meta).

    The policy_model is returned in train mode; the caller decides whether
    to also load a frozen reference copy.
    """
    _, TextConditionedUNet, SimpleDDPMScheduler, CLIPTextEncoder = _import_tcd()

    ckpt_path = resolve_checkpoint(base_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device)

    ckpt_cfg = ckpt.get("config", {}) or {}
    text_dim = ckpt_cfg.get("text_dim", 512)
    clip_model = ckpt_cfg.get("clip_model", "openai/clip-vit-base-patch32")

    model = TextConditionedUNet(text_dim=text_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    text_encoder = CLIPTextEncoder(model_name=clip_model, freeze=True).to(device)
    text_encoder.eval()

    scheduler = SimpleDDPMScheduler(num_timesteps=1000)

    meta = {
        "ckpt_path": ckpt_path,
        "text_dim": text_dim,
        "clip_model": clip_model,
        "epoch": ckpt.get("epoch"),
        "loss": ckpt.get("loss"),
    }
    return model, text_encoder, scheduler, meta


def clone_frozen(model: torch.nn.Module) -> torch.nn.Module:
    """Return a deep-copied, frozen, eval-mode clone of `model` on the same device."""
    import copy

    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def pick_device(preferred: str = "cuda") -> str:
    """Resolve a device string with sensible fallbacks (cuda -> mps -> cpu)."""
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred in ("cuda", "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
