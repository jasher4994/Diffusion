"""Evaluation harness for diffusion checkpoints.

Generates a fixed-seed sample grid for a given checkpoint and writes the
per-run directory contract described in plans/PLAN_RL.md:

    runs/<run_id>/
      config.json
      samples/step_<N>/
        prompt_<i>_seed_<j>.png        # individual samples
        grid.png                       # composite (rows=prompts, cols=seeds)
        metrics.json                   # snapshot computed on this sample set

Usable standalone via `python -m rl.eval --base-ckpt ... --run-id ...` to
seed the dashboard with a baseline run before any training happens.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from rl import config as rl_config
from rl import metrics as rl_metrics
from rl.model_loader import load_pretrained, pick_device


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    scheduler,
    prompts: Sequence[str],
    n_seeds: int,
    cfg_scale: float,
    device: str,
    image_size: int = rl_config.IMAGE_SIZE,
    base_seed: int = 0,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Generate `len(prompts) * n_seeds` samples deterministically.

    Returns:
        images:  [N, 1, H, W] in [-1, 1] (raw model output, clamped).
        prompt_per_sample: length-N list of the prompt used for each row.
        seed_per_sample:   length-N list of the seed used for each sample.
    """
    model.eval()
    images = []
    prompt_per_sample: list[str] = []
    seed_per_sample: list[int] = []

    # Encode every prompt once.
    text_emb = text_encoder(list(prompts))  # [P, 512]

    for j in range(n_seeds):
        seed = base_seed + j
        # Deterministic initial noise *per (prompt, seed)*: stack into one batch.
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_T = torch.randn(
            (len(prompts), 1, image_size, image_size),
            generator=gen,
        ).to(device)

        # Run the existing sampler.
        x0 = _sample_from(scheduler, model, x_T, text_emb, cfg_scale, device)
        images.append(x0.cpu())
        prompt_per_sample.extend(list(prompts))
        seed_per_sample.extend([seed] * len(prompts))

    return torch.cat(images, dim=0), prompt_per_sample, seed_per_sample


def _sample_from(scheduler, model, x_T, text_emb, cfg_scale, device):
    """Run scheduler.sample_text but starting from a provided x_T.

    The upstream `sample_text` always re-rolls noise internally; we
    replicate its loop to inject our own deterministic start.
    """
    from tqdm import tqdm

    img = x_T.to(device)
    b = img.shape[0]
    for i in tqdm(
        reversed(range(scheduler.num_timesteps)),
        total=scheduler.num_timesteps,
        desc=f"sample[B={b}]",
        leave=False,
    ):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = scheduler.p_sample_text(model, img, t, text_emb, cfg_scale)
        img = torch.clamp(img, -2.0, 2.0)
    return img


# ---------------------------------------------------------------------------
# Sample-grid writing
# ---------------------------------------------------------------------------


def _to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a single [1,H,W] tensor in [-1,1] to a PIL L-mode image."""
    arr = ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).squeeze(0).cpu().numpy()
    return Image.fromarray(arr, mode="L")


def write_grid(
    images: torch.Tensor,
    prompts_per_sample: list[str],
    seeds_per_sample: list[int],
    out_dir: Path,
) -> Path:
    """Write per-sample PNGs and a composite `grid.png`.

    Grid layout: rows = unique prompts (in first-seen order),
                 cols = unique seeds (sorted ascending).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_prompts: list[str] = []
    for p in prompts_per_sample:
        if p not in unique_prompts:
            unique_prompts.append(p)
    unique_seeds = sorted(set(seeds_per_sample))

    # Build a lookup: (prompt, seed) -> tensor index
    idx_of = {
        (p, s): i for i, (p, s) in enumerate(zip(prompts_per_sample, seeds_per_sample))
    }

    h = w = images.shape[-1]
    rows, cols = len(unique_prompts), len(unique_seeds)
    grid = Image.new("L", (cols * w, rows * h), color=255)

    for r, prompt in enumerate(unique_prompts):
        safe = "".join(c if c.isalnum() else "_" for c in prompt)[:60]
        for c, seed in enumerate(unique_seeds):
            i = idx_of[(prompt, seed)]
            tile = _to_pil(images[i])
            tile.save(out_dir / f"prompt_{r:02d}_{safe}_seed_{seed}.png")
            grid.paste(tile, (c * w, r * h))

    grid_path = out_dir / "grid.png"
    grid.save(grid_path)

    # Manifest helps the dashboard render labels.
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows": unique_prompts,
                "cols": unique_seeds,
                "tile_size": h,
            },
            indent=2,
        )
    )
    return grid_path


# ---------------------------------------------------------------------------
# Run-directory ops
# ---------------------------------------------------------------------------


@dataclass
class EvalArtifacts:
    sample_dir: Path
    metrics_path: Path
    grid_path: Path
    metrics: dict


def evaluate_checkpoint(
    *,
    model: torch.nn.Module,
    text_encoder: torch.nn.Module,
    scheduler,
    prompts: Sequence[str],
    n_seeds: int,
    cfg_scale: float,
    device: str,
    run_dir: Path,
    step: int,
    base_seed: int = 0,
) -> EvalArtifacts:
    """Generate samples + write PNGs + write metrics.json. Returns paths."""
    images, p_per, s_per = generate_samples(
        model,
        text_encoder,
        scheduler,
        prompts=prompts,
        n_seeds=n_seeds,
        cfg_scale=cfg_scale,
        device=device,
        base_seed=base_seed,
    )
    sample_dir = run_dir / "samples" / f"step_{step:06d}"
    grid_path = write_grid(images, p_per, s_per, sample_dir)
    m = rl_metrics.all_image_metrics(images.to(device), prompts=p_per)
    m["step"] = step
    metrics_path = sample_dir / "metrics.json"
    metrics_path.write_text(json.dumps(m, indent=2))
    return EvalArtifacts(sample_dir, metrics_path, grid_path, m)


def write_run_skeleton(run_dir: Path, cfg_dict: dict) -> None:
    """Create run_dir and write its config.json + an empty log.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    (run_dir / "log.jsonl").touch()
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-ckpt",
        default=rl_config.DEFAULT_BASE_CKPT,
        help="Local path or 'hf:repo:filename'.",
    )
    p.add_argument(
        "--run-id", default="base", help="Directory name under runs/ to write into."
    )
    p.add_argument("--runs-dir", default=rl_config.RUNS_DIR)
    p.add_argument("--prompts", nargs="+", default=list(rl_config.TRAINED_PROMPTS))
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument(
        "--step",
        type=int,
        default=0,
        help="Step label for this snapshot (use 0 for a 'base' eval).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--base-seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Loading model on {device} from {args.base_ckpt}")
    model, text_encoder, scheduler, meta = load_pretrained(
        args.base_ckpt, device=device
    )
    print(f"Loaded ckpt meta: {meta}")

    run_dir = Path(args.runs_dir) / args.run_id
    cfg_record = {
        "run_id": args.run_id,
        "base_ckpt": args.base_ckpt,
        "ckpt_meta": meta,
        "eval_prompts": list(args.prompts),
        "eval_n_seeds": args.n_seeds,
        "cfg_scale": args.cfg_scale,
        "kind": "eval-only",  # marks runs that have no training log
    }
    write_run_skeleton(run_dir, cfg_record)

    print(f"Generating {len(args.prompts) * args.n_seeds} samples...")
    art = evaluate_checkpoint(
        model=model,
        text_encoder=text_encoder,
        scheduler=scheduler,
        prompts=args.prompts,
        n_seeds=args.n_seeds,
        cfg_scale=args.cfg_scale,
        device=device,
        run_dir=run_dir,
        step=args.step,
        base_seed=args.base_seed,
    )
    print(f"Wrote samples to: {art.sample_dir}")
    print(f"Metrics: {json.dumps(art.metrics, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
