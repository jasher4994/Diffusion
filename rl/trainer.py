"""GRPO trainer for diffusion fine-tuning.

Per outer step:
  1. ROLLOUT (no grad): pick `prompts_per_step` prompts, generate `group_size` samples
     per prompt with the current policy over T_inf respaced steps. Store the trajectory
     (x_t, x_prev, log_prob_old) for every step.
  2. SCORE: reward(x_0) → group-relative advantage A.
  3. UPDATE (with grad): replay each stored (x_t, step) through the policy with grad,
     compute log_prob_new, ratio, PPO-clipped surrogate, closed-form Gaussian KL to
     the frozen reference. Sum across timesteps via gradient accumulation. One Adam step.

Periodically calls rl/eval.py to dump a sample grid into samples/step_<N>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from rl import config as rl_config
from rl.config import RunConfig
from rl.model_loader import clone_frozen, load_pretrained, pick_device
from rl.rewards import REWARDS
from rl.scheduler_ext import RLScheduler
from rl.eval import evaluate_checkpoint, write_run_skeleton


# ---------------------------------------------------------------------------
# Trajectory storage
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Trajectory:
    """Stored data from one rollout (all on CPU to keep memory tight)."""

    x_ts: list[torch.Tensor]  # length T_inf, each [B, C, H, W]
    x_prevs: list[torch.Tensor]  # length T_inf, each [B, C, H, W]
    log_probs_old: list[torch.Tensor]  # length T_inf, each [B]
    text_emb: torch.Tensor  # [B, text_dim]
    prompt_idx: torch.Tensor  # [B] — which prompt each row corresponds to
    final_x0: torch.Tensor  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    rls: RLScheduler,
    text_emb: torch.Tensor,
    prompt_idx: torch.Tensor,
    cfg_scale: float,
    device: str,
    image_size: int,
    seed: int,
) -> Trajectory:
    B = text_emb.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((B, 1, image_size, image_size), generator=gen, device=device)

    x_ts: list[torch.Tensor] = []
    x_prevs: list[torch.Tensor] = []
    log_probs_old: list[torch.Tensor] = []

    for step_idx in range(rls.num_inference_steps):
        x_t_in = x  # what we feed the model
        x_prev, log_prob, _mean, _std = rls.p_step_with_logprob(
            model,
            x,
            step_idx,
            text_emb,
            cfg_scale=cfg_scale,
            generator=gen,
        )
        x_ts.append(x_t_in.detach().cpu())
        x_prevs.append(x_prev.detach().cpu())
        log_probs_old.append(log_prob.detach().cpu())
        x = x_prev
        x = torch.clamp(x, -2.0, 2.0)  # matches sample_text

    return Trajectory(
        x_ts=x_ts,
        x_prevs=x_prevs,
        log_probs_old=log_probs_old,
        text_emb=text_emb.detach().cpu(),
        prompt_idx=prompt_idx.detach().cpu(),
        final_x0=x.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Advantage
# ---------------------------------------------------------------------------


def group_relative_advantages(
    rewards: torch.Tensor, prompt_idx: torch.Tensor
) -> torch.Tensor:
    """A_i = r_i - mean(r_j : prompt_idx[j] == prompt_idx[i])."""
    adv = torch.zeros_like(rewards)
    for p in prompt_idx.unique():
        mask = prompt_idx == p
        adv[mask] = rewards[mask] - rewards[mask].mean()
    return adv


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------


def update_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rls: RLScheduler,
    traj: Trajectory,
    advantages: torch.Tensor,  # [B]
    *,
    eps_clip: float,
    beta: float,
    cfg_scale: float,
    device: str,
) -> dict:
    """One GRPO gradient step. Returns metrics dict."""
    model.train()
    advantages = advantages.to(device)
    text_emb = traj.text_emb.to(device)
    T = rls.num_inference_steps

    sum_loss_pg = 0.0
    sum_loss_kl = 0.0
    sum_ratio = 0.0
    sum_clip_frac = 0.0
    sum_logp_diff_sq = 0.0
    n_pg_steps = 0  # number of (step) contributions with non-zero log-prob

    for step_idx in range(T):
        x_t = traj.x_ts[step_idx].to(device)
        x_prev = traj.x_prevs[step_idx].to(device)
        log_prob_old = traj.log_probs_old[step_idx].to(device)

        # Re-score the action under the CURRENT policy (with grad).
        _x_prev, log_prob_new, mean_new, std = rls.p_step_with_logprob(
            model,
            x_t,
            step_idx,
            text_emb,
            cfg_scale=cfg_scale,
            generator=None,
            prev_sample=x_prev,
        )

        # Reference policy's mean (no grad).
        with torch.no_grad():
            t_model = torch.full(
                (x_t.shape[0],),
                int(rls.timesteps[step_idx]),
                device=device,
                dtype=torch.long,
            )
            eps_ref = ref_model(x_t, t_model, text_emb)
            if cfg_scale > 1.0:
                uncond = torch.zeros_like(text_emb)
                eps_ref_u = ref_model(x_t, t_model, uncond)
                eps_ref = eps_ref_u + cfg_scale * (eps_ref - eps_ref_u)
            mean_ref, variance = rls.posterior_mean_variance(x_t, eps_ref, step_idx)

        # Skip the last (deterministic) step for both PG and KL.
        if step_idx == T - 1:
            continue

        # --- Policy gradient term -----------------------------------------
        # ratio = π_θ / π_θ_old, advantage broadcasts across batch.
        log_ratio = log_prob_new - log_prob_old
        ratio = log_ratio.exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - eps_clip, 1 + eps_clip) * advantages
        loss_pg_step = -torch.min(unclipped, clipped).mean()

        # --- KL term -------------------------------------------------------
        # For 𝒩(μ_θ, σ²·I) and 𝒩(μ_ref, σ²·I), KL = ‖μ_θ - μ_ref‖² / (2σ²).
        kl_per_sample = ((mean_new - mean_ref) ** 2).flatten(1).sum(dim=1) / (
            2.0 * variance
        )
        loss_kl_step = kl_per_sample.mean()

        loss_step = loss_pg_step + beta * loss_kl_step
        # Average over timesteps so the overall loss magnitude is independent of T.
        (loss_step / (T - 1)).backward()

        # --- Diagnostics ---------------------------------------------------
        sum_loss_pg += loss_pg_step.detach().item()
        sum_loss_kl += loss_kl_step.detach().item()
        sum_ratio += ratio.detach().mean().item()
        sum_clip_frac += (
            ((ratio.detach() < 1 - eps_clip) | (ratio.detach() > 1 + eps_clip))
            .float()
            .mean()
            .item()
        )
        sum_logp_diff_sq += (log_ratio.detach() ** 2).mean().item()
        n_pg_steps += 1

    denom = max(1, n_pg_steps)
    return {
        "loss_pg": sum_loss_pg / denom,
        "loss_kl": sum_loss_kl / denom,
        "ratio_mean": sum_ratio / denom,
        "clip_frac": sum_clip_frac / denom,
        "logp_diff_sq_mean": sum_logp_diff_sq / denom,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: RunConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = pick_device(cfg.device)
    print(f"[trainer] device={device}")

    # Load policy + frozen reference.
    model, text_encoder, base_scheduler, meta = load_pretrained(
        cfg.base_ckpt, device=device
    )
    print(f"[trainer] loaded base ckpt: {meta}")
    ref_model = clone_frozen(model)

    rls = RLScheduler(base_scheduler, num_inference_steps=cfg.t_inf)
    print(
        f"[trainer] using {rls.num_inference_steps} respaced timesteps "
        f"(every {1000 // cfg.t_inf}-th)"
    )

    reward_fn = REWARDS[cfg.reward_name]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Run skeleton.
    run_dir = Path(rl_config.RUNS_DIR) / cfg.run_id
    cfg_record = cfg.to_dict()
    cfg_record["ckpt_meta"] = meta
    cfg_record["kind"] = "training"
    write_run_skeleton(run_dir, cfg_record)
    log_path = run_dir / "log.jsonl"

    # Encode trained prompts once.
    all_prompts = list(rl_config.TRAINED_PROMPTS)
    all_text_emb = text_encoder(all_prompts)  # [P_all, text_dim]

    def baseline_eval(step: int) -> dict:
        model.eval()
        art = evaluate_checkpoint(
            model=model,
            text_encoder=text_encoder,
            scheduler=base_scheduler,
            prompts=cfg.eval_prompts,
            n_seeds=cfg.eval_n_seeds,
            cfg_scale=cfg.cfg_scale,
            device=device,
            run_dir=run_dir,
            step=step,
            base_seed=0,
        )
        model.train()
        return art.metrics

    print(
        f"[trainer] running step-0 baseline eval ({len(cfg.eval_prompts)*cfg.eval_n_seeds} samples)..."
    )
    eval0 = baseline_eval(0)
    print(
        f"[trainer] step 0 baseline: symmetry_l2_mean={eval0['symmetry_l2_mean']:.4f}"
    )

    for step in tqdm(range(1, cfg.n_steps + 1), desc="grpo"):
        t_start = time.time()
        # ---- Pick prompts for this rollout ----
        idx = torch.tensor(
            [random.randrange(len(all_prompts)) for _ in range(cfg.prompts_per_step)],
            dtype=torch.long,
        )
        # Repeat each prompt `group_size` times.
        prompt_idx = idx.repeat_interleave(cfg.group_size)  # [B]
        text_emb = all_text_emb[prompt_idx].to(device)  # [B, text_dim]

        # ---- Rollout (no grad) ----
        rollout_seed = cfg.seed * 100003 + step
        traj = rollout(
            model,
            rls,
            text_emb,
            prompt_idx,
            cfg_scale=cfg.cfg_scale,
            device=device,
            image_size=rl_config.IMAGE_SIZE,
            seed=rollout_seed,
        )

        # ---- Reward + advantage ----
        # Per-sample reward on the final clamped image. We work in image space [-1,1].
        rewards = reward_fn(traj.final_x0.to(device)).detach()  # [B]
        advantages = group_relative_advantages(rewards.cpu(), traj.prompt_idx)

        # ---- Update ----
        optimizer.zero_grad(set_to_none=True)
        upd = update_step(
            model,
            ref_model,
            rls,
            traj,
            advantages,
            eps_clip=cfg.eps_clip,
            beta=cfg.beta,
            cfg_scale=cfg.cfg_scale,
            device=device,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        ).item()
        optimizer.step()

        # ---- Log ----
        row = {
            "step": step,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std(unbiased=False).item(),
            "advantage_abs_mean": advantages.abs().mean().item(),
            "kl_to_ref": upd["loss_kl"],
            "loss_pg": upd["loss_pg"],
            "loss_kl": upd["loss_kl"],
            "loss_total": upd["loss_pg"] + cfg.beta * upd["loss_kl"],
            "ratio_mean": upd["ratio_mean"],
            "clip_frac": upd["clip_frac"],
            "logp_diff_sq_mean": upd["logp_diff_sq_mean"],
            "grad_norm": grad_norm,
            "lr": cfg.lr,
            "wall_secs": time.time() - t_start,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        # ---- Periodic eval ----
        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            print(f"\n[trainer] step {step} eval...")
            m = baseline_eval(step)
            print(
                f"[trainer] step {step}: reward={row['reward_mean']:.3f}  "
                f"kl={row['kl_to_ref']:.3f}  "
                f"sym_l2={m['symmetry_l2_mean']:.4f}"
            )

    # ---- Final checkpoint ----
    ckpt_path = run_dir / "checkpoints" / f"step_{cfg.n_steps:06d}.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": cfg.n_steps}, ckpt_path)
    print(f"[trainer] saved final checkpoint to {ckpt_path}")
    print("[trainer] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--t-inf", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n-seeds", type=int, default=8)
    p.add_argument(
        "--eval-prompts",
        nargs="+",
        default=None,
        help="Override eval prompts (default: all 5 trained prompts).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward", default="vsym_l2")
    args = p.parse_args()

    cfg = RunConfig(
        run_id=args.run_id,
        beta=args.beta,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        n_steps=args.n_steps,
        t_inf=args.t_inf,
        lr=args.lr,
        cfg_scale=args.cfg_scale,
        eval_every=args.eval_every,
        eval_n_seeds=args.eval_n_seeds,
        eval_prompts=(
            args.eval_prompts if args.eval_prompts else list(rl_config.TRAINED_PROMPTS)
        ),
        device=args.device,
        seed=args.seed,
        reward_name=args.reward,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
