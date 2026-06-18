# Plan: Exploring RL for Shaping Diffusion Model Outputs

> **Status (2026-06-18):** Phase A–F complete. The β baseline pilot ran and
> confirmed all hypotheses — see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (Experiment 1)
> and [PILOT_RESULTS.md](PILOT_RESULTS.md). Key finding: `vsym_l2`'s optimum is a
> blank symmetric blob, so no β yields a usable model. Active direction is now
> **reward redesign** (Experiment 2), not further β tuning.

## Purpose — read this first

This is an **exploratory, learning-driven project**, not a feature-delivery task. The goal is to build genuine understanding of how reinforcement learning (specifically GRPO) reshapes a pretrained diffusion model, and to share that understanding with others. We choose **vertical symmetry as the reward task** because it is concrete, accessible, visually obvious, and rich enough to expose interesting RL dynamics (notably, reward hacking via mode collapse). Symmetry is the *vehicle*; understanding is the *destination*.

Concretely, success means:
1. Being able to *see* what β (the KL-to-base coefficient) actually does to a diffusion model, not just read about it.
2. Producing a repeatable harness so others can swap in their own rewards / hyperparameters and explore further.
3. A small portfolio of trained models and side-by-side visualizations that tell a story.

Completing a single training run is not success. Understanding *why* runs differ is success.

---

## Algorithm: GRPO (with rationale)

We use **GRPO** (Group Relative Policy Optimization). The standard choice for RL-finetuning diffusion is DDPO / DPOK (PPO with a learned critic). GRPO replaces the critic with a group baseline — for each prompt, sample K trajectories and use `mean(reward)` as the baseline, `(r_i − μ) / (σ + ε)` as the advantage. This is a particularly good fit for diffusion because:

- Sampling K rollouts per prompt is cheap relative to training a critic on noisy intermediate states `x_t`.
- A critic on `x_t` is hard to learn well (the value depends on the whole remaining trajectory).
- Removing the critic removes a major source of hyperparameter sensitivity.

We acknowledge in any write-up that this is a deliberate alternative to DDPO; we are not claiming GRPO is universally better, only that it fits this exploration well.

---

## What we are studying

Primary axis: **β** (KL-to-reference coefficient). Pilot values: `{0.0, 0.04, 0.4}` — pure reward chasing, typical PPO value, heavy regularization.

Secondary axis (deferred until after pilot): **K** (group size). Candidates: `{4, 8, 16}`.

Reward (fixed for now, single function): `vsym_l2(x) = −‖x − hflip(x)‖₂`, the negated L2 distance between the image and its horizontal-flip. Deliberately **without anti-collapse safeguards**, because watching collapse happen at low β is itself one of the most pedagogically valuable outcomes. Additional rewards can be plugged in later via `rl/rewards.py`'s registry.

Reference model: the pretrained text-conditional UNet from the existing repo (also published at `huggingface.co/jamesaasher/quickdraw-text-diffusion`). The reference is frozen for the entire run — no periodic refresh.

---

## Architecture

```
rl/
  config.py            # defaults: β=0.04, K=8, T_inf=50, lr=1e-5, n_steps=200, eps_clip=0.2
  scheduler_ext.py     # p_step_with_logprob, posterior_mean_variance, respaced DDPM
  rewards.py           # vsym_l2 (registered in a dict for future extension)
  trainer.py           # GRPO loop; reads run-config; writes runs/<id>/
  sweep.py             # orchestrates pilot then staircase
  metrics.py           # symmetry score, KL-to-base, diversity, per-class breakdown
  eval.py              # fixed-seed sample grids + metric computation on a checkpoint
dashboard/
  server.py            # FastAPI: GET /runs, /runs/{id}/metrics, /runs/{id}/samples/{step}
  static/
    index.html         # Bootstrap 5 layout
    app.js             # jQuery + Chart.js: list runs, plot curves, show sample grids
runs/                  # outputs (per the on-disk contract below)
plans/
  PLAN_RL.md           # this document
```

Why a separate `rl/` folder (not `grpo/`): leaves room to add another algorithm later for comparison, without renaming.

Why a custom FastAPI + Bootstrap + jQuery dashboard (not Streamlit/Gradio): user preference; also gives full control over layout and lets the dashboard double as a teaching artifact.

---

## On-disk run contract

Every run writes a self-contained directory the dashboard consumes:

```
runs/<run_id>/
  config.json          # full hyperparameters: β, K, reward, T_inf, base_ckpt, seed, lr, ...
  log.jsonl            # one JSON per training step:
                       #   {step, reward_mean, reward_std, kl_to_ref,
                       #    clip_frac, diversity, per_class_reward, lr}
  checkpoints/step_<N>.pt
  samples/step_<N>/    # PNG grid: rows = fixed prompts, cols = fixed seeds
  final_metrics.json   # summary computed once at end of run
```

Run IDs are human-readable, e.g. `pilot_beta0.04_k8_2026-05-27_14-30`.

The dashboard reads only from this directory structure — the trainer and harness communicate purely through these files, no shared in-memory state.

---

## Phase-by-phase work

### Phase A — Eval harness (built first)

Build the metrics and dashboard *before* the trainer so the trainer is forced to log against a contract that already exists.

- `rl/metrics.py`: pure functions taking `(x: Tensor[B,1,64,64], prompts, class_ids, ref_model?)` → scalar/dict.
  - `symmetry_l2`, `symmetry_l1`, `symmetry_ssim` (multiple so we can cross-check the chosen reward).
  - `kl_to_ref` — closed-form Gaussian KL averaged over a held-out prompt set.
  - `diversity` — mean pairwise pixel L2 across the sample batch (the collapse detector).
  - `per_class_reward` — break the above down by class id.
- `rl/eval.py`: load a checkpoint, generate `N_prompts × M_seeds` samples at fixed seeds, run all metrics, save `samples/step_<N>/` + a metrics dict.
- `dashboard/`: FastAPI app with three endpoints (`/runs`, `/runs/{id}/metrics`, `/runs/{id}/samples/{step}`) and a single-page UI showing (a) a run picker, (b) overlaid metric curves for selected runs, (c) sample grids stepping through time.

**Validation for Phase A**: run `rl/eval.py` on the pretrained base checkpoint; dashboard should display its (flat, single-point) curves and a sample grid. If this works, the contract is sound.

### Phase B — Scheduler extension

Add to `rl/scheduler_ext.py` (do not modify `text_conditional_diffusion/scheduler.py`):

- `posterior_mean_variance(x_t, x0_pred, t)` — `(mean, var)` of `q(x_{t-1} | x_t, x0)`.
- `predict_x0_from_eps(x_t, eps, t)`.
- `p_step_with_logprob(model, x_t, t, text_emb, cfg_scale, generator)` → `(x_prev, log_prob, mean, std)`.
- `make_respaced_timesteps(num_inference_steps)` — subsamples the 1000-step DDPM schedule down to ~50 steps with adjusted αs.

**Validation for Phase B**: parity test. With a fixed seed, full-DDPM sampling via `p_step_with_logprob` (ignoring the log-prob output) must match the existing `p_sample_text` within 1e-5.

### Phase C — Rewards

`rl/rewards.py`: a `REWARDS: dict[str, Callable]` registry. Implements `vsym_l2` only. Each callable signature: `(x0: Tensor[B,1,H,W], prompts: list[str], meta: dict) → Tensor[B]`. Unit-tested on synthetic symmetric (all-zero, mirrored noise) and asymmetric (gradient) inputs.

### Phase D — GRPO trainer

`rl/trainer.py` reads a run-config dict and runs end-to-end:

1. Load `policy` (trainable) and `ref` (frozen, `eval()`, no grad) from the base checkpoint. Frozen CLIP shared.
2. **Outer loop** (n_steps total): each iteration is one rollout + update cycle.
3. **Rollout**: sample `B_prompts` prompts × K rollouts each → 50-step respaced DDPM trajectories via `p_step_with_logprob`. Cache `{x_t, t, text_emb, x_prev, log_prob_old}` per step on CPU. Compute terminal reward on `x_0`.
4. **Advantage**: per-prompt group normalization `A_i = (r_i − μ) / (σ + ε)`. Broadcast across all timesteps of that rollout.
5. **Update** (a few PPO inner epochs): recompute `log_prob_new`, importance ratio `ρ`, clipped surrogate `−min(ρA, clip(ρ, 1−ε, 1+ε) A)`, plus closed-form Gaussian `β · KL(policy ‖ ref)` per step. Adam, grad-clip 1.0.
6. **Logging**: append one JSONL row per outer step with all metrics. Every `eval_every` steps, call `rl/eval.py` to dump a sample grid + metric snapshot.
7. **Checkpointing**: same cadence as eval.

### Phase E — Sweep runner & the pilot

`rl/sweep.py` reads a YAML/JSON listing run configs, launches them sequentially, names directories deterministically.

**Pilot (3 runs)**:

| Run | β | K | n_steps | T_inf | Hypothesis |
|---|---|---|---|---|---|
| `pilot_beta0.00_k8` | 0.0 | 8 | 200 | 50 | Reward spikes; diversity collapses; samples become near-blank symmetric blobs |
| `pilot_beta0.04_k8` | 0.04 | 8 | 200 | 50 | Steady reward gain; diversity holds; samples become recognizable + symmetric |
| `pilot_beta0.40_k8` | 0.4 | 8 | 200 | 50 | Reward barely moves; KL stays near zero; samples ≈ base |

After the pilot, **stop, inspect the dashboard, write down observations**, then decide what to do next (extend β grid, sweep K, add an anti-collapse reward, etc.). This pause is a project requirement, not optional.

### Phase F — Iteration & write-up

Driven by what the pilot shows. Examples of likely follow-ups:
- Add `vsym_plus_clip` to demonstrate anti-collapse.
- Sweep K at the best β.
- Try a longer training run at the best config to see whether trends continue.
- Compare with DDPO (much bigger scope — only if curiosity demands).

---

## Pilot success criteria

The pilot is a success if **all three** are true:
1. The three runs produce visibly different sample grids in the dashboard at step 200.
2. The metric curves (`reward_mean`, `kl_to_ref`, `diversity`) tell internally consistent stories — e.g., β=0 shows high reward, high KL, low diversity; β=0.4 shows low reward gain, low KL, stable diversity.
3. We can articulate, in plain English in `runs/PILOT_NOTES.md`, what we learned and what we want to vary next.

If any of these fails, the next action is debugging, not expanding the sweep.

---

## Risks & how we'll handle them

- **Compute**: pilot is 3 runs × ~30-60 min each. If a run exceeds 90 min, reduce `n_steps` or `T_inf` and re-run rather than waiting.
- **Reward hacking via blank canvases at low β**: expected and desired — it is part of what we want to show. Diversity metric must catch it; sample grid must visualize it.
- **Trajectory memory blowup** (`B_prompts × K × T_inf × 1 × 64 × 64`): cache on CPU; if still too big, fall back to recomputing trajectories in the PPO inner loop.
- **Trainer / harness drift**: the on-disk contract is the single source of truth. Anything not written to `runs/<id>/` cannot influence the report.

---

## Out of scope (for now)

- DDIM (deferred; respaced DDPM is the rollout sampler).
- DDPO / value-network methods (potential follow-up only).
- Distributed training (single-machine, single-process; trainer must run on the same hardware as the original 4-hr training without modification).
- ~~Reward functions other than `vsym_l2`.~~ **Now in scope** as of Experiment 2 —
  the pilot showed `vsym_l2` alone collapses, so reward redesign is the next step
  (see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)).
- Multi-axis sweeps before the pilot is digested.

---

## Open decisions (resolved)

- Algorithm: **GRPO**.
- Reward: **`vsym_l2`** only; no anti-collapse safeguard in pilot.
- β grid for pilot: **{0.0, 0.04, 0.4}** at K=8.
- Dashboard: **FastAPI + Bootstrap + jQuery + Chart.js**, custom localhost app.
- Base checkpoint: pretrained text-conditional UNet, either from local `text_conditional_diffusion/checkpoints/` or downloaded via `huggingface_hub` from `jamesaasher/quickdraw-text-diffusion`.
- Reference model: frozen for the whole run; no periodic refresh.
- Folder names: `rl/`, `dashboard/`, `runs/`.
- Build order: **harness before trainer**.
- After pilot: **stop and reflect** before any sweep expansion.
