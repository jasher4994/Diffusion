# RL-for-Diffusion: How the pieces fit together

## 1. Code & data dependencies (modules)

```mermaid
flowchart LR
    subgraph base["text_conditional_diffusion/ (frozen / read-only)"]
        TCD_MODEL["model.py<br/>TextConditionedUNet"]
        TCD_SCHED["scheduler.py<br/>DDPMScheduler<br/>(p_sample_text)"]
        TCD_TEXT["text_encoder.py<br/>CLIPTextEncoder"]
        TCD_CFG["config.py"]
    end

    subgraph rl["rl/ (new)"]
        RL_CFG["config.py<br/>RunConfig + TRAINED_PROMPTS"]
        RL_LOAD["model_loader.py<br/>load_pretrained + clone_frozen"]
        RL_METRICS["metrics.py<br/>symmetry_l2/l1/ssim<br/>diversity, KL"]
        RL_SCHED["scheduler_ext.py (Phase B)<br/>p_step_with_logprob<br/>respaced timesteps"]
        RL_REWARDS["rewards.py (Phase C)<br/>REWARDS registry<br/>vsym_l2"]
        RL_TRAIN["trainer.py (Phase D)<br/>GRPO loop"]
        RL_EVAL["eval.py<br/>generate + grid + metrics"]
        RL_SWEEP["sweep.py (Phase E)<br/>3-run β pilot"]
    end

    subgraph dash["dashboard/"]
        DASH_API["server.py<br/>FastAPI /api/runs/..."]
        DASH_UI["static/{index.html,app.js}<br/>Bootstrap + jQuery + Chart.js"]
    end

    HF[("HF Hub<br/>jamesaasher/quickdraw-text-diffusion")]
    RUNS[("runs/&lt;id&gt;/<br/>config.json, log.jsonl,<br/>samples/step_N/, checkpoints/")]

    HF --> RL_LOAD
    TCD_MODEL --> RL_LOAD
    TCD_SCHED --> RL_LOAD
    TCD_SCHED --> RL_SCHED
    TCD_TEXT --> RL_LOAD
    TCD_CFG --> RL_LOAD

    RL_LOAD --> RL_EVAL
    RL_LOAD --> RL_TRAIN
    RL_METRICS --> RL_EVAL
    RL_METRICS --> RL_TRAIN
    RL_METRICS --> RL_REWARDS
    RL_SCHED --> RL_TRAIN
    RL_REWARDS --> RL_TRAIN
    RL_CFG --> RL_EVAL
    RL_CFG --> RL_TRAIN
    RL_CFG --> RL_SWEEP
    RL_TRAIN --> RL_SWEEP
    RL_EVAL --> RL_TRAIN

    RL_EVAL --> RUNS
    RL_TRAIN --> RUNS
    RUNS --> DASH_API
    DASH_API --> DASH_UI
```

## 2. Per-step GRPO training loop (runtime)

```mermaid
flowchart TD
    A["Sample K prompts × 1 (or 1 prompt × K)<br/>same prompt → group"] --> B
    B["Rollout: T_inf=50 respaced steps<br/>p_step_with_logprob → x_0<br/>store (x_t, t, a_t, log π_θ_old) per step"] --> C
    C["Reward: r = vsym_l2(x_0)<br/>per-sample scalar"]
    C --> D["Group baseline:<br/>A_i = r_i − mean(r_group)<br/>(optionally / std)"]
    D --> E["Recompute log π_θ(a_t|x_t,t)<br/>on stored trajectory"]
    E --> F["PPO surrogate:<br/>L_pg = −E[min(ρ·A, clip(ρ,1±ε)·A)]<br/>where ρ = π_θ / π_θ_old"]
    E --> G["Ref-model log π_ref<br/>(frozen clone, no grad)"]
    G --> H["KL = closed-form Gauss<br/>KL(π_θ ∥ π_ref) per step"]
    F --> I["L = L_pg + β · KL"]
    H --> I
    I --> J["Adam step on θ<br/>grad-clip 1.0"]
    J --> K{"step % eval_every<br/>== 0?"}
    K -- yes --> L["rl/eval.py:<br/>generate 40 samples,<br/>write samples/step_N/"]
    K -- no --> A
    L --> A
```

## 3. β-pilot orchestration (Phase E)

```mermaid
flowchart LR
    SWEEP["sweep.py"] --> R1["pilot_beta0.00_k8<br/>200 steps"]
    SWEEP --> R2["pilot_beta0.04_k8<br/>200 steps"]
    SWEEP --> R3["pilot_beta0.40_k8<br/>200 steps"]
    R1 --> RUNS[("runs/pilot_*/")]
    R2 --> RUNS
    R3 --> RUNS
    RUNS --> DASH["dashboard<br/>overlay 3 runs<br/>reward / KL / diversity / per-pixel std"]
    DASH --> WRITEUP["Phase F: write-up<br/>β trade-off observations"]
```

## 4. On-disk contract per run

```
runs/<run_id>/
├── config.json              # RunConfig snapshot
├── log.jsonl                # one row per training step
│                            # {step, reward_mean, kl_to_ref, loss_pg, loss_kl, grad_norm, lr}
├── checkpoints/
│   └── step_<N>.pt          # policy weights (optional, periodic)
├── samples/
│   └── step_<N>/
│       ├── grid.png         # 5 prompts × 8 seeds composite
│       ├── manifest.json    # rows=prompts, cols=seeds, tile_size
│       └── metrics.json     # symmetry_l2/l1/ssim_mean + by_prompt, diversity, per_pixel_std
└── final_metrics.json
```
