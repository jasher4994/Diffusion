# What we learned in this session

Built up from "where does GRPO fit?" to "the policy *is* the per-pixel Gaussian."

---

## 1. Pretraining vs GRPO — same net, different loop

```mermaid
flowchart TB
    subgraph PRE["Pretraining"]
        P1["real x_0"] --> P2["add noise"]
        P2 --> P3["UNet → ε̂"]
        P3 --> P4["MSE loss"]
        P4 --> P5["backprop"]
        P5 -.-> P1
    end

    subgraph GRPO["GRPO fine-tune"]
        G1["self-generate"] --> G2["score r"]
        G2 --> G3["advantage A"]
        G3 --> G4["PG + KL loss"]
        G4 --> G5["backprop"]
        G5 -.-> G1
    end

    PRE -.->|"init weights"| GRPO
```

Same UNet, same backprop, same optimizer. **Only the loss formula changes.**

---

## 2. Why we can't backprop through symmetry

```mermaid
flowchart LR
    THETA["θ"] --> UNET["UNet"]
    UNET --> MU["μ_θ"]
    MU --> SAMP["sample"]
    SAMP --> X0["x_0"]
    X0 --> SYM["symmetry"]

    RAND["randn()"] -.->|"breaks chain"| SAMP

    style RAND fill:#fdd,stroke:#900
    style SAMP fill:#fdd,stroke:#900
```

`randn()` is not a function of θ. Chain rule stops there. → use policy gradients instead.

---

## 3. The policy = per-pixel Gaussian

```mermaid
flowchart TB
    IN["x_t, t, text"] --> UNET["UNet_θ"]
    UNET --> EPS["ε̂<br/>per-pixel"]
    EPS --> S1["x̂_0"]
    S1 --> S2["μ̃"]
    IN --> S2
    BETA["β̃"] --> G["π_θ = 𝒩(μ̃, β̃·I)"]
    S2 --> G
    G --> SAMPLE["sample x_prev"]

    style G fill:#dfd,stroke:#070,stroke-width:3px
```

UNet → per-pixel noise. Scheduler → per-pixel mean. **That Gaussian is the policy.**

---

## 4. How log π_θ(action) is computed

```mermaid
flowchart LR
    XT["x_t<br/>stored"] --> FWD["UNet fwd<br/>w/ grad"]
    FWD --> MU["μ̃_θ"]
    XPREV["x_prev<br/>stored"] --> DIFF["x_prev − μ̃"]
    MU --> DIFF
    DIFF --> SQ["sum sq"]
    SQ --> LOGP["log π_θ"]

    style LOGP fill:#dfd,stroke:#070,stroke-width:3px
```

Scaled negative MSE between action and predicted mean. **Differentiable in θ.**

---

## 5. The GRPO loss

```mermaid
flowchart TB
    R["reward r"] --> A["A = r − mean"]
    LOGP["log π_θ"] --> RATIO["ρ"]
    LOGPOLD["log π_old"] --> RATIO
    A --> MULT["ρ·A"]
    RATIO --> MULT
    A --> CLIP["clip(ρ)·A"]
    RATIO --> CLIP
    MULT --> MIN["min"]
    CLIP --> MIN
    MIN --> LPG["L_PG"]

    MUNEW["μ_θ"] --> KL["β · KL"]
    MUREF["μ_ref"] --> KL

    LPG --> LOSS["loss"]
    KL --> LOSS
    LOSS --> BACK["backprop"]

    style LOSS fill:#dfd,stroke:#070,stroke-width:3px
```

PG term + KL anchor. Clip caps per-step jumps; KL caps long-term drift.

---

## 6. What's differentiable?

```mermaid
flowchart TB
    Q1["smooth?"]
    Q1 -- no --> NO1["not diff"]
    Q1 -- yes --> Q2["depends on θ?"]
    Q2 -- no --> NO2["constant"]
    Q2 -- yes --> YES["grad flows"]

    style YES fill:#dfd
    style NO1 fill:#fdd
    style NO2 fill:#fed
```

Dice rolls fail **Q2**, not Q1.

- **not diff**: argmax, if, round, discrete sample
- **constant**: stored tensors, rewards, randn output
- **grad flows**: UNet, μ_θ, log π_θ, KL

---

## 7. Full GRPO step

```mermaid
flowchart LR
    subgraph ROLL["Rollout — no grad"]
        R1["50 sample steps"] --> R2["store traj"]
        R2 --> R3["reward → A"]
    end

    subgraph UPD["Update — with grad"]
        U1["replay through UNet"] --> U2["log π_θ"]
        U2 --> U3["PG + KL"]
        U3 --> U4["backprop"]
    end

    ROLL ==>|"frozen data"| UPD
```

Rollout = make data. Update = feed back, weight by advantage, backprop.

---

## TL;DR

| Concept | Resolution |
|---|---|
| GRPO vs pretraining | Same net + backprop, different scalar |
| Why not `-symmetry().backward()`? | randn breaks the gradient chain |
| What's the policy? | Per-pixel Gaussian the UNet+scheduler defines |
| What's P(action)? | Density of that Gaussian at the sampled image |
| UNet output? | Per-pixel unit-variance noise tensor |
| Noise → Gaussian? | Scheduler arithmetic, no learning |
| Where does gradient flow? | μ̃_θ → log π_θ → ρ → loss |
