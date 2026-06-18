# Experiment Log

Running log of RL experiments on the QuickDraw text-conditional diffusion model.
Newest experiments are appended at the bottom. Each entry records the question,
setup, result, and the decision it led to. Full per-run artifacts live in
`runs/<id>/` (gitignored); curves are viewable in the dashboard.

Status legend: ✅ done · 🔄 running · 📋 planned

---

## Experiment 1 — β baseline pilot ✅

**Question.** Does the KL-to-reference coefficient β control the trade-off between
chasing a symmetry reward and staying close to the base model — and can we *see*
reward hacking happen?

**Setup.** GRPO fine-tune toward vertical symmetry. Reward `vsym_l2(x) =
−mean((x − hflip(x))²)`, no anti-collapse safeguard. Three runs varying only β ∈
{0.0, 0.04, 0.4}. K=8, prompts_per_step=2, n_steps=200, T_inf=50 (respaced from
1000), lr=1e-5, eps_clip=0.2, grad_clip=1.0, cfg_scale=5.0, eval_every=25,
eval_n_seeds=8, fp32. Base = `jamesaasher/quickdraw-text-diffusion`
`text_diffusion_final_epoch_100.pt`. 4×V100 in parallel, 98 min wall.

**Result.** All three hypotheses confirmed.

| β | diversity 0→200 | symmetry 0→200 | KL→ref | grad_norm 1→200 | verdict |
|---|---|---|---|---|---|
| 0.00 | 46.6 → **7.9** | −0.55 → −0.11 | ~159 | 3.8 → 0.009 | reward-hacked: symmetric but **collapsed** |
| 0.04 | 46.6 → 46.7 | −0.55 → −0.50 | ~0.26 | 3.8 → 5.6 | healthy: mild gain, diversity intact |
| 0.40 | 46.6 → 47.9 | −0.55 → −0.54 | ~0.24 | 3.8 → 135 | pinned to base, KL leash dominates |

- **β=0** reward-hacks via mode collapse — near-blank symmetric blobs maximize the
  reward; diversity craters between step 50→75.
- **β=0.04** holds diversity while moving symmetry slightly.
- **β=0.40** barely leaves the base model.
- **grad_norm is a second, independent collapse signal**: β=0 decays to a dead
  fixed point (~0.009); β=0.40 grows (~135) as the KL penalty and reward fight.

**Conclusion / decision.** The pilot is pedagogically successful but produces **no
usable model**: there is no β that yields "recognizable *and* symmetric." Root cause
is the **reward, not the hyperparameter** — `vsym_l2`'s global optimum is *any*
symmetric image, and blank blobs are the easiest to reach. β only trades collapse
speed against no movement. → Next experiment must change the *reward shape*, not β.

Full write-up: [PILOT_RESULTS.md](PILOT_RESULTS.md) · raw notes: `runs/PILOT_NOTES.md`.

**Caveat carried forward.** `metrics.symmetry_l2` applies a `.sqrt()` that the
`rewards.vsym_l2` training signal does not, and eval uses held-out prompts at
T=1000 vs training prompts at T=50. The two live on different scales and are not
directly comparable; reconcile before reading small eval moves.

---

## Experiment 2 — reward redesign: escaping the collapse-vs-nothing dilemma 📋

**Question.** Can a reward that does **not** make a blank image optimal produce a
model that is both *recognizable* and *symmetric* — i.e. a genuine usable result,
not just a pedagogical failure mode?

**Hypothesis.** The collapse in Experiment 1 is a property of `vsym_l2`'s optimum,
not of GRPO or β. A reward that ties symmetry to *content* (so blanking yields no
gain) should let the model improve symmetry while staying recognizable, even at
low/zero β.

**Plan.** Hold β fixed at **0.04** (the healthy regime from Exp 1) and vary only the
**reward function**, so reward *shape* is the single variable:

1. `vsym_l2` — **control** (reproduces Exp 1's β=0.04 run).
2. `vsym_scale_inv` — scale-invariant symmetry: normalize the symmetry error by
   image content variance, e.g. `−‖x − hflip(x)‖² / (‖x − x̄‖² + ε)`. A blank image
   has ~zero content variance, so blanking no longer maximizes reward. Cheapest fix,
   no new model.
3. `vsym_plus_clip` — composite: `vsym_l2 + λ · CLIP(image, prompt)`, adding an
   on-prompt term via CLIP image–text similarity. Most directly targets
   "recognizable *and* symmetric"; cost is the CLIP image encoder in the rollout.

**Prerequisite fixes (do first).**
- Reconcile the reward/eval scale mismatch (sqrt + shared prompt set) so symmetry is
  comparable across train and eval.
- Add periodic checkpointing (trainer currently saves only the final step) so we can
  inspect pre-/post-collapse models.

**Success criterion.** At least one reward yields a run whose **diversity holds**
(≈ base, no collapse) **and** whose **eval symmetry improves** meaningfully over the
`vsym_l2` control — with sample grids that are visibly on-prompt *and* more symmetric.

**Status.** Planned. Reward implementations to be added to `rl/rewards.py`'s
registry; comparison run via `rl.sweep` over `--reward` at fixed β.
