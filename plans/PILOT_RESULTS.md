# Pilot Results — β sweep {0.0, 0.04, 0.4}

Closes Phase F of [PLAN_RL.md](PLAN_RL.md). GRPO fine-tune of the pretrained
text-conditional QuickDraw diffusion model toward vertical symmetry
(`vsym_l2`, no anti-collapse safeguard). 3 runs, K=8, 200 steps, T_inf=50,
fp32, on 4×V100 in parallel (98 min wall).

Full notes + curves live in `runs/PILOT_NOTES.md` (gitignored) and the dashboard.

## Outcome — all three hypotheses confirmed

β cleanly controls the trade-off between chasing the reward and staying near base.

| β | diversity 0→200 | symmetry 0→200 | KL→ref | verdict |
|---|---|---|---|---|
| 0.00 | 46.6 → **7.9** | −0.55 → −0.11 | ~159 | reward-hacked: symmetric but **collapsed** |
| 0.04 | 46.6 → 46.7 | −0.55 → −0.50 | ~0.26 | healthy: mild gain, diversity intact |
| 0.40 | 46.6 → 47.9 | −0.55 → −0.54 | ~0.24 | pinned to base, KL leash dominates |

- **β=0** reward-hacks via mode collapse — degenerate near-blank symmetric blobs
  maximize the reward; diversity craters between step 50→75. Caught exactly as predicted.
- **β=0.04** is the healthy middle: diversity holds while symmetry improves slightly.
- **β=0.40** barely moves from base.
- **grad_norm is a second collapse signal**: β=0 decays to ~0.009 (dead fixed point),
  β=0.04 stays moderate (actively learning), β=0.40 grows to ~135 (KL-vs-reward tug-of-war).

## Open question for next chapter

Training reward improved for all three β, but held-out eval symmetry only moved for β=0.
Likely a prompt-set + scale mismatch (training reward: train prompts @ T=50; eval: 5 held-out
prompts @ T=1000; plus `metrics.symmetry_l2` takes a sqrt that `rewards.vsym_l2` does not).
Reconcile before expanding the sweep.

## Candidate next steps

- `vsym_plus_clip` anti-collapse reward; re-run β=0 to show collapse can be cured.
- Sweep K ∈ {4, 8, 16} at β=0.04.
- Add periodic checkpointing (trainer currently saves only the final step) to capture
  a pre-collapse β=0 model.
