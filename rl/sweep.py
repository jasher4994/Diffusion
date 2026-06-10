"""Orchestrate the β-pilot (or any user-supplied sweep) over GRPO runs.

Default sweep is the 3-run β pilot from PLAN_RL.md:
  pilot_beta0.00_k8, pilot_beta0.04_k8, pilot_beta0.40_k8

Two execution modes:

  sequential (default): call `train(cfg)` directly in-process, one after the
                        other. Use this on a single-GPU box (or MPS).

  --parallel:           spawn one subprocess per run with CUDA_VISIBLE_DEVICES
                        pinned. Use this when you have N >= len(runs) GPUs
                        (e.g. the 4xV100 Azure box).

Logs from parallel subprocesses are tee'd to runs/<id>/stdout.log so you can
tail any one of them while the sweep runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from rl import config as rl_config
from rl.config import RunConfig
from rl.trainer import train


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------


def default_pilot_configs(
    base: RunConfig,
    betas: list[float],
    prefix: str = "pilot",
) -> list[RunConfig]:
    """Build one RunConfig per beta, inheriting everything else from `base`."""
    cfgs: list[RunConfig] = []
    for b in betas:
        # File-friendly id: "pilot_beta0.04_k8".
        rid = f"{prefix}_beta{b:.2f}_k{base.group_size}"
        cfgs.append(replace(base, run_id=rid, beta=b))
    return cfgs


# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------


def run_sequential(cfgs: list[RunConfig]) -> list[tuple[str, float, int]]:
    """Run configs one after another in the current process. Returns
    (run_id, wall_secs, exit_code) per run.  exit_code is 0 on success,
    1 on exception."""
    results: list[tuple[str, float, int]] = []
    for cfg in cfgs:
        print(f"\n{'=' * 70}\n[sweep] starting {cfg.run_id} (beta={cfg.beta})\n{'=' * 70}")
        t0 = time.time()
        try:
            train(cfg)
            code = 0
        except Exception as e:  # noqa: BLE001
            print(f"[sweep] run {cfg.run_id} FAILED: {e!r}")
            code = 1
        results.append((cfg.run_id, time.time() - t0, code))
    return results


def run_parallel(cfgs: list[RunConfig], cuda_devices: list[int]) -> list[tuple[str, float, int]]:
    """Spawn one subprocess per config, each pinned to its own GPU."""
    if len(cuda_devices) < len(cfgs):
        raise ValueError(
            f"Need at least {len(cfgs)} CUDA devices for parallel mode, "
            f"got {len(cuda_devices)}."
        )

    procs: list[tuple[RunConfig, subprocess.Popen, Path, float]] = []
    for cfg, dev in zip(cfgs, cuda_devices):
        run_dir = Path(rl_config.RUNS_DIR) / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "stdout.log"
        cmd = _build_trainer_cmd(cfg)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[sweep] launching {cfg.run_id} on cuda:{dev}  (log -> {log_path})")
        f = log_path.open("w")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        procs.append((cfg, p, log_path, time.time()))

    results: list[tuple[str, float, int]] = []
    for cfg, p, log_path, t0 in procs:
        code = p.wait()
        wall = time.time() - t0
        status = "OK" if code == 0 else f"FAIL (exit {code})"
        print(f"[sweep] {cfg.run_id}: {status} in {wall / 60:.1f} min  (log {log_path})")
        results.append((cfg.run_id, wall, code))
    return results


def _build_trainer_cmd(cfg: RunConfig) -> list[str]:
    """Build the python -m rl.trainer CLI for a config."""
    cmd = [
        sys.executable, "-m", "rl.trainer",
        "--run-id", cfg.run_id,
        "--beta", str(cfg.beta),
        "--group-size", str(cfg.group_size),
        "--prompts-per-step", str(cfg.prompts_per_step),
        "--n-steps", str(cfg.n_steps),
        "--t-inf", str(cfg.t_inf),
        "--lr", str(cfg.lr),
        "--cfg-scale", str(cfg.cfg_scale),
        "--eval-every", str(cfg.eval_every),
        "--eval-n-seeds", str(cfg.eval_n_seeds),
        "--device", cfg.device,
        "--seed", str(cfg.seed),
        "--reward", cfg.reward_name,
    ]
    if cfg.eval_prompts and cfg.eval_prompts != list(rl_config.TRAINED_PROMPTS):
        cmd.append("--eval-prompts")
        cmd.extend(cfg.eval_prompts)
    return cmd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="pilot", help="Run-id prefix (default: pilot).")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.04, 0.4],
                   help="KL coefficients to sweep (default: 0.0 0.04 0.4).")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--t-inf", type=int, default=50)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n-seeds", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward", default="vsym_l2")
    p.add_argument("--parallel", action="store_true",
                   help="Spawn one subprocess per run, pinning each to its own GPU.")
    p.add_argument("--cuda-devices", type=int, nargs="+", default=None,
                   help="GPU ids to use in --parallel mode (default: 0..N-1).")
    args = p.parse_args()

    base = RunConfig(
        run_id="<set-by-sweep>",
        beta=0.0,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        n_steps=args.n_steps,
        t_inf=args.t_inf,
        lr=args.lr,
        cfg_scale=args.cfg_scale,
        eval_every=args.eval_every,
        eval_n_seeds=args.eval_n_seeds,
        device=args.device,
        seed=args.seed,
        reward_name=args.reward,
    )
    cfgs = default_pilot_configs(base, args.betas, prefix=args.prefix)

    print(f"[sweep] {len(cfgs)} runs queued: {[c.run_id for c in cfgs]}")
    print(f"[sweep] mode: {'parallel' if args.parallel else 'sequential'}")

    t0 = time.time()
    if args.parallel:
        devs = args.cuda_devices if args.cuda_devices is not None else list(range(len(cfgs)))
        results = run_parallel(cfgs, devs)
    else:
        results = run_sequential(cfgs)
    wall = time.time() - t0

    # Summary
    print(f"\n{'=' * 70}\n[sweep] complete in {wall / 60:.1f} min\n{'=' * 70}")
    print(f"{'run_id':<40s} {'wall_min':>10s} {'status':>8s}")
    for rid, w, code in results:
        print(f"{rid:<40s} {w / 60:>10.1f} {'OK' if code == 0 else 'FAIL':>8s}")
    return 0 if all(c == 0 for _, _, c in results) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
