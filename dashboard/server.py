"""FastAPI dashboard for inspecting RL training runs.

Serves the static UI under `/` and a tiny JSON API under `/api/...`.
Reads exclusively from `runs/<run_id>/` directories on disk (the contract
defined in `plans/PLAN_RL.md`). Start with:

    uvicorn dashboard.server:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Diffusion RL Dashboard", docs_url="/api/docs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    """Resolve and safety-check a run directory under RUNS_DIR."""
    d = (RUNS_DIR / run_id).resolve()
    if RUNS_DIR.resolve() not in d.parents and d != RUNS_DIR.resolve():
        raise HTTPException(400, "invalid run_id")
    if not d.exists() or not d.is_dir():
        raise HTTPException(404, f"run not found: {run_id}")
    return d


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _sample_steps(run: Path) -> list[int]:
    """Return sorted list of step numbers that have samples/step_<N>/ dirs."""
    samples_dir = run / "samples"
    if not samples_dir.is_dir():
        return []
    steps: list[int] = []
    for d in samples_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_"):
            try:
                steps.append(int(d.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(steps)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/runs")
def list_runs() -> list[dict]:
    """Summary of every run on disk (id, config, available sample steps)."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg = _read_json(d / "config.json") or {}
        out.append(
            {
                "run_id": d.name,
                "config": cfg,
                "n_log_rows": sum(1 for _ in (d / "log.jsonl").open())
                if (d / "log.jsonl").exists()
                else 0,
                "sample_steps": _sample_steps(d),
            }
        )
    return out


@app.get("/api/runs/{run_id}/config")
def run_config(run_id: str) -> dict:
    cfg = _read_json(_run_dir(run_id) / "config.json")
    if cfg is None:
        raise HTTPException(404, "config.json not found")
    return cfg


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str) -> list[dict]:
    """Full training-step log as an array of rows."""
    return _read_jsonl(_run_dir(run_id) / "log.jsonl")


@app.get("/api/runs/{run_id}/samples")
def run_samples(run_id: str) -> list[dict]:
    """List of every saved sample step with its metrics + grid URL."""
    run = _run_dir(run_id)
    out = []
    for step in _sample_steps(run):
        step_dir = run / "samples" / f"step_{step:06d}"
        out.append(
            {
                "step": step,
                "grid_url": f"/runs/{run_id}/samples/step_{step:06d}/grid.png",
                "manifest": _read_json(step_dir / "manifest.json"),
                "metrics": _read_json(step_dir / "metrics.json"),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Static & sample-image serving
# ---------------------------------------------------------------------------

# Serve raw run artefacts (sample PNGs, manifests) under /runs/...
if RUNS_DIR.exists():
    app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")

# Serve the UI.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
