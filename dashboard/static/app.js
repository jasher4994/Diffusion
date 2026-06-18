/* Diffusion RL Dashboard – jQuery + Chart.js front-end. */
(function () {
  "use strict";

  const PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
  ];

  // run_id -> {color, log, samples, config}
  const RUN_CACHE = new Map();
  const CHARTS = {};
  let SELECTED = new Set();

  // ----- Reward registry (mirrors rl/rewards.py) -------------------------
  // KaTeX source for the per-sample reward each run optimizes. Keep in sync
  // with the REWARDS dict in rl/rewards.py.
  const REWARD_FORMULAS = {
    vsym_l2: {
      tex: "r(x) = -\\operatorname{mean}_{\\text{pixels}}\\big( (x - \\mathrm{hflip}(x))^2 \\big)",
      note: "Vertical-symmetry L2. Optimum is ANY symmetric image — a blank blob maximizes it, so it can reward-hack via collapse.",
    },
    vsym_scale_inv: {
      tex: "r(x) = -\\dfrac{\\lVert x - \\mathrm{hflip}(x) \\rVert^2}{\\lVert x - \\bar{x} \\rVert^2 + \\varepsilon}",
      note: "Scale-invariant symmetry: blank images have ~zero content variance, so blanking no longer helps.",
    },
    vsym_plus_clip: {
      tex: "r(x) = -\\operatorname{mean}\\big((x - \\mathrm{hflip}(x))^2\\big) + \\lambda\\,\\mathrm{CLIP}(x,\\text{prompt})",
      note: "Composite: symmetry plus an on-prompt CLIP term to keep samples recognizable.",
    },
  };

  // GRPO objective (mirrors rl/trainer.update_step). Shown with each run's β.
  function grpoLossTex(beta) {
    const b = typeof beta === "number" ? beta : 0;
    return (
      "\\mathcal{L} = -\\min\\!\\big(\\rho\\,A,\\ \\mathrm{clip}(\\rho, 1-\\epsilon, 1+\\epsilon)\\,A\\big)" +
      " + " + formatBeta(b) + "\\cdot \\mathrm{KL}(\\pi_\\theta \\,\\Vert\\, \\pi_{\\text{ref}})"
    );
  }

  function formatBeta(b) {
    // Render β as its numeric value so the leash strength is explicit per run.
    return "\\beta{=}" + b;
  }

  function renderTex(tex, el) {
    if (window.katex) {
      try {
        window.katex.render(tex, el, { throwOnError: false, displayMode: true });
        return;
      } catch (e) {
        /* fall through to text */
      }
    }
    el.textContent = tex;
  }

  function repaintObjective() {
    const $body = $("#objective-body").empty();
    const ids = [...SELECTED];
    if (ids.length === 0) {
      $body.html('<em class="text-muted">Select a run to see its reward and loss function.</em>');
      return;
    }
    // Shared definitions (advantage + ratio) shown once.
    const $defs = $(`
      <div class="run-objective">
        <div class="formula-label">GRPO objective (per timestep, summed over the trajectory)</div>
        <div class="formula-block" data-tex="${grpoLossTex(0).replace(/"/g, "&quot;")}" data-shared="1"></div>
        <div class="text-muted small">
          ratio <span data-itex="\\rho = \\pi_\\theta / \\pi_{\\theta_{\\text{old}}}"></span>,
          group-relative advantage <span data-itex="A_i = (r_i - \\mu)/(\\sigma + \\varepsilon)"></span>,
          clip <span data-itex="\\epsilon = 0.2"></span>. The shared form above uses each run's own β below.
        </div>
      </div>
    `);
    $body.append($defs);

    ids.forEach((runId) => {
      const run = RUN_CACHE.get(runId);
      if (!run) return;
      const cfg = run.config || {};
      const rname = cfg.reward_name || "vsym_l2";
      const reward = REWARD_FORMULAS[rname] || {
        tex: "r(x) = \\text{" + rname + "}",
        note: "",
      };
      const $r = $(`
        <div class="run-objective">
          <div>
            <span class="run-color-swatch" style="background:${run.color}"></span>
            <strong>${runId}</strong>
            <span class="text-muted small">&nbsp;reward=<code>${rname}</code> · β=${cfg.beta} · K=${cfg.group_size} · T_inf=${cfg.t_inf}</span>
          </div>
          <div class="formula-label mt-2">Reward</div>
          <div class="formula-block" data-tex="${reward.tex.replace(/"/g, "&quot;")}"></div>
          ${reward.note ? `<div class="text-muted small">${reward.note}</div>` : ""}
          <div class="formula-label mt-2">Loss (this run's β)</div>
          <div class="formula-block" data-tex="${grpoLossTex(cfg.beta).replace(/"/g, "&quot;")}"></div>
        </div>
      `);
      $body.append($r);
    });

    // Render all collected TeX.
    $body.find(".formula-block").each(function () {
      renderTex($(this).attr("data-tex"), this);
    });
    $body.find("[data-itex]").each(function () {
      if (window.katex) {
        try {
          window.katex.render($(this).attr("data-itex"), this, { throwOnError: false });
        } catch (e) {
          this.textContent = $(this).attr("data-itex");
        }
      }
    });
  }

  // ----- Chart helpers ---------------------------------------------------

  function makeChart(canvasId, yLabel) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: { datasets: [] },
      options: {
        animation: false,
        parsing: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", title: { display: true, text: "step" } },
          y: { title: { display: true, text: yLabel } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function initCharts() {
    CHARTS.reward = makeChart("chart-reward", "reward_mean");
    CHARTS.kl = makeChart("chart-kl", "kl_to_ref");
    CHARTS.diversity = makeChart("chart-diversity", "diversity");
    CHARTS.pps = makeChart("chart-pps", "per_pixel_std");
  }

  function seriesFor(metricKey, runId) {
    const run = RUN_CACHE.get(runId);
    if (!run) return [];
    // Try log.jsonl first (training rows), then fall back to per-sample metrics.
    const points = [];
    (run.log || []).forEach((row) => {
      const v = row[metricKey];
      if (typeof v === "number" && typeof row.step === "number") {
        points.push({ x: row.step, y: v });
      }
    });
    if (points.length === 0) {
      (run.samples || []).forEach((s) => {
        const v = s.metrics && s.metrics[metricKey];
        if (typeof v === "number") points.push({ x: s.step, y: v });
      });
    }
    return points;
  }

  function repaintCharts() {
    const mapping = {
      reward: "reward_mean",
      kl: "kl_to_ref",
      diversity: "diversity",
      pps: "per_pixel_std",
    };
    Object.entries(mapping).forEach(([chartKey, metricKey]) => {
      const chart = CHARTS[chartKey];
      chart.data.datasets = [...SELECTED].map((runId) => {
        const run = RUN_CACHE.get(runId);
        return {
          label: runId,
          data: seriesFor(metricKey, runId),
          // For sample-only metrics, fall back to symmetry_l2_mean if the
          // canonical metric isn't there (eval-only runs).
          borderColor: run.color,
          backgroundColor: run.color,
          pointRadius: 3,
          tension: 0.15,
        };
      });
      // Fallback: if the reward chart is empty for an eval-only run, plot
      // symmetry_l2_mean from samples instead.
      if (chartKey === "reward" && chart.data.datasets.every((d) => d.data.length === 0)) {
        chart.data.datasets = [...SELECTED].map((runId) => {
          const run = RUN_CACHE.get(runId);
          const pts = (run.samples || [])
            .map((s) => ({ x: s.step, y: s.metrics && s.metrics.symmetry_l2_mean }))
            .filter((p) => typeof p.y === "number");
          return {
            label: runId + " (symmetry_l2)",
            data: pts,
            borderColor: run.color,
            backgroundColor: run.color,
            pointRadius: 3,
          };
        });
      }
      chart.update();
    });
  }

  // ----- Sample grids ----------------------------------------------------

  function repaintSamples() {
    const $sec = $("#samples-section").empty();
    [...SELECTED].forEach((runId) => {
      const run = RUN_CACHE.get(runId);
      if (!run || !run.samples || run.samples.length === 0) return;
      const samples = run.samples;
      const sliderId = "slider-" + runId.replace(/[^a-z0-9]/gi, "_");
      const multiStep = samples.length > 1;

      // Navigation row: prev/next buttons + slider (or just a label if single-step).
      const navHtml = multiStep ? `
        <div class="d-flex align-items-center gap-2 mt-2">
          <button class="btn btn-sm btn-outline-secondary" id="${sliderId}-prev">&laquo;</button>
          <input type="range" class="form-range flex-grow-1" id="${sliderId}"
                 min="0" max="${samples.length - 1}" step="1"
                 value="${samples.length - 1}">
          <button class="btn btn-sm btn-outline-secondary" id="${sliderId}-next">&raquo;</button>
        </div>
      ` : `
        <div class="mt-2 text-muted small">Only one snapshot — train a run for a timeline.</div>
      `;

      const $card = $(`
        <div class="card mb-3">
          <div class="card-header d-flex justify-content-between align-items-center">
            <span>
              <span class="run-color-swatch" style="background:${run.color}"></span>
              <strong>${runId}</strong>
            </span>
            <span class="step-label" id="${sliderId}-label"></span>
          </div>
          <div class="card-body">
            <div class="row">
              <div class="col-md-9">
                <img class="grid-img" id="${sliderId}-img" alt="sample grid">
                ${navHtml}
              </div>
              <div class="col-md-3">
                <pre class="cfg" id="${sliderId}-metrics"></pre>
              </div>
            </div>
          </div>
        </div>
      `);
      $sec.append($card);

      let currentIdx = samples.length - 1;
      function render(idx) {
        currentIdx = Math.max(0, Math.min(samples.length - 1, idx));
        const s = samples[currentIdx];
        $(`#${sliderId}-img`).attr("src", s.grid_url);
        $(`#${sliderId}-label`).text(`step ${s.step}  (${currentIdx + 1} / ${samples.length})`);
        $(`#${sliderId}-metrics`).text(JSON.stringify(s.metrics, null, 2));
        if (multiStep) $(`#${sliderId}`).val(currentIdx);
      }
      render(currentIdx);

      if (multiStep) {
        $(`#${sliderId}`).on("input", function () { render(parseInt(this.value, 10)); });
        $(`#${sliderId}-prev`).on("click", () => render(currentIdx - 1));
        $(`#${sliderId}-next`).on("click", () => render(currentIdx + 1));
      }
    });
  }

  // ----- Data loading ----------------------------------------------------

  function ensureRunLoaded(runId, color) {
    if (RUN_CACHE.has(runId)) {
      RUN_CACHE.get(runId).color = color;
      return Promise.resolve();
    }
    return Promise.all([
      $.getJSON(`/api/runs/${encodeURIComponent(runId)}/log`),
      $.getJSON(`/api/runs/${encodeURIComponent(runId)}/samples`),
      $.getJSON(`/api/runs/${encodeURIComponent(runId)}/config`),
    ]).then(([log, samples, config]) => {
      RUN_CACHE.set(runId, { color, log, samples, config });
    });
  }

  function refreshAll() {
    return $.getJSON("/api/runs").then((runs) => {
      const $list = $("#runs-list").empty();
      if (runs.length === 0) {
        $list.html('<em class="text-muted">No runs found. Generate one with <code>python -m rl.eval</code>.</em>');
        return;
      }
      runs.forEach((r, i) => {
        const color = PALETTE[i % PALETTE.length];
        const id = "run-" + i;
        const checked = SELECTED.has(r.run_id) ? "checked" : "";
        $list.append(`
          <div class="form-check">
            <input class="form-check-input run-toggle" type="checkbox" id="${id}"
                   data-run-id="${r.run_id}" data-color="${color}" ${checked}>
            <label class="form-check-label" for="${id}">
              <span class="run-color-swatch" style="background:${color}"></span>
              ${r.run_id}
              <small class="text-muted d-block ms-3">
                ${r.n_log_rows} log rows · ${r.sample_steps.length} sample steps
              </small>
            </label>
          </div>
        `);
      });

      // Auto-select first run on initial load.
      if (SELECTED.size === 0 && runs.length > 0) {
        SELECTED.add(runs[0].run_id);
        $list.find(`[data-run-id="${runs[0].run_id}"]`).prop("checked", true);
      }
      return reloadSelected();
    });
  }

  function reloadSelected() {
    const $checked = $("#runs-list .run-toggle:checked");
    const loaders = [];
    $checked.each(function () {
      const runId = $(this).data("run-id");
      const color = $(this).data("color");
      loaders.push(ensureRunLoaded(runId, color));
    });
    return Promise.all(loaders).then(() => {
      repaintObjective();
      repaintCharts();
      repaintSamples();
    });
  }

  // ----- Init ------------------------------------------------------------

  $(function () {
    initCharts();
    refreshAll();

    $(document).on("change", ".run-toggle", function () {
      const runId = $(this).data("run-id");
      if (this.checked) SELECTED.add(runId);
      else SELECTED.delete(runId);
      reloadSelected();
    });

    $("#refresh-btn").on("click", function () {
      // Force re-fetch of everything currently selected.
      [...SELECTED].forEach((id) => RUN_CACHE.delete(id));
      refreshAll();
    });
  });
})();
