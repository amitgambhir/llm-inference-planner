# llm-inference-planner — Claude Code context

## Overview

Three-layer LLM inference planning, benchmarking, and deployment decisioning tool:

1. **Capacity Planner** (`planner/`, `api/`, `ui/`) — size a deployment before you have a GPU; roofline model → replicas, cost, confidence, benchmark plan
2. **Load benchmarking** (`collect/run_bench.py`) — TTFT/throughput/latency under concurrent load
3. **Quality-aware evaluation** (`evaluate/run_eval.py` + `analyze/deployment_advisor.py`) — production deployment recommendation balancing latency, cost, and quality

All three layers are additive and decoupled.

## Running tests

A globally installed pytest plugin tries to bind a socket in this environment. Bypass it:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

345 tests across 14 files:

| File | Count | What it covers |
| --- | --- | --- |
| `tests/test_capacity.py` | 55 | roofline model: prefill, decode, KV budget, replica rounding, efficiency curve integration |
| `tests/test_api.py` | 35 | FastAPI acceptance tests with isolated SQLite + mocked subprocess |
| `tests/test_ingest_anchor.py` | 29 | ingest_anchor pipeline, confidence rubric, geometry_source downgrade |
| `tests/test_catalog.py` | 29 | GPU/model catalog loading, get_gpu, get_model, resolve_gpu |
| `tests/test_benchmark_plan.py` | 27 | test matrix ordering, step count, priority logic |
| `tests/test_deployment_advisor.py` | 26 | load_deployment, compute_tradeoff, recommend, render |
| `tests/test_report.py` | 24 | mode badges, confidence bands, cost envelope section, anchor evidence |
| `tests/test_explain.py` | 17 | napkin-math explainer: sections, parameterised peak_rps, users/cost conditionals, report integration |
| `tests/test_run_eval.py` | 18 | dataset loading, score normalization, metric selection, sidecar writing |
| `tests/test_cost.py` | 18 | cost envelope, on-demand vs reserved, GPU-hour math |
| `tests/test_efficiency.py` | 16 | mfu_prefill + bw_eff_decode curves: monotonicity, bounds, fallback |
| `tests/test_compare.py` | 19 | cheapest/safest/best-latency scoring, H200 vs H100 note |
| `tests/test_validation.py` | 18 | public benchmark fit, accuracy targets, adapter routing, CV, sensitivity |
| `tests/test_intake.py` | 14 | demand resolver: three input modes, precedence, users passthrough, WorkloadError, divergence warning |

## Key files

### Capacity planner

| File | Role |
| --- | --- |
| `catalog/gpus.yaml` | Peak FLOPS, memory bandwidth, VRAM, arch, memory_type, MFU defaults for each GPU SKU |
| `catalog/models.yaml` | Parameter count, hidden dim, layers, KV heads; includes llama-3.3-70b and llama-4-maverick |
| `catalog/costs.yaml` | On-demand + 1-yr reserved cost per GPU-hour |
| `catalog/anchors.yaml` | Measured throughput anchors; written by `ingest_anchor.py` |
| `catalog/benchmarks_public.yaml` | Public benchmark points (schema v2) — vLLM `level`, TRT-LLM `shape`, distribution `validate`, `sanity` points; Phase B stubs (measured: null) pre-staged |
| `catalog/runtimes.yaml` | Supported inference engines with display names and engine confound notes |
| `catalog/runpod_phase_b.sh` | Phase B runbook — copy to RunPod 1×H100-SXM pod; runs 5 ISL/OSL offline benchmarks and streams `output_tps` per pair |
| `catalog/compute_engine_factor.py` | Phase C helper — prints per-pair TRT-LLM/vLLM ratio table + median to paste into `efficiency_constants.yaml` |
| `catalog/phase_c_refit.py` | Phase C automation — pins `engine_factor`, narrows `PARAM_BOUNDS`, runs full refit, prints CV + sensitivity + suggested test targets |
| `planner/intake.py` | Multi-mode demand resolver — `DemandSpec`, `WorkloadError`, `resolve_demand()`; three input modes (requests_per_day / avg_rps / users×prompts); `users` passes through to cost layer |
| `planner/capacity.py` | Roofline model — prefill (compute-bound) + decode (bandwidth-bound); `plan()` accepts `users`; `CapacityEstimate` carries `users`, `gpu_mem_gb`, `headroom_factor`; CLI: `--avg-rps`, `--users`, `--prompts-per-user-per-day`, `--explain` |
| `planner/catalog.py` | GPU/model catalog loader — reads `catalog/gpus.yaml` + `catalog/models.yaml` |
| `planner/cost.py` | Cost envelope from catalog pricing; `CostVariant` carries `cost_per_user_per_month` (populated when `estimate.users` is set) |
| `planner/benchmark_plan.py` | Ordered test matrix: ISL sweep, concurrency sweep, precision compare, KV check |
| `planner/confidence.py` | THREE-tier rubric: HIGH (±10%), MEDIUM (±20%), DEFAULT (±25%) |
| `planner/efficiency.py` | Regime-aware efficiency curves: `mfu_prefill` (size + ISL + MoE) and `bw_eff_decode` (batch amortization only; KV counted once in `decode_ceiling`) |
| `planner/efficiency_constants.yaml` | Tunable constants for efficiency curves; updated by `validate.fit()` |
| `planner/explain.py` | Napkin-math explainer — `render_napkin_math(est, cost=None)`; six sections; every number read from `CapacityEstimate`/`CostEstimate`, no hardcoded constants |
| `planner/validate.py` | `fit()` + `report()` + `cv_leave_one_gpu_out()` + `parameter_sensitivity()`; dual-roofline per-point adapter; fit_role filtering (level+shape in fit, validate reported separately) |
| `planner/ingest_anchor.py` | Reads benchmark JSON, writes calibration anchor to `catalog/anchors.yaml` |
| `planner/compare.py` | Multi-config comparison: cheapest, safest, best_latency |
| `planner/report.py` | Markdown report with mode badge; `include_napkin_math: bool = False` appends "## How we got here" section |
| `api/db.py` | SQLAlchemy 2.0 ORM: ScenarioRow, CapacityEstimateRow, BenchmarkRunRow, RecommendationRow, ReportRow |
| `api/schemas.py` | Pydantic v2 request/response schemas |
| `api/jobs.py` | Background benchmark subprocess runner |
| `api/main.py` | `create_app()` factory + 9 REST endpoints |

### Benchmark + analysis pipeline

| File | Role |
| --- | --- |
| `collect/run_bench.py` | Async load benchmark — OpenAI-compatible `/v1/completions`; supports `--token` (Bearer) and `--basic-auth USER:PASS` |
| `evaluate/run_eval.py` | Offline quality evaluator — DeepEval + LLM-judge |
| `analyze/deployment_advisor.py` | Deployment decision engine — 4 pure functions + CLI |
| `analyze/report.py` | Markdown report from latency results (stdlib only) |
| `datasets/*.jsonl` | Eval datasets — `schema_version: 1`, workloads: chat/rag/long_context |
| `results/quality/` | Quality sidecars written by `run_eval.py` |
| `results/real/` | Gitignored — populated by `run_bench.py` |

## Architecture invariants

### Multi-mode demand resolver

`planner/intake.py` converts three demand input modes into a single `(requests_per_day, users)` pair before the roofline runs:

| Mode | Fields | Conversion |
| --- | --- | --- |
| Direct | `requests_per_day` | passthrough |
| RPS | `avg_rps` | `× 86,400` |
| User-based | `users + prompts_per_user_per_day` | `users × prompts` |

Precedence: `requests_per_day` → `avg_rps` → `users × prompts`. Multiple sources accepted; a `warnings.warn` is emitted if two sources diverge >20%. No source → `WorkloadError`. `users` is a cross-cutting optional in every mode — it never enters the roofline physics but flows through to `CostVariant.cost_per_user_per_month`.

### Roofline model

`capacity.py` models two phases independently:

- **Prefill** (compute-bound): `prefill_time_ms = (isl × params × 2) / (peak_flops × mfu × tp)`
- **Decode** (bandwidth-bound): `decode_time_ms = (params × bytes_per_param) / (memory_bandwidth × mfu × tp)`
- TTFT estimate = max(prefill_time, decode_time) × peak_multiplier
- `max_concurrent_seqs` (KV budget): `floor(vram_bytes × kv_fraction / kv_bytes_per_seq)`
- Replicas = `ceil(requests_per_day × peak_multiplier / (seconds_per_day / ttft_secs) / max_concurrent_seqs)`

### Confidence rubric

`confidence.py` assigns three tiers based on `geometry_source` and whether anchor data exists:

- `geometry_source="measured"` + anchor present → starts at HIGH
- `geometry_source="measured"` + no anchor → MEDIUM
- `geometry_source="estimated"` → always downgrades one level from above

Band widths: HIGH ±10%, MEDIUM ±20%, DEFAULT ±25%.

The third tier is named `DEFAULT` (not `LOW`) — it is the baseline for any estimate that hasn't been upgraded by measured geometry or anchor data. The band tightened from ±50% to ±25% as the roofline model was calibrated against public benchmarks via `validate.fit()`.

### create_app() factory isolation

`api/main.py` exports `create_app(db_url, results_dir, anchors_file)` — a factory, not a module-level singleton. Tests inject `tmp_path` paths so no test ever touches the production SQLite DB or `catalog/anchors.yaml`. The default module-level `app = create_app()` is only used by uvicorn.

All per-app config is stored on `app.state` (`session_factory`, `results_dir`, `anchors_file`) so endpoints and background tasks use the same test-isolated paths via `request.app.state`.

### BackgroundTasks + TestClient

Starlette's `TestClient` runs `BackgroundTasks` synchronously — the task completes before the HTTP response is returned. Tests can assert `status == "done"` immediately after `POST /benchmarks/run`. This behavior is deterministic; no polling needed in tests.

### session_factory in background tasks

Background jobs (`api/jobs.py`) run after the request session has closed. `run_job()` accepts `session_factory` as a parameter and opens its own session inside the task body. Routes pass `request.app.state.session_factory` to `background_tasks.add_task()` — never a live session object.

### Quality sidecar coupling

Each quality sidecar (`results/quality/<tag>.json`) carries a `latency_tag` backlink to the latency result it was paired with. Hard errors:

- `latency_tag` in sidecar ≠ the tag being loaded — stale sidecar can silently corrupt a recommendation
- `meta.model` in sidecar ≠ `meta.model` in latency result — different model, incomparable

### Cross-profile dataset validation

`compute_tradeoff` hard-errors if profiles carry different `_dataset` values. Quality scores from different eval sets are not comparable. This check lives in `compute_tradeoff`, not in `load_deployment`.

### Dataset schema

`load_dataset` in `run_eval.py` validates:

- Required fields: `schema_version`, `id`, `workload`, `prompt`, `expected`
- `schema_version` must equal `1` — any other value is a hard error
- Valid workloads: `"chat"`, `"rag"`, `"long_context"`
- RAGAS fields (`contexts`, `ground_truth`) are V2 only — no V1 code or dependency

### DeploymentProfile contract

`load_deployment` flattens the nested latency JSON into a normalized in-memory dict:

- `metrics.ttft_ms.p50` → `latency.ttft_ms_p50`
- `metrics.ttft_ms.p95` → `latency.ttft_ms_p95`
- `metrics.throughput_tokens_per_sec` → `latency.throughput_tokens_per_sec`
- `quality` is `None` when no sidecar exists (warn, not error)
- `_dataset` is `None` when no sidecar; set to `meta.dataset` from the sidecar when one is loaded

### Real results only

`load_deployment` reads from `results/real/` (gitignored, populated by `run_bench.py`) and `results/quality/` (quality sidecars). There is no `results/synthetic/` in this repo — synthetic reference data lives in `llm-inference-bench`.

## Known gotchas

**DeepEval env var mutation.** When `--eval-endpoint` or `--eval-token` are provided, `run_deepeval()` sets `OPENAI_API_KEY` and/or `OPENAI_BASE_URL` as process-level environment variables. Safe for the CLI (runs to completion and exits), but a footgun if called from a long-running server or a test suite that parallelizes eval runs.

**Hallucination normalization differs by evaluator path.** The LLM-judge prompt scores hallucination 1–5 where `5=none` (higher-is-better). After dividing by 5, `normalize_score("hallucination", ...)` is NOT applied. On the DeepEval path, `HallucinationMetric.score` returns a rate (lower-is-better), so `normalize_score` inverts via `1 - score`. The two paths are intentionally asymmetric.

**DeepEval is a lazy import.** `run_deepeval` imports DeepEval inside the function body. This keeps the 314 unit tests fast — they never trigger a network call or require an `OPENAI_API_KEY`.

**app.state.anchors_file in tests.** `ingest_anchor()` defaults to writing `catalog/anchors.yaml`. The API's `/ingest/{run_id}` endpoint passes `request.app.state.anchors_file` to override this. Tests pass `tmp_path / "test_anchors.yaml"` via `create_app(anchors_file=...)` so the real catalog is never modified during the test suite.

**geometry_source="estimated" downgrades confidence.** When a model is specified via the UI's custom-model form (not from the catalog), `geometry_source` is set to `"estimated"`. This automatically downgrades the confidence tier by one level. The effect is intentional and visible in the ConfidenceBadge.

**Basic auth credentials in the stored command.** When the UI benchmark plan page sends an endpoint URL containing `user:pass@host`, `api/main.py` strips the credentials from the URL and appends `--basic-auth user:pass` to the subprocess command. The clean URL (without credentials) is what gets stored in the `BenchmarkRunRow.command` column. `api/jobs.py` uses `shlex.split()` (not `.split()`) so passwords with special characters tokenize correctly.

**mock_subprocess fixture writes fake result JSON.** `tests/test_api.py` patches `subprocess.run` in `api.jobs` to write a `FAKE_RESULT` dict (valid latency JSON) to the expected `--output-dir/<tag>.json` path before returning `CompletedProcess(returncode=0)`. This enables the full ingest → recommendation pipeline to be exercised in tests without a real GPU.

**`--requests-per-day` is now optional.** Any of the three demand modes (`--requests-per-day`, `--avg-rps`, or `--users + --prompts-per-user-per-day`) satisfies the CLI. Supplying none raises `WorkloadError` before the catalog is even loaded.

**`users` flows through as a passthrough field on `CapacityEstimate`.** `plan()` accepts `users: int | None = None`; it is stored on the estimate unchanged. It never enters the prefill/decode/sizing math. `compute_cost()` reads `estimate.users` to populate `CostVariant.cost_per_user_per_month`; if `users` is None, that field is None.

**`render_napkin_math` reads `gpu_mem_gb` from `CapacityEstimate`, not from a `GpuProfile`.** `plan()` sets `gpu_mem_gb = gpu.mem_gb` on the estimate so the explainer is self-contained. If constructing a `CapacityEstimate` outside of `plan()` (e.g. in tests), set `gpu_mem_gb` explicitly or the KV-budget section will show `0 GB`.

**bw_eff_decode has no hard floor — the natural floor is `base × batch_floor`.** The old hard floors (0.05, 0.20) are removed. KV traffic is now counted once in `decode_ceiling`'s `bytes_per_step`, so no separate `g_kv` penalty is needed in `bw_eff_decode`. The floor emerges naturally from `batch_floor` (currently ~0.80), giving `floor ≈ bw_base × 0.80`.

**fit_role taxonomy controls what enters the efficiency fit.** `catalog/benchmarks_public.yaml` tags every benchmark point with one of four roles: `level` (vLLM reference — pins absolute efficiency level; enters `fit()`), `shape` (TRT-LLM ISL/size sweep — pins curve shape + `engine_factor[trtllm]`; enters `fit()`), `validate` (distribution-dataset points excluded from fit; predicted and reported at widened tolerance), `sanity` (excluded from fit and default report). The `report()` function filters to `fit_roles=("level",)` by default; pass `fit_roles=("level","shape")` for the full fit set, or `fit_roles=None` to report all points.

**Engine confound policy — never mix TRT-LLM throughput into vLLM estimates.** TRT-LLM consistently runs 1.5–2× faster than vLLM on identical hardware. If TRT-LLM `total_output_tps` numbers were used as `level` points to calibrate the efficiency constants, all subsequent vLLM estimates would be systematically over-predicted. The engine field and `fit_role: shape` tag enforce this separation at the data level.

**`fit_roles` default is a tuple, not None.** `report()` signature is `fit_roles: Optional[tuple] = ("level",)`. Passing `fit_roles=None` means "no filter — return all points". The tuple default (not `None`) ensures the common case (level-only report) requires no argument, while "all points" requires an explicit `None`.

## Deployment (Vercel + Render)

The app is deployable as a hosted web service. Key files:

| File | Purpose |
| --- | --- |
| `render.yaml` | Render blueprint — creates a Python web service + free Postgres DB, wires `DATABASE_URL` automatically |
| `vercel.json` | Tells Vercel the Next.js root is `ui/` |

**Environment variables:**

| Var | Where set | Value |
| --- | --- | --- |
| `DATABASE_URL` | Render (auto from blueprint) | Postgres connection string; `postgres://` prefix is rewritten to `postgresql://` at startup |
| `ALLOWED_ORIGIN` | Render dashboard (manual) | Vercel deployment URL, e.g. `https://your-app.vercel.app` |
| `RESULTS_DIR` | `render.yaml` | `/tmp/results` — ephemeral, fine for estimates; real benchmark run files don't survive restarts |
| `BACKEND_URL` | Vercel dashboard | Render service URL, e.g. `https://llm-inference-bench-api.onrender.com` |

**Render `postgres://` gotcha.** Render sets `DATABASE_URL` with the `postgres://` scheme. SQLAlchemy 2.x requires `postgresql://`. The startup code in `api/main.py` rewrites it automatically before passing to `create_app()`.

**CORS.** `ALLOWED_ORIGIN` can be a comma-separated list for multiple allowed origins. The default (`*`) is fine for local dev and is the Render default until you set the env var manually.

**Ephemeral benchmark results.** `POST /benchmarks/run` writes result JSON to `RESULTS_DIR` (`/tmp/results` on Render). Files are lost on restart. Estimates, plans, and recommendations (stored in Postgres) survive. The `POST /ingest/{run_id}` step must be called in the same process lifetime as the benchmark run.

## Three advisors, three layers

| Advisor / Layer | Question answered |
| --- | --- |
| `planner/capacity.py` + `api/` + `ui/` | "How many replicas do I need before I run anything?" |
| `analyze/deployment_advisor.py` | "Which quantization/precision/config should I deploy, given quality requirements?" |

Each answers a distinct question at a different stage of the deployment lifecycle. Neither replaces the other.
