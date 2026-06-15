# llm-inference-planner

> **Size a deployment before you touch any hardware. Validate it when you do.**

Most teams provision GPUs by copying a blog post or over-provisioning by 2–3× to buy safety margin. This tool replaces that guesswork with a physics-based capacity planner calibrated against public benchmarks — then closes the loop with a real benchmark harness and quality-aware comparison.

1. **Describe your workload** (requests/day, token lengths, latency SLO) — get a replica estimate, cost envelope, and confidence level *before* you have a GPU.
2. **Run the generated benchmark plan** against your actual endpoint — real measurements replace the estimate, confidence upgrades automatically.
3. **Compare deployment options** on latency, cost, *and* quality together — not just the fastest option, but the best option that meets your quality bar.

Ships as a **web app** (Next.js + FastAPI, deployable on Vercel + Render) and as a **CLI**.

---

## Key outputs

| Stage | What you get |
| --- | --- |
| Before hardware | Replica count, cost envelope (on-demand + reserved), confidence tier (HIGH ±10% / MEDIUM ±20% / DEFAULT ±25%), ordered benchmark plan |
| After benchmarking | Validated latency profiles (TTFT p50/p95/p99, throughput, failure rate), upgraded confidence |
| Deployment decision | Ranked recommendation across latency, cost, and quality — with eliminated options explained |

---

## Table of contents

1. [Quick start — no GPU needed](#1-quick-start--no-gpu-needed)
2. [Quick start — against a running endpoint](#2-quick-start--against-a-running-endpoint)
3. [Capacity Planner](#3-capacity-planner)
   - 3.1 [CLI](#31-cli)
   - 3.2 [API + UI](#32-api--ui)
   - 3.3 [`planner/` modules](#33-planner-modules)
   - 3.4 [`api/` REST endpoints](#34-api-rest-endpoints)
   - 3.5 [`catalog/` GPU and model catalog](#35-catalog-gpu-and-model-catalog)
   - 3.6 [`ui/` four-screen web interface](#36-ui-four-screen-web-interface)
4. [Benchmark harness](#4-benchmark-harness)
5. [Quality-aware comparison](#5-quality-aware-comparison)
6. [Project structure](#6-project-structure)
7. [Deployment (Vercel + Render)](#7-deployment-vercel--render)
8. [Contributing](#8-contributing)
9. [License](#9-license)

---

## 1. Quick start — no GPU needed

```bash
pip install -r requirements.txt

python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --requests-per-day 50000 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime
```

Output: replica count, confidence tier, TTFT estimate (ms), max concurrent sequences (KV budget), binding constraint (`compute` / `bandwidth` / `kv_budget`), and cost envelope.

To compare multiple GPU/dtype configurations:

```bash
python planner/compare.py --configs configs.json
```

---

## 2. Quick start — against a running endpoint

```bash
python collect/run_bench.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --isl 1024 --osl 256 \
  --concurrency 20 --duration 90 \
  --tag my_first_run

python planner/ingest_anchor.py results/real/my_first_run.json
```

The second command ingests the measurement as a calibration anchor. The next `planner/capacity.py` call on the same `(model, gpu, dtype)` automatically upgrades to HIGH confidence and uses the measured MFU.

For authenticated endpoints:

```bash
--token $API_KEY          # Bearer token
--basic-auth user:pass    # HTTP Basic Auth
```

---

## 3. Capacity Planner

Size a deployment *before* you have a GPU. Given a workload description, the planner applies a roofline model to estimate replica count, cost envelope, and confidence — then generates an ordered benchmark plan to validate the estimate.

### 3.1 CLI

Three equivalent ways to specify demand — pick the one that matches how you think about your workload:

```bash
# Option A — direct request volume
python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --requests-per-day 50000 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime

# Option B — average RPS
python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --avg-rps 0.58 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime

# Option C — user base × prompts/day
python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --users 25000 --prompts-per-user-per-day 2 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime
```

Add `--users` to any mode to unlock `$/user/month` in the cost output. Add `--explain` to append a plain-language walk of the sizing arithmetic:

```bash
python planner/capacity.py \
  --model llama-3.1-8b --gpu h100_sxm --dtype fp8 \
  --avg-rps 0.58 --users 25000 --isl 1024 --osl 256 \
  --ttft-slo-ms 500 --traffic-class realtime \
  --explain
```

### 3.2 API + UI

```bash
# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
cd ui && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Four screens:

| Screen | Route | Purpose |
| --- | --- | --- |
| Scenario Builder | `/` | Workload + GPU/model/dtype selection; supports catalog models and custom model spec |
| Estimate | `/estimate` | Replica range chart, confidence badge, binding constraint, warnings |
| Benchmark Plan | `/benchmark-plan` | Ordered test matrix with copy-ready `run_bench.py` commands |
| Report | `/report` | Recommendation summary, validation status badge, Markdown export |

### 3.3 `planner/` modules

| Module | Role |
| --- | --- |
| `planner/intake.py` | Multi-mode demand resolver — accepts `requests_per_day`, `avg_rps`, or `users × prompts_per_user_per_day`; returns `(requests_per_day, users)`; raises `WorkloadError` if no source supplied |
| `planner/capacity.py` | Roofline model — prefill (compute-bound) + decode (bandwidth-bound) → replicas, TTFT, KV budget; `plan()` accepts `users`; `CapacityEstimate` carries `users`, `gpu_mem_gb`, `headroom_factor` |
| `planner/cost.py` | On-demand and 1-yr reserved cost envelope from `catalog/costs.yaml`; `CostVariant` includes `cost_per_user_per_month` when `users` is set |
| `planner/explain.py` | Napkin-math explainer — `render_napkin_math(est, cost=None)` walks the sizing arithmetic in plain language; every number read from `CapacityEstimate`/`CostEstimate` |
| `planner/benchmark_plan.py` | Ordered test matrix — ISL sweep, concurrency sweep, precision compare, KV check |
| `planner/confidence.py` | Three-tier rubric: HIGH ±10%, MEDIUM ±20%, DEFAULT ±25%; `geometry_source="estimated"` downgrades one level |
| `planner/efficiency.py` | Regime-aware MFU and bandwidth efficiency curves — `mfu_prefill` (size + ISL + MoE) and `bw_eff_decode` (batch amortization; KV counted once in `decode_ceiling`) |
| `planner/efficiency_constants.yaml` | Tunable curve constants; calibrated by `validate.fit()` against public benchmarks |
| `planner/validate.py` | `fit()`, `report()`, `cv_leave_one_gpu_out()`, `parameter_sensitivity()`; dual-roofline per-point adapter; fit_role filtering |
| `planner/ingest_anchor.py` | Reads a completed benchmark JSON → writes calibration anchor to `catalog/anchors.yaml` |
| `planner/compare.py` | Multi-config comparison: cheapest, safest (confidence), best latency |
| `planner/report.py` | Markdown report with mode badge (`estimate_only` / `partially_validated` / `validated_by_benchmark`); `include_napkin_math=True` appends "How we got here" section |

### 3.4 `api/` REST endpoints

FastAPI with SQLite persistence (Postgres-ready). Uses `create_app()` factory for test isolation.

| Endpoint | What it does |
| --- | --- |
| `POST /scenarios` | Create a planning scenario |
| `GET /scenarios/{id}` | Fetch a scenario |
| `POST /scenarios/{id}/estimate` | Run the roofline estimator |
| `GET /scenarios/{id}/benchmark-plan` | Generate the ordered benchmark plan |
| `POST /benchmarks/run` | Enqueue a benchmark run as a background task |
| `GET /benchmarks/{run_id}` | Poll job status |
| `POST /ingest/{run_id}` | Ingest a completed run into the anchor catalog |
| `GET /scenarios/{id}/recommendation` | Final recommendation; `mode` escalates as benchmark runs complete |
| `GET /scenarios/{id}/report` | Fetch the Markdown report |

### 3.5 `catalog/` GPU and model catalog

| File | Contents |
| --- | --- |
| `catalog/gpus.yaml` | Peak FLOPS, memory bandwidth, VRAM, arch, memory_type, MFU defaults — h100_sxm, h200_sxm, a100_80gb_sxm, l40s, l4 |
| `catalog/models.yaml` | Param counts, hidden dim, layers, KV heads — llama-3.1-8b/70b, llama-3.3-70b, llama-4-maverick, gpt-oss-20b |
| `catalog/costs.yaml` | On-demand + 1-yr reserved cost per GPU-hour |
| `catalog/anchors.yaml` | Measured throughput anchors written by `ingest_anchor.py` |
| `catalog/benchmarks_public.yaml` | Public benchmark points (schema v2) — vLLM `level`, TRT-LLM `shape`, distribution `validate`, `sanity`; Phase B calibration stubs pre-staged |
| `catalog/runtimes.yaml` | Supported engines with engine confound notes |
| `catalog/runpod_phase_b.sh` | Phase B runbook — 5 ISL/OSL offline benchmarks on RunPod H100 |
| `catalog/phase_c_refit.py` | Phase C automation — pin `engine_factor`, refit, CV, sensitivity, print test targets |

### 3.6 `ui/` four-screen web interface

Next.js 14 (App Router) + Tailwind CSS + Recharts.

| Component | Purpose |
| --- | --- |
| `ReplicaRangeChart` | Confidence-colored bar chart (low/recommended/high) |
| `ConfidenceBadge` | Tier + band percentage, color-coded green/yellow/red |
| `ModeBadge` | Validation status — yellow (`estimate_only`), blue (`partially_validated`), green (`validated_by_benchmark`) |

---

## 4. Benchmark harness

`collect/run_bench.py` fires concurrent load at any OpenAI-compatible `/v1/completions` endpoint.

```bash
python collect/run_bench.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --isl 1024 --osl 256 \
  --concurrency 20 --duration 90 \
  --tag h100_fp8_c20
```

Captures TTFT (p50/p95/p99), end-to-end latency, throughput (tokens/s, req/s), and failure rate. Embeds the full environment contract in every result file (GPU, runtime, workload shape) so runs are reproducible and comparable.

After the run, ingest the anchor:

```bash
python planner/ingest_anchor.py results/real/h100_fp8_c20.json
```

---

## 5. Quality-aware comparison

Answers: *"Which quantization / precision / config should I deploy, given my quality bar?"*

```bash
# 1. Evaluate quality on a deployment
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/real/vllm_fp8.json \
  --dataset datasets/rag.jsonl \
  --evaluator deepeval \
  --eval-model gpt-4o

# 2. Compare deployments
python analyze/deployment_advisor.py \
  --tags vllm_fp16 vllm_fp8 vllm_int4 \
  --baseline vllm_fp16 \
  --quality-threshold 0.10
```

Three built-in eval datasets under `datasets/`: `chat.jsonl`, `rag.jsonl`, `long_context.jsonl`.

---

## 6. Project structure

```text
llm-inference-planner/
├── catalog/                        # GPU, model, pricing, and benchmark data
│   ├── gpus.yaml                   # FLOPS, bandwidth, VRAM, arch, memory_type
│   ├── models.yaml                 # Params, hidden dim, layers, KV heads
│   ├── costs.yaml                  # On-demand + 1-yr reserved cost per GPU-hour
│   ├── anchors.yaml                # Measured throughput anchors (ingest_anchor.py)
│   ├── benchmarks_public.yaml      # Public benchmark points; level/shape/validate/sanity roles
│   ├── runtimes.yaml               # Supported inference engines
│   ├── runpod_phase_b.sh           # Phase B: RunPod benchmark runbook
│   ├── compute_engine_factor.py    # Phase C: compute median TRT-LLM/vLLM ratio
│   └── phase_c_refit.py            # Phase C: pin engine_factor, refit, CV, sensitivity
│
├── planner/                        # Capacity planner — pure Python, no GPU needed
│   ├── intake.py                   # Multi-mode demand resolver (requests_per_day / avg_rps / users×prompts)
│   ├── capacity.py                 # Roofline model: prefill + decode → replicas, TTFT, KV budget
│   ├── catalog.py                  # GPU/model catalog loader — reads gpus.yaml + models.yaml
│   ├── cost.py                     # Cost envelope; cost_per_user_per_month when users is set
│   ├── explain.py                  # Napkin-math explainer — render_napkin_math(est, cost=None)
│   ├── benchmark_plan.py           # Ordered test matrix generator
│   ├── confidence.py               # HIGH/MEDIUM/DEFAULT confidence rubric
│   ├── efficiency.py               # Regime-aware MFU + bandwidth efficiency curves
│   ├── efficiency_constants.yaml   # Tunable curve constants; recalibrated by validate.fit()
│   ├── validate.py                 # fit(), report(), CV, sensitivity; dual-roofline adapter
│   ├── ingest_anchor.py            # Benchmark result → calibration anchor
│   ├── compare.py                  # Multi-config comparison
│   └── report.py                   # Markdown report with mode badge; --include_napkin_math
│
├── api/                            # FastAPI REST layer with SQLite/Postgres persistence
│   ├── db.py                       # SQLAlchemy 2.0 ORM
│   ├── schemas.py                  # Pydantic v2 schemas
│   ├── jobs.py                     # Background benchmark subprocess runner
│   └── main.py                     # create_app() factory + 9 endpoints
│
├── ui/                             # Next.js 14 App Router + Tailwind + Recharts
│   ├── app/
│   │   ├── page.tsx                # Screen 1: Scenario Builder
│   │   ├── estimate/page.tsx       # Screen 2: Replica range chart + confidence
│   │   ├── benchmark-plan/page.tsx # Screen 3: Ordered test matrix
│   │   └── report/page.tsx         # Screen 4: Recommendation + Markdown export
│   └── components/
│       ├── ConfidenceBadge.tsx
│       ├── ModeBadge.tsx
│       ├── ReplicaRangeChart.tsx
│       └── CopyButton.tsx
│
├── collect/
│   └── run_bench.py                # Async benchmark harness; OpenAI-compatible; streaming TTFT
│
├── evaluate/
│   └── run_eval.py                 # Quality evaluator — DeepEval + LLM-judge
│
├── analyze/
│   ├── report.py                   # JSON → Markdown latency report (stdlib only)
│   └── deployment_advisor.py       # Latency + quality + cost → deployment recommendation
│
├── datasets/
│   ├── chat.jsonl                  # 15 enterprise chat eval prompts
│   ├── rag.jsonl                   # 15 RAG eval prompts
│   └── long_context.jsonl          # 15 long-document analysis prompts
│
├── tests/                          # 345 tests
│   ├── test_capacity.py            # 55: roofline model, prefill/decode, KV budget
│   ├── test_api.py                 # 35: FastAPI acceptance tests, isolated SQLite
│   ├── test_ingest_anchor.py       # 29: ingest_anchor + confidence rubric
│   ├── test_catalog.py             # 29: GPU/model catalog loading and lookup
│   ├── test_benchmark_plan.py      # 27: test matrix ordering, step count, priority
│   ├── test_deployment_advisor.py  # 26: load_deployment, compute_tradeoff, recommend
│   ├── test_report.py              # 24: mode badges, confidence bands, cost envelope
│   ├── test_explain.py             # 17: napkin-math sections, peak_rps param, cost/user conditionals
│   ├── test_run_eval.py            # 18: dataset loading, score normalization, sidecar
│   ├── test_cost.py                # 18: cost envelope, on-demand vs reserved
│   ├── test_efficiency.py          # 16: MFU + bandwidth efficiency curves
│   ├── test_compare.py             # 19: cheapest/safest/best-latency scoring
│   ├── test_validation.py          # 18: fit, accuracy, adapter routing, CV, sensitivity
│   └── test_intake.py              # 14: three demand modes, precedence, WorkloadError, divergence warning
│
├── results/
│   ├── real/                       # Populated by run_bench.py (gitignored)
│   └── quality/                    # Quality sidecars from run_eval.py
│
├── examples/
│   ├── local-vllm/QUICKSTART.md    # Local vLLM / Docker / SGLang
│   └── baseten/QUICKSTART.md       # Benchmark a Baseten-deployed model
│
├── render.yaml                     # Render deployment blueprint
└── vercel.json                     # Vercel config (Next.js root = ui/)
```

---

## 7. Deployment (Vercel + Render)

```bash
# API on Render (render.yaml blueprint)
# UI on Vercel (vercel.json)
```

| Var | Where | Value |
| --- | --- | --- |
| `DATABASE_URL` | Render (auto) | Postgres connection string |
| `ALLOWED_ORIGIN` | Render (manual) | Vercel deployment URL |
| `RESULTS_DIR` | `render.yaml` | `/tmp/results` |
| `BACKEND_URL` | Vercel (manual) | Render service URL |

---

## 8. Contributing

1. **Physics before empiricism.** New efficiency curve constants need a citable source (public benchmark, vendor perf overview, or a `fit_role: level` point in `catalog/benchmarks_public.yaml`). Ad-hoc constants get rejected.
2. **Engine separation.** Never mix TRT-LLM throughput numbers into vLLM estimates. Use the `engine_factor` mechanism in `efficiency_constants.yaml`.
3. **Tests first.** Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` before committing. 345 tests, all must pass.

---

## 9. License

Apache 2.0.
