# Roofline Model — How `planner/capacity.py` Works

This document traces every formula and decision in the capacity planner from raw
inputs to the final replica recommendation. Read it top-to-bottom: each section
builds on the previous one. The goal is to make the physics reviewable without
having to read Python.

---

## TLDR — The approach in plain language

### The core insight: two phases, two bottlenecks

LLM inference is not a single workload. It has two fundamentally different phases, and they saturate completely different parts of the hardware:

- **Prefill** processes all the input tokens at once. This is an enormous matrix multiplication — the kind that fills tensor cores. Throughput here is limited by **peak FLOPS**, not memory. More FLOPS → faster prefill.
- **Decode** generates one output token per request per step. The weight matrices are the same size as in prefill, but the batch of *new* activations is tiny — often just one token. There is almost no arithmetic to amortize the cost of loading those weights. Throughput here is limited by **HBM memory bandwidth**, not FLOPS. More bandwidth → faster decode.

Sizing a deployment with a single throughput number ignores this split. A workload with very long inputs (ISL >> OSL) needs far more compute headroom; a high-throughput generation workload (many concurrent long outputs) needs far more memory bandwidth. Treating them the same leads to systematic over- or under-provisioning.

### Why roofline, not utilization guessing

The roofline model (originally from HPC) asks a simpler question: *what is the theoretical maximum throughput this hardware can deliver for this workload?* Given peak FLOPS and peak HBM bandwidth, you can compute a hard ceiling for each phase — before you know anything about software efficiency. Actual throughput will be some fraction of that ceiling, but it can never exceed it.

This is more principled than starting with an assumed GPU utilization (e.g. "assume 60% utilization") which has no physical grounding and is usually wrong in both directions depending on workload shape.

### Calibrated efficiency, not flat constants

The gap between hardware ceiling and measured throughput is real, but it is *predictable*. Rather than applying a flat MFU constant from the GPU catalog, the planner uses regime-aware efficiency curves:

- **Prefill MFU** rises with model size (larger GEMMs keep tensor cores fed) and with input length (longer sequences amortize the fixed cost of loading weights). MoE models take a penalty for routing overhead and expert load imbalance.
- **Decode bandwidth efficiency** rises with batch size (more sequences amortize the weight read across more work).

These curves are fitted against citable public benchmark points — not invented. When you have a real measurement from your own hardware (a calibration anchor), that overrides everything. Confidence tiers express how much of the estimate rests on measurement versus inference:

| Tier | Source | Replica range |
|---|---|---|
| HIGH | Anchor from your GPU, near-ISL match | ±10% |
| MEDIUM | Anchor on same model, different hardware or ISL | ±20% |
| DEFAULT | Efficiency curves only, no measurement | ±25% |

### Three constraints, take the max

Once the per-GPU ceilings are known, replica count is determined by whichever of three independent constraints binds first:

1. **Prefill-driven** — how many replicas are needed to absorb the peak input token rate?
2. **Decode-driven** — how many replicas are needed to sustain the peak output token rate?
3. **KV-memory (concurrency)-driven** — each replica can hold only so many sequences in its KV cache simultaneously. Little's Law converts peak RPS and average request latency into a required concurrency, and that caps how many replicas are needed for this reason alone.

The binding constraint is reported explicitly so you know *why* you need the replicas you need. A headroom factor (40% for realtime traffic, 10% for batch) is applied on top for traffic spikes and rolling restarts.

---

## Inputs

| Source | What it provides |
|---|---|
| `catalog/gpus.yaml` | peak TFLOPS per dtype, HBM bandwidth (GB/s), VRAM (GB), GPU arch (`hopper`/`ampere`/`ada`/`blackwell`), memory type (`hbm`/`gddr`), default MFU + bw_eff fallbacks |
| `catalog/models.yaml` | param counts (total + active), num_layers, d_model, KV head config, native dtype, optional resident_weights_gb, geometry_source |
| `catalog/anchors.yaml` | measured TTFT + throughput from past benchmark runs (optional but improves confidence) |
| `catalog/benchmarks_public.yaml` | citable public benchmark points (MLPerf, vendor perf overviews); used by `validate.fit()` to tune efficiency_constants.yaml; schema_version 2 with fit_role field |
| `planner/efficiency_constants.yaml` | MFU base values by GPU arch + dtype, bandwidth efficiency base by memory type, model-size saturation constants (size_floor, size_scale), ISL saturation constants (isl_floor, isl_scale), MoE factor, engine_factor per serving engine (vllm=1.0 reference, trtllm=~1.3) |
| `catalog/runtimes.yaml` | Runtime engine profiles (vLLM, TRT-LLM, SGLang, managed); reference engine designation; engine_factor (relative to vLLM=1.0) |
| Scenario / CLI | requests_per_day (or avg_rps or users×prompts — resolved by `planner/intake.py` before `plan()` is called), peak_multiplier, ISL, OSL, TTFT SLO, dtype, tp, traffic_class, gpu_mem_util; optional: users (cross-cutting, unlocks $/user/month), prefix_cache_len + prefix_cache_hit_rate (reduce effective prefill ISL; KV budget unchanged), max_num_seqs (scheduler batch cap) |

**Dtype byte widths used throughout:**

| dtype | bytes/param |
|---|---|
| fp32 | 4 |
| bf16 / fp16 | 2 |
| fp8 / int8 | 1 |
| mxfp4 / int4 | 0.5 |

If the requested dtype is not in this table, or is not supported by the chosen GPU,
`plan()` raises `CatalogError` immediately before any calculation begins.

**KV bytes per token** (from model card geometry):

```
kv_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × kv_dtype_bytes
```

The factor of 2 is the K and V tensors. `kv_dtype_bytes` defaults to 1 (fp8
KV cache is vLLM's default for H100).

---

## Pre-stage — Demand resolution (`planner/intake.py`)

Before `plan()` is called, `resolve_demand(DemandSpec)` converts one of three input modes into a single `requests_per_day` value:

| Mode | Input fields | Conversion |
|---|---|---|
| Direct | `requests_per_day` | passthrough |
| RPS | `avg_rps` | `avg_rps × 86,400` |
| User-based | `users`, `prompts_per_user_per_day` | `users × prompts_per_user_per_day` |

Precedence when multiple modes are supplied: `requests_per_day` → `avg_rps` → `users × prompts`. A `warnings.warn` is emitted if two sources diverge by more than 20%. Zero sources → `WorkloadError`.

`users` is a cross-cutting optional — it is carried through to `CapacityEstimate.users` unchanged and never enters the roofline math. It unlocks `CostVariant.cost_per_user_per_month` in `cost.py`.

---

## Stage 1 — Traffic normalisation

```
avg_rps         = requests_per_day / 86,400
peak_rps        = avg_rps × peak_multiplier
effective_isl   = ISL − floor(prefix_cache_len × prefix_cache_hit_rate)   (= ISL when prefix cache not set)
input_tps_peak  = peak_rps × effective_isl    (tokens/s of prefill at peak)
output_tps_peak = peak_rps × OSL              (tokens/s of decode at peak)
```

The planner sizes for **peak**, not average.
`peak_multiplier` (default 3×) models the ratio of peak hour to average hour.

**Prefix cache** (`--prefix-cache-len`, `--prefix-cache-hit-rate`): when a system prompt or shared context is cached, those tokens do not need to be recomputed. `effective_isl` captures the tokens that must actually be prefilled. The KV budget (Stage 2) always uses the full `ISL` — cached tokens still occupy VRAM so they remain resident for reuse. When prefix cache is not configured, `effective_isl = ISL`.

---

## Stage 2 — KV cache budget

**Per-GPU weight memory:**

```
# When resident_weights_gb is set in the catalog (preferred for mixed-precision models):
weights_bytes_per_gpu = resident_weights_gb × 1e9 / tp

# Fallback when not set:
weights_bytes_per_gpu = total_params × dtype_bytes / tp
```

`resident_weights_gb` is used when set because mixed-precision checkpoints (e.g. a
MoE model with 4-bit experts and bf16 attention/norm layers) cannot be accurately
represented by a single dtype multiplied by total params. For gpt-oss-20b: the
catalog sets `resident_weights_gb = 13.0` vs the fallback of `20.9B × 0.5 = 10.45 GB`.

With tensor parallelism `tp`, each GPU holds `1/tp` of the weights.

**Usable memory per GPU:**

```
usable_mem = vram_gb × 1e9 × gpu_mem_util   (default gpu_mem_util = 0.90)
```

The 10% headroom covers CUDA context, activation buffers, and framework overhead.
An additional 0.5 GB fixed constant (`FIXED_OVERHEAD_BYTES`) is subtracted for
PyTorch/vLLM housekeeping.

**KV cache budget (TP group):**

```
kv_shard_factor = min(tp, num_kv_heads)
kv_cache_budget = (usable_mem − weights_bytes_per_gpu − 0.5 GB) × kv_shard_factor
```

KV heads shard across TP ranks, but only up to `num_kv_heads` ways. When `tp > num_kv_heads`
(e.g. tp=8 on Qwen3-30B-A3B which has 4 KV heads), excess ranks replicate KV rather than
extending the budget. Using `tp` directly would over-count the available KV pool.

For most models with `num_kv_heads ≥ tp`, `kv_shard_factor = tp` and the formula is
equivalent to the original.

**Maximum concurrent sequences:**

```
max_kv_tokens           = kv_cache_budget / kv_bytes_per_token
effective_context_tokens = (ISL + OSL)                            # full-attention models
max_concurrent_seqs     = floor(max_kv_tokens / effective_context_tokens)
```

This is the hard ceiling on how many requests can be in flight simultaneously
in one replica. If it drops below 4, the planner emits a warning.

**Sliding-window interleaved attention (Gemma 2/3/4).**
Models with `sliding_window` and `global_layer_every_n` set in `catalog/models.yaml`
use a mixed-layer formula. Local layers only need to store up to `sliding_window`
tokens of KV; global layers store the full sequence. The planner averages across
all layers:

```
n                        = global_layer_every_n          # e.g. 6 for Gemma 3/4
global_layers            = num_layers // n
local_layers             = num_layers − global_layers
effective_context_tokens = (global_layers × (ISL+OSL)
                            + local_layers × min(ISL+OSL, sliding_window))
                           / num_layers
max_concurrent_seqs      = floor(max_kv_tokens / effective_context_tokens)
```

For Gemma 3 12B at ISL=32768, OSL=512 (window=1024, n=6):

- Without fix: effective_context = 33280 tokens/layer → baseline concurrency
- With fix: effective_context ≈ (8 × 33280 + 40 × 1024) / 48 ≈ 6413 → **5× more concurrent sequences**

`KvBudget.effective_context_tokens` stores the computed value for downstream use
(explainer, report). For full-attention models it equals `ISL + OSL` unchanged.

**Hard error:** If weights alone do not fit in usable memory at the chosen `tp`,
the planner raises `CatalogError` rather than emitting a nonsense number.

---

## Stage 2b — Scheduler batch cap (`--max-num-seqs`)

vLLM's `--max-num-seqs` flag limits how many sequences the scheduler will run concurrently, independently of available KV cache VRAM. When set, the planner applies:

```
effective_max_seqs = min(max_concurrent_seqs, max_num_seqs)
```

`effective_max_seqs` replaces `max_concurrent_seqs` in all downstream calculations: `eff_batch` (Stage 5b), `replicas_concurrency` (Stage 6). The KV budget itself is unchanged — the cap is a scheduler policy, not a memory constraint.

When `max_num_seqs ≥ max_concurrent_seqs` (or `--max-num-seqs` is not set), this stage is a no-op.

---

## Stage 3 — Efficiency curves

Rather than applying flat MFU and bandwidth efficiency values, the planner uses
**regime-aware monotonic curves** (`planner/efficiency.py`) whose constants are
stored in `planner/efficiency_constants.yaml` and tuned by `validate.fit()`.

### MFU for prefill — `mfu_prefill(model, gpu, dtype, isl)`

```
mfu = clamp(base × f_size × f_isl × f_moe,  low=0.08,  high=base)
```

| Factor | Formula | Physics |
|---|---|---|
| `base` | `mfu_base[arch][dtype]` | Asymptotic tensor-core utilisation for large model at long ISL |
| `f_size` | `size_floor + (1−size_floor) × (1−exp(−active_params / size_scale))` | Larger active GEMMs keep tensor cores fed; small models under-utilise |
| `f_isl` | `isl_floor + (1−isl_floor) × (1−exp(−isl / isl_scale))` | Longer prefill = more arithmetic per weight byte; saturates above `isl_scale` |
| `f_moe` | `moe_factor` (if MoE), else `1.0` | Routing overhead + smaller per-expert GEMMs + load imbalance |

Falls back to `gpu.default_mfu_prefill` when GPU `arch` is absent or unmapped
(e.g. custom GPU specs in tests).

Fitted constants after `validate.fit()` on the public benchmark set (hopper fp8 example):
`base=0.530`, `size_floor=0.482`, `size_scale=5.86 B params`, `isl_floor=0.392`, `isl_scale=512`, `moe_factor=0.80`.

### Bandwidth efficiency for decode — `bw_eff_decode(gpu, eff_batch)`

```
bw_eff = clamp(base × g_batch,  low=base×batch_floor,  high=base)
```

| Factor | Formula | Physics |
|---|---|---|
| `base` | `bw_base[memory_type]` | Asymptotic HBM or GDDR utilisation at large batch |
| `g_batch` | `batch_floor + (1−batch_floor) × (1−exp(−eff_batch / batch_scale))` | Weight-amortisation: with `batch_floor` near 1.0 this is ≈1.0 for practical serving batch sizes |

KV reads are already counted explicitly in `decode_ceiling`'s `bytes_per_step`
(the `eff_batch × kv_bytes_per_token × avg_ctx / tp` term). Applying a `g_kv`
penalty here would double-count them. The natural floor is `base × batch_floor`.

### Bandwidth efficiency for prefill — `bw_eff_prefill(gpu)`

```
bw_eff_prefill = bw_base[memory_type]
```

Used for the bandwidth ceiling in `prefill_ceiling()` when ISL is below the ridge
point (bandwidth-bound prefill). Prefill processes one request at a time (effective
batch=1 for weight reads), so no batch or KV adjustment is applied.

### Precedence

```
measured anchor  >  efficiency curve  >  hard floor (0.08 for MFU; bw_eff has no separate hard floor — natural floor = base × batch_floor)
```

When a HIGH-confidence anchor exists, `plan()` uses the measured MFU directly
and bypasses the efficiency curve for prefill. The decode `bw_eff` is not yet
calibrated from anchors — it always uses the curve output.

---

## Stage 4 — Confidence and calibration

The confidence rubric sets the **replica range band** and selects the `mfu` value
(anchor-measured or curve-predicted) used in Stage 5.

### Anchor search (in priority order)

The scenario concurrency passed to the rubric is **`eff_batch`** — the expected steady-state operating batch (`floor(effective_max_seqs × batch_efficiency)`), not the KV-cache capacity ceiling. Using the KV ceiling would let a single low-concurrency measurement (e.g. c=0.6 from a 2 RPS single-replica run) grant HIGH confidence to plans that operate at hundreds of concurrent sequences and were never measured at that load.

1. **Exact family** — anchors where `(model, gpu, dtype)` all match.
   - ISL distance = `|anchor_isl − scenario_isl| / scenario_isl`
   - Concurrency ratio = `max(anchor.concurrency, eff_batch) / min(anchor.concurrency, eff_batch)`
   - If ISL distance ≤ 20% **and** concurrency ratio ≤ 10×: **HIGH** confidence; MFU taken from `derived_mfu_prefill`.
   - Otherwise (ISL extrapolated OR concurrency ratio > 10×): **MEDIUM**; MFU from efficiency curve.

   The 10× ratio gate (`HIGH_CONCURRENCY_RATIO`) is ratio-based, not normalized, because normalized distance `abs(a−b)/max(a,b)` caps at 1.0 and cannot distinguish c=0.6→eff_batch=59 (0.99) from c=10→eff_batch=59 (0.83) at any useful threshold. The ratio (98× vs 5.9×) cleanly separates them.

2. **Same model, different GPU/dtype** — model measured elsewhere: **MEDIUM**,
   efficiency curve for both MFU and bw_eff.

3. **No anchor for this model** — pure roofline from efficiency curves: **DEFAULT**.

### Geometry downgrade

If `model.geometry_source == "estimated"` (custom model entered via param count,
not from the catalog), confidence is downgraded one level:
- HIGH → MEDIUM
- MEDIUM → DEFAULT

This is because inferred geometry (d_model, num_layers, head config) can be
materially wrong for MoE or non-standard architectures.

### Confidence tiers and band factors

| Tier | Band | Meaning |
|---|---|---|
| HIGH | ±10% | Calibrated from a near-ISL anchor on the same hardware |
| MEDIUM | ±20% | Extrapolated or cross-GPU anchor |
| DEFAULT | ±25% | No measurement; pure efficiency-curve estimate |

Note: The former `LOW (±50%)` tier was renamed `DEFAULT (±25%)` and the band was
tightened when the efficiency curves replaced flat GPU-catalog defaults. The curves
are significantly more accurate than ad-hoc constants, so the no-anchor uncertainty
is narrower than before.

---

## Stage 5 — Throughput ceilings

### 5a. Prefill ceiling (compute-bound vs bandwidth-bound)

**FLOPs per prefill token:**

```
q_proj_dim      = num_q_heads × head_dim
flops_per_token = 2 × active_params                         (linear layers: matmuls)
                + 2 × num_layers × ISL × q_proj_dim         (attention: grows with context)
```

The attention term uses `q_proj_dim` (the Q-head output dimension), not `d_model`.
For standard transformers they are equal, but for GQA-heavy or non-standard
architectures they can differ. Example: Qwen3-30B-A3B has `d_model = 2048` but
`num_q_heads × head_dim = 32 × 128 = 4096` — using `d_model` would halve the
attention FLOPs and overestimate prefill throughput.

**Compute-bound ceiling (TP group):**

```
compute_tps = peak_flops_tflops × 1e12 × mfu × tp / flops_per_token
```

`mfu` comes from `mfu_prefill()` (efficiency curve) or a measured anchor.

**Bandwidth-bound ceiling (TP group):**

```
weight_bytes = active_params × dtype_bytes
bw_tps = ISL × hbm_bandwidth_gbps × 1e9 × bw_eff_prefill × tp / weight_bytes
```

`bw_eff_prefill` = `bw_base[memory_type]` from `efficiency_constants.yaml`.

This captures the "streaming" regime at short ISL: if the batch is so small that
the GPU finishes loading weights before the tensor cores saturate, throughput is
limited by how fast weights stream from HBM, not by TFLOPS.

**Prefill ceiling = min(compute_tps, bw_tps)**

**Ridge point** — the ISL at which the two bounds are equal:

```
ridge_point ≈ peak_flops × mfu / (bandwidth × bw_eff_prefill)
```

For H100 fp8 with fitted constants (Phase A): `1979e12 × 0.53 / (3350e9 × 0.39) ≈ 803 tokens`.

- **ISL > ridge point**: compute-bound; `bw_tps` does not bind.
- **ISL < ridge point**: bandwidth-bound; adding more TFLOPS does not help.

Both ceilings scale linearly with `tp`.

### 5b. Decode ceiling (bandwidth-bound, with compute guard)

**Effective batch:**

```
eff_batch = max(1, floor(effective_max_seqs × batch_efficiency))
```

`batch_efficiency = 0.70` (default). vLLM continuous batching is never 100% full
in steady state — requests arrive and complete asynchronously. 0.70 is the
empirically-observed fill fraction. Sizing at 100% would under-provision by ~18%.

`effective_max_seqs = min(max_concurrent_seqs, max_num_seqs)` (see Stage 2b). When `--max-num-seqs` is tighter than the KV budget, it is the binding limit on batch size.

**Average KV context during decode:**

```
avg_ctx = ISL + OSL // 2
```

**Decode bw_eff from the efficiency curve:**

```
g_batch   ≈ 1.0  for practical serving batch sizes   (batch_floor = 0.80 post-fit)
decode_bw = clamp(bw_base[memory_type] × g_batch,  low=base×batch_floor,  high=bw_base)
```

KV bytes are already in `bytes_per_step` (see the decode ceiling formula below).
No separate `g_kv` factor is applied — doing so would double-count KV traffic.

**Accuracy note:** The roofline models the **hardware ceiling**. vLLM serving
throughput at medium batch (e.g. 32–128 concurrent) runs ~50% of that ceiling due
to PagedAttention block scatter and scheduler overhead — a structural gap not
capturable by the roofline. At batch=1 and at very large batch (max_concurrent_seqs),
the model is much more accurate.

**MoE expert coverage (decode weight term):**

For dense models, each decode step reads `active_params` weights. For MoE models
at batch > 1, different sequences route to different experts. By the birthday problem,
the union of experts touched per step grows with batch:

```
distinct_frac(B) = 1 − (1 − experts_per_token / num_experts) ^ B
```

At batch=64 for gpt-oss-20b (32 experts, top-4): `1 − (0.875)^64 ≈ 99.99%` of
the expert pool is touched per step. For batch=1 it's just `4/32 = 12.5%`.

The weight bytes term is therefore:

```
# Dense models:
weight_bytes_step = active_params × dtype_bytes

# MoE models:
r               = experts_per_token / num_experts
dense_params    = (active_params − r × total_params) / (1 − r)   # non-expert layers
expert_pool     = total_params − dense_params
weight_bytes_step = (dense_params + distinct_frac(eff_batch) × expert_pool) × dtype_bytes
```

At large batch, MoE decode approaches dense-model memory pressure — the `active_params`
shortcut significantly underestimates bytes/step and over-estimates throughput.

**Decode ceiling formula (TP group):**

```
achievable_bw   = hbm_bandwidth_gbps × 1e9 × decode_bw
bytes_per_step  = weight_bytes_step / tp
                + eff_batch × kv_bytes_per_token × avg_ctx / tp
bw_tps          = eff_batch × achievable_bw / bytes_per_step
```

**Compute ceiling (decode):**

```
compute_tps = peak_flops × 1e12 × mfu × tp / (2 × active_params)
```

**Decode ceiling = min(bw_tps, compute_tps)**

The compute ceiling binds only at very large batch on small models. For typical
serving workloads, `bw_tps` binds first.

**Key note — weight + KV are summed, not max'd.** They share the same HBM bus.

---

## Stage 6 — Replica sizing

Three independent constraints produce three replica counts; the binding one wins.

### Prefill-driven replicas

```
replicas_prefill = ceil(input_tps_peak / prefill_ceiling_per_replica)
```

`input_tps_peak = peak_rps × effective_isl` — already reduced by prefix caching at Stage 1.

### Decode-driven replicas

```
replicas_decode = ceil(output_tps_peak / decode_ceiling_per_replica)
```

### Concurrency-driven replicas (Little's Law)

```
ttft_rough_s     = effective_isl / prefill_ceiling
itl_s            = eff_batch / decode_ceiling        (inter-token latency at eff_batch)
avg_latency_s    = ttft_rough_s + OSL × itl_s
peak_concurrent  = peak_rps × avg_latency_s          (Little's Law: L = λW)
replicas_concurrency = ceil(peak_concurrent / effective_max_seqs)
```

`itl_s` uses `eff_batch` (not `max_concurrent_seqs`) because `decode_ceiling` was
computed at `eff_batch`. Using `max_concurrent_seqs` would inflate the latency
estimate and overstate `replicas_concurrency`.

### Headroom and binding constraint

```
base_replicas = max(replicas_prefill, replicas_decode, replicas_concurrency)
replicas      = ceil(base_replicas × headroom_factor)
```

| Traffic class | Headroom | Rationale |
|---|---|---|
| realtime | 1.40× | 40% spare for traffic spikes and rolling restarts |
| mixed | 1.25× | |
| batch | 1.10× | Batch jobs tolerate short queuing |

The fields `binding_constraint` and `headroom_factor` are both stored on `CapacityEstimate` (the latter for use by `render_napkin_math`).

### Total GPUs

```
total_gpus = replicas × tp
```

`tp_used` is stored in the estimate for display and downstream use.

### TPOT (inter-token latency per request)

```
tpot_ms = (eff_batch / decode_ceiling) × 1000
```

---

## Stage 7 — Confidence range widening

```
replicas_low  = max(1, ceil(replicas × (1 − band_factor)))
replicas_high =       ceil(replicas × (1 + band_factor))
```

Band factors by tier:

| Tier | band_factor | replicas range |
|---|---|---|
| HIGH | 0.10 | point ±10% |
| MEDIUM | 0.20 | point ±20% |
| DEFAULT | 0.25 | point ±25% |

---

## Stage 8 — TTFT estimate

Uses an **M/M/1 queuing model** (single-server queue, Poisson arrivals, exponential
service times) applied per replica.

```
rho             = min(input_tps_peak / (replicas × prefill_ceiling), 0.94)
ttft_compute_s  = effective_isl / prefill_ceiling_per_replica
queue_wait_s    = ttft_compute_s × rho / (1 − rho)
ttft_ms         = (ttft_compute_s + queue_wait_s) × 1000
```

`rho` is the prefill utilization per replica. It is clamped at 0.94 to prevent
division-by-zero as utilization approaches 1.

The M/M/1 model is a simplification: real vLLM schedulers use continuous batching
and chunked prefill, which change queuing dynamics. This estimate is appropriate
for order-of-magnitude planning and SLO checks; validate with `run_bench.py` before
production capacity commitment.

---

## Confidence rubric summary

```
anchor (model, gpu, dtype) + ISL within ±20% + concurrency ratio ≤ 10×  →  HIGH    (±10% range)
anchor (model, gpu, dtype) + ISL beyond ±20% OR concurrency ratio > 10×  →  MEDIUM  (±20% range)
anchor same model, different gpu/dtype                                    →  MEDIUM  (±20% range)
no anchor for this model                                                  →  DEFAULT (±25% range)

geometry_source == "estimated"  →  downgrade one level
  HIGH → MEDIUM, MEDIUM → DEFAULT

Scenario concurrency = eff_batch (steady-state operating batch), not KV-cache ceiling.
Concurrency ratio    = max(anchor.c, eff_batch) / min(anchor.c, eff_batch)   [ratio-based, not normalized]
```

---

## Efficiency curve calibration — `planner/validate.py`

The efficiency curves are data-driven: `validate.fit()` runs coordinate descent
over the constants in `planner/efficiency_constants.yaml` to minimize median
relative error against citable public benchmark points.

### Benchmark catalog — `catalog/benchmarks_public.yaml`

Schema version 2. Points are tagged with a `fit_role`:

| fit_role | Engine | Used in fit | Purpose |
|---|---|---|---|
| `level` | vLLM | Yes — absolute level | Pins the efficiency LEVEL; currently 2 Trelis A100 latency points |
| `shape` | TRT-LLM | Yes — curve shape + engine_factor | ISL/size sweep; `engine_factor[trtllm]` absorbs the vLLM→TRT-LLM throughput lift |
| `validate` | vLLM | No — predicted, not fit | Distribution-dataset points (CNN-DM mean ISL); predicted and reported at widened tolerance |
| `sanity` | Various | No | Coarse order-of-magnitude check only; engine/attribution unclear |

**Engine confound policy:** Different serving engines realize different fractions of
peak for the same hardware. vLLM is the reference engine (`engine_factor=1.0`);
TRT-LLM achieves ~1.3× higher throughput on the same hardware. The `engine_factor`
is stored in `efficiency_constants.yaml` and fitted from the `shape` benchmark points.
User predictions are always made at their actual runtime's engine factor.

### Fit benchmark set (Phase A — provisional)

| fit_role | Source | GPU | Model | Scenario |
|---|---|---|---|---|
| `level` | Trelis one-click-llms | A100 SXM | Llama-3.1-8B fp8 | latency, batch=1 |
| `level` | Trelis one-click-llms | A100 SXM | Llama-3.1-8B fp8 | latency, batch=64 |
| `shape` | TRT-LLM v0.20 (12 pts) | H100/H200 SXM | Llama-3.1-8B/3.3-70B/4-Maverick fp8 | offline ISL sweep |
| `validate` | Red Hat MLPerf v5.1 (vLLM 0.10.0) | H100/L40S | Llama-3.1-8B fp8 | offline + server |

Red Hat points use CNN-DailyMail (mean ISL=778, distribution dataset). At mean ISL the
workload is prefill-infeasible on H100 (needs 61.5k input tok/s; ceiling is 37k), so
the uniform-dataset roofline cannot correctly predict them. They are predicted and checked
at widened tolerance (`fit_role: validate`), not included in the fit objective.

Phase B stubs are pre-staged in `catalog/benchmarks_public.yaml` (`measured: null`; skipped
by `load_public_benchmarks` until filled). Fill the 5 measured values from
`catalog/runpod_phase_b.sh` output, then run `catalog/phase_c_refit.py` to pin
`engine_factor[trtllm]` as the direct matched-pair median and promote those stubs to
`fit_role: level`.

### Post-fit accuracy (Phase A — level + shape fit set, provisional)

After `validate.fit()` on level + shape points:

**Median: ~30%  |  P90: ~205%  |  Max: ~227%**

The large p90/max is dominated by the Llama-3.3-70B and Llama-4-Maverick TRT-LLM points:
the roofline over-predicts MoE/large-dense throughput at high batch because MoE dispatch
latency and expert routing overhead at tp=8 are not modeled. The A100 batch=64 point also
has ~2× structural gap from vLLM serving overhead. These will improve in Phase C.

Validate points (Red Hat, distribution): median ~19%.

### Fitted constants (as of last `validate.fit()` run — Phase A provisional)

| Constant | Value | Controls |
|---|---|---|
| `mfu_base.hopper.fp8` | 0.48–0.53 | H100/H200 fp8 asymptotic prefill MFU |
| `bw_base.hbm` | 0.39–0.40 | HBM (A100/H100/H200) bandwidth efficiency |
| `bw_base.gddr` | 0.35 | GDDR6/6X (L40S/L4) bandwidth efficiency |
| `engine_factor.trtllm` | ~1.29 | TRT-LLM throughput lift over vLLM reference |
| `size_floor` | 0.48–0.80 | Minimum MFU scaling for tiny models |
| `size_scale` | 4–6 B | Active params at which MFU reaches half of (base−floor) |
| `isl_floor` | 0.39–0.49 | Minimum MFU scaling for very short ISL |
| `isl_scale` | 512 | ISL at which MFU reaches half of (base−floor) |

`kv_scale` has been removed — KV traffic is counted once in `decode_ceiling`'s
`bytes_per_step`, not in `bw_eff_decode`.

To refit after adding benchmark points:
```bash
python3 -c "from planner.validate import fit, load_public_benchmarks; fit(load_public_benchmarks())"
```

### Phase B → C calibration workflow

**Phase B** (requires RunPod 1×H100-SXM pod):
```bash
# On the pod:
bash catalog/runpod_phase_b.sh 2>&1 | tee phase_b_results.txt
# Fill the 5 `measured: null` values in catalog/benchmarks_public.yaml
# Also fill `engine_version`
```

**Phase C** (local, after Phase B YAML is filled):
```bash
python catalog/phase_c_refit.py
# Automatically: pins engine_factor, narrows PARAM_BOUNDS, runs refit (seed=42),
# prints leave-one-GPU-out CV, parameter sensitivity, and suggested test targets.
# Then commit the updated constants + YAML + tests.
```

---

## Known limitations and exceptions

### What the model does NOT include

| Gap | Impact | Mitigation |
|---|---|---|
| **USL / TP communication overhead** | NVLink within-node (<5% at tp≤8); InfiniBand cross-node 15–30% | Use anchors from target hardware; add USL correction when tp>8 |
| **Prefix caching hit rate** | Implemented via `--prefix-cache-len` + `--prefix-cache-hit-rate`; reduces effective prefill ISL; KV budget uses full ISL | Set both flags to model your system prompt cache |
| **Chunked prefill** | Chunks ISL > 4096 into smaller pieces; changes TTFT profile | Planner emits a warning; TTFT is a worst-case without chunking |
| **MoE routing imbalance** | Expert load uneven across ranks; effective FLOPS per token varies | MoE coverage fraction assumes uniform routing; add imbalance factor for skewed workloads |
| **Speculative decoding** | Can multiply decode throughput by acceptance rate | Not modelled; requires OSL + acceptance rate adjustments |
| **Disaggregated prefill/decode (llm-d)** | Separate P and D pools allow independent scaling | Not yet implemented |
| **Decode anchor calibration** | `bw_eff` calibrated by `validate.fit()` against public benchmarks, but NOT from user's own `anchors.yaml` data yet | Even at HIGH confidence, decode uses the curve output, not the user's anchor |
| **vLLM serving overhead at medium batch** | PagedAttention scatter + scheduler overhead reduce throughput ~50% vs hardware ceiling at batch 32–256 | Roofline systematically over-predicts in this regime; use DEFAULT confidence band (±25%) |
| **Engine factor** | TRT-LLM achieves ~1.3× higher throughput than vLLM on the same hardware | `engine_factor` is implemented in `efficiency_constants.yaml` (trtllm≈1.29, vllm=1.0). Phase C will pin the value from matched-pair calibration runs. |
| **Quantisation accuracy loss** | fp8/mxfp4 may degrade quality; planner is throughput-only | Use `evaluate/run_eval.py` + `analyze/deployment_advisor.py` |

### Edge cases that produce warnings or errors

| Condition | Behaviour |
|---|---|
| Unknown dtype string | Hard `CatalogError` in `plan()` before any calculation |
| dtype not supported by GPU (e.g. fp4 on A100) | Hard `CatalogError` in `plan()` before any calculation |
| Weights don't fit at chosen dtype + tp | Hard `CatalogError` from `kv_budget()` |
| `max_concurrent_seqs < 4` | Warning: KV budget very tight; suggest higher tp or larger GPU |
| ISL ≥ 4096 | Warning: chunked prefill required for vLLM |
| ISL > `model.context_len` | Warning: ISL exceeds native training context; RoPE scaling or equivalent required |
| `model.global_head_dim` set | Informational warning: global-layer KV differs from local estimate; states direction (over/under) and percentage |
| `geometry_source == "estimated"` | Warning: verify num_layers, d_model before production; confidence downgraded |
| TTFT estimate > SLO | Warning with diagnosis: queue-bound vs compute-bound |
| Confidence == DEFAULT | Warning: validate on live GPU before committing infrastructure |
| Balanced prefill/decode load (ratio < 2×) | Warning: max() assumption may undercount — see note below |

**Balanced prefill/decode warning.** `size_replicas` uses `max(replicas_prefill, replicas_decode)`,
which assumes the binding workload owns the full GPU. For colocated vLLM this is optimistic
when both phases are significant: prefill stresses compute; decode stresses bandwidth; they
share the same GPU. When neither dominates by ≥ 2×, the planner emits a warning and recommends
benchmarking before committing capacity.

The principled fix is to replace the two single-resource ceilings with two *aggregate* utilizations:

```
total_compute_util = input_tps_peak/compute_ceiling_prefill + output_tps_peak/compute_ceiling_decode
total_bw_util      = input_tps_peak/bw_ceiling_prefill      + output_tps_peak/bw_ceiling_decode
replicas           = ceil(max(total_compute_util, total_bw_util) × headroom)
```

### Accuracy expectations by confidence tier

| Tier | Expected accuracy | When |
|---|---|---|
| HIGH | ±10–15% | Calibrated anchor within 20% ISL on same GPU+dtype |
| MEDIUM | ±20–30% | Extrapolated or cross-GPU calibration |
| DEFAULT | ±25–45% | No anchor; efficiency-curve estimate only |

The target accuracy is ±20% for throughput sizing. **HIGH confidence meets this
target. MEDIUM may miss it by a small margin. DEFAULT should be treated as a
directional estimate** — use it to pick a GPU SKU and TP strategy, then run
`run_bench.py` to get an anchor before finalising infrastructure commitments.

### Numerical gotchas

- **`resident_weights_gb` vs `total_params × dtype_bytes`** — when set, the catalog
  value always wins. For MoE models with mixed-precision checkpoints the two can
  differ by 20%+, directly affecting KV budget and concurrency ceiling.

- **`kv_cache_budget` caps at `num_kv_heads`** — the budget scales with
  `min(tp, num_kv_heads)`, not `tp`. At tp=8 on a 4-KV-head model, the budget
  is the same as at tp=4. Weight memory still divides by `tp`.

- **MoE decode uses `active_params` only for compute ceiling** — the bandwidth
  (weight bytes) term uses the birthday-problem coverage fraction at the serving
  batch. At large batch the effective weight bytes approach `total_params × dtype_bytes`.

- **`bw_eff_prefill` vs `decode_bw`** — prefill uses `bw_base[memory_type]` directly;
  decode applies `g_batch` amortization only (no `g_kv` — KV bytes are counted once in
  `bytes_per_step`). Both values are printed in the assumptions section of the output.

- **TP-group vs per-GPU throughput** — `prefill_ceiling` and `decode_ceiling`
  return TP-group throughput (all `tp` GPUs combined). `size_replicas` uses these
  directly. Do **not** multiply by `tp` a second time.

- **`q_proj_dim` vs `d_model` in attention FLOPs** — use `num_q_heads × head_dim`
  for the O(n²d) attention term, not `d_model`. They are equal for standard
  transformers but differ for some GQA-heavy models.

- **`bw_eff_decode` natural floor is `base × batch_floor`** — there is no separate
  hard floor. The old 0.05/0.22 hard floors are removed because KV pressure is now
  accounted for in `bytes_per_step`, not in `bw_eff_decode`.

---

## How calibration feeds back

After running `collect/run_bench.py`, call:

```bash
python planner/ingest_anchor.py results/real/<tag>.json
```

This writes a new entry to `catalog/anchors.yaml` with `derived_mfu_prefill`
computed from the measured throughput. The next `plan()` call on the same
`(model, gpu, dtype, ISL)` combination will:

1. Find the anchor via the exact-family search.
2. Promote to HIGH confidence if ISL distance ≤ 20%.
3. Use the measured MFU instead of the efficiency curve.
4. Narrow the replica range to ±10%.

Note: `bw_eff` for decode is not yet calibrated from user anchors — it always uses
the efficiency curve output even at HIGH confidence. This is a known gap.

This is the intended workflow: estimate first (DEFAULT), benchmark, ingest,
re-estimate (HIGH).

---

## Post-stage — Napkin-math explainer (`planner/explain.py`)

`render_napkin_math(est: CapacityEstimate, cost: CostEstimate | None = None) -> str`
walks the full sizing chain in plain language. Every number is read from the
estimate — no hardcoded constants.

**Six sections (cost section omitted when `cost=None`):**

| Section | What it shows |
|---|---|
| 1. Traffic normalisation | `req/day ÷ 86,400 = avg RPS × peak_mult = peak RPS` |
| 2. Token demand | `× ISL = input tok/s (prefill)` and `× OSL = output tok/s (decode)` |
| 3. Per-replica ceilings | `prefill_tps_gpu` at `mfu_used`; `decode_tps_gpu` at `bw_eff_used` |
| 4. KV-budget concurrency | `gpu_mem_gb − weights_gb = kv_gb ÷ kv_per_seq_kb = max_concurrent_seqs` |
| 5. Replica sizing | prefill/decode/concurrency driven counts → binding constraint → `× headroom_factor → replicas` |
| 6. Cost (optional) | `replicas × tp × 24h × $/hr = $/day → $/month`; `$/user/month` when `est.users` is set |

**Fields read from `CapacityEstimate`:** `traffic.*`, `kv_budget.*`, `prefill_tps_gpu`, `decode_tps_gpu`, `mfu_used`, `bw_eff_used`, `replicas_prefill`, `replicas_decode`, `replicas_concurrency`, `binding_constraint`, `headroom_factor`, `gpu_mem_gb`, `tp_used`, `replicas`, `users`.

`render_napkin_math` is called automatically by `render_report(..., include_napkin_math=True)` (appends `## How we got here`) and directly by `planner/capacity.py` CLI when `--explain` is passed.

---

## Quick reference — formula sheet

```
# KV bytes per token
kv_bpt = 2 × L × H_kv × D_head × kv_dtype_bytes

# Weights per GPU (use resident_weights_gb when available)
w_bytes = resident_weights_gb × 1e9 / tp          # if set in catalog
        = total_params × dtype_bytes / tp          # fallback

# KV shard factor (caps at num_kv_heads, not tp)
kv_shard_factor = min(tp, num_kv_heads)

# KV cache budget (TP group)
kv_budget = (vram × gpu_mem_util − w_bytes − 0.5 GB) × kv_shard_factor

# Effective context per layer per sequence (sliding-window models)
# full-attention: effective_ctx = ISL + OSL
# interleaved (Gemma 2/3/4): average of global (full) and local (capped) layers
n              = global_layer_every_n
global_layers  = L // n
local_layers   = L − global_layers
effective_ctx  = (global_layers × (ISL+OSL) + local_layers × min(ISL+OSL, sliding_window)) / L

# Max concurrent sequences
max_seqs = floor(kv_budget / (kv_bpt × effective_ctx))

# Effective ISL for prefill (prefix cache reduces compute, not KV storage)
effective_isl = ISL − floor(prefix_cache_len × prefix_cache_hit_rate)   (= ISL when not set)

# Effective max sequences (scheduler cap; = max_seqs when --max-num-seqs not set)
effective_max_seqs = min(max_seqs, max_num_seqs)

# Efficiency curves (planner/efficiency.py, constants in efficiency_constants.yaml)
mfu         = clamp(base × f_size × f_isl × f_moe,  low=0.08,  high=base)
              f_size = size_floor + (1−size_floor) × (1−exp(−active_params / size_scale))
              f_isl  = isl_floor  + (1−isl_floor)  × (1−exp(−isl / isl_scale))
              f_moe  = moe_factor (MoE) or 1.0 (dense)

bw_eff_prefill = bw_base[memory_type]

decode_bw   = clamp(bw_base[memory_type] × g_batch,  low=base×batch_floor,  high=bw_base)
# KV bytes counted once in bytes_per_step, not in bw_eff

# Prefill ceiling (TP group)
q_proj_dim  = num_q_heads × head_dim
compute_tps = TFLOPS × 1e12 × mfu × tp / (2×active_params + 2×L×ISL×q_proj_dim)
bw_tps      = ISL × BW_gbps × 1e9 × bw_eff_prefill × tp / (active_params × dtype_bytes)
prefill_tps = min(compute_tps, bw_tps)

# Effective batch
eff_batch = max(1, floor(effective_max_seqs × 0.70))

# Average KV context
avg_ctx = ISL + OSL // 2

# KV ratio (diagnostic only; not used in bw_eff_decode — KV counted in bytes_per_step)
kv_ratio = kv_bpt × avg_ctx × eff_batch / (active_params × dtype_bytes)

# Decode weight bytes per step (MoE-aware)
distinct_frac  = 1 − (1 − experts_per_token/num_experts) ^ eff_batch   # MoE only
dense_params   = (active_params − r × total_params) / (1 − r)          # r = k/n
weight_bytes_step = (dense_params + distinct_frac × (total_params − dense_params)) × dtype_bytes
                  = active_params × dtype_bytes                         # dense models

# Decode ceiling (TP group)
bytes_per_step  = weight_bytes_step / tp + eff_batch × kv_bpt × avg_ctx / tp
bw_tps          = eff_batch × BW_gbps × 1e9 × decode_bw / bytes_per_step
compute_tps     = TFLOPS × 1e12 × mfu × tp / (2 × active_params)
decode_tps      = min(bw_tps, compute_tps)

# Sizing
replicas_prefill     = ceil(peak_rps × effective_isl / prefill_tps)
replicas_decode      = ceil(peak_rps × OSL / decode_tps)
itl_s                = eff_batch / decode_tps
avg_latency_s        = effective_isl/prefill_tps + OSL × itl_s
replicas_concurrency = ceil(peak_rps × avg_latency_s / effective_max_seqs)
replicas             = ceil(max(...) × headroom)
total_gpus           = replicas × tp

# TPOT
tpot_ms = eff_batch / decode_tps × 1000

# TTFT (M/M/1)
rho      = min(peak_rps × effective_isl / (replicas × prefill_tps), 0.94)
ttft_ms  = (effective_isl / prefill_tps) × (1 / (1 − rho)) × 1000

# Confidence band
replicas_low  = max(1, ceil(replicas × (1 − band_factor)))
replicas_high = ceil(replicas × (1 + band_factor))
  HIGH=0.10, MEDIUM=0.20, DEFAULT=0.25
```
