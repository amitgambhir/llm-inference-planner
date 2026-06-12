# Quality-Aware Benchmarking — Design Spec

**Date:** 2026-06-12  
**Status:** Approved  

---

## Problem

Every inference benchmark tool measures TTFT, throughput, and latency. Almost none answer the question a production AI PM or architect actually cares about:

> "Is the faster deployment actually good enough?"

Without a quality signal, the tool recommends the fastest option. A production team would choose the best quality-per-cost option. This feature closes that gap.

---

## Decision Summary

| Question | Decision |
|---|---|
| Online vs offline quality eval | Offline — different cadences (100k requests vs 50–500 prompts) |
| Dataset format | JSONL with `{schema_version, id, workload, prompt, expected}`, optional RAGAS fields |
| Primary evaluator | DeepEval |
| Secondary evaluator | LLM-as-judge (`--evaluator llm-judge`) |
| Cost model | User-supplied `--cost-per-million-tokens` (primary), throughput ratio (fallback) |
| Approach | Dedicated pipeline: `run_eval.py` → quality sidecar → `deployment_advisor.py` |
| New tradeoff CLI name | `deployment_advisor.py` (decision engine, not analytics) |

---

## Architecture

### Pipeline

```
collect/run_bench.py   →  results/real/<tag>.json          (existing, unchanged)
evaluate/run_eval.py   →  results/quality/<tag>.json       (new quality sidecar)
analyze/deployment_advisor.py --tags t1 t2 t3              (new decision engine)
```

The existing pipeline (`report.py`, `playbook/advisor.py`) is untouched. This feature is purely additive.

### Files Added

```
evaluate/run_eval.py              quality evaluator CLI
analyze/deployment_advisor.py     deployment decision engine CLI
datasets/chat.jsonl               ~50 eval prompts, chat workload
datasets/rag.jsonl                ~50 eval prompts, RAG workload
datasets/long_context.jsonl       ~50 eval prompts, long-context workload
results/quality/.gitkeep          new output directory
```

### Files Modified

```
requirements.txt    add: deepeval
README.md           add: quality-aware benchmarking section
```

---

## Data Schemas

### Dataset Row (JSONL, one object per line)

```json
{
  "schema_version": 1,
  "id": "rag_001",
  "workload": "rag",
  "prompt": "...",
  "expected": "..."
}
```

Optional RAGAS-extension fields (V2, no migration needed — evaluator detects via duck-typing):

```json
  "contexts": ["retrieved chunk 1", "retrieved chunk 2"],
  "ground_truth": "..."
```

**Workload values:** `"chat"`, `"rag"`, `"long_context"`

### Quality Sidecar (`results/quality/<tag>.json`)

```json
{
  "meta": {
    "tag": "vllm_l4fp8_isl2k_c10",
    "latency_tag": "vllm_l4fp8_isl2k_c10",
    "evaluator": "deepeval",
    "model": "llama-3.1-8b",
    "dataset": "datasets/rag.jsonl",
    "num_samples": 50,
    "timestamp": "2026-06-12T00:00:00+00:00"
  },
  "metrics": {
    "answer_relevancy": 0.93,
    "faithfulness": 0.91,
    "hallucination_rate": 0.04,
    "overall_score": 0.923
  },
  "cost": {
    "per_million_tokens": 0.80,
    "throughput_proxy_tokens_per_sec": 262
  }
}
```

`latency_tag` is a backlink to the latency result this eval was paired with, making the coupling explicit in the data rather than inferred from filename convention.

### DeploymentProfile (in-memory contract)

The shared language across the pipeline. Produced by `load_deployment()`, consumed by `compute_tradeoff()` and `recommend()`. Neither `quality` nor `cost` are required keys — their absence is handled gracefully.

```json
{
  "tag": "vllm_l4fp8_isl2k_c10",
  "model": "llama-3.1-8b",
  "latency": {
    "ttft_ms_p50": 115,
    "ttft_ms_p95": 133,
    "throughput_tokens_per_sec": 262
  },
  "quality": {
    "overall_score": 0.923,
    "metrics": {
      "answer_relevancy": 0.93,
      "faithfulness": 0.91,
      "hallucination_rate": 0.04
    }
  },
  "cost": {
    "per_million_tokens": 0.80,
    "throughput_proxy_tokens_per_sec": 262
  }
}
```

A `validate_profile(d)` function enforces the contract and fails fast on missing or mismatched fields. New metric dimensions (security, compliance) are new optional top-level keys — the contract expands without breaking existing consumers.

---

## Component: `evaluate/run_eval.py`

### CLI

```bash
python evaluate/run_eval.py \
  --endpoint http://localhost:8000/v1/completions \
  --model llama-3.1-8b \
  --latency-result results/real/vllm_l4fp8_isl2k_c10.json \
  --dataset datasets/rag.jsonl \
  --evaluator deepeval \
  --cost-per-million-tokens 0.80 \
  --output-dir results/quality
```

`--latency-result` is explicit (not inferred from tag). This is the link between the two halves of the pipeline and must be unambiguous.

`--evaluator` accepts `deepeval` (default) or `llm-judge`.

`--cost-per-million-tokens` is optional. When omitted, the quality sidecar records `per_million_tokens: null` and the throughput proxy is used for cost comparison.

`--dry-run` validates inputs and prints what would be run without hitting the endpoint or DeepEval.

### Behavior

1. Load dataset JSONL. Validate `schema_version` and required fields. Fail fast on malformed rows.
2. Send each prompt to the endpoint asynchronously (concurrency=5 — deliberately low to avoid interfering with a parallel load test or warming the KV cache).
3. Collect `(prompt, expected, response)` triples.
4. Detect workload type from dataset rows. Activate DeepEval metrics per workload:
   - `chat`, `long_context`: `AnswerRelevancyMetric`, `GEval(criteria="correctness")`
   - `rag`: adds `FaithfulnessMetric`, `HallucinationMetric` when `contexts` field is present
5. Run DeepEval evaluation. Aggregate per-metric scores. `overall_score` = mean of all active metrics.
6. Read `throughput_tokens_per_sec` from the latency result file as throughput proxy.
7. Write quality sidecar JSON. Tag name derived from the latency result filename.

### LLM-Judge Path

When `--evaluator llm-judge`, requires `--judge-endpoint` and `--judge-model`. Sends a structured scoring prompt to any OpenAI-compatible endpoint:

```
Score this response on three dimensions (1–5 each):
  correctness: does it answer the question accurately?
  helpfulness: is it useful and complete?
  hallucination: 5=none, 1=severe fabrication

Question: {prompt}
Expected answer: {expected}
Response: {response}

Return JSON: {"correctness": N, "helpfulness": N, "hallucination": N}
```

Scores are normalized to 0–1 and averaged to `overall_score`. Output schema is identical to the DeepEval path.

---

## Component: `analyze/deployment_advisor.py`

### CLI

```bash
python analyze/deployment_advisor.py \
  --tags vllm_a100fp16 vllm_l4fp8 vllm_l4int4 \
  --baseline vllm_a100fp16 \
  --quality-threshold 0.10 \
  --output markdown
```

`--quality-threshold` is the maximum acceptable quality drop relative to baseline (default: 0.10 = 10%). Deployments exceeding this are eliminated before ranking.

`--output` accepts `markdown` (default, terminal-friendly) or `json`.

`--dry-run` validates that all tags can be loaded and are comparable before computing anything.

### Internal Structure: Four Pure Functions

The CLI is a thin wrapper. All logic lives in four pure functions with defined inputs and outputs.

**`load_deployment(tag, latency_dirs, quality_dir) -> DeploymentProfile`**

Finds latency JSON (real overrides synthetic, same precedence as `report.py`). Merges quality sidecar if present. Validates internal consistency:
- `model` must match between latency and quality files
- `latency_tag` in quality sidecar must match the latency file's tag
- `dataset` must be the same across all profiles being compared (different datasets = incomparable quality scores)

**`compute_tradeoff(profiles, baseline_tag) -> TradeoffTable`**

For each non-baseline profile, computes three relative metrics vs baseline:

- **Latency improvement:** `(baseline_ttft_p50 - tag_ttft_p50) / baseline_ttft_p50`
- **Quality delta:** `tag_overall_score - baseline_overall_score` (absolute, shown as %)
- **Cost reduction:** `(baseline_cost - tag_cost) / baseline_cost`
  - Uses `per_million_tokens` when available on both profiles
  - Falls back to `(tag_throughput - baseline_throughput) / tag_throughput` as proxy
  - Shows "N/A" when neither is available

**`recommend(table, quality_threshold) -> Recommendation`**

1. Eliminate any profile where `quality_delta < -quality_threshold`
2. Among survivors (including baseline), rank by latency improvement descending
3. If only baseline survives, recommend baseline with note "no alternative met the quality threshold"
4. If no quality data present on any profile, warn and rank by latency only

**`render(recommendation, output_format) -> str`**

Produces the terminal card or JSON blob. Terminal format:

```
=== Deployment Recommendation ===

Recommended: vllm_l4fp8

  Latency Improvement:   33%  (115ms → 77ms TTFT p50)
  Cost Reduction:        24%  ($1.20 → $0.91 per 1M tokens)
  Quality Delta:        -1.7% (0.950 → 0.933)

Eliminated: vllm_l4int4 — quality drop 17.9% exceeds threshold (10.0%)

Tradeoff Table:
  Tag              TTFT p50   Tok/s   Quality   Cost/1M    Status
  vllm_a100fp16    1.2s       210     0.950     $1.20      baseline
  vllm_l4fp8       0.8s       290     0.933     $0.91      RECOMMENDED
  vllm_l4int4      0.4s       520     0.780     $0.50      eliminated
```

---

## Error Handling

| Condition | Behavior |
|---|---|
| Tag has no latency result | Hard error, stop |
| Tag has no quality sidecar | Warn, include in table with quality "N/A", exclude from quality-gated ranking |
| `latency_tag` in quality sidecar doesn't match loaded latency tag | Warn with both values, continue |
| `model` differs between latency and quality files | Hard error |
| `dataset` differs across profiles being compared | Hard error ("comparing quality scores from different datasets is invalid") |
| `model` differs across profiles being compared | Hard error |
| `--cost-per-million-tokens` not provided and no throughput data | Cost column shows "N/A" |
| `import deepeval` fails | Clear error: "DeepEval not installed. Run: pip install deepeval" |
| DeepEval metric evaluation fails on a sample | Log warning, skip sample, continue with remaining |

---

## DeepEval Metric Selection by Workload

| Workload | Metrics |
|---|---|
| `chat` | `AnswerRelevancyMetric`, `GEval(criteria="correctness")` |
| `long_context` | `AnswerRelevancyMetric`, `GEval(criteria="correctness")` |
| `rag` (no `contexts` field) | `AnswerRelevancyMetric`, `GEval(criteria="correctness")` |
| `rag` (with `contexts` field) | above + `FaithfulnessMetric`, `HallucinationMetric` |

---

## RAGAS Extension Path (V2)

No schema migration needed. Dataset rows gain optional `contexts` and `ground_truth` fields. The evaluator detects these at runtime via duck-typing. RAGAS metrics (faithfulness, answer relevancy, context recall, context precision) activate when both fields are present. The quality sidecar schema is unchanged — RAGAS metrics are additional keys under `metrics`.

---

## Testing

No test framework exists in the repo. For V1:

- Both scripts support `--dry-run` for input validation without network calls or DeepEval
- The synthetic benchmark results in `results/synthetic/` serve as the latency fixture for integration testing `deployment_advisor.py`
- Quality sidecars for synthetic results are generated once and committed to `results/quality/` as fixtures

---

## Relation to Existing Advisors

`playbook/advisor.py` answers: **"Given my workload and hardware, what vLLM configuration flags should I use?"**

`analyze/deployment_advisor.py` answers: **"Given my quality requirements and cost constraints, which deployment configuration should I choose?"**

Two advisors, two levels of the stack. Neither replaces the other.
