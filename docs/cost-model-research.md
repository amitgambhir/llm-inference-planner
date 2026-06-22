# Cost Model Research: llm-inference-planner vs gpu-calc-v2

**Date:** 2026-06-22  
**Status:** Deferred — current model retained; improvements noted for future work

---

## Background

After wiring up cost estimation into the `POST /estimate` API (commit `0c82a6b`), we evaluated whether the costing approach in the adjacent `gpu-calc-v2` repo offered meaningful improvements over the current model.

---

## Current Model (`planner/cost.py`)

Simple physics-driven approach:

```
gpu_hours_day = replicas × tp × 24
cost_month    = gpu_hours_day × rate × 30
cost_per_1M   = cost_month / (total_tokens_day × 30 / 1e6)
cost_per_req  = cost_day / requests_per_day
```

- Rates from `catalog/costs.yaml` (static, manually maintained)
- On-demand and 1-yr reserved variants both computed
- Per-token and per-request costs derived from actual traffic volume (not capacity)
- GPU count is physics-driven from the roofline model — not a manual input
- No utilization assumption; replicas already sized for peak traffic via `peak_multiplier`

---

## gpu-calc-v2 Approach (SemiAnalysis TCO methodology)

### What it does differently

| Feature | gpu-calc-v2 | Current model |
|---|---|---|
| **TCO components** | GPU + storage + network egress + ops engineering + support uplift (e.g. 3% AWS) + goodput loss | GPU only |
| **Live pricing** | Cloudflare Worker fetches real-time $/hr from 8 providers; 30-min cache + localStorage override | Static `catalog/costs.yaml` |
| **Provider comparison** | AWS, Azure, GCP, Lambda, CoreWeave, RunPod, Nebius, Vast.ai | Single rate per GPU SKU |
| **Utilization multiplier** | User-specified target (default 70% cloud, 80% on-prem); warns if effective cost > 2× raw rate | Implicit via peak_multiplier in sizing |
| **On-demand vs reserved** | Defined in types, not fully computed in practice | Both computed and displayed |
| **Per-token formula** | Exists in `cost.ts` but unwired in the UI | Wired to actual traffic volume |
| **CAPEX / on-prem** | `hardware_cost / (36 months × 730 hours/month)` | Not modelled |
| **GPU count source** | Manual input | Physics-driven from roofline |

### Key insight: gpu-calc-v2 takes GPU count as a manual input

The cluster cost module in gpu-calc-v2 accepts an arbitrary GPU count from the user. This is fine for TCO modeling of a known fleet, but it's a different use case from llm-inference-planner, which *derives* the GPU count from traffic × roofline physics. The two tools are complementary, not competing.

---

## Recommendations (Priority Order)

### High value — worth implementing

1. **Multi-provider pricing table** — replace the single rate in `catalog/costs.yaml` with a provider × GPU matrix so users can compare CoreWeave vs Lambda vs AWS H100 side by side. This is the single most actionable improvement: a solution architect picking a cloud provider would immediately use this.

   Approach: extend `catalog/costs.yaml` with a `providers` key per GPU SKU; add a `provider` field to `ScenarioCreate`; expose a provider dropdown in the UI.

2. **Utilization multiplier** — add an optional `utilization_target` parameter (default 1.0, i.e. no change) to `compute_cost()`. At 70% utilization, effective $/token rises by ~43%. This makes the cost output more conservative and closer to real-world numbers where GPUs sit idle between bursts.

   Note: this partially overlaps with `peak_multiplier` in sizing. Document the distinction clearly: `peak_multiplier` sizes for burst traffic headroom; `utilization_target` accounts for idle time between bursts.

### Medium value — consider later

3. **Live pricing API** — a lightweight proxy (Cloudflare Worker or small Lambda) that refreshes provider spot/on-demand rates. Useful once the tool has external users who care about current market rates.

4. **CAPEX / on-prem model** — `hardware_cost / (amortization_months × 730 hours/month)`. Relevant for enterprise SA conversations comparing cloud-rental vs buy. Can be added as a third `CostVariant` alongside on-demand and reserved.

### Low value — skip

5. **Full TCO (storage, network egress, ops, support uplift)** — meaningful for data center total cost analysis but adds noise to a pre-deployment sizing conversation. These numbers require org-specific inputs (team size, data volumes, SLA tier) that users don't have at the scenario-building stage.

---

## Decision

Current model retained as-is (`replicas × tp × 24 × rate`). It is simpler, more accurate for its use case (cost derived from sizing physics, not assumptions), and already superior to gpu-calc-v2 on the per-token and per-request metrics.

The two highest-value improvements (multi-provider pricing and utilization multiplier) are the natural next step for cost model iteration.
