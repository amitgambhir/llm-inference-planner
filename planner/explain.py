"""
planner/explain.py — plain-language sizing walk for capacity estimates.

render_napkin_math() reads entirely from CapacityEstimate (and an optional
CostEstimate) — no hardcoded constants anywhere.
"""
from __future__ import annotations

from typing import Optional

from planner.capacity import CapacityEstimate
from planner.cost import CostEstimate


def render_napkin_math(
    est: CapacityEstimate,
    cost: Optional[CostEstimate] = None,
) -> str:
    """Return a plain-language walk of the sizing arithmetic.

    Every number comes from est or cost — no magic constants.
    Format: one bullet per step, grouped into numbered sections.
    The cost section (section 6) is omitted when cost is None.
    The $/user/month line is omitted when est.users is None.
    """
    t = est.traffic
    kv = est.kv_budget
    lines: list[str] = []

    # ── 1. Traffic normalization ──────────────────────────────────────────
    peak_mult = t.peak_rps / t.avg_rps if t.avg_rps else 0.0
    lines += [
        "**1. Traffic normalization**",
        f"- {t.requests_per_day:,}/day ÷ 86,400 = {t.avg_rps:,.1f} avg RPS "
        f"× {peak_mult:.1f} peak = {t.peak_rps:,.1f} peak RPS",
        "",
    ]

    # ── 2. Token demand ───────────────────────────────────────────────────
    isl = round(t.input_tps_avg / t.avg_rps) if t.avg_rps else 0
    osl = round(t.output_tps_avg / t.avg_rps) if t.avg_rps else 0
    lines += [
        "**2. Token demand**",
        f"- × {isl:,} tokens/request = {t.input_tps_peak:,.0f} input tok/s (prefill demand)",
        f"- × {osl:,} tokens/request = {t.output_tps_peak:,.0f} output tok/s (decode demand)",
        "",
    ]

    # ── 3. Per-replica ceilings ───────────────────────────────────────────
    lines += [
        "**3. Per-replica ceilings**",
        f"- Prefill ceiling: {est.prefill_tps_gpu:,.0f} tok/s/replica "
        f"(at MFU {est.mfu_used:.2f}, {est.confidence})",
        f"- Decode ceiling:  {est.decode_tps_gpu:,.0f} tok/s/replica "
        f"(at BW eff {est.bw_eff_used:.2f})",
        "",
    ]

    # ── 4. KV-budget concurrency ──────────────────────────────────────────
    weights_gb = kv.weights_resident_bytes / 1e9
    kv_gb = kv.kv_cache_budget_bytes / 1e9
    kv_per_seq_kb = (
        kv.kv_cache_budget_bytes / kv.max_concurrent_seqs / 1024
        if kv.max_concurrent_seqs > 0 else 0.0
    )
    lines += [
        "**4. KV-budget concurrency**",
        f"- {est.gpu_mem_gb:.0f} GB − {weights_gb:.1f} GB weights = {kv_gb:.1f} GB KV budget",
        f"- ÷ {kv_per_seq_kb:.0f} KB/sequence = "
        f"{kv.max_concurrent_seqs} max concurrent sequences/replica",
        "",
    ]

    # ── 5. Replica sizing ─────────────────────────────────────────────────
    lines += [
        "**5. Replica sizing**",
        f"- Prefill → {est.replicas_prefill} replicas | "
        f"Decode → {est.replicas_decode} | "
        f"Concurrency → {est.replicas_concurrency}",
        f"- Binding constraint: {est.binding_constraint}",
        f"- × {est.headroom_factor:.2f} headroom → {est.replicas} replicas",
        "",
    ]

    # ── 6. Cost (only when CostEstimate provided) ─────────────────────────
    if cost is not None:
        od = cost.on_demand
        lines += [
            "**6. Cost**",
            f"- {est.replicas} × {est.tp_used}× GPU × 24h × "
            f"${od.price_usd_per_gpu_hour}/hr = "
            f"${od.cost_day_usd:,.0f}/day → ${od.cost_month_usd:,.0f}/month",
        ]
        if est.users and od.cost_per_user_per_month is not None:
            lines.append(
                f"- ${od.cost_per_user_per_month:.2f}/user/month "
                f"across {est.users:,} users"
            )
        lines.append("")

    return "\n".join(lines)
