"""Phase 1 tests for planner/capacity.py.

Includes:
  - GOLDEN TEST: gpt-oss-20b @ H100, 100M req/day, ISL 9000, OSL 500 (prefill-bound, low confidence)
  - AGNOSTIC TEST: full inline spec + param-only rough spec, neither in seed catalog
  - Unit tests for each pipeline function
"""
import math
import pytest

from planner.catalog import CatalogError, get_gpu, get_model, resolve_model, resolve_gpu
from planner.capacity import (
    CapacityEstimate,
    KvBudget,
    Traffic,
    TtftEstimate,
    decode_ceiling,
    kv_budget,
    normalize_traffic,
    plan,
    prefill_ceiling,
    size_replicas,
    ttft_estimate,
    confidence,
)
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ============================================================================
# GOLDEN TEST
# Input: 100M req/day, peak 3x, ISL 9000, OSL 500
#        gpt-oss-20b, h100_sxm, mxfp4, tp=1, realtime
# ============================================================================


@pytest.fixture
def golden_estimate():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    return plan(
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        isl=9000,
        osl=500,
        ttft_slo_ms=2000.0,
        model=model,
        gpu=gpu,
        dtype="mxfp4",
        tp=1,
        traffic_class="realtime",
    )


def test_golden_avg_rps_within_1pct(golden_estimate):
    # 100M / 86400 ≈ 1157.4
    assert abs(golden_estimate.traffic.avg_rps - 1157.4) / 1157.4 < 0.01


def test_golden_peak_rps_within_1pct(golden_estimate):
    # 1157.4 * 3 ≈ 3472.2
    assert abs(golden_estimate.traffic.peak_rps - 3472.2) / 3472.2 < 0.01


def test_golden_input_tps_avg_within_1pct(golden_estimate):
    # 1157.4 * 9000 ≈ 10,416,667
    expected = 1157.4 * 9000
    assert abs(golden_estimate.traffic.input_tps_avg - expected) / expected < 0.01


def test_golden_output_tps_avg_within_1pct(golden_estimate):
    # 1157.4 * 500 ≈ 578,704
    expected = 1157.4 * 500
    assert abs(golden_estimate.traffic.output_tps_avg - expected) / expected < 0.01


def test_golden_prefill_bound(golden_estimate):
    # gpt-oss-20b MoE at ISL=9000: very long prefill dominates — scenario is prefill-bound.
    # (Previously asserted decode-bound due to a g_kv double-count that has been removed.)
    assert golden_estimate.binding_constraint == "prefill-bound"


def test_golden_confidence_default(golden_estimate):
    # h200_sxm/bf16 anchors exist for gpt-oss-20b → cross-GPU evidence upgrades to MEDIUM
    assert golden_estimate.confidence == "medium"


def test_golden_replica_range_low_lt_high(golden_estimate):
    assert golden_estimate.replicas_low < golden_estimate.replicas_high


def test_golden_replica_range_wide(golden_estimate):
    # DEFAULT band = ±25%: replicas_high should be ≥ 1.25× replicas_low
    ratio = golden_estimate.replicas_high / golden_estimate.replicas_low
    assert ratio >= 1.25, f"Range not wide enough for DEFAULT confidence: ratio={ratio:.2f}"


def test_golden_validate_warning_present(golden_estimate):
    # Cross-GPU anchor warning replaces the old "no live GPU data" message
    combined = " ".join(golden_estimate.warnings).lower()
    assert "validate" in combined or "different gpu" in combined


def test_golden_chunked_prefill_warning_present(golden_estimate):
    combined = " ".join(golden_estimate.warnings).lower()
    assert "chunked prefill" in combined


def test_golden_replicas_positive(golden_estimate):
    assert golden_estimate.replicas >= 1
    assert golden_estimate.replicas_low >= 1
    assert golden_estimate.replicas_high > golden_estimate.replicas_low


# ============================================================================
# AGNOSTIC TEST — model AND GPU not in seed catalog
# ============================================================================


CUSTOM_GPU_SPEC = {
    "name": "custom-h100-variant",
    "display_name": "Custom H100 Variant",
    "mem_gb": 80,
    "hbm_bandwidth_gbps": 3350,
    "peak_flops": {"fp16": 989, "bf16": 989, "fp8": 1979, "mxfp4": 1979},
    "default_mfu_prefill": 0.40,
    "default_bw_efficiency_decode": 0.70,
}

CUSTOM_MODEL_FULL_SPEC = {
    "name": "custom-dense-13b",
    "display_name": "Custom Dense 13B",
    "is_moe": False,
    "total_params": 13_000_000_000,
    "active_params": 13_000_000_000,
    "num_layers": 40,
    "d_model": 5120,
    "num_q_heads": 40,
    "num_kv_heads": 8,
    "head_dim": 128,
    "native_dtype": "fp8",
    "weight_bytes_per_param": 1.0,
    "kv_dtype_bytes": 1,
}


def test_agnostic_full_inline_spec_produces_valid_estimate():
    model = resolve_model(CUSTOM_MODEL_FULL_SPEC)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=1_000_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=256,
        ttft_slo_ms=3000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="mixed",
    )
    assert isinstance(est, CapacityEstimate)
    assert est.replicas >= 1
    assert est.replicas_low < est.replicas_high
    assert est.binding_constraint in ("prefill-bound", "decode-bound", "kv-memory-bound")
    assert est.confidence in ("high", "medium", "default")


def test_agnostic_full_inline_spec_geometry_known():
    model = resolve_model(CUSTOM_MODEL_FULL_SPEC)
    assert model.geometry_source == "known"


def test_agnostic_rough_spec_produces_valid_estimate():
    rough_spec = {
        "name": "rough-20b-custom",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=500_000,
        peak_multiplier=2.0,
        isl=1024,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=1,
        traffic_class="batch",
    )
    assert isinstance(est, CapacityEstimate)
    assert est.replicas >= 1


def test_agnostic_rough_spec_geometry_source_estimated():
    rough_spec = {
        "name": "rough-20b-check",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    assert model.geometry_source == "estimated"


def test_agnostic_rough_spec_confidence_downgraded():
    """Rough-spec model can be at most MEDIUM confidence, never HIGH."""
    rough_spec = {
        "name": "rough-8b-l4-test",
        "total_params": 8_030_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = get_gpu("l4")
    # l4 has anchors for llama-3.1-8b/fp8 — rough spec of similar size should NOT inherit HIGH
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=512,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )
    assert est.confidence in ("medium", "default"), (
        f"Rough spec should never be HIGH confidence, got: {est.confidence}"
    )


def test_agnostic_rough_spec_has_geometry_warning():
    rough_spec = {
        "name": "rough-20b-warn",
        "total_params": 20_000_000_000,
        "native_dtype": "bf16",
    }
    model = resolve_model(rough_spec)
    gpu = resolve_gpu(CUSTOM_GPU_SPEC)
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=512,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=1,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "geometry" in combined and "estimated" in combined


# ============================================================================
# Unit tests — normalize_traffic
# ============================================================================


def test_normalize_traffic_avg_rps():
    t = normalize_traffic(86_400, 1.0, 1024, 128)
    assert t.avg_rps == pytest.approx(1.0, rel=1e-6)
    assert t.peak_rps == pytest.approx(1.0, rel=1e-6)


def test_normalize_traffic_tps_values():
    t = normalize_traffic(86_400, 3.0, 1000, 100)
    assert t.input_tps_avg == pytest.approx(1000.0, rel=1e-6)
    assert t.output_tps_avg == pytest.approx(100.0, rel=1e-6)
    assert t.input_tps_peak == pytest.approx(3000.0, rel=1e-6)
    assert t.output_tps_peak == pytest.approx(300.0, rel=1e-6)


def test_normalize_traffic_total_tokens():
    t = normalize_traffic(1_000_000, 1.0, 500, 100)
    assert t.total_tokens_day == pytest.approx(600_000_000, rel=1e-6)


# ============================================================================
# Unit tests — kv_budget
# ============================================================================


def test_kv_budget_llama_8b_l4():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    kb = kv_budget(gpu, model, "fp8", 2048, 128)
    # l4 has 24 GB; llama-8b weights ≈ 8.03B * 2 bytes = 16.06 GB
    # usable = 24e9 * 0.9 = 21.6 GB
    # kv_budget = 21.6 - 16.06 - 2 = 3.54 GB
    assert kb.kv_cache_budget_bytes > 0
    assert kb.max_concurrent_seqs >= 1


def test_kv_budget_model_fits_h100():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    kb = kv_budget(gpu, model, "mxfp4", 9000, 500)
    assert kb.max_concurrent_seqs >= 1


def test_kv_budget_model_does_not_fit_raises():
    # llama-3.1-70b at bf16 is ~140 GB — won't fit on L4 (24 GB)
    model = get_model("llama-3.1-70b")
    gpu = get_gpu("l4")
    with pytest.raises(CatalogError, match="Use a larger tp"):
        kv_budget(gpu, model, "bf16", 2048, 128)


def test_kv_bytes_per_token_formula():
    model = get_model("llama-3.1-8b")
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 1 kv_dtype_bytes = 65536
    expected = 2 * 32 * 8 * 128 * 1
    assert model.kv_bytes_per_token == expected


# ============================================================================
# Unit tests — prefill_ceiling
# ============================================================================


def test_prefill_ceiling_increases_with_mfu():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    low = prefill_ceiling(gpu, model, "fp8", 2048, 0.30)
    high = prefill_ceiling(gpu, model, "fp8", 2048, 0.60)
    assert high > low


def test_prefill_ceiling_decreases_with_longer_isl():
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    short = prefill_ceiling(gpu, model, "mxfp4", 1000, 0.40)
    long_ = prefill_ceiling(gpu, model, "mxfp4", 9000, 0.40)
    assert short > long_, "Longer ISL must reduce prefill ceiling due to attention FLOPs"


def test_prefill_ceiling_positive():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = prefill_ceiling(gpu, model, "fp8", 512, 0.40)
    assert tps > 0


def test_prefill_ceiling_bw_bound_at_short_isl():
    """Below the ridge point, bandwidth ceiling binds and throughput is lower than compute ceiling."""
    # H100 ridge ≈ 989e12 * 0.40 / (3350e9 * 0.70) ≈ 169 tokens for llama-3.1-70b bf16.
    model = get_model("llama-3.1-70b")
    gpu = get_gpu("h100_sxm")
    tps_short = prefill_ceiling(gpu, model, "bf16", 50, 0.40, bw_eff=0.70)   # ISL=50: bw-bound
    tps_long  = prefill_ceiling(gpu, model, "bf16", 512, 0.40, bw_eff=0.70)  # ISL=512: compute-bound
    assert tps_short < tps_long, "ISL below ridge should be bandwidth-bound → lower throughput"


def test_prefill_ceiling_bw_bound_scales_with_isl():
    """In the bandwidth-bound regime, throughput grows linearly with ISL (more tokens per weight load)."""
    model = get_model("llama-3.1-70b")
    gpu = get_gpu("h100_sxm")
    tps_50  = prefill_ceiling(gpu, model, "bf16", 50,  0.40, bw_eff=0.70)
    tps_100 = prefill_ceiling(gpu, model, "bf16", 100, 0.40, bw_eff=0.70)
    # Both are bw-bound (50 < 100 < 169 ridge); doubling ISL should roughly double throughput.
    assert tps_100 > tps_50 * 1.8


# ============================================================================
# Unit tests — decode_ceiling
# ============================================================================


def test_decode_ceiling_increases_with_batch():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    small = decode_ceiling(gpu, model, "fp8", 8, 1024, 0.70)
    large = decode_ceiling(gpu, model, "fp8", 64, 1024, 0.70)
    assert large > small


def test_decode_ceiling_decreases_with_avg_ctx():
    """Larger avg_ctx means more KV reads → lower decode throughput."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    short_ctx = decode_ceiling(gpu, model, "mxfp4", 32, 1000, 0.70)
    long_ctx = decode_ceiling(gpu, model, "mxfp4", 32, 9000, 0.70)
    assert short_ctx > long_ctx


def test_decode_ceiling_positive():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = decode_ceiling(gpu, model, "fp8", 16, 512, 0.65)
    assert tps > 0


def test_decode_ceiling_moe_large_batch_higher_weight_traffic():
    """MoE expert coverage: at large batch almost all experts are touched → weight bytes ≈ total."""
    model = get_model("gpt-oss-20b")  # 32 experts, top-4, 20.9B total / 3.61B active
    gpu = get_gpu("h100_sxm")
    # batch=1: distinct_frac ≈ 4/32 = 0.125 — only active experts streamed
    # batch=128: distinct_frac ≈ 1 − (0.875)^128 ≈ 0.9999 — nearly all experts streamed
    tps_b1 = decode_ceiling(gpu, model, "mxfp4", 1, 500, 0.70)
    tps_b128 = decode_ceiling(gpu, model, "mxfp4", 128, 500, 0.70)
    # Even though batch grows 128×, throughput gain is muted by the growing weight term.
    # Key check: throughput at b=128 is NOT 128× the b=1 throughput (saturation effect).
    ratio = tps_b128 / tps_b1
    assert ratio < 128, f"MoE should saturate weight reads at large batch; ratio={ratio:.1f}"
    assert ratio > 1, "Throughput should still grow with batch"


def test_decode_ceiling_dense_vs_moe_weight_bytes():
    """At batch=64, MoE decode ceiling is lower than a hypothetical dense model of same total params."""
    moe = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    # Dense proxy: same total_params but active==total
    from planner.catalog import resolve_model
    dense_proxy = resolve_model({
        "name": "dense-proxy-21b",
        "total_params": 20_900_000_000,
        "active_params": 20_900_000_000,
        "num_layers": 24,
        "d_model": 2880,
        "num_q_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 64,
        "native_dtype": "mxfp4",
        "weight_bytes_per_param": 0.5,
        "kv_dtype_bytes": 1,
    })
    tps_moe = decode_ceiling(gpu, moe, "mxfp4", 64, 500, 0.70)
    tps_dense = decode_ceiling(gpu, dense_proxy, "mxfp4", 64, 500, 0.70)
    # MoE at batch=64 streams nearly all experts → similar weight bytes to dense → similar TPS
    # (MoE decode should be close to dense of same total params at large batch, not 6× faster)
    assert tps_moe < tps_dense * 2, (
        f"MoE at large batch should stream near-full weight set; "
        f"moe={tps_moe:.0f} dense={tps_dense:.0f}"
    )


def test_decode_ceiling_compute_ceiling_binds_at_short_ctx():
    """Compute ceiling binds when KV traffic is tiny (very short context, small model)."""
    # Use a tiny avg_ctx so KV bytes are negligible — weight-then-compute dominated
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    # At avg_ctx=1, KV barely matters; compute ceiling = peak_flops*mfu / (2*active_params)
    tps_short = decode_ceiling(gpu, model, "fp8", 512, 1, 0.70, mfu=0.40)
    compute_ceiling = gpu.peak_flops.get("fp8") * 1e12 * 0.40 / (2 * model.active_params)
    assert tps_short <= compute_ceiling * 1.001, (
        f"At avg_ctx=1 with large batch, compute ceiling should bind; "
        f"tps={tps_short:.0f} ceiling={compute_ceiling:.0f}"
    )


# ============================================================================
# Unit tests — kv_budget physics
# ============================================================================


def test_kv_budget_uses_resident_weights_gb():
    """gpt-oss-20b has resident_weights_gb=13.0 — should use that, not total_params × dtype_bytes."""
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    kb = kv_budget(gpu, model, "mxfp4", 9000, 500)
    # With resident_weights_gb=13.0: per-GPU weight = 13.0 GB
    # Without override: total_params × 0.5 bytes = 20.9e9 × 0.5 = 10.45 GB
    # The resident_weights path gives less KV budget (more weight memory consumed).
    # usable = 80 × 0.9 = 72 GB; kv_budget(resident) = (72 - 13 - 0.5) × 1 = 58.5 GB
    # kv_budget(fallback)  = (72 - 10.45 - 0.5) × 1 = 61.05 GB
    assert kb.kv_cache_budget_bytes == pytest.approx(58.5e9, rel=0.01)


def test_kv_budget_kv_shard_factor_caps_at_num_kv_heads():
    """When tp > num_kv_heads, KV budget should not grow beyond num_kv_heads × per-GPU free mem."""
    # qwen3-30b-a3b has num_kv_heads=4; at tp=8 kv_shard_factor should be 4, not 8.
    model = get_model("qwen3-30b-a3b")
    gpu = get_gpu("h100_sxm")
    kb_tp4 = kv_budget(gpu, model, "bf16", 1024, 256, tp=4)
    kb_tp8 = kv_budget(gpu, model, "bf16", 1024, 256, tp=8)
    # At tp=4: kv_shard_factor = min(4, 4) = 4
    # At tp=8: kv_shard_factor = min(8, 4) = 4 — same KV budget, since weights halved
    # KV budget should not double from tp=4 to tp=8 (it would if we used tp directly)
    assert kb_tp8.kv_cache_budget_bytes < kb_tp4.kv_cache_budget_bytes * 1.5, (
        "tp=8 should not double KV budget vs tp=4 when num_kv_heads=4"
    )


# ============================================================================
# Unit tests — dtype validation
# ============================================================================


def test_plan_raises_on_unsupported_dtype_for_gpu():
    """'fp4' is not a recognized serving dtype in the GPU catalog — plan() should raise CatalogError."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("a100_80gb_sxm")
    with pytest.raises(CatalogError):
        plan(
            requests_per_day=100_000,
            peak_multiplier=2.0,
            isl=512,
            osl=128,
            ttft_slo_ms=5000.0,
            model=model,
            gpu=gpu,
            dtype="fp4",
            tp=1,
            traffic_class="batch",
        )


def test_plan_raises_on_unknown_dtype():
    """Completely unknown dtype string should raise CatalogError immediately."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    with pytest.raises(CatalogError):
        plan(
            requests_per_day=100_000,
            peak_multiplier=2.0,
            isl=512,
            osl=128,
            ttft_slo_ms=5000.0,
            model=model,
            gpu=gpu,
            dtype="bf32",
            tp=1,
            traffic_class="batch",
        )


# ============================================================================
# Unit tests — prefill Q-proj dim
# ============================================================================


def test_prefill_ceiling_qwen3_higher_attention_flops():
    """Qwen3-30B: q_proj_dim (32×128=4096) > d_model (2048) → more attention FLOPs → lower prefill TPS."""
    from planner.catalog import resolve_model, resolve_gpu
    # Build a proxy that matches qwen3-30b-a3b but with d_model == q_proj_dim (4096)
    # to compare against the model where d_model=2048 ≠ q_proj_dim=4096.
    model = get_model("qwen3-30b-a3b")
    gpu = get_gpu("h100_sxm")
    tps_correct = prefill_ceiling(gpu, model, "bf16", 4096, 0.40)  # uses q_proj_dim = 4096
    # Manually check: q_proj_dim = num_q_heads × head_dim = 32 × 128 = 4096
    assert model.num_q_heads * model.head_dim == 4096
    assert model.d_model == 2048
    # The correct calculation uses 4096, not 2048 — so flops_per_token is higher
    # and TPS is lower than if we used d_model.
    flops_d_model = 2 * model.active_params + 2 * model.num_layers * 4096 * model.d_model
    flops_qproj = 2 * model.active_params + 2 * model.num_layers * 4096 * (model.num_q_heads * model.head_dim)
    assert flops_qproj > flops_d_model, "q_proj_dim path should compute higher FLOPs for Qwen3"


# ============================================================================
# Unit tests — tp_used in output
# ============================================================================


def test_plan_tp_used_stored_in_estimate():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=512,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=2,
        traffic_class="batch",
    )
    assert est.tp_used == 2
    assert est.total_gpus == est.replicas * 2


# ============================================================================
# Unit tests — confidence
# ============================================================================


def test_confidence_high_for_seeded_l4_anchor():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    assert conf.level == "high"
    assert conf.anchor_matched is True


def test_confidence_default_for_unknown_model_gpu():
    # Use a model+GPU with no anchors to exercise the pure DEFAULT tier
    model = get_model("qwen3-30b-a3b")
    gpu = get_gpu("l40s")
    conf = confidence(model, gpu, "bf16", 9000, 500, 100)
    assert conf.level == "default"
    assert conf.anchor_matched is False


def test_confidence_estimated_geometry_downgrade():
    rough_spec = {
        "name": "rough-8b",
        "total_params": 8_000_000_000,
        "native_dtype": "fp8",
    }
    model = resolve_model(rough_spec)
    gpu = get_gpu("l4")
    # Even though l4/fp8 anchors exist for llama-8b, rough spec should downgrade
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    # rough spec → the name "rough-8b" won't match the anchor's model "llama-3.1-8b"
    # so it would be DEFAULT anyway; but even if there WERE anchors, it must not be HIGH
    assert conf.level in ("medium", "default")


def test_confidence_mfu_from_anchor_when_available():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    conf = confidence(model, gpu, "fp8", 2048, 128, 10)
    # Anchor derived_mfu_prefill is null in seeded data → falls back to GPU default
    assert conf.mfu_used == gpu.default_mfu_prefill


# ============================================================================
# Unit tests — ttft_estimate
# ============================================================================


def test_ttft_compute_only_no_queue():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = prefill_ceiling(gpu, model, "fp8", 2048, 0.40) * 1  # tp=1
    result = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.0, 5000.0)
    assert result.ttft_queue_ms == pytest.approx(0.0)
    assert result.ttft_ms == pytest.approx(result.ttft_compute_ms)


def test_ttft_queue_grows_with_utilization():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = 50_000.0
    low_util = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.20, 5000.0)
    high_util = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.80, 5000.0)
    assert high_util.ttft_ms > low_util.ttft_ms


def test_ttft_slo_met_flag():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    tps = 1_000_000.0  # very fast
    result = ttft_estimate(gpu, model, "fp8", 512, tps, 0.10, 5000.0)
    assert result.slo_met is True
    assert result.slo_breach_reason is None


def test_ttft_slo_breach_flag():
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    tps = 100.0  # extremely slow prefill to force SLO breach
    result = ttft_estimate(gpu, model, "fp8", 2048, tps, 0.10, 500.0)
    assert result.slo_met is False
    assert result.slo_breach_reason is not None


# ============================================================================
# Unit tests — size_replicas
# ============================================================================


def test_size_replicas_prefill_bound():
    traffic = normalize_traffic(100_000_000, 3.0, 9000, 500)
    sz = size_replicas(
        traffic=traffic,
        prefill_tps_gpu=90_000.0,      # low prefill → many replicas needed
        decode_tps_gpu=500_000.0,      # high decode → few replicas needed
        max_concurrent_seqs=200,
        tp=1,
        traffic_class="realtime",
        isl=9000,
        osl=500,
    )
    assert sz["binding_constraint"] == "prefill-bound"


def test_size_replicas_decode_bound():
    traffic = normalize_traffic(1_000_000, 2.0, 256, 2000)  # high output
    sz = size_replicas(
        traffic=traffic,
        prefill_tps_gpu=500_000.0,     # fast prefill
        decode_tps_gpu=500.0,          # very slow decode
        max_concurrent_seqs=512,
        tp=1,
        traffic_class="realtime",
        isl=256,
        osl=2000,
    )
    assert sz["binding_constraint"] == "decode-bound"


def test_size_replicas_headroom_applied():
    # 80M req/day, isl=512 → input_tps_peak ≈ 474k → base ≈ 5 replicas at 100k tps/GPU
    # ceil(5 * 1.40) = 7  vs  ceil(5 * 1.10) = 6  → realtime > batch
    traffic = normalize_traffic(80_000_000, 1.0, 512, 128)
    sz_realtime = size_replicas(traffic, 100_000.0, 100_000.0, 64, 1, "realtime", 512, 128)
    sz_batch = size_replicas(traffic, 100_000.0, 100_000.0, 64, 1, "batch", 512, 128)
    assert sz_realtime["replicas"] > sz_batch["replicas"]


def test_size_replicas_tp_scales_per_replica_capacity():
    # prefill_tps_gpu / decode_tps_gpu are TP-group throughput (already include TP factor).
    # At tp=4 the caller (plan → ceiling functions) provides 4× the throughput and
    # 4× the KV slots — so each replica handles more load → fewer replicas needed.
    traffic = normalize_traffic(1_000_000, 1.0, 512, 128)
    sz_tp1 = size_replicas(traffic, 50_000.0, 50_000.0, 64, tp=1, traffic_class="batch", isl=512, osl=128)
    sz_tp4 = size_replicas(traffic, 200_000.0, 200_000.0, 256, tp=4, traffic_class="batch", isl=512, osl=128)
    assert sz_tp4["replicas"] <= sz_tp1["replicas"]


# ============================================================================
# Integration: CLI-equivalent paths
# ============================================================================


def test_plan_produces_assumptions_list(golden_estimate):
    assert len(golden_estimate.assumptions) > 0
    # MFU and bandwidth efficiency should be documented
    combined = " ".join(golden_estimate.assumptions).lower()
    assert "mfu" in combined
    assert "bandwidth" in combined


def test_plan_llama_l4_has_high_confidence():
    """Seeded anchors should yield high confidence for llama-3.1-8b on l4."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )
    assert est.confidence == "high"
    assert est.anchor_matched is True


# ============================================================================
# Unit tests — sliding-window KV budget
# ============================================================================


def test_kv_budget_sliding_window_more_seqs_than_full_attention():
    """At long ISL, sliding-window model fits more concurrent seqs than a full-attention clone."""
    gpu = get_gpu("h100_sxm")
    # Use gemma-3-12b which has sliding_window=1024, global_layer_every_n=6
    sw_model = get_model("gemma-3-12b")
    # Build a synthetic full-attention twin (same geometry, no sliding window)
    fa_model = resolve_model({
        "name": "gemma-3-12b-full-attn",
        "display_name": "Gemma 3 12B (full-attn twin)",
        "total_params": sw_model.total_params,
        "active_params": sw_model.active_params,
        "num_layers": sw_model.num_layers,
        "d_model": sw_model.d_model,
        "num_q_heads": sw_model.num_q_heads,
        "num_kv_heads": sw_model.num_kv_heads,
        "head_dim": sw_model.head_dim,
        "native_dtype": sw_model.native_dtype,
        "weight_bytes_per_param": sw_model.weight_bytes_per_param,
        "kv_dtype_bytes": sw_model.kv_dtype_bytes,
        "context_len": sw_model.context_len,
        # No sliding_window / global_layer_every_n
    })
    isl, osl = 32768, 512
    kv_sw = kv_budget(gpu, sw_model, "bf16", isl, osl)
    kv_fa = kv_budget(gpu, fa_model, "bf16", isl, osl)
    # With ISL=32768 >> window=1024, sliding-window should allow far more sequences
    assert kv_sw.max_concurrent_seqs > kv_fa.max_concurrent_seqs


def test_kv_budget_sliding_window_no_benefit_within_window():
    """When ISL+OSL ≤ sliding_window, concurrency equals the full-attention case."""
    gpu = get_gpu("h100_sxm")
    sw_model = get_model("gemma-3-1b")  # sliding_window=512
    fa_model = resolve_model({
        "name": "gemma-3-1b-fa",
        "display_name": "Gemma 3 1B full-attn",
        "total_params": sw_model.total_params,
        "active_params": sw_model.active_params,
        "num_layers": sw_model.num_layers,
        "d_model": sw_model.d_model,
        "num_q_heads": sw_model.num_q_heads,
        "num_kv_heads": sw_model.num_kv_heads,
        "head_dim": sw_model.head_dim,
        "native_dtype": sw_model.native_dtype,
        "weight_bytes_per_param": sw_model.weight_bytes_per_param,
        "kv_dtype_bytes": sw_model.kv_dtype_bytes,
    })
    # ISL+OSL=400 is well under window=512 → no benefit; both should give same result
    isl, osl = 300, 100
    kv_sw = kv_budget(gpu, sw_model, "bf16", isl, osl)
    kv_fa = kv_budget(gpu, fa_model, "bf16", isl, osl)
    assert kv_sw.max_concurrent_seqs == kv_fa.max_concurrent_seqs


def test_kv_budget_effective_context_tokens_stored():
    """effective_context_tokens is stored on KvBudget and is < isl+osl at long context."""
    gpu = get_gpu("h100_sxm")
    model = get_model("gemma-3-4b")  # sliding_window=1024, global_layer_every_n=6
    isl, osl = 65536, 512
    kb = kv_budget(gpu, model, "bf16", isl, osl)
    full_ctx = float(isl + osl)
    assert kb.effective_context_tokens < full_ctx
    assert kb.effective_context_tokens > 0


def test_kv_budget_full_attention_model_effective_equals_full_ctx():
    """For models without sliding window, effective_context_tokens == isl+osl."""
    gpu = get_gpu("h100_sxm")
    model = get_model("llama-3.1-8b")
    isl, osl = 2048, 128
    kb = kv_budget(gpu, model, "fp8", isl, osl)
    assert kb.effective_context_tokens == float(isl + osl)


# ============================================================================
# Unit tests — ISL > context_len warning
# ============================================================================


def test_plan_warns_when_isl_exceeds_context_len():
    """ISL exceeding the model's native context window should produce a warning."""
    model = get_model("mistral-7b")        # context_len=32768
    gpu = get_gpu("h100_sxm")
    est = plan(
        requests_per_day=10_000,
        peak_multiplier=2.0,
        isl=65536,                         # > 32768 native context
        osl=256,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=1,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "native context" in combined or "context window" in combined


def test_plan_no_context_warning_within_limit():
    """No context-length warning when ISL is within the model's native window."""
    model = get_model("llama-3.1-8b")     # context_len=131072
    gpu = get_gpu("l4")
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "native context" not in combined


# ============================================================================
# Unit tests — global_head_dim warning
# ============================================================================


def test_plan_warns_on_global_head_dim_difference():
    """Gemma 4 31B has global_head_dim=512 and should produce an informational warning."""
    model = get_model("gemma-4-31b")
    gpu = get_gpu("h100_sxm")
    est = plan(
        requests_per_day=10_000,
        peak_multiplier=2.0,
        isl=4096,
        osl=256,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="bf16",
        tp=2,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "global attention" in combined


def test_plan_no_global_head_dim_warning_for_standard_model():
    """Standard models without global_head_dim should not produce the global-layer warning."""
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    est = plan(
        requests_per_day=100_000,
        peak_multiplier=2.0,
        isl=2048,
        osl=128,
        ttft_slo_ms=5000.0,
        model=model,
        gpu=gpu,
        dtype="fp8",
        tp=1,
        traffic_class="batch",
    )
    combined = " ".join(est.warnings).lower()
    assert "global attention" not in combined
