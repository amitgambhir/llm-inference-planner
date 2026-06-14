"""
tests/test_efficiency.py — unit tests for planner/efficiency.py curve functions.

Tests verify monotonicity, bounds, fallback behavior, and physical plausibility
of the two curves.  No GPU required; no anchors loaded.
"""
import pytest

from planner.catalog import get_gpu, resolve_gpu
from planner.efficiency import bw_eff_decode, bw_eff_prefill, mfu_prefill

import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# Custom GPU spec WITHOUT arch/memory_type — used to test fallback behavior.
_CUSTOM_GPU_SPEC = {
    "name": "custom-no-arch",
    "display_name": "Custom GPU (no arch)",
    "mem_gb": 80,
    "hbm_bandwidth_gbps": 3350,
    "peak_flops": {"bf16": 989, "fp8": 1979},
    "default_mfu_prefill": 0.42,
    "default_bw_efficiency_decode": 0.68,
}


# ============================================================================
# mfu_prefill — monotonicity and bounds
# ============================================================================


def test_mfu_prefill_increases_with_isl():
    """Longer ISL → more compute per weight byte → higher MFU (f_isl factor)."""
    from planner.catalog import get_model
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    mfu_short = mfu_prefill(model, gpu, "bf16", 64)
    mfu_long = mfu_prefill(model, gpu, "bf16", 4096)
    assert mfu_long > mfu_short, f"MFU should increase with ISL: {mfu_short:.3f} → {mfu_long:.3f}"


def test_mfu_prefill_increases_with_active_params():
    """Larger model → bigger GEMMs → higher MFU (f_size factor)."""
    from planner.catalog import get_model
    small = get_model("llama-3.1-8b")     # 8B active
    large = get_model("llama-3.1-70b")   # 70B active
    gpu = get_gpu("h100_sxm")
    mfu_small = mfu_prefill(small, gpu, "bf16", 512)
    mfu_large = mfu_prefill(large, gpu, "bf16", 512)
    assert mfu_large > mfu_small, f"MFU should increase with model size: {mfu_small:.3f} → {mfu_large:.3f}"


def test_mfu_prefill_moe_penalty():
    """MoE model should have lower MFU than a dense model of similar active size."""
    from planner.catalog import get_model
    moe = get_model("gpt-oss-20b")      # MoE: 3.61B active, is_moe=True
    dense = get_model("llama-3.1-8b")   # dense: 8.03B active (larger, but density matters)
    gpu = get_gpu("h100_sxm")
    # Use mxfp4 for MoE (its native dtype), bf16 for dense
    mfu_moe = mfu_prefill(moe, gpu, "mxfp4", 512)
    mfu_dense = mfu_prefill(dense, gpu, "bf16", 512)
    # MoE should be penalized relative to its own base
    # Load a reference: same moe model treated as dense (force is_moe=False via direct call)
    from planner.catalog import ModelProfile
    dense_proxy = ModelProfile(
        name=moe.name, display_name=moe.display_name,
        is_moe=False, total_params=moe.total_params, active_params=moe.active_params,
        num_layers=moe.num_layers, d_model=moe.d_model, num_q_heads=moe.num_q_heads,
        num_kv_heads=moe.num_kv_heads, head_dim=moe.head_dim,
        native_dtype=moe.native_dtype, weight_bytes_per_param=moe.weight_bytes_per_param,
    )
    mfu_moe_if_dense = mfu_prefill(dense_proxy, gpu, "mxfp4", 512)
    assert mfu_moe < mfu_moe_if_dense, (
        f"MoE penalty should lower MFU vs dense equivalent: {mfu_moe:.3f} vs {mfu_moe_if_dense:.3f}"
    )


def test_mfu_prefill_custom_gpu_fallback():
    """Custom GPU without arch → falls back to gpu.default_mfu_prefill."""
    from planner.catalog import get_model, resolve_gpu
    model = get_model("llama-3.1-8b")
    gpu = resolve_gpu(_CUSTOM_GPU_SPEC)
    result = mfu_prefill(model, gpu, "bf16", 512)
    assert result == pytest.approx(0.42), (
        f"Expected fallback to default_mfu_prefill=0.42, got {result:.3f}"
    )


def test_mfu_prefill_lower_bounded():
    """MFU must never fall below the hard floor (0.08)."""
    from planner.catalog import get_model
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("l4")
    # Degenerate inputs: very short ISL → near-zero f_isl
    result = mfu_prefill(model, gpu, "bf16", 1)
    assert result >= 0.08, f"MFU below hard floor: {result:.4f}"


def test_mfu_prefill_upper_bounded_by_base():
    """MFU must never exceed the arch+dtype base constant."""
    from planner.catalog import get_model
    from planner.efficiency import _load_constants
    model = get_model("llama-3.1-70b")   # large model → f_size → 1.0
    gpu = get_gpu("h100_sxm")
    c = _load_constants()
    base = c["mfu_base"]["hopper"]["bf16"]
    result = mfu_prefill(model, gpu, "bf16", 16384)  # very long ISL → f_isl → 1.0
    assert result <= base + 1e-9, f"MFU {result:.4f} exceeds base {base:.4f}"


def test_mfu_prefill_dtype_fallback_to_bf16():
    """If arch supports dtype via fallback to bf16, returns a valid value (not 0)."""
    from planner.catalog import get_model
    model = get_model("llama-3.1-8b")
    gpu = get_gpu("h100_sxm")
    # fp16 is in hopper map, should work; even unlisted dtypes fall back to bf16
    result_fp16 = mfu_prefill(model, gpu, "fp16", 512)
    result_bf16 = mfu_prefill(model, gpu, "bf16", 512)
    assert result_fp16 == pytest.approx(result_bf16, rel=1e-6), (
        "fp16 and bf16 should have same MFU on H100 (same base constant)"
    )


# ============================================================================
# bw_eff_decode — monotonicity, bounds (no kv_ratio — KV counted in decode_ceiling)
# ============================================================================


def test_bw_eff_decode_increases_with_batch():
    """Larger batch → better weight amortization → higher bandwidth efficiency."""
    gpu = get_gpu("h100_sxm")
    eff_small = bw_eff_decode(gpu, eff_batch=1)
    eff_large = bw_eff_decode(gpu, eff_batch=64)
    assert eff_large > eff_small, (
        f"bw_eff should increase with batch: {eff_small:.3f} → {eff_large:.3f}"
    )


def test_bw_eff_decode_custom_gpu_fallback():
    """Custom GPU without memory_type → falls back to gpu.default_bw_efficiency_decode."""
    gpu = resolve_gpu(_CUSTOM_GPU_SPEC)
    result = bw_eff_decode(gpu, eff_batch=32)
    assert result == pytest.approx(0.68), (
        f"Expected fallback to default_bw_efficiency_decode=0.68, got {result:.3f}"
    )


def test_bw_eff_decode_lower_bounded_by_batch_floor():
    """bw_eff_decode at batch=1 must be ≥ base × batch_floor (no hard floor needed)."""
    from planner.efficiency import _load_constants
    gpu = get_gpu("h100_sxm")
    c = _load_constants()
    base = c["bw_base"]["hbm"]
    batch_floor = c["batch_floor"]
    result = bw_eff_decode(gpu, eff_batch=1)
    assert result >= base * batch_floor - 1e-9, (
        f"bw_eff {result:.4f} below base*batch_floor {base*batch_floor:.4f}"
    )


def test_bw_eff_decode_upper_bounded_by_base():
    """bw_eff_decode must never exceed the memory_type base."""
    from planner.efficiency import _load_constants
    gpu = get_gpu("h100_sxm")
    c = _load_constants()
    base = c["bw_base"]["hbm"]
    result = bw_eff_decode(gpu, eff_batch=10000)
    assert result <= base + 1e-9, f"bw_eff {result:.4f} exceeds base {base:.4f}"


def test_bw_eff_decode_gddr_lower_than_hbm():
    """GDDR GPU should achieve lower base bw_eff than HBM GPU at equivalent conditions."""
    hbm_gpu = get_gpu("h100_sxm")
    gddr_gpu = get_gpu("l4")
    eff_hbm = bw_eff_decode(hbm_gpu, eff_batch=64)
    eff_gddr = bw_eff_decode(gddr_gpu, eff_batch=64)
    assert eff_gddr < eff_hbm, (
        f"GDDR should have lower bw_eff than HBM: {eff_gddr:.3f} vs {eff_hbm:.3f}"
    )


# ============================================================================
# bw_eff_prefill — memory type routing
# ============================================================================


def test_bw_eff_prefill_hbm_returns_base():
    """H100 (HBM) should return the hbm base bandwidth efficiency."""
    from planner.efficiency import _load_constants
    gpu = get_gpu("h100_sxm")
    c = _load_constants()
    expected = c["bw_base"]["hbm"]
    result = bw_eff_prefill(gpu)
    assert result == pytest.approx(expected), (
        f"Expected hbm base {expected:.3f}, got {result:.3f}"
    )


def test_bw_eff_prefill_gddr_returns_base():
    """L4 (GDDR) should return the gddr base bandwidth efficiency."""
    from planner.efficiency import _load_constants
    gpu = get_gpu("l4")
    c = _load_constants()
    expected = c["bw_base"]["gddr"]
    result = bw_eff_prefill(gpu)
    assert result == pytest.approx(expected), (
        f"Expected gddr base {expected:.3f}, got {result:.3f}"
    )


def test_bw_eff_prefill_custom_gpu_fallback():
    """Custom GPU without memory_type → falls back to gpu.default_bw_efficiency_decode."""
    gpu = resolve_gpu(_CUSTOM_GPU_SPEC)
    result = bw_eff_prefill(gpu)
    assert result == pytest.approx(0.68)
