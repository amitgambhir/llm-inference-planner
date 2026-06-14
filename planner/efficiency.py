"""
planner/efficiency.py — regime-aware MFU and bandwidth efficiency defaults.

Replaces the flat GPU-catalog constants (0.40 / 0.70) with monotonic curves
keyed on physical drivers:

  mfu_prefill  : (model, GPU arch, dtype, ISL)   — prefill compute utilization
  bw_eff_decode: (GPU memory type, eff_batch, kv_ratio) — decode HBM utilization
  bw_eff_prefill: (GPU memory type)              — prefill weight-stream efficiency

All forms are monotonic, bounded, and physically motivated:
  - mfu_prefill saturates at the arch+dtype "base" for large models and long ISL.
  - bw_eff_decode increases with batch (weight amortization) and decreases with
    kv_ratio (PagedAttention scatter). Replaces the old step-threshold 3/10 block.

Precedence in plan():
  measured anchor  >  efficiency curve  >  hard floor (0.08 / 0.20)

Constants live in planner/efficiency_constants.yaml and are tuned by
validate.fit() against catalog/benchmarks_public.yaml. A refit is a data edit,
not a code change. Use reload_constants() after fit() to pick up the new values.

Custom GPU specs (used in tests) that lack `arch` or `memory_type` silently fall
back to gpu.default_mfu_prefill / gpu.default_bw_efficiency_decode.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from planner.catalog import GpuProfile, ModelProfile

_CONSTANTS_PATH = Path(__file__).parent / "efficiency_constants.yaml"
_CONSTANTS: dict[str, Any] = {}


def _load_constants() -> dict[str, Any]:
    global _CONSTANTS
    if not _CONSTANTS:
        _CONSTANTS = yaml.safe_load(_CONSTANTS_PATH.read_text())
    return _CONSTANTS


def reload_constants() -> None:
    """Force-reload efficiency_constants.yaml from disk (call after validate.fit())."""
    global _CONSTANTS
    _CONSTANTS = {}
    _load_constants()


# ---------------------------------------------------------------------------
# MFU for prefill
# ---------------------------------------------------------------------------


def mfu_prefill(
    model: "ModelProfile",
    gpu: "GpuProfile",
    dtype: str,
    isl: int,
    constants: Optional[dict] = None,
) -> float:
    """Regime-aware MFU for the prefill compute ceiling.

    return = clamp(base × f_size × f_isl × f_moe,  low=0.08,  high=base)

    Factors (all in (0, 1]):
      base   — asymptotic MFU for this GPU arch + dtype (large model, ISL >> scale)
      f_size — model-size saturation: larger active GEMMs → tensor cores stay fed
      f_isl  — ISL saturation: longer prefill → more arithmetic per weight byte
      f_moe  — MoE penalty: routing overhead + smaller per-expert GEMMs + imbalance

    Falls back to gpu.default_mfu_prefill when gpu.arch is absent or unmapped
    (e.g. custom GPU specs in tests that pre-date the arch field).
    """
    c = constants if constants is not None else _load_constants()

    arch = getattr(gpu, "arch", None)
    mfu_base_map: dict = c.get("mfu_base", {})
    if arch is None or arch not in mfu_base_map:
        return gpu.default_mfu_prefill

    arch_map: dict = mfu_base_map[arch]
    base: float = arch_map.get(dtype, arch_map.get("bf16", gpu.default_mfu_prefill))

    size_floor: float = float(c["size_floor"])
    size_scale: float = float(c["size_scale"])
    isl_floor: float = float(c["isl_floor"])
    isl_scale: float = float(c["isl_scale"])
    moe_factor: float = float(c["moe_factor"])

    f_size = size_floor + (1.0 - size_floor) * (1.0 - math.exp(-model.active_params / size_scale))
    f_isl = isl_floor + (1.0 - isl_floor) * (1.0 - math.exp(-isl / isl_scale))
    f_moe = moe_factor if model.is_moe else 1.0

    return max(0.08, min(base, base * f_size * f_isl * f_moe))


# ---------------------------------------------------------------------------
# Bandwidth efficiency for decode
# ---------------------------------------------------------------------------


def bw_eff_decode(
    gpu: "GpuProfile",
    eff_batch: int,
    kv_ratio: float,
    constants: Optional[dict] = None,
) -> float:
    """Bandwidth efficiency for the decode phase — absorbs batch and KV-scatter effects.

    return = clamp(base × g_batch × g_kv,  low=0.20,  high=base)

    Factors:
      base    — asymptotic efficiency by GPU memory type (HBM vs GDDR)
      g_batch — weight amortization: larger batch → better HBM utilization
      g_kv    — scatter degradation: high KV ratio → degraded effective bandwidth

    This is a smooth, differentiable replacement for the old step-threshold block
    that applied 15%/30% penalties at kv_ratio 3/10. The optimizer in validate.fit()
    can tune kv_scale continuously rather than jumping at fixed thresholds.

    Falls back to gpu.default_bw_efficiency_decode when gpu.memory_type is absent.
    """
    c = constants if constants is not None else _load_constants()

    memory_type = getattr(gpu, "memory_type", None)
    bw_base_map: dict = c.get("bw_base", {})
    if memory_type is None or memory_type not in bw_base_map:
        return gpu.default_bw_efficiency_decode

    base: float = float(bw_base_map[memory_type])
    batch_floor: float = float(c["batch_floor"])
    batch_scale: float = float(c["batch_scale"])
    kv_scale: float = float(c["kv_scale"])

    # Weight amortization (g_batch): the batch_floor/batch_scale curve saturates very quickly
    # for typical serving batch sizes (≥1); the main driver of per-user throughput variation
    # is KV-cache scatter (g_kv), not weight-read amortization. batch_floor is kept in the
    # YAML and the optimizer is constrained to keep it near 1.0.
    g_batch = batch_floor + (1.0 - batch_floor) * (1.0 - math.exp(-eff_batch / batch_scale))
    g_kv = 1.0 / (1.0 + kv_ratio / kv_scale)

    # Floor of 0.05: allows KV-dominated regimes (large ISL, high concurrency) to predict
    # correctly. The old 0.20 floor was too high — it prevented matching measured vLLM
    # throughput at high concurrency where effective bandwidth efficiency can drop to 5-10%.
    return max(0.05, min(base, base * g_batch * g_kv))


# ---------------------------------------------------------------------------
# Bandwidth efficiency for prefill weight streaming
# ---------------------------------------------------------------------------


def bw_eff_prefill(
    gpu: "GpuProfile",
    constants: Optional[dict] = None,
) -> float:
    """Base HBM efficiency for prefill weight-streaming (bandwidth-floor path).

    Used for the bandwidth ceiling in prefill_ceiling() when ISL < ridge point.
    Prefill processes one request at a time (effective batch=1 for weight reads),
    so no batch or kv_ratio adjustment is applied — just the memory-type base.

    Falls back to gpu.default_bw_efficiency_decode when gpu.memory_type is absent.
    """
    c = constants if constants is not None else _load_constants()

    memory_type = getattr(gpu, "memory_type", None)
    bw_base_map: dict = c.get("bw_base", {})
    if memory_type is None or memory_type not in bw_base_map:
        return gpu.default_bw_efficiency_decode

    return float(bw_base_map[memory_type])
