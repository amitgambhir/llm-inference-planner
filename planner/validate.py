"""
planner/validate.py — fit and validate efficiency curves against public benchmarks.

Two public functions:
  report(points, constants) -> ValidationReport  : per-point rel-error + aggregate stats
  fit(points, train_frac, seed) -> FittedConstants : tune efficiency_constants.yaml

Typical usage:
  from planner.validate import fit, load_public_benchmarks, report
  pts = load_public_benchmarks()
  r = report(pts)                    # report with current constants (level points only)
  fitted = fit(pts)                  # tune constants, writes efficiency_constants.yaml
  r2 = report(pts)                   # re-report with fitted values

The benchmark data lives in catalog/benchmarks_public.yaml.  The efficiency
constants are in planner/efficiency_constants.yaml.  A refit is a data change
(efficiency_constants.yaml is updated); the code is never touched.

Metric routing (scenario × metric → prediction path):
  offline  × agg_throughput_tps_per_gpu : decode_ceiling(max_concurrent_seqs) / tp
  offline  × total_output_tps           : decode_ceiling(max_concurrent_seqs)   (group tps, no division)
  latency  × per_user_decode_tps        : decode_ceiling(batch) / batch

fit_role policy (see catalog/benchmarks_public.yaml engine confound policy):
  level  — vLLM reference points; pins the absolute efficiency LEVEL; used in fit()
  shape  — TRT-LLM points; curve shape only; excluded from level fit, reported separately
  sanity — excluded from fit and default report; coarse sanity check only
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from planner.catalog import (
    CatalogError,
    get_gpu,
    get_model,
)
from planner.capacity import decode_ceiling, kv_budget
import planner.efficiency as eff

_BENCHMARKS_PATH = Path(__file__).parent.parent / "catalog" / "benchmarks_public.yaml"
_CONSTANTS_PATH = Path(__file__).parent / "efficiency_constants.yaml"
_FLAT_VALLEY_THRESHOLD = 0.01  # delta_error below this flags an underdetermined parameter


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkPoint:
    model: str
    gpu: str
    dtype: str
    tp: int
    isl: int
    osl: int
    batch: Optional[int]           # None = offline (use max_concurrent_seqs)
    scenario: str                  # "offline" | "server" | "latency"
    metric: str                    # "agg_throughput_tps_per_gpu" | "per_user_decode_tps" | "total_output_tps"
    measured: float
    source: str
    url: Optional[str] = None
    engine: Optional[str] = None   # "vllm" | "trtllm" | "sglang" | ...
    engine_version: Optional[str] = None
    fit_role: str = "level"        # "level" | "shape" | "validate" | "sanity"
    kv_frac: float = 0.90          # GPU memory fraction reserved for KV cache
    dataset: str = "uniform"       # "uniform" (fixed ISL/OSL, stdev=0) | "distribution" (mean ISL/OSL)
    kv_dtype: Optional[str] = None # KV cache dtype; None = same as weights dtype
    pp: int = 1                    # pipeline parallel degree (comms not modeled; for record only)


@dataclass
class PointResult:
    model: str
    gpu: str
    dtype: str
    scenario: str
    metric: str
    predicted: Optional[float]
    measured: float
    rel_error: float
    error_msg: Optional[str] = None


@dataclass
class ValidationReport:
    median_rel_error: float
    p90_rel_error: float
    max_rel_error: float
    n_points: int
    points: list[PointResult] = field(default_factory=list)


@dataclass
class FittedConstants:
    constants: dict
    train_median_rel_error: float
    holdout_median_rel_error: Optional[float]
    n_train: int
    n_holdout: int


@dataclass
class CrossValidationResult:
    left_out_gpu: str
    train_median_rel_error: float
    holdout_median_rel_error: float
    n_train: int
    n_holdout: int


@dataclass
class SensitivityResult:
    param: str
    base_error: float
    plus15_error: float
    minus15_error: float
    delta_error: float  # max(|plus - base|, |minus - base|)
    is_flat: bool       # delta_error < _FLAT_VALLEY_THRESHOLD; signature of underdetermined param


# ---------------------------------------------------------------------------
# Load benchmark data
# ---------------------------------------------------------------------------


def load_public_benchmarks() -> list[BenchmarkPoint]:
    """Load and parse catalog/benchmarks_public.yaml (schema_version 1 or 2)."""
    raw = yaml.safe_load(_BENCHMARKS_PATH.read_text())
    schema_ver = raw.get("schema_version")
    if schema_ver not in (1, 2):
        raise ValueError(
            f"Unsupported benchmarks_public.yaml schema_version: {schema_ver!r}. "
            "Expected 1 or 2."
        )
    points: list[BenchmarkPoint] = []
    for p in raw.get("points", []):
        if p.get("measured") is None:
            continue  # pending stub (Phase B placeholders not yet filled in)
        points.append(BenchmarkPoint(
            model=p["model"],
            gpu=p["gpu"],
            dtype=p["dtype"],
            tp=int(p.get("tp", 1)),
            isl=int(p["isl"]),
            osl=int(p["osl"]),
            batch=int(p["batch"]) if p.get("batch") is not None else None,
            scenario=p.get("scenario", "offline"),
            metric=p["metric"],
            measured=float(p["measured"]),
            source=p.get("source", ""),
            url=p.get("url"),
            engine=p.get("engine"),
            engine_version=p.get("engine_version"),
            fit_role=p.get("fit_role", "level"),
            kv_frac=float(p.get("kv_frac", 0.90)),
            dataset=p.get("dataset", "uniform"),
            kv_dtype=p.get("kv_dtype"),
            pp=int(p.get("pp", 1)),
        ))
    return points


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _dual_roofline_out_tps(
    gpu_profile,
    model_profile,
    dtype: str,
    isl: int,
    osl: int,
    batch: int,
    tp: int = 1,
    constants: Optional[dict] = None,
) -> float:
    """Output tokens/sec via dual-roofline over the whole request lifecycle.

    min(compute_lifecycle, decode_bw_at_base_efficiency)

    compute_lifecycle: FLOPs across isl prefill tokens + osl decode tokens, per output token.
    decode_bw: decode_ceiling at base HBM efficiency (no g_kv — KV counted once in bytes_per_step).
    """
    mfu = eff.mfu_prefill(model_profile, gpu_profile, dtype, isl, constants=constants)
    base_bw_eff = eff.bw_eff_prefill(gpu_profile, constants=constants)

    # Lifecycle compute ceiling: total FLOPs for one output token (prefill + decode amortised)
    q_proj_dim = model_profile.num_q_heads * model_profile.head_dim
    flops_pf_per_input = (
        2 * model_profile.active_params
        + 2 * model_profile.num_layers * isl * q_proj_dim
    )
    flops_dec_per_output = 2 * model_profile.active_params
    flops_per_output_token = (isl * flops_pf_per_input + osl * flops_dec_per_output) / osl
    peak_flops = gpu_profile.peak_flops.get(dtype)
    if peak_flops is None:
        raise CatalogError(
            f"GPU '{gpu_profile.name}' does not support dtype '{dtype}'."
        )
    compute_out_tps = peak_flops * 1e12 * mfu * tp / flops_per_output_token

    avg_ctx = isl + osl // 2
    decode_out_tps = decode_ceiling(
        gpu_profile, model_profile, dtype,
        batch, avg_ctx, base_bw_eff, tp=tp, mfu=mfu,
    )

    return min(compute_out_tps, decode_out_tps)


def _predict_point(point: BenchmarkPoint, constants: Optional[dict] = None) -> float:
    """Predict the benchmark metric using per-point conditions (metric, scenario, kv_frac, tp, engine).

    Routing:
      offline / server → _dual_roofline_out_tps at reconstructed max_concurrent_seqs
      latency          → decode_ceiling at stated batch (base HBM efficiency)

    engine_factor scales the raw prediction: vllm=1.0 (reference), trtllm=~1.5, etc.
    Server points use the offline formula — they are typically fit_role: validate with
    widened tolerance (SLO throttling isn't modelled here).
    """
    model_profile = get_model(point.model)
    gpu_profile = get_gpu(point.gpu)

    kb = kv_budget(
        gpu_profile, model_profile, point.dtype, point.isl, point.osl,
        gpu_mem_util=point.kv_frac, tp=point.tp,
    )
    batch = point.batch if point.batch is not None else kb.max_concurrent_seqs

    if point.scenario in ("offline", "server"):
        raw = _dual_roofline_out_tps(
            gpu_profile, model_profile, point.dtype, point.isl, point.osl,
            batch, tp=point.tp, constants=constants,
        )
    elif point.scenario == "latency":
        base_bw_eff = eff.bw_eff_prefill(gpu_profile, constants=constants)
        mfu_val = eff.mfu_prefill(model_profile, gpu_profile, point.dtype, point.isl, constants=constants)
        avg_ctx = point.isl + point.osl // 2
        raw = decode_ceiling(
            gpu_profile, model_profile, point.dtype,
            batch, avg_ctx, base_bw_eff, tp=point.tp, mfu=mfu_val,
        )
    else:
        raise ValueError(
            f"Unknown scenario: '{point.scenario}'. Supported: offline, server, latency"
        )

    c = constants if constants is not None else eff._load_constants()
    engine_factors: dict = c.get("engine_factor", {})
    engine_factor = engine_factors.get(point.engine, 1.0) if point.engine else 1.0

    pred = raw * engine_factor

    if point.metric == "total_output_tps":
        return pred
    if point.metric == "agg_throughput_tps_per_gpu":
        return pred / point.tp
    if point.metric == "per_user_decode_tps":
        return pred / max(batch, 1)
    raise ValueError(
        f"Unknown metric: '{point.metric}'. "
        "Supported: total_output_tps, agg_throughput_tps_per_gpu, per_user_decode_tps"
    )


# Test-access alias (private function exposed for unit tests only)
_predict_point_for_testing = _predict_point


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _median(vals: list[float]) -> float:
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 == 1 else (vals[mid - 1] + vals[mid]) / 2.0


def report(
    points: list[BenchmarkPoint],
    constants: Optional[dict] = None,
    fit_roles: Optional[tuple] = ("level",),
) -> ValidationReport:
    """Compute per-point relative errors and aggregate statistics.

    rel_error = |predicted - measured| / measured  (0.0 = perfect, 1.0 = 100% off)
    A prediction failure (unknown model/GPU, CatalogError) counts as rel_error=1.0.

    fit_roles: restrict to points with these fit_role values.  Default ("level",) — the
    vLLM reference points that ground the absolute efficiency level.  Pass None to
    report all points regardless of fit_role.
    """
    if fit_roles is None:
        active = list(points)
    else:
        active = [p for p in points if p.fit_role in fit_roles]

    results: list[PointResult] = []
    for pt in active:
        try:
            predicted = _predict_point(pt, constants=constants)
            rel_error = abs(predicted - pt.measured) / pt.measured
            error_msg = None
        except (CatalogError, ValueError, ZeroDivisionError) as e:
            predicted = None
            rel_error = 1.0
            error_msg = str(e)
        results.append(PointResult(
            model=pt.model, gpu=pt.gpu, dtype=pt.dtype,
            scenario=pt.scenario, metric=pt.metric,
            predicted=predicted, measured=pt.measured,
            rel_error=rel_error, error_msg=error_msg,
        ))

    errors = sorted(r.rel_error for r in results)
    n = len(errors)
    if n == 0:
        return ValidationReport(0.0, 0.0, 0.0, 0, results)  # no matching points

    median_err = _median(errors)
    p90_idx = min(n - 1, int(math.ceil(0.9 * n)) - 1)
    p90_err = errors[max(0, p90_idx)]
    return ValidationReport(
        median_rel_error=median_err,
        p90_rel_error=p90_err,
        max_rel_error=errors[-1],
        n_points=n,
        points=results,
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

# Parameter bounds for coordinate descent: (dotted_yaml_key, lo, hi)
# Nested keys use '.' separator matching the YAML structure.
#
# Strategy: fix the LEVELS (tight ±10% bounds around literature priors),
# fit the SHAPES (ISL/size/batch curves) and engine_factor freely.
# This prevents the optimizer trading engine_factor × MFU_BASE.
#
# MFU priors: asymptotic large-model long-ISL MFU from published benchmarks.
# BW priors: empirically calibrated base HBM/GDDR efficiency (current fitted values).
PARAM_BOUNDS: list[tuple[str, float, float]] = [
    # --- Levels: tight ±10% around literature priors ---
    ("mfu_base.hopper.bf16",  0.45, 0.55),   # prior 0.50
    ("mfu_base.hopper.fp8",   0.43, 0.53),   # prior 0.48
    ("mfu_base.hopper.mxfp4", 0.45, 0.55),   # prior 0.50
    ("mfu_base.ampere.bf16",  0.45, 0.55),   # prior 0.50
    ("mfu_base.ampere.fp8",   0.405, 0.495), # prior 0.45
    ("mfu_base.ada.bf16",     0.405, 0.495), # prior 0.45
    ("mfu_base.ada.fp8",      0.405, 0.495), # prior 0.45
    ("bw_base.hbm",           0.39, 0.49),   # prior 0.44 (empirically fitted)
    ("bw_base.gddr",          0.34, 0.42),   # prior 0.38 (empirically fitted)
    # --- Shapes: free within physically meaningful ranges ---
    ("size_floor",            0.20, 0.90),
    ("size_scale",            1e9,  100e9),
    ("isl_floor",             0.20, 0.90),
    ("isl_scale",             64.0, 8192.0),
    ("moe_factor",            0.60, 1.00),
    ("batch_floor",           0.80, 1.00),   # prior 0.80; lower bound intentionally tight
    ("batch_scale",           1.0,  64.0),
    # --- Engine factor: vllm=1.0 fixed; only trtllm is free ---
    ("engine_factor.trtllm",  1.00, 3.00),
    # NOTE: fp16, mxfp4, and Blackwell arch entries are absent because no public
    # benchmark points cover those dtypes/arches yet. Add bounds here when benchmarks exist.
]


def _get_param(c: dict, dotted_key: str) -> float:
    val: object = c
    for k in dotted_key.split("."):
        val = val[k]  # type: ignore[index]
    return float(val)  # type: ignore[arg-type]


def _set_param(c: dict, dotted_key: str, value: float) -> dict:
    c = copy.deepcopy(c)
    keys = dotted_key.split(".")
    node: dict = c
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value
    return c


def _objective(constants: dict, train_points: list[BenchmarkPoint]) -> float:
    """Median relative error on train_points.

    fit_roles=None: the caller has already pre-filtered the list to the desired
    roles; passing None tells report() to score every point it receives rather
    than re-filtering to level-only.
    """
    r = report(train_points, constants=constants, fit_roles=None)
    return r.median_rel_error


def _coordinate_descent(
    train_points: list[BenchmarkPoint],
    init_constants: dict,
) -> tuple[dict, float]:
    """Pure coordinate descent over train_points. No I/O, no side effects.

    Returns (best_constants_dict, best_median_rel_error).
    Used by fit() (which adds disk write) and cv_leave_one_gpu_out() (no disk write).
    """
    c = copy.deepcopy(init_constants)
    best_error = _objective(c, train_points)

    STEP_FRACS = [0.20, 0.10, 0.05, 0.02]
    MAX_OUTER = 60
    for _outer in range(MAX_OUTER):
        improved = False
        for param, lo, hi in PARAM_BOUNDS:
            try:
                current_val = _get_param(c, param)
            except (KeyError, TypeError):
                continue
            for frac in STEP_FRACS:
                for sign in (+1, -1):
                    new_val = max(lo, min(hi, current_val * (1.0 + sign * frac)))
                    if abs(new_val - current_val) < 1e-12:
                        continue
                    trial_c = _set_param(c, param, new_val)
                    trial_error = _objective(trial_c, train_points)
                    if trial_error < best_error - 1e-7:
                        best_error = trial_error
                        c = trial_c
                        current_val = new_val
                        improved = True
        if not improved:
            break

    return c, best_error


def fit(
    points: list[BenchmarkPoint],
    train_frac: float = 0.70,
    seed: int = 0,
) -> FittedConstants:
    """Coordinate-descent minimisation of median relative error over train split.

    Fits on level + shape points jointly:
      level — vLLM reference; pins absolute efficiency level (engine_factor[vllm]=1.0)
      shape — TRT-LLM uniform sweep; pins ISL/size curve shape + engine_factor[trtllm]

    validate and sanity points are excluded from the objective.
    Writes fitted constants to efficiency_constants.yaml and reloads in-process cache.
    """
    fit_points = [p for p in points if p.fit_role in ("level", "shape")]

    rng = random.Random(seed)
    shuffled = list(fit_points)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_frac))
    train = shuffled[:n_train]
    holdout = shuffled[n_train:]

    init_c: dict = yaml.safe_load(_CONSTANTS_PATH.read_text())
    best_c, best_error = _coordinate_descent(train, init_c)

    _CONSTANTS_PATH.write_text(yaml.dump(best_c, default_flow_style=False, sort_keys=True))
    eff.reload_constants()

    holdout_error: Optional[float] = None
    if holdout:
        holdout_report = report(holdout, constants=best_c, fit_roles=None)
        holdout_error = holdout_report.median_rel_error

    return FittedConstants(
        constants=best_c,
        train_median_rel_error=best_error,
        holdout_median_rel_error=holdout_error,
        n_train=len(train),
        n_holdout=len(holdout),
    )


def cv_leave_one_gpu_out(
    points: list[BenchmarkPoint],
) -> list[CrossValidationResult]:
    """Leave-one-GPU-out cross-validation over fit-eligible (level + shape) points.

    For each GPU that has at least one fit-eligible point: fit on all other GPUs,
    evaluate on the held-out GPU. Tests generalization across hardware.
    """
    fit_points = [p for p in points if p.fit_role in ("level", "shape")]
    gpus = sorted({p.gpu for p in fit_points})
    init_c: dict = eff._load_constants()
    results: list[CrossValidationResult] = []

    for gpu_name in gpus:
        train = [p for p in fit_points if p.gpu != gpu_name]
        holdout = [p for p in fit_points if p.gpu == gpu_name]
        if len(train) < 2 or not holdout:
            continue
        best_c, train_err = _coordinate_descent(train, init_c)
        holdout_r = report(holdout, constants=best_c, fit_roles=None)
        results.append(CrossValidationResult(
            left_out_gpu=gpu_name,
            train_median_rel_error=train_err,
            holdout_median_rel_error=holdout_r.median_rel_error,
            n_train=len(train),
            n_holdout=len(holdout),
        ))

    return results


def parameter_sensitivity(
    points: list[BenchmarkPoint],
    constants: Optional[dict] = None,
    perturb_frac: float = 0.15,
) -> list[SensitivityResult]:
    """Perturb each fitted param ±perturb_frac and report change in median error.

    A flat valley (delta_error < 0.01) flags an underdetermined parameter.
    Call after fit() to verify identifiability of the current dataset + param set.
    """
    c = constants if constants is not None else eff._load_constants()
    fit_points = [p for p in points if p.fit_role in ("level", "shape")]
    base_error = _objective(c, fit_points)
    results: list[SensitivityResult] = []

    for param, lo, hi in PARAM_BOUNDS:
        try:
            current = _get_param(c, param)
        except (KeyError, TypeError):
            continue
        plus_val = max(lo, min(hi, current * (1.0 + perturb_frac)))
        minus_val = max(lo, min(hi, current * (1.0 - perturb_frac)))
        plus_err = _objective(_set_param(c, param, plus_val), fit_points)
        minus_err = _objective(_set_param(c, param, minus_val), fit_points)
        delta = max(abs(plus_err - base_error), abs(minus_err - base_error))
        results.append(SensitivityResult(
            param=param,
            base_error=base_error,
            plus15_error=plus_err,
            minus15_error=minus_err,
            delta_error=delta,
            is_flat=(delta < _FLAT_VALLEY_THRESHOLD),
        ))

    return results
