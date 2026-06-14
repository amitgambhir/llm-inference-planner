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
    fit_role: str = "level"        # "level" | "shape" | "sanity"
    kv_frac: float = 0.90          # GPU memory fraction reserved for KV cache


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
        ))
    return points


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict_point(point: BenchmarkPoint, constants: Optional[dict] = None) -> float:
    """Predict the benchmark metric using the roofline model + efficiency curves.

    For offline scenario: batch = max_concurrent_seqs (fully saturated).
    For latency scenario: batch is the stated concurrent batch size.
    """
    model_profile = get_model(point.model)
    gpu_profile = get_gpu(point.gpu)

    kb = kv_budget(
        gpu_profile, model_profile, point.dtype, point.isl, point.osl,
        gpu_mem_util=point.kv_frac, tp=point.tp,
    )

    predict_batch = kb.max_concurrent_seqs if point.batch is None else point.batch
    avg_ctx = point.isl + point.osl // 2

    mfu_val = eff.mfu_prefill(model_profile, gpu_profile, point.dtype, point.isl, constants=constants)
    decode_bw = eff.bw_eff_decode(gpu_profile, predict_batch, constants=constants)

    tps_group = decode_ceiling(
        gpu_profile, model_profile, point.dtype,
        predict_batch, avg_ctx, decode_bw, tp=point.tp, mfu=mfu_val,
    )

    if point.metric == "total_output_tps":
        return tps_group          # group-level throughput — no further division
    if point.metric == "agg_throughput_tps_per_gpu":
        return tps_group / point.tp
    if point.metric == "per_user_decode_tps":
        return tps_group / max(predict_batch, 1)
    raise ValueError(
        f"Unknown metric: '{point.metric}'. "
        "Supported: total_output_tps, agg_throughput_tps_per_gpu, per_user_decode_tps"
    )


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
PARAM_BOUNDS: list[tuple[str, float, float]] = [
    ("mfu_base.hopper.bf16",  0.15, 0.65),
    ("mfu_base.hopper.fp8",   0.15, 0.65),
    ("mfu_base.hopper.mxfp4", 0.15, 0.65),
    ("mfu_base.ampere.bf16",  0.15, 0.65),
    ("mfu_base.ampere.fp8",   0.10, 0.55),
    ("mfu_base.ada.bf16",     0.10, 0.55),
    ("mfu_base.ada.fp8",      0.10, 0.55),
    ("size_floor",            0.20, 0.90),
    ("size_scale",            1e9,  100e9),
    ("isl_floor",             0.20, 0.90),
    ("isl_scale",             64.0, 8192.0),
    ("moe_factor",            0.60, 1.00),
    ("bw_base.hbm",           0.40, 0.90),
    ("bw_base.gddr",          0.35, 0.80),
    # batch_floor near 1.0: g_batch ≈ 1.0 for all practical batch sizes.
    # The weight-amortization effect is already captured by the batch × step_rate structure.
    ("batch_floor",           0.80, 1.00),
    ("batch_scale",           1.0,  64.0),
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
    """Median relative error on train_points."""
    r = report(train_points, constants=constants)
    return r.median_rel_error


def fit(
    points: list[BenchmarkPoint],
    train_frac: float = 0.70,
    seed: int = 0,
) -> FittedConstants:
    """Coordinate-descent minimisation of median relative error over train split.

    Tunes efficiency_constants.yaml and writes back the fitted values so that
    subsequent report() calls and the pytest suite use calibrated numbers.

    Holdout split: honesty check for overfitting.  With small datasets (< 10 points)
    the holdout is too small to be statistically meaningful — add more benchmark
    points to catalog/benchmarks_public.yaml before trusting the holdout score.

    Returns FittedConstants with train + holdout median errors for inspection.
    """
    # Only fit on vLLM "level" points — they pin the absolute efficiency level.
    # shape/sanity points use a different engine factor; fitting on them would
    # conflate engine-level differences with hardware efficiency constants.
    level_points = [p for p in points if p.fit_role == "level"]

    rng = random.Random(seed)
    shuffled = list(level_points)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_frac))
    train = shuffled[:n_train]
    holdout = shuffled[n_train:]

    c: dict = yaml.safe_load(_CONSTANTS_PATH.read_text())
    best_error = _objective(c, train)

    # Coordinate descent: cycle through all parameters, try ±{frac} steps.
    # Stop when no step improves the objective or MAX_OUTER iterations reached.
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
                    trial_error = _objective(trial_c, train)
                    if trial_error < best_error - 1e-7:
                        best_error = trial_error
                        c = trial_c
                        current_val = new_val
                        improved = True
        if not improved:
            break

    # Write fitted constants back to disk; reload the in-process cache.
    _CONSTANTS_PATH.write_text(yaml.dump(c, default_flow_style=False, sort_keys=True))
    eff.reload_constants()

    holdout_error: Optional[float] = None
    if holdout:
        holdout_report = report(holdout, constants=c)
        holdout_error = holdout_report.median_rel_error

    return FittedConstants(
        constants=c,
        train_median_rel_error=best_error,
        holdout_median_rel_error=holdout_error,
        n_train=len(train),
        n_holdout=len(holdout),
    )
