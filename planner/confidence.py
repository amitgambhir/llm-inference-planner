"""
planner/confidence.py — single source of truth for confidence rubric.

Moved out of capacity.py so ingest_anchor.py and future Phase 4 scenario
persistence can share the same rubric without importing the full planner.

The rubric is a pure function over (anchors, scenario, model.geometry_source):
  HIGH    : anchor for (model, gpu, dtype) within ±20% ISL
  MEDIUM  : same (model, gpu, dtype) anchors exist but ISL extrapolated >20%,
            OR anchors for same model on a different gpu/dtype
  DEFAULT : no anchor for this model at all — roofline + validated efficiency curves

Extrapolation distance is computed for (model, gpu, dtype) matches and drives
both the label threshold and the band_factor used for range widening:
  HIGH    → band_factor = 0.10  (anchor calibrated, narrow)
  MEDIUM  → band_factor = 0.20  (anchor present but extrapolated)
  DEFAULT → band_factor = 0.25  (no anchor; curves validated to ±15% median)

One-level downgrade when model.geometry_source == "estimated":
  HIGH → MEDIUM → DEFAULT.
The MFU/bw_eff source and confidence label are decided together — always the same call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from planner.catalog import Anchor, GpuProfile, ModelProfile, _get_catalog

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_ISL_THRESHOLD = 0.20       # ±20% ISL → HIGH
MEDIUM_ISL_THRESHOLD = 1.0      # ±100% ISL (same family) → MEDIUM
HIGH_CONCURRENCY_RATIO = 10.0   # anchor must be within 10× of scenario concurrency for HIGH
                                 # guards against sub-1 anchors calibrating high-batch plans

BAND_FACTORS = {
    "high": 0.10,    # anchor calibrated, ISL within ±20%
    "medium": 0.20,  # anchor exists but ISL extrapolated; tightened now curves are validated
    "default": 0.25, # no anchor; efficiency curves validated to median ±15% on public data
}


# ---------------------------------------------------------------------------
# Extrapolation distance
# ---------------------------------------------------------------------------


@dataclass
class ExtrapolationDistance:
    """How far the nearest anchor is from the scenario.

    isl_distance and concurrency_distance are normalised fractions
    (0.0 = exact match, 1.0 = 100% off).  overall = max of the two,
    used for the threshold comparison.
    """
    isl_distance: float
    concurrency_distance: float
    nearest_anchor: Anchor

    @property
    def overall(self) -> float:
        return max(self.isl_distance, self.concurrency_distance)


# ---------------------------------------------------------------------------
# ConfidenceResult
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceResult:
    level: str                              # "high" | "medium" | "low"
    band_factor: float                      # ±fraction for range widening
    mfu_used: float
    bw_eff_used: float
    anchor_matched: bool
    geometry_source: str
    best_anchor: Optional[Anchor] = None
    extrapolation_distance: Optional[ExtrapolationDistance] = None
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core rubric — pure function over an explicit anchor list
# ---------------------------------------------------------------------------


def compute_confidence_from_anchors(
    anchors: list[Anchor],
    model: ModelProfile,
    gpu: GpuProfile,
    dtype: str,
    isl: int,
    osl: int,
    concurrency: int,
) -> ConfidenceResult:
    """Pure function: determine confidence and calibrated coefficients.

    Accepts an explicit anchor list so it can be called in tests or after
    ingest with a freshly loaded set without touching the module singleton.
    """
    notes: list[str] = []

    # ── Search 1: exact (model, gpu, dtype) family ────────────────────────
    exact_family = [
        a for a in anchors
        if a.model == model.name and a.gpu == gpu.name and a.dtype == dtype
    ]

    if exact_family:
        # Nearest by ISL distance
        nearest = min(exact_family, key=lambda a: abs(a.isl - isl))
        isl_dist = abs(nearest.isl - isl) / max(isl, 1)
        conc_dist = abs(nearest.concurrency - concurrency) / max(concurrency, 1)
        # Ratio-based concurrency gate: how many times apart are the two concurrency values?
        # 10× allows c=10 anchor to cover eff_batch≤100 (typical L4 range), while blocking
        # sub-1 anchors (c=0.6) from calibrating plans at eff_batch≥6.
        _conc_ratio = max(nearest.concurrency, concurrency) / max(
            min(nearest.concurrency, concurrency), 1e-9
        )
        ext_dist = ExtrapolationDistance(
            isl_distance=isl_dist,
            concurrency_distance=conc_dist,
            nearest_anchor=nearest,
        )

        if isl_dist <= HIGH_ISL_THRESHOLD and _conc_ratio <= HIGH_CONCURRENCY_RATIO:
            level = "high"
            mfu = nearest.derived_mfu_prefill or gpu.default_mfu_prefill
            notes.append(
                f"Calibrated from anchor: {nearest.source} "
                f"(ISL distance {isl_dist:.0%}, concurrency distance {conc_dist:.0%})."
            )
        else:
            level = "medium"
            mfu = gpu.default_mfu_prefill
            if isl_dist > HIGH_ISL_THRESHOLD:
                notes.append(
                    f"Same (model, gpu, dtype) anchor exists but ISL is extrapolated "
                    f"{isl_dist:.0%} beyond the calibrated point ({nearest.source}). "
                    "MFU is GPU default, not calibrated."
                )
            else:
                notes.append(
                    f"Anchor at concurrency {nearest.concurrency:.1f} is {_conc_ratio:.0f}× "
                    f"from scenario concurrency {concurrency} — too far to calibrate "
                    f"high-batch behavior. MFU is GPU default. ({nearest.source})"
                )
        anchor_matched = (level == "high")
        best_anchor = nearest

    else:
        ext_dist = None
        best_anchor = None
        anchor_matched = False
        mfu = gpu.default_mfu_prefill

        # ── Search 2: same model, different gpu/dtype ────────────────────
        same_model = [a for a in anchors if a.model == model.name]
        if same_model:
            level = "medium"
            best_anchor = same_model[0]
            notes.append(
                f"Anchors exist for '{model.name}' on a different GPU/dtype "
                f"({best_anchor.gpu}/{best_anchor.dtype}). "
                "MFU/bw_eff are GPU defaults, not calibrated for this hardware."
            )
        else:
            level = "default"
            notes.append(
                f"No anchors found for model '{model.name}'. "
                "Estimate uses regime-aware efficiency curves (validated to ±15% median "
                "on public benchmarks). Validate on a live GPU before committing infrastructure."
            )

    bw_eff = gpu.default_bw_efficiency_decode

    # ── Geometry downgrade ────────────────────────────────────────────────
    if model.geometry_source == "estimated":
        original_level = level
        if level == "high":
            level = "medium"
        elif level == "medium":
            level = "default"
        # "default" is the lowest tier — no further downgrade
        if level != original_level:
            notes.append(
                f"Model geometry was estimated from param count (not a model card). "
                f"Confidence downgraded from {original_level.upper()} to {level.upper()}."
            )
        else:
            notes.append(
                "Model geometry estimated from param count; already at DEFAULT confidence."
            )

    band_factor = BAND_FACTORS[level]

    return ConfidenceResult(
        level=level,
        band_factor=band_factor,
        mfu_used=mfu,
        bw_eff_used=bw_eff,
        anchor_matched=anchor_matched,
        geometry_source=model.geometry_source,
        best_anchor=best_anchor,
        extrapolation_distance=ext_dist,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Catalog-aware wrapper (used by capacity.py and ingest_anchor.py)
# ---------------------------------------------------------------------------


def confidence(
    model: ModelProfile,
    gpu: GpuProfile,
    dtype: str,
    isl: int,
    osl: int,
    concurrency: int,
) -> ConfidenceResult:
    """Look up live catalog anchors and run the rubric."""
    cat = _get_catalog()
    return compute_confidence_from_anchors(
        cat.anchors, model, gpu, dtype, isl, osl, concurrency
    )
