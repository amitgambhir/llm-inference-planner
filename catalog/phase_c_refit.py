#!/usr/bin/env python3
"""Phase C: pin engine_factor[trtllm], run final refit, print CV + sensitivity.

Run this after all 5 Phase B measured values are filled into benchmarks_public.yaml:
    python catalog/phase_c_refit.py

What it does:
  1. Computes median(trtllm/vllm) from matched H100 offline uniform pairs
  2. Updates efficiency_constants.yaml: engine_factor.trtllm = median
  3. Updates PARAM_BOUNDS in validate.py: engine_factor.trtllm = [median*0.85, median*1.15]
  4. Runs coordinate-descent refit (seed=42, train_frac=0.80)
  5. Prints full fit-set accuracy, validate-point accuracy, CV, sensitivity
  6. Prints the tightened test targets for Phase C

You still need to manually:
  - Review the printed targets and update tests/test_validation.py
  - git commit the final constants + YAML + tests
"""
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml

CONSTANTS_PATH = ROOT / "planner" / "efficiency_constants.yaml"
VALIDATE_PATH  = ROOT / "planner" / "validate.py"

# ── Step 1: compute median engine_factor ──────────────────────────────────────
print("=" * 60)
print("Step 1: computing engine_factor[trtllm] from matched pairs")
print("=" * 60)

from planner.validate import load_public_benchmarks

pts = load_public_benchmarks()

pairs: dict = {}
for p in pts:
    if (
        p.model == "llama-3.1-8b"
        and p.gpu == "h100_sxm"
        and p.scenario == "offline"
        and p.dataset == "uniform"
        and p.engine in ("vllm", "trtllm")
    ):
        pairs.setdefault((p.isl, p.osl), {})
        pairs[(p.isl, p.osl)][p.engine] = p.measured

ratios = []
print(f"{'ISL':>6} {'OSL':>5}  {'vLLM':>10}  {'TRT-LLM':>10}  {'ratio':>7}")
print("-" * 50)
for (isl, osl), engines in sorted(pairs.items()):
    if "vllm" in engines and "trtllm" in engines:
        r = engines["trtllm"] / engines["vllm"]
        ratios.append(r)
        print(f"{isl:>6} {osl:>5}  {engines['vllm']:>10.1f}  {engines['trtllm']:>10.1f}  {r:>7.3f}")
    else:
        print(f"{isl:>6} {osl:>5}  (incomplete — missing {{'vllm','trtllm'} - set(engines)})")

if not ratios:
    sys.exit(
        "\nNo matched pairs found. "
        "Fill in all 5 Phase B measured values in catalog/benchmarks_public.yaml first."
    )

ef = statistics.median(ratios)
ef_lo = round(ef * 0.85, 4)
ef_hi = round(ef * 1.15, 4)
print(f"\nMedian engine_factor[trtllm] = {ef:.4f}  (bounds [{ef_lo}, {ef_hi}])")

# ── Step 2: update efficiency_constants.yaml ──────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: updating efficiency_constants.yaml")
print("=" * 60)

c = yaml.safe_load(CONSTANTS_PATH.read_text())
c.setdefault("engine_factor", {})
c["engine_factor"]["trtllm"] = ef
c["engine_factor"]["vllm"] = 1.0
CONSTANTS_PATH.write_text(yaml.dump(c, default_flow_style=False, sort_keys=True))
print(f"  engine_factor.trtllm → {ef:.4f}")

# ── Step 3: update PARAM_BOUNDS in validate.py ────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: narrowing PARAM_BOUNDS engine_factor.trtllm in validate.py")
print("=" * 60)

src = VALIDATE_PATH.read_text()
old_pat = re.compile(
    r'(\("engine_factor\.trtllm",\s*)[\d.]+(\s*,\s*)[\d.]+(\s*\),)'
)
new_line = f'("engine_factor.trtllm",  {ef_lo}, {ef_hi}),'
match = old_pat.search(src)
if match:
    new_src = old_pat.sub(
        f'("engine_factor.trtllm",  {ef_lo}, {ef_hi}),',
        src,
        count=1,
    )
    VALIDATE_PATH.write_text(new_src)
    print(f"  PARAM_BOUNDS engine_factor.trtllm → [{ef_lo}, {ef_hi}]")
else:
    print("  WARNING: could not find engine_factor.trtllm in PARAM_BOUNDS — update manually.")

# ── Step 4: refit ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: running coordinate-descent refit (seed=42, train_frac=0.80)")
print("=" * 60)

import importlib
import planner.efficiency as _eff
import planner.validate as _validate

_eff.reload_constants()
importlib.reload(_validate)
from planner.validate import (
    cv_leave_one_gpu_out,
    fit,
    load_public_benchmarks,
    parameter_sensitivity,
    report,
)

pts = load_public_benchmarks()
fitted = fit(pts, train_frac=0.80, seed=42)
print(f"  train median: {fitted.train_median_rel_error:.1%}")
print(f"  holdout median: {fitted.holdout_median_rel_error}")
print(f"  n_train={fitted.n_train}  n_holdout={fitted.n_holdout}")

# ── Step 5: accuracy report ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: accuracy")
print("=" * 60)

r_fit = report(pts, fit_roles=("level", "shape"))
print(f"  Fit set (level+shape):  median={r_fit.median_rel_error:.1%}  "
      f"p90={r_fit.p90_rel_error:.1%}  max={r_fit.max_rel_error:.1%}  "
      f"n={r_fit.n_points}")

r_val = report(pts, fit_roles=("validate",))
print(f"  Validate (distribution): median={r_val.median_rel_error:.1%}  "
      f"n={r_val.n_points}")

# ── Step 6: CV ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: leave-one-GPU-out CV")
print("=" * 60)

cv_results = cv_leave_one_gpu_out(pts)
for cr in cv_results:
    flag = "  *** > 100% ***" if cr.holdout_median_rel_error > 1.0 else ""
    print(f"  left_out={cr.left_out_gpu:20}  "
          f"train={cr.train_median_rel_error:.1%}  "
          f"holdout={cr.holdout_median_rel_error:.1%}  "
          f"n_holdout={cr.n_holdout}{flag}")

# ── Step 7: sensitivity ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: parameter sensitivity (±15%)")
print("=" * 60)

sens = parameter_sensitivity(pts)
flat = [s for s in sens if s.is_flat]
print(f"  {len(sens)} params, {len(flat)} flat")
for s in sorted(sens, key=lambda x: x.delta_error):
    flag = "  FLAT" if s.is_flat else ""
    print(f"  {s.param:35}  delta={s.delta_error:.4f}{flag}")

# ── Step 8: suggested Phase C test targets ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8: suggested Phase C test targets")
print("=" * 60)
print("  Update tests/test_validation.py test_no_anchor_curves_hit_target:")
print(f"    assert r.median_rel_error <= {min(0.20, round(r_fit.median_rel_error * 1.2, 2))}")
print(f"    assert r.p90_rel_error    <= {min(0.60, round(r_fit.p90_rel_error    * 1.2, 2))}")
print(f"    assert r.max_rel_error    <= {min(1.20, round(r_fit.max_rel_error    * 1.2, 2))}")
print("\nDone. Review the numbers above, then commit:")
print("  git add planner/efficiency_constants.yaml planner/validate.py \\")
print("          catalog/benchmarks_public.yaml tests/test_validation.py")
print('  git commit -m "feat: Phase C — pin engine_factor[trtllm], final refit + CV"')
