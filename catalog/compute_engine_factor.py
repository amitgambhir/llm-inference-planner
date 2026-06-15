#!/usr/bin/env python3
"""Phase C helper: compute engine_factor[trtllm] from matched vLLM / TRT-LLM pairs.

Usage (after Phase B results are added to benchmarks_public.yaml):
    python catalog/compute_engine_factor.py

Prints the per-pair ratio and the median, which becomes engine_factor.trtllm.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        key = (p.isl, p.osl)
        pairs.setdefault(key, {})
        pairs[key][p.engine] = p.measured

ratios = []
print(f"{'ISL':>6} {'OSL':>5}  {'vLLM':>10}  {'TRT-LLM':>10}  {'ratio':>7}")
print("-" * 48)
for (isl, osl), engines in sorted(pairs.items()):
    if "vllm" in engines and "trtllm" in engines:
        r = engines["trtllm"] / engines["vllm"]
        ratios.append(r)
        print(f"{isl:>6} {osl:>5}  {engines['vllm']:>10.1f}  {engines['trtllm']:>10.1f}  {r:>7.3f}")
    else:
        missing = {"vllm", "trtllm"} - set(engines)
        print(f"{isl:>6} {osl:>5}  (missing {missing})")

if not ratios:
    print("\nNo matched pairs found. Add Phase B vLLM points to benchmarks_public.yaml first.")
    sys.exit(1)

median_ef = statistics.median(ratios)
print("-" * 48)
print(f"\nMedian engine_factor[trtllm] = {median_ef:.4f}")
print(f"\nPaste into planner/efficiency_constants.yaml:")
print(f"  engine_factor:")
print(f"    trtllm: {median_ef:.4f}")
print(f"    vllm: 1.0")
print(f"\nPaste into planner/validate.py PARAM_BOUNDS:")
lo = round(median_ef * 0.85, 4)
hi = round(median_ef * 1.15, 4)
print(f'    ("engine_factor.trtllm",  {lo}, {hi}),   # pinned ±15% around measured median {median_ef:.3f}')
