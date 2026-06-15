"""Tests for planner/explain.py — napkin-math sizing explainer."""
import pytest

from planner.catalog import get_gpu, get_model
from planner.capacity import plan
from planner.cost import compute_cost
from planner.explain import render_napkin_math
from planner.report import render_report
import planner.catalog as _catalog_module


@pytest.fixture(autouse=True)
def reset_catalog():
    _catalog_module._catalog = None
    yield
    _catalog_module._catalog = None


# ── Golden estimate fixture ───────────────────────────────────────────────────


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


@pytest.fixture
def golden_cost(golden_estimate):
    return compute_cost(golden_estimate, "h100_sxm", tp=1)


# ── render_napkin_math returns a non-empty string ─────────────────────────────


def test_render_napkin_math_returns_string(golden_estimate):
    result = render_napkin_math(golden_estimate)
    assert isinstance(result, str)
    assert len(result) > 100


# ── Correct values appear in output ──────────────────────────────────────────


def test_napkin_math_contains_avg_rps(golden_estimate):
    result = render_napkin_math(golden_estimate)
    # 100M / 86400 ≈ 1157.4
    assert "1,157" in result or "1157" in result


def test_napkin_math_contains_peak_rps(golden_estimate):
    result = render_napkin_math(golden_estimate)
    # 1157.4 * 3 ≈ 3472
    assert "3,472" in result or "3472" in result


def test_napkin_math_contains_replicas(golden_estimate):
    result = render_napkin_math(golden_estimate)
    assert str(golden_estimate.replicas) in result


def test_napkin_math_contains_binding_constraint(golden_estimate):
    result = render_napkin_math(golden_estimate)
    assert golden_estimate.binding_constraint in result


def test_napkin_math_contains_mfu(golden_estimate):
    result = render_napkin_math(golden_estimate)
    assert f"{golden_estimate.mfu_used:.2f}" in result


def test_napkin_math_contains_headroom_factor(golden_estimate):
    result = render_napkin_math(golden_estimate)
    assert f"{golden_estimate.headroom_factor:.2f}" in result


# ── Parameterised: changing peak_multiplier changes peak RPS in output ────────


@pytest.mark.parametrize("peak_mult", [2.0, 3.0, 5.0])
def test_peak_rps_reflects_peak_multiplier(peak_mult):
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    est = plan(
        requests_per_day=100_000_000,
        peak_multiplier=peak_mult,
        isl=9000,
        osl=500,
        ttft_slo_ms=2000.0,
        model=model,
        gpu=gpu,
        dtype="mxfp4",
        tp=1,
        traffic_class="realtime",
    )
    result = render_napkin_math(est)
    expected_peak = est.traffic.peak_rps
    # The rendered peak RPS must match the estimate's actual peak
    assert f"{expected_peak:,.1f}" in result


# ── Cost section absent when cost=None ───────────────────────────────────────


def test_cost_section_absent_without_cost_arg(golden_estimate):
    result = render_napkin_math(golden_estimate, cost=None)
    assert "Cost" not in result
    assert "$/hr" not in result


def test_cost_section_present_with_cost_arg(golden_estimate, golden_cost):
    result = render_napkin_math(golden_estimate, cost=golden_cost)
    assert "Cost" in result
    assert "/day" in result


# ── Per-user line absent when users=None ─────────────────────────────────────


def test_per_user_line_absent_when_users_none(golden_estimate, golden_cost):
    assert golden_estimate.users is None
    result = render_napkin_math(golden_estimate, cost=golden_cost)
    assert "/user/month" not in result


def test_per_user_line_present_when_users_set(golden_cost):
    model = get_model("gpt-oss-20b")
    gpu = get_gpu("h100_sxm")
    est = plan(
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
        users=500_000,
    )
    cost = compute_cost(est, "h100_sxm", tp=1)
    result = render_napkin_math(est, cost=cost)
    assert "/user/month" in result
    assert "500,000" in result


# ── include_napkin_math in render_report ──────────────────────────────────────


def test_report_without_napkin_math_has_no_section(golden_estimate):
    report = render_report(
        estimate=golden_estimate,
        scenario_label="test",
        model_name="gpt-oss-20b",
        gpu_name="h100_sxm",
        dtype="mxfp4",
        isl=9000,
        osl=500,
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        include_napkin_math=False,
    )
    assert "How we got here" not in report


def test_report_with_napkin_math_has_section(golden_estimate):
    report = render_report(
        estimate=golden_estimate,
        scenario_label="test",
        model_name="gpt-oss-20b",
        gpu_name="h100_sxm",
        dtype="mxfp4",
        isl=9000,
        osl=500,
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        include_napkin_math=True,
    )
    assert "How we got here" in report
    assert str(golden_estimate.replicas) in report


def test_report_napkin_math_section_contains_binding_constraint(golden_estimate):
    report = render_report(
        estimate=golden_estimate,
        scenario_label="test",
        model_name="gpt-oss-20b",
        gpu_name="h100_sxm",
        dtype="mxfp4",
        isl=9000,
        osl=500,
        requests_per_day=100_000_000,
        peak_multiplier=3.0,
        include_napkin_math=True,
    )
    assert golden_estimate.binding_constraint in report
