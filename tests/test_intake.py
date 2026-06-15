"""Tests for planner/intake.py — multi-mode demand resolver."""
import warnings
import pytest

from planner.intake import DemandSpec, WorkloadError, resolve_demand


# ---------------------------------------------------------------------------
# All three modes resolve to (requests_per_day, users)
# ---------------------------------------------------------------------------


def test_direct_mode_returns_rpd():
    rpd, users = resolve_demand(DemandSpec(requests_per_day=8_640_000))
    assert rpd == pytest.approx(8_640_000)
    assert users is None


def test_avg_rps_mode_converts_to_rpd():
    rpd, users = resolve_demand(DemandSpec(avg_rps=100))
    assert rpd == pytest.approx(8_640_000)
    assert users is None


def test_users_mode_multiplies_prompts():
    rpd, users = resolve_demand(DemandSpec(users=50_000, prompts_per_user_per_day=2))
    assert rpd == pytest.approx(100_000)
    assert users == 50_000


def test_three_equivalent_modes_produce_same_rpd():
    rpd_direct, _ = resolve_demand(DemandSpec(requests_per_day=8_640_000))
    rpd_rps, _ = resolve_demand(DemandSpec(avg_rps=100))
    rpd_users, _ = resolve_demand(DemandSpec(users=86_400, prompts_per_user_per_day=100))
    assert rpd_direct == pytest.approx(rpd_rps)
    assert rpd_direct == pytest.approx(rpd_users)


# ---------------------------------------------------------------------------
# users is passed through in all modes
# ---------------------------------------------------------------------------


def test_users_passed_through_in_direct_mode():
    _, users = resolve_demand(DemandSpec(requests_per_day=100_000_000, users=500_000))
    assert users == 500_000


def test_users_passed_through_in_rps_mode():
    _, users = resolve_demand(DemandSpec(avg_rps=1157, users=500_000))
    assert users == 500_000


def test_users_is_none_when_not_supplied():
    _, users = resolve_demand(DemandSpec(requests_per_day=1_000))
    assert users is None


# ---------------------------------------------------------------------------
# Precedence: requests_per_day wins over avg_rps
# ---------------------------------------------------------------------------


def test_precedence_rpd_beats_avg_rps():
    # avg_rps=1 → 86_400; requests_per_day=999_999 wins
    rpd, _ = resolve_demand(DemandSpec(requests_per_day=999_999, avg_rps=1))
    assert rpd == pytest.approx(999_999)


def test_precedence_avg_rps_beats_users_mode():
    # avg_rps=1 → 86_400; users mode → 1 × 1 = 1; avg_rps wins
    rpd, _ = resolve_demand(DemandSpec(avg_rps=1, users=1, prompts_per_user_per_day=1))
    assert rpd == pytest.approx(86_400)


# ---------------------------------------------------------------------------
# Error: no demand source
# ---------------------------------------------------------------------------


def test_no_source_raises_workload_error():
    with pytest.raises(WorkloadError, match="No demand source"):
        resolve_demand(DemandSpec())


def test_users_only_without_prompts_raises():
    # users alone is not a demand source — prompts_per_user_per_day is required
    with pytest.raises(WorkloadError):
        resolve_demand(DemandSpec(users=10_000))


# ---------------------------------------------------------------------------
# Warning when two sources diverge >20%
# ---------------------------------------------------------------------------


def test_diverging_sources_emit_warning():
    # requests_per_day=100_000, avg_rps=10 → 864_000 (764% divergence)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_demand(DemandSpec(requests_per_day=100_000, avg_rps=10))
    assert len(caught) == 1
    assert "diverge" in str(caught[0].message).lower()


def test_close_sources_no_warning():
    # requests_per_day=864_000, avg_rps=10 → exactly same — no warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_demand(DemandSpec(requests_per_day=864_000, avg_rps=10))
    assert len(caught) == 0


def test_warning_uses_primary_value():
    # Primary (requests_per_day) should still be returned despite warning
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        rpd, _ = resolve_demand(DemandSpec(requests_per_day=1_000, avg_rps=1000))
    assert rpd == pytest.approx(1_000)
