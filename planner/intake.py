"""
planner/intake.py — multi-mode demand resolver.

Accepts three demand input modes and resolves them to a single
(requests_per_day, users) pair for downstream use by the capacity planner.
users is a cross-cutting optional that unlocks $/user/month in any mode.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional


class WorkloadError(ValueError):
    """Raised when demand input is invalid (no source provided)."""


@dataclass
class DemandSpec:
    requests_per_day: Optional[float] = None
    avg_rps: Optional[float] = None
    users: Optional[int] = None
    prompts_per_user_per_day: Optional[float] = None


def resolve_demand(spec: DemandSpec) -> tuple[float, Optional[int]]:
    """Return (requests_per_day, users).

    Accepts at least ONE demand source; raises WorkloadError if none provided.
    Precedence: requests_per_day → avg_rps → users × prompts_per_user_per_day.
    When multiple sources are present the highest-precedence one wins; a warning
    is emitted if any secondary source diverges from the primary by more than 20%.
    users is passed through regardless of which demand mode was used.
    """
    rpd_direct = spec.requests_per_day
    rpd_rps = spec.avg_rps * 86_400 if spec.avg_rps is not None else None
    rpd_users = (
        spec.users * spec.prompts_per_user_per_day
        if spec.users is not None and spec.prompts_per_user_per_day is not None
        else None
    )

    candidates: list[tuple[str, Optional[float]]] = [
        ("requests_per_day", rpd_direct),
        ("avg_rps", rpd_rps),
        ("users×prompts", rpd_users),
    ]
    provided = [(name, val) for name, val in candidates if val is not None]

    if not provided:
        raise WorkloadError(
            "No demand source provided. Supply one of: --requests-per-day, "
            "--avg-rps, or --users + --prompts-per-user-per-day."
        )

    primary_name, rpd = provided[0]

    for name, val in provided[1:]:
        divergence = abs(val - rpd) / max(rpd, 1.0)
        if divergence > 0.20:
            warnings.warn(
                f"Demand source '{primary_name}' ({rpd:,.0f} req/day) and "
                f"'{name}' ({val:,.0f} req/day) diverge by {divergence:.0%}. "
                f"Using '{primary_name}'.",
                stacklevel=2,
            )

    return rpd, spec.users
