#!/usr/bin/env python3
"""Auto-update mechanical metrics in README.md and CLAUDE.md.

Updates:
  - Test count (from pytest --collect-only)
  - GPU count (from catalog/gpus.yaml)
  - Model count (from catalog/models.yaml)

Prints a summary of every substitution made. Exit 0 always — callers
decide whether to block on warnings.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


# ── Gather metrics ────────────────────────────────────────────────────────────

def count_tests() -> int | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             "--no-header", "--tb=no"],
            capture_output=True, text=True, cwd=REPO,
            env={**__import__("os").environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            timeout=30,
        )
        # Last non-blank line is "N tests selected" or "N passed, ..."
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            m = re.match(r"(\d+)\s+test", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def count_yaml_top_level_keys(path: Path) -> int | None:
    try:
        import yaml  # type: ignore[import]
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            return len(data)
    except Exception:
        pass
    return None


# ── Patch helpers ─────────────────────────────────────────────────────────────

def patch_file(path: Path, substitutions: list[tuple[str, str]]) -> list[str]:
    """Apply regex substitutions to a file. Returns list of human-readable changes."""
    original = path.read_text()
    current = original
    changes: list[str] = []

    for pattern, replacement in substitutions:
        new, n = re.subn(pattern, replacement, current)
        if n and new != current:
            changes.append(f"  {path.name}: {pattern!r} → {replacement!r}")
            current = new

    if current != original:
        path.write_text(current)

    return changes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    n_tests = count_tests()
    n_gpus = count_yaml_top_level_keys(REPO / "catalog" / "gpus.yaml")
    n_models = count_yaml_top_level_keys(REPO / "catalog" / "models.yaml")

    if n_tests is None:
        print("[update-docs] WARNING: could not count tests — skipping test count update")
    if n_gpus is None:
        print("[update-docs] WARNING: could not count GPUs from catalog/gpus.yaml")
    if n_models is None:
        print("[update-docs] WARNING: could not count models from catalog/models.yaml")

    readme = REPO / "README.md"
    claude = REPO / "CLAUDE.md"

    all_changes: list[str] = []

    # ── README.md ──
    readme_subs: list[tuple[str, str]] = []
    if n_tests is not None:
        readme_subs.append((r"\b\d+\s+tests,\s+all\s+must\s+pass\b",
                            f"{n_tests} tests, all must pass"))
    if n_gpus is not None:
        readme_subs.append((r"\b\d+\s+GPUs:\s+NVIDIA\b",
                            f"{n_gpus} GPUs: NVIDIA"))
    if n_models is not None:
        readme_subs.append((r"\b\d+\s+catalog\s+models\b",
                            f"{n_models} catalog models"))
    all_changes += patch_file(readme, readme_subs)

    # ── CLAUDE.md ──
    claude_subs: list[tuple[str, str]] = []
    if n_tests is not None:
        claude_subs.append((r"\b\d+\s+tests\s+across\s+\d+\s+files\b",
                            f"{n_tests} tests across 14 files"))
    if n_gpus is not None:
        claude_subs.append((r"\b\d+\s+GPUs:\s+NVIDIA\s+\(H100",
                            f"{n_gpus} GPUs: NVIDIA (H100"))
    if n_models is not None:
        claude_subs.append((r"\b\d+\s+entries\s+including\s+Llama\b",
                            f"{n_models} entries including Llama"))
    all_changes += patch_file(claude, claude_subs)

    if all_changes:
        print("[update-docs] Auto-updated metrics in docs:")
        for c in all_changes:
            print(c)
    else:
        print("[update-docs] Docs metrics already up to date.")

    if n_tests is not None:
        print(f"[update-docs] Current: {n_tests} tests, {n_gpus} GPUs, {n_models} models")


if __name__ == "__main__":
    main()
