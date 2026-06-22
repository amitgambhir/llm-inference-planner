#!/usr/bin/env bash
# Point git at the committed hooks directory.
# Run once after cloning: bash scripts/setup_hooks.sh

set -euo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/pre-push

echo "✓ Git hooks installed (.githooks/pre-commit, .githooks/pre-push)"
echo "  Run 'git config --unset core.hooksPath' to remove."
