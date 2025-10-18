#!/usr/bin/env bash
# Local runner that refreshes the snapshot database, reuses Stage 2 scores,
# re-scores remaining apps, and rebuilds the delta table.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."   # must be set before running
#   ./run_snapshot_refresh.sh

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY environment variable is not set." >&2
  exit 1
fi

# Determine repository root (handles execution from any directory)
REPO_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# Activate virtual environment if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "[1/5] Running Stage 1 scraper (top-free)..."
python app_store_scraper_v2.py \
  --collection top-free \
  --all-categories \
  --limit 100 \
  --note "local cron top-free"

echo "[2/5] Running Stage 1 scraper (top-paid)..."
python app_store_scraper_v2.py \
  --collection top-paid \
  --all-categories \
  --limit 100 \
  --note "local cron top-paid"

echo "[3/5] Reusing existing Stage 2 scores..."
python scripts/reuse_stage2_scores.py --log-level INFO

echo "[4/5] Running Stage 2 scoring..."
python app_stage2_analysis.py

echo "[5/5] Building snapshot deltas..."
python analysis/build_deltas.py

echo "Snapshot refresh complete."
echo "Database updated at: exports/app_store_apps_v2.db"
echo "If you track the database in Git, remember to commit/push manually (beware of large file sizes)."
