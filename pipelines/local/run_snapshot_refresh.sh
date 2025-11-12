#!/usr/bin/env bash
# Local runner that refreshes the snapshot database, reuses Stage 2 scores,
# re-scores remaining apps, and rebuilds the delta table.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."   # must be set before running
#   ./pipelines/local/run_snapshot_refresh.sh

set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY environment variable is not set." >&2
  exit 1
fi

# Determine repository root (handles execution from any directory)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# Activate virtual environment if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "[1/5] Running Stage 1 scraper (top-free)..."
python -m src.local.stage1.app_store_scraper_v2 \
  --collection top-free \
  --all-categories \
  --limit 100 \
  --note "local cron top-free"

echo "[2/5] Running Stage 1 scraper (top-paid)..."
python -m src.local.stage1.app_store_scraper_v2 \
  --collection top-paid \
  --all-categories \
  --limit 100 \
  --note "local cron top-paid"

echo "[3/5] Reusing existing Stage 2 scores..."
python -m src.local.scripts.reuse_stage2_scores --log-level INFO

echo "[4/5] Running Stage 2 scoring..."
python -m src.local.stage2.app_stage2_analysis

echo "[5/5] Building snapshot deltas..."
python -m src.local.analysis.build_deltas

echo "Snapshot refresh complete."
echo "Database updated at: exports/app_store_apps_v2.db"
echo "If you track the database in Git, remember to commit/push manually (beware of large file sizes)."
