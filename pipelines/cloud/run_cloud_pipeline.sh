#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

collections=("top-free" "top-paid")

for collection in "${collections[@]}"; do
  echo "[Cloud Pipeline] Running Stage 1 scrape (${collection} all categories)..."
  python -m src.cloud.stage1.app_store_scraper_v2_cloud \
    --collection "$collection" \
    --all-categories \
    --country us \
    --limit 100 \
    --note "${collection} all categories cloud refresh"
done

echo "[Cloud Pipeline] Running Stage 2 scoring..."
python -m src.cloud.scripts.reuse_stage2_scores_cloud
python -m src.cloud.stage2.app_stage2_analysis_cloud

echo "[Cloud Pipeline] Building snapshot deltas..."
python -m src.cloud.analysis.build_deltas_cloud

echo "[Cloud Pipeline] Generating embeddings..."
python -m src.cloud.analysis.generate_embeddings_cloud

echo "[Cloud Pipeline] Computing neighbours..."
python -m src.cloud.analysis.build_neighbors_cloud --all-runs

echo "[Cloud Pipeline] Clustering embeddings..."
python -m src.cloud.analysis.build_clusters_cloud --all-runs

echo "[Cloud Pipeline] Completed."
