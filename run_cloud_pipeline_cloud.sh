#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

collections=("top-free" "top-paid")

for collection in "${collections[@]}"; do
  echo "[Cloud Pipeline] Running Stage 1 scrape (${collection} all categories)..."
  python "$SCRIPT_DIR/app_store_scraper_v2_cloud.py" \
    --collection "$collection" \
    --all-categories \
    --country us \
    --limit 100 \
    --note "${collection} all categories cloud refresh"
done

echo "[Cloud Pipeline] Running Stage 2 scoring..."
python "$SCRIPT_DIR/app_stage2_analysis_cloud.py"

echo "[Cloud Pipeline] Building snapshot deltas..."
python "$SCRIPT_DIR/analysis/build_deltas_cloud.py"

echo "[Cloud Pipeline] Generating embeddings..."
python "$SCRIPT_DIR/analysis/generate_embeddings_cloud.py"

echo "[Cloud Pipeline] Computing neighbours..."
python "$SCRIPT_DIR/analysis/build_neighbors_cloud.py" --all-runs

echo "[Cloud Pipeline] Clustering embeddings..."
python "$SCRIPT_DIR/analysis/build_clusters_cloud.py" --all-runs

echo "[Cloud Pipeline] Completed."
