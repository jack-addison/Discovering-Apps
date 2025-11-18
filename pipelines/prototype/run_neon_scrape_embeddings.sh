#!/usr/bin/env bash
# Neon pipeline helper: scrape top 500 apps per category (free + paid) and refresh embeddings.

set -euo pipefail

if [[ -z "${PROTOTYPE_DATABASE_URL:-}" ]]; then
  echo "ERROR: PROTOTYPE_DATABASE_URL must be set to your Neon/Postgres connection string." >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY must be set before generating embeddings." >&2
  exit 1
fi

DATE_NOTE="$(date -u '+%Y-%m-%d')"

echo ">>> Scraping top-free (all categories, limit 500)..."
python -m src.prototype.app_store_scraper_neon \
  --collection top-free \
  --all-categories \
  --limit 500 \
  --note "Top free run ${DATE_NOTE}"

echo ">>> Scraping top-paid (all categories, limit 500)..."
python -m src.prototype.app_store_scraper_neon \
  --collection top-paid \
  --all-categories \
  --limit 500 \
  --note "Top paid run ${DATE_NOTE}"

echo ">>> Generating embeddings for all snapshots..."
python -m src.prototype.analysis.generate_embeddings_neon \
  --model text-embedding-3-small \
  --batch-size 200 \
  --all-snapshots

echo ">>> Clustering all embedded snapshots..."
python -m src.prototype.analysis.cluster_all \
  --scope-label all \
  --model text-embedding-3-small \
  --clusters 60 \
  --min-cluster-size 4 \
  --top-keywords 6

echo ">>> Building UMAP projection..."
python -m src.prototype.analysis.build_umap \
  --scope-label all \
  --model text-embedding-3-small \
  --run-id "$(psql "$PROTOTYPE_DATABASE_URL" -At -c 'SELECT id FROM scrape_runs ORDER BY id DESC LIMIT 1')" \
  --run-id "$(psql "$PROTOTYPE_DATABASE_URL" -At -c 'SELECT id FROM scrape_runs ORDER BY id DESC OFFSET 1 LIMIT 1')"

echo ">>> Rebuilding snapshot deltas..."
python -m src.prototype.analysis.build_deltas_neon

echo "Pipeline complete."
