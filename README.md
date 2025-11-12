# Apple App Store Scraper

Stage 1 delivers a lightweight Python scraper that harvests metadata for iOS apps and stores it in a SQLite database under `exports/app_store_apps.db`.

## Features
- Keyword search, single chart pulls, or full-category sweeps over (`top-free`, `top-paid`, `top-grossing`).
- Captures category, review score, description, and an estimated download count (rating count proxy).
- Persists results with metadata such as developer, price, languages, artwork URL, plus JSON chart membership data per app.
- Idempotent upserts so repeated runs refresh existing rows.

> Apple does not expose real download totals. The `number_of_downloads` column stores the public rating count as a conservative proxy. Stage 2 can swap in a truer metric if a source becomes available.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prerequisites
- Python 3.9+ (project tested with 3.9)
- `sqlite3` CLI (optional, useful for inspecting the database)
- OpenAI API key (only required for Stage 2 scoring)

### Repository layout
- `src/local/stage1/app_store_scraper.py` – Stage 1 scraper pulling App Store metadata (latest snapshot only).
- `src/local/stage1/app_store_scraper_v2.py` – Snapshot-aware Stage 1 scraper capturing historical runs and richer metadata.
- `src/local/stage2/app_stage2_analysis.py` – Stage 2 LLM scoring script that enriches the local SQLite database.
- `src/local/analysis/` – Local analysis utilities (`build_deltas.py`, `generate_embeddings.py`, `build_neighbors.py`, `build_clusters.py`).
- `src/local/scripts/reuse_stage2_scores.py` – Reuses Stage 2 scores when metadata is unchanged.
- `src/cloud/stage1/app_store_scraper_v2_cloud.py` – Cloud-connected Stage 1 scraper (writes snapshots to SQLiteCloud).
- `src/cloud/stage2/app_stage2_analysis_cloud.py` – Stage 2 scoring script targeting SQLiteCloud.
- `src/cloud/analysis/` – Cloud variants of the analysis pipelines (deltas, embeddings, neighbours, clusters).
- `src/cloud/scripts/reuse_stage2_scores_cloud.py` – Reuse logic for Stage 2 scores in SQLiteCloud.
- `src/prototype/` – Experimental PostgreSQL tooling (Neon schema, migration scripts, Postgres-native scraper).
- `src/prototype/analysis/` – Neon analysis jobs (dissatisfied-app selection, embeddings, clustering).
- `apps/local/` – Visualization entry points for local data (`streamlit_app.py`, `visualize_scores.py`, `visualize_scores_interactive.py`).
- `apps/cloud/` – Cloud Streamlit dashboards (`streamlit_app_cloud.py`, `streamlit_app_neon.py`).
- `pipelines/local/run_snapshot_refresh.sh` – Local cron-friendly pipeline runner.
- `pipelines/cloud/run_cloud_pipeline.sh` – Cloud ingestion & analysis orchestrator.
- `config/cloud.py` – SQLiteCloud connection details and helper.
- `artifacts/experiments/` – Sample outputs from experiment scripts.
- `exports/app_store_apps.db` / `exports/app_store_apps_v2.db` – Sample SQLite databases populated via the scrapers.
- `requirements.txt` – Python dependencies.
- `visualizations/` – Generated charts (PNG/HTML).

Run a keyword scrape:

```bash
python -m src.local.stage1.app_store_scraper --search-term "productivity" --limit 25
```

Gather a chart:

```bash
python -m src.local.stage1.app_store_scraper --collection top-free --country us --limit 50
```

Harvest the top 100 apps for every category (default limit is 100):

```bash
python -m src.local.stage1.app_store_scraper --collection top-free --all-categories
```

### Snapshot-aware scraper (Stage 1b)
Use the v2 script when you want to keep historical runs and additional metadata (release dates, version info, price tier, screenshots, etc.). Each invocation writes to `exports/app_store_apps_v2.db` by default and records a new entry in `scrape_runs`, `app_snapshots`, and `app_rankings`.

```bash
python -m src.local.stage1.app_store_scraper_v2 \
  --collection top-free \
  --all-categories \
  --country us \
  --limit 100 \
  --note "top-free all categories"
```

Switch to `--search-term` for keyword scrapes, or provide `--output-db` to store snapshots in a different location without touching the original Stage 1 database.

The SQLite database lives at `exports/app_store_apps.db`. Inspect it with the built-in shell:

```bash
sqlite3 exports/app_store_apps.db "SELECT name, category, review_score, number_of_downloads FROM apps LIMIT 5;"
```

When you sweep every category, the `chart_memberships` column stores a JSON array like:

```json
[{"chart_type": "top-free", "category_id": "6018", "category_name": "Books", "rank": 3}]
```

## Stage 2: LLM Scoring

Use the Stage 2 script to enrich the snapshot database with estimated build time and success potential:

```bash
export OPENAI_API_KEY=sk-...
python -m src.local.stage2.app_stage2_analysis --run-id 5 --run-id 6
```

### Getting an OpenAI API key
- Create an OpenAI account (https://platform.openai.com/) and visit **Dashboard → API keys**.
- Generate a new secret key and copy it immediately; you can only view it once.
- **Do not** commit the key to source control. Store it in a password manager or `.env` file that remains local.

### Supplying the key to the script
- **macOS/Linux (bash/zsh)**  
  ```bash
  export OPENAI_API_KEY="sk-your-key"
  python -m src.local.stage2.app_stage2_analysis
  ```
- **macOS/Linux (Fish shell)**  
  ```fish
  set -x OPENAI_API_KEY "sk-your-key"
  python -m src.local.stage2.app_stage2_analysis
  ```
- **Windows PowerShell**  
  ```powershell
  setx OPENAI_API_KEY "sk-your-key"
  # restart your shell, then:
  python -m src.local.stage2.app_stage2_analysis
  ```
- **Windows CMD (temporary for session)**  
  ```cmd
  set OPENAI_API_KEY=sk-your-key
  python -m src.local.stage2.app_stage2_analysis
  ```
- For reusable workflows, place the export in your shell profile or use a local `.env` file with a loader (e.g., `python -m dotenv run -- python -m src.local.stage2.app_stage2_analysis`), but keep that file out of version control.

Key notes:
- Targets `exports/app_store_apps_v2.db` by default (override with `--db-path`). Use one or more `--run-id` arguments to limit which scrape snapshots are scored.
- The script adds `build_time_estimate`, `success_score`, and `success_reasoning` columns if they are missing, then iterates apps (skipping already scored rows unless you pass `--force`).
- It prompts an OpenAI model (default `gpt-4.1-mini`) with app metadata, asking for an MVP build-time estimate and a 0–100 success score. Adjust the model with `--model`.
- Progress logs appear every 20 apps; you can limit runs with `--max-apps` for dry runs.
- Handle rate limits automatically with retry/back-off (configure via `--max-retries` and `--retry-wait`).

### How the scoring prompt works
- **Build time estimate**: The LLM receives the app’s description, category, pricing, chart position, and language coverage, then estimates the weeks a small senior team would need to ship a minimal viable product. Infrastructure is assumed to be greenfield, so complex integrations or rich content tend to raise the estimate.
- **Success score (0–100)**: The model balances market signals (rating average, rating volume/download proxy, chart ranks, developer reputation) against the qualitative description. High review count + strong ratings + popular categories push the score upward; sparse feedback or niche ideas drag it down.
- **Reasoning string**: Each response includes a short justification for GPT’s numbers. We store only the numeric fields in SQLite, but the reasoning is logged for debugging.
- **Quick-win definition**: Throughout the tooling, “quick wins” refers to `build_time_estimate ≤ 12` and `success_score ≥ 70`. This threshold powers the shading in the plots and the leaderboard filters.

- **Batch experiment**: `python -m src.local.experiments.batch_stage2_experiment` tests grouping apps (default 20 at a time, first 40 snapshots) and writes the raw model responses to `artifacts/experiments/batch_stage2_results.json` for comparison without touching the database.
- **Cheaper-model experiment**: `python -m src.local.experiments.cheaper_model_experiment` re-scores the first 40 snapshots with a cheaper model (default `gpt-3.5-turbo`) and stores results in `artifacts/experiments/cheaper_model_results.json` for side-by-side comparison.
- **Feature extraction**: `python -m src.local.analysis.build_deltas` materialises per-app snapshot deltas (e.g., success/rating/rank changes) into the `app_snapshot_deltas` table inside `exports/app_store_apps_v2.db`.

## Stage 3: Visualising Outcomes

- Static preview: `python -m apps.local.visualize_scores` renders `visualizations/success_vs_build_time.png`.
- Interactive dashboard: `python -m apps.local.visualize_scores_interactive --open` writes an HTML scatter plot to `visualizations/success_vs_build_time.html` and opens it in your browser. Categories are split into free/paid variants, and you can use flags such as `--min-ratings 500`, `--max-build-time 16`, `--min-success 70`, or `--quick-wins-only` to focus on specific cohorts.
- Streamlit app: `streamlit run apps/local/streamlit_app.py` launches an interactive workspace with filter controls (category, price tier, scrape run selection, rating volume, build time, success score, quick wins toggle), configurable 2D/3D scatter plots (choose axes, colour, bubble size), category summary bars, distribution box plots, a quick-win leaderboard, a similarity cluster explorer, and a revamped Opportunity Finder. The latter now exposes demand dissatisfaction (raw and percentile scores), execution floor controls (success score or success-per-week), cohort metric cards, concise result tables with expandable snapshots, embedded “similar apps” suggestions, and per-category standouts. The sidebar can be widened by tweaking the injected CSS in `apps/local/streamlit_app.py`.
- Hosted demo: visit [discovering-apps-jack.streamlit.app](https://discovering-apps-jack.streamlit.app) (cloud-backed) to explore the dashboard without running it locally. The hosted instance uses `apps/cloud/streamlit_app_cloud.py`, which reads from the remote SQLiteCloud database.
- Neon-backed prototype: `streamlit run apps/cloud/streamlit_app_neon.py` reads directly from the Neon PostgreSQL database (set `PROTOTYPE_DATABASE_URL` or `NEON_DATABASE_URL` before launching). The app currently exposes:
  - **Clusters tab** – browse embedding clusters, filter within a cluster by rating threshold / rating count / price tier, and stack-rank members via an opportunity score (rating volume × rating gap) with descriptions inline.
  - **Apps tab** – simple dropdown to inspect any app scraped into Neon.
  - **Deltas tab** – day-level view of rating and rating-count deltas plus top positive/negative movers, backed by `app_snapshot_deltas`.
- Hover a point to inspect the app’s scores, ratings volume, review average, and price. Toggle categories via the legend to declutter the view. The shaded quadrant highlights quick wins (high success score, low build effort).

## Similarity embeddings (experimental)

Generate text embeddings for snapshot descriptions to group comparable apps or build “similar apps” features:

```bash
export OPENAI_API_KEY=sk-...
python -m src.local.analysis.generate_embeddings --batch-size 50 --run-id 12
```

Key details:
- Defaults to the `text-embedding-3-small` model (cheaper, high recall). Override with `--model`.
- Stores vectors in `app_snapshot_embeddings` keyed by `(run_id, track_id, model)` with a description hash to avoid re-embedding unchanged rows.
- Supports `--run-id`, `--max-apps`, and `--force` flags; runs in batches with automatic retry handling.
- Combine the resulting vectors with cosine similarity or your preferred ANN index to surface competitive cohorts in the dashboard.

Compute nearest-neighbour tables (used by the Streamlit Opportunity Finder):

```bash
python -m src.local.analysis.build_neighbors --run-id 12 --top-k 5
```

- Uses the embeddings from `src/local/analysis/generate_embeddings.py` (defaults to `text-embedding-3-small`).
- Writes results into `app_snapshot_neighbors` with similarity scores, refreshed per run or across all runs (`--all-runs`).
- Tune `--min-similarity` or `--top-k` to balance precision vs breadth.

Build keyword-labelled clusters for the Streamlit similarity tab:

```bash
python -m src.local.analysis.build_clusters --clusters 20 --all-runs
```

- Clusters leverage normalised embeddings; keywords are extracted from member descriptions (basic stop-word filtering included).
- Results populate `app_snapshot_clusters` and `app_snapshot_cluster_members`, which drive the “Similarity clusters” tab.
- Adjust `--min-cluster-size` to drop or retain small cohorts.

### Cloud workflow

For the hosted dashboard, use the cloud-specific scripts (they target the SQLiteCloud database defined in `config/cloud.py`).

Typical end-to-end refresh:

```bash
# Example: refresh top-free all categories in the cloud DB
./pipelines/cloud/run_cloud_pipeline.sh --collection top-free --all-categories --country us --limit 100 --note "Top free cloud refresh"
```

The helper script executes:
- `src/cloud/stage1/app_store_scraper_v2_cloud.py` – Stage 1 scrape into SQLiteCloud.
- `src/cloud/stage2/app_stage2_analysis_cloud.py` – Stage 2 scoring.
- `src/cloud/scripts/reuse_stage2_scores_cloud.py` – Reuse prior Stage 2 scores before re-scoring.
- `src/cloud/analysis/build_deltas_cloud.py` – Snapshot deltas.
- `src/cloud/analysis/generate_embeddings_cloud.py` – Embeddings.
- `src/cloud/analysis/build_neighbors_cloud.py` – Similarity neighbours.
- `src/cloud/analysis/build_clusters_cloud.py` – Cluster labels.

Adjust flags as needed (e.g., run multiple scrapes separately for top-paid, keyword searches, etc.).

## Prototype: Neon/PostgreSQL Migration

The `src/prototype/` package provides a path to run the full pipeline on Postgres (e.g., a Neon project) instead of SQLite.

1. **Provision the database**  
   Create a Neon branch/database and copy the connection string. For local development you can tunnel through the Neon VS Code extension or run the local proxy (example DSN: `postgres://neon:npg@localhost:5432/<database_name>`). Export it as `PROTOTYPE_DATABASE_URL`.

2. **Create the schema**  
   Run `exports/schema_postgres.sql` against your Neon database (via the Neon SQL console or `psql`):  
   ```bash
   psql "$PROTOTYPE_DATABASE_URL" -f exports/schema_postgres.sql
   ```

3. **Copy data from SQLite**  
   Use the migration helper to bulk copy the existing `exports/app_store_apps_v2.db` into Postgres:  
   ```bash
   python -m src.prototype.migrate_sqlite_to_postgres
   ```
   The script ensures the schema exists (unless disabled), truncates destination tables, and inserts rows in batches. Pass `--postgres-dsn` if you prefer not to rely on the environment variable.  
   If you’d rather use `pgloader`, customize `src/prototype/pgloader.load.template` with absolute paths and the Neon DSN, then run `pgloader`.

4. **Point tooling at Neon**  
   Once migrated, you can connect the Streamlit app or Stage 2 scripts to Postgres by reading from `PROTOTYPE_DATABASE_URL` (future prototype work will add direct Postgres entry points). Until then, SQLite remains the default for the main pipeline.

5. **Neon-native scraper (optional)**  
   `python -m src.prototype.app_store_scraper_neon --collection top-free --all-categories --limit 400`  
   pulls up to the top 400 apps per category directly into Neon (the script falls back to whatever the feed exposes if fewer than 400 entries exist). Supply `--search-term` for keyword scrapes.
6. **Snapshot deltas in Neon**  
   `python -m src.prototype.analysis.build_deltas_neon`  
   recomputes run-to-run changes (ratings, price, etc.) inside Postgres so the Streamlit deltas tab has fresh data.

### Dissatisfied app pipeline (Neon)

Use this lightweight pipeline to surface categories where lots of users are unhappy, without running Stage 2:

1. Flag high-volume, low-rated apps per category:
   ```bash
   python -m src.prototype.analysis.select_dissatisfied --rating-quantile 0.7 --rating-threshold 3
   ```
   Adjust the quantile/threshold as needed; only apps above the category’s volume percentile and below the rating cutoff are stored in `app_snapshot_dissatisfied`.

2. Generate embeddings for the flagged cohort (or every snapshot via `--all-snapshots`, always skipping rows that already have vectors):
   ```bash
   export OPENAI_API_KEY=sk-...
   python -m src.prototype.analysis.generate_embeddings_neon --model text-embedding-3-small
   ```

3. Cluster embeddings to find theme groups:
   ```bash
   # For just the dissatisfied cohort
   python -m src.prototype.analysis.cluster_dissatisfied --clusters 20 --scope-label dissatisfied

   # Or cluster every embedded snapshot in Neon
   python -m src.prototype.analysis.cluster_all --clusters 40 --scope-label all
   ```
   Clusters are written back to `app_snapshot_clusters` / `app_snapshot_cluster_members`, scoped by the label you choose.

## Next steps
- Schedule recurring Stage 1 scrapes (cron, CI) so your dataset stays fresh.
- Feed the Stage 2 scores into downstream analyses (e.g., prioritisation dashboards or deeper GPT summaries).

## Workflow guidance
1. Run Stage 1 to refresh the SQLite database (`python -m src.local.stage1.app_store_scraper` or `..._v2`).
2. Run Stage 2 to score new rows (`python -m src.local.stage2.app_stage2_analysis`).
3. Re-generate visuals (`python -m apps.local.visualize_scores*`) or view the Streamlit dashboard to analyse quick wins.

## Contributing & license
- Pull requests/issues welcomed for bug fixes, new data sources, or visualisations.
- Choose and add an open-source license (e.g., MIT, Apache-2.0) that fits your needs.
