# Project Handoff Guide: Apple App Store Opportunity Explorer

_Last updated: $(date '+%Y-%m-%d')._

## 1. Project overview
This repository builds a dataset and analysis pipeline around Apple App Store apps with the goal of identifying high-upside, low-effort opportunities. The workflow comprises three stages:

1. **Stage 1 (Scraping)** – harvest metadata from the App Store.
2. **Stage 2 (LLM scoring)** – use OpenAI models to assign build-effort and success-likelihood scores.
3. **Stage 3 (Visualisation)** – explore the results via static/interactive charts and a hosted Streamlit dashboard.

The latest stack focuses on the **v2 snapshot pipeline**, which keeps historical runs for longitudinal analysis.

## 2. Repository layout
- `src/local/stage1/app_store_scraper.py`: original Stage 1 scraper (single snapshot, table `apps`).
- `src/local/stage1/app_store_scraper_v2.py`: enhanced Stage 1 scraper producing `exports/app_store_apps_v2.db` with tables `scrape_runs`, `app_snapshots`, `app_rankings`.
- `src/local/stage2/app_stage2_analysis.py`: LLM scoring script targeting the v2 database (`app_snapshots`). Adds `build_time_estimate`, `success_score`, `success_reasoning`. Supports optional `--run-id` filters.
- `src/local/analysis/`: local analysis utilities (`build_deltas.py`, `generate_embeddings.py`, `build_neighbors.py`, `build_clusters.py`).
- `src/local/scripts/reuse_stage2_scores.py`: reuse Stage 2 scores when metadata is unchanged.
- `src/local/experiments/`: batch and cheaper-model experiments (`batch_stage2_experiment.py`, `cheaper_model_experiment.py`).
- `src/cloud/stage1/app_store_scraper_v2_cloud.py`: Stage 1 scraper configured for SQLiteCloud output.
- `src/cloud/stage2/app_stage2_analysis_cloud.py`: Stage 2 scoring script targeting SQLiteCloud.
- `src/cloud/analysis/`: cloud-ready analysis scripts (`build_deltas_cloud.py`, `generate_embeddings_cloud.py`, `build_neighbors_cloud.py`, `build_clusters_cloud.py`).
- `src/cloud/scripts/reuse_stage2_scores_cloud.py`: reuse Stage 2 scores in SQLiteCloud.
- `src/prototype/`: experimental PostgreSQL tooling (Neon schema + migration utilities + Postgres-native scraper).
- `src/prototype/analysis/`: Neon-native analysis jobs (dissatisfied app selection, embeddings, clustering).
- `apps/local/`: Streamlit dashboard and visualization entry points for local data (`streamlit_app.py`, `visualize_scores.py`, `visualize_scores_interactive.py`).
- `apps/cloud/`: Streamlit dashboards for hosted environments (`streamlit_app_cloud.py`, `streamlit_app_neon.py`).
- `pipelines/local/run_snapshot_refresh.sh`: shell helper for the local pipeline.
- `pipelines/cloud/run_cloud_pipeline.sh`: shell helper that runs the full cloud ingestion/analysis pipeline in sequence.
- `config/cloud.py`: SQLiteCloud connection helper.
- `artifacts/experiments/`: sample experiment outputs.
- `exports/app_store_apps_v2.db`: example snapshot DB (bundled with repo).
- `exports/app_store_apps.db`: legacy DB from original scraper.
- `requirements.txt`: Python dependencies (requests, openai, pandas, matplotlib, plotly, streamlit, etc.).
- `AI_HANDOFF.md`: this handoff document.

## 3. Current workflow
1. **Scrape** (Stage 1b recommended):
   ```bash
   python -m src.local.stage1.app_store_scraper_v2 --collection top-free --all-categories --limit 100 --note "Top free run"
   python -m src.local.stage1.app_store_scraper_v2 --collection top-paid --all-categories --limit 100 --note "Top paid run"
   ```
   Add `--search-term` for keyword scrapes. Each run appends to `app_store_apps_v2.db`.

2. **Score** (Stage 2):
   ```bash
   export OPENAI_API_KEY=sk-...
   python -m src.local.stage2.app_stage2_analysis --run-id <run> [--run-id <run> ...]
   ```
   Omitting `--run-id` processes all snapshots lacking scores. `--force` re-scores everything. Requires `OPENAI_API_KEY`.

3. **Explore** (Stage 3):
   - Local: `streamlit run apps/local/streamlit_app.py`
   - Hosted (SQLiteCloud-backed): `https://discovering-apps-jack.streamlit.app`
   - Neon prototype: `streamlit run apps/cloud/streamlit_app_neon.py` (requires `PROTOTYPE_DATABASE_URL`).
   - The dashboard includes run selectors, configurable scatter axes/bubbles, 3D view, category summaries, distributions, quick-win table, similarity clusters, and an Opportunity Finder (demand dissatisfaction, execution-floor slider, concise results table, per-category highlights).

## 4. Predictive analysis roadmap
The plan is to evolve from descriptive scoring to predictive indicators, focusing on these threads:

1. **Historical snapshots** – Run the v2 scraper/scorer on a schedule (daily/weekly) to build a time series.
2. **Competitive metrics** – From `app_rankings`, compute features such as competitor arrival rate, rank volatility, incumbent resilience, feature diversity.
3. **Feature extraction** – Consider NLP/LLM prompts to derive user stories, content themes, monetization types, etc., for richer feature vectors.
4. **Prediction tasks** – Prototype models to forecast:
   - Derivative attractiveness (which apps to clone).
   - Impact of competitor arrival on incumbents.
   - Survival time of new entrants in specific categories.
5. **Validation** – Manual scoring exercises and historical case studies to ensure features align with intuition.
6. **Integration** – Feed predictive outputs back into the Streamlit app (e.g., risk indicators, competitive overlays).

## 5. Outstanding considerations
- `visualize_scores_interactive.py` still references the legacy DB. Update if interactive HTML previews should use snapshots.
- Ensure `.gitignore` is configured as desired if large databases stay in repo.
- Automate Stage 1/Stage 2 runs (cron, CI) once interval is chosen.
- Select and add a license (README currently reminds to do so).

## 6. Quick tips for future contributors
- Use `pip install -r requirements.txt` inside a virtual environment (`python3 -m venv .venv`).
- Scoring requires `OPENAI_API_KEY` exported in the environment.
- To limit Stage 2 processing: pass `--run-id` or `--max-apps`.
- Streamlit selectboxes have custom keys to avoid duplication errors; maintain that convention when adding controls.
- Reasoning is stored only in Stage 2 results (and shown in quick-win table). Scatter tooltips deliberately omit it.

## 7. Useful commands
```bash
# View available scrape runs
sqlite3 exports/app_store_apps_v2.db "SELECT id, created_at, source, note FROM scrape_runs ORDER BY id DESC";

# Count scored apps per run
sqlite3 exports/app_store_apps_v2.db "SELECT run_id, COUNT(*) FROM app_snapshots WHERE success_score IS NOT NULL GROUP BY run_id ORDER BY run_id DESC";

# Launch Streamlit against production DB
streamlit run apps/local/streamlit_app.py

# Launch Neon prototype dashboard
streamlit run apps/cloud/streamlit_app_neon.py
```

## 8. Prototype Postgres (Neon) workflow
- Schema lives in `exports/schema_postgres.sql`; run it against your Neon database:  
  `psql "$PROTOTYPE_DATABASE_URL" -f exports/schema_postgres.sql`
- Copy existing SQLite data with the helper:  
  `python -m src.prototype.migrate_sqlite_to_postgres`  
  (reads `PROTOTYPE_DATABASE_URL`; override with `--postgres-dsn` as needed).
- A pgloader template is available at `src/prototype/pgloader.load.template` for bulk migrations.
- Example local DSN via Neon extension: `postgres://neon:npg@localhost:5432/<database_name>`.
- Scrape directly into Neon (pulls up to the top 400 apps per category):  
  `python -m src.prototype.app_store_scraper_neon --collection top-free --all-categories --limit 400`
- Dissatisfied pipeline (no Stage 2 needed):
  1. Flag unhappy but high-volume apps per category:  
     `python -m src.prototype.analysis.select_dissatisfied --rating-quantile 0.7 --rating-threshold 3`
  2. Generate embeddings for flagged apps (or every snapshot via `--all-snapshots`):  
     `python -m src.prototype.analysis.generate_embeddings_neon --model text-embedding-3-small`
  3. Cluster them for themes:  
     `python -m src.prototype.analysis.cluster_dissatisfied --scope-label dissatisfied`
- General deltas & clustering:  
  - Rebuild snapshot deltas: `python -m src.prototype.analysis.build_deltas_neon`  
  - Cluster every embedded snapshot: `python -m src.prototype.analysis.cluster_all --scope-label all`
- Neon Streamlit app tabs: clusters (with in-cluster opportunity filters and scores), an all-app dropdown, and a deltas view built on `app_snapshot_deltas`.

---
This document should be refreshed whenever workflows change (new prediction models, additional tables, etc.).
