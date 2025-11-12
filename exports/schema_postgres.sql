-- PostgreSQL schema for the Apple App Store snapshot pipeline.
-- Generated to mirror the existing SQLite structures in exports/app_store_apps_v2.db.

CREATE TABLE IF NOT EXISTS scrape_runs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    country TEXT,
    collection TEXT,
    search_term TEXT,
    limit_requested INTEGER,
    all_categories BOOLEAN,
    note TEXT
);

CREATE TABLE IF NOT EXISTS app_snapshots (
    run_id INTEGER NOT NULL REFERENCES scrape_runs(id),
    track_id BIGINT NOT NULL,
    name TEXT,
    description TEXT,
    release_date TIMESTAMPTZ,
    current_version_release_date TIMESTAMPTZ,
    version TEXT,
    primary_genre_id INTEGER,
    primary_genre_name TEXT,
    genre_ids TEXT,
    genres TEXT,
    content_advisory_rating TEXT,
    price DOUBLE PRECISION,
    formatted_price TEXT,
    currency TEXT,
    is_free BOOLEAN,
    has_in_app_purchases BOOLEAN,
    seller_name TEXT,
    seller_url TEXT,
    developer_id TEXT,
    bundle_id TEXT,
    average_user_rating DOUBLE PRECISION,
    average_user_rating_current DOUBLE PRECISION,
    user_rating_count INTEGER,
    user_rating_count_current INTEGER,
    rating_count_list TEXT,
    language_codes TEXT,
    minimum_os_version TEXT,
    file_size_bytes BIGINT,
    screenshot_urls TEXT,
    ipad_screenshot_urls TEXT,
    appletv_screenshot_urls TEXT,
    app_store_url TEXT,
    artwork_url TEXT,
    chart_memberships TEXT,
    scraped_at TIMESTAMPTZ,
    build_time_estimate DOUBLE PRECISION,
    success_score DOUBLE PRECISION,
    success_reasoning TEXT,
    PRIMARY KEY (run_id, track_id)
);

CREATE TABLE IF NOT EXISTS app_rankings (
    run_id INTEGER NOT NULL REFERENCES scrape_runs(id),
    track_id BIGINT NOT NULL,
    chart_type TEXT NOT NULL,
    category_id TEXT,
    category_name TEXT,
    rank INTEGER,
    PRIMARY KEY (run_id, track_id, chart_type, category_id)
);

CREATE TABLE IF NOT EXISTS app_snapshot_embeddings (
    run_id INTEGER NOT NULL,
    track_id BIGINT NOT NULL,
    model TEXT NOT NULL,
    description_sha256 TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    vector_length INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, track_id, model)
);

CREATE TABLE IF NOT EXISTS app_snapshot_neighbors (
    run_id INTEGER NOT NULL,
    track_id BIGINT NOT NULL,
    neighbor_run_id INTEGER NOT NULL,
    neighbor_track_id BIGINT NOT NULL,
    model TEXT NOT NULL,
    similarity DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, track_id, neighbor_run_id, neighbor_track_id, model)
);

CREATE TABLE IF NOT EXISTS app_snapshot_clusters (
    id SERIAL PRIMARY KEY,
    scope TEXT NOT NULL,
    model TEXT NOT NULL,
    label TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    size INTEGER NOT NULL,
    avg_success DOUBLE PRECISION,
    avg_build DOUBLE PRECISION,
    avg_demand DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS app_snapshot_cluster_members (
    cluster_id INTEGER NOT NULL REFERENCES app_snapshot_clusters(id),
    run_id INTEGER NOT NULL,
    track_id BIGINT NOT NULL,
    distance DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cluster_id, run_id, track_id)
);

CREATE TABLE IF NOT EXISTS app_snapshot_deltas (
    run_id INTEGER NOT NULL,
    run_created_at TIMESTAMPTZ,
    track_id BIGINT NOT NULL,
    name TEXT,
    category TEXT,
    price DOUBLE PRECISION,
    currency TEXT,
    average_user_rating DOUBLE PRECISION,
    user_rating_count INTEGER,
    build_time_estimate DOUBLE PRECISION,
    success_score DOUBLE PRECISION,
    success_reasoning TEXT,
    best_rank DOUBLE PRECISION,
    description TEXT,
    version TEXT,
    developer TEXT,
    prev_run_id INTEGER,
    prev_run_created_at TIMESTAMPTZ,
    prev_success_score DOUBLE PRECISION,
    prev_build_time DOUBLE PRECISION,
    prev_rating DOUBLE PRECISION,
    prev_rating_count DOUBLE PRECISION,
    prev_price DOUBLE PRECISION,
    prev_currency TEXT,
    prev_rank DOUBLE PRECISION,
    is_new_track BOOLEAN,
    delta_success DOUBLE PRECISION,
    delta_build_time DOUBLE PRECISION,
    delta_rating DOUBLE PRECISION,
    delta_rating_count DOUBLE PRECISION,
    delta_price DOUBLE PRECISION,
    delta_rank DOUBLE PRECISION,
    price_changed BOOLEAN,
    days_since_prev DOUBLE PRECISION,
    PRIMARY KEY (run_id, track_id)
);

CREATE INDEX IF NOT EXISTS idx_app_deltas_track ON app_snapshot_deltas(track_id);
CREATE INDEX IF NOT EXISTS idx_app_deltas_run ON app_snapshot_deltas(run_id);

CREATE TABLE IF NOT EXISTS app_snapshot_dissatisfied (
    run_id INTEGER NOT NULL,
    track_id BIGINT NOT NULL,
    category TEXT,
    price DOUBLE PRECISION,
    average_user_rating DOUBLE PRECISION,
    user_rating_count INTEGER,
    rating_percentile DOUBLE PRECISION,
    threshold_rating DOUBLE PRECISION,
    flagged_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (run_id, track_id)
);

CREATE INDEX IF NOT EXISTS idx_dissatisfied_run ON app_snapshot_dissatisfied(run_id);
