#!/usr/bin/env python3
"""Cloud variant of embedding generation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import sqlitecloud
from openai import OpenAI

from analysis.generate_embeddings import (  # type: ignore
    DEFAULT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_WAIT,
    SnapshotRecord,
    ensure_embedding_table,
    fetch_snapshots,
    compute_hash,
    build_embedding_payload,
    chunked,
    embed_batch,
    upsert_embeddings,
)
from cloud_config import CONNECTION_URI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings directly in SQLiteCloud.")
    parser.add_argument("--connection-uri", default=CONNECTION_URI)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-wait", type=int, default=DEFAULT_RETRY_WAIT)
    parser.add_argument("--max-apps", type=int)
    parser.add_argument("--run-id", type=int, action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Environment variable OPENAI_API_KEY must be set.")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with sqlitecloud.connect(args.connection_uri) as conn:
        ensure_embedding_table(conn)
        snapshots = fetch_snapshots(
            conn,
            model=args.model,
            run_ids=args.run_id,
            limit=args.max_apps,
        )
        to_process = []
        skipped = 0
        for record in snapshots:
            desc_hash = compute_hash(record.description)
            if not args.force and record.existing_hash == desc_hash:
                skipped += 1
                continue
            to_process.append((record, desc_hash))
        logging.info("Snapshots to embed: %d (skipped %d unchanged)", len(to_process), skipped)
        if not to_process:
            return

        client = OpenAI()
        total_batches = (len(to_process) + args.batch_size - 1) // args.batch_size
        processed = 0
        for batch_index, batch in enumerate(chunked(to_process, args.batch_size), start=1):
            payloads = [build_embedding_payload(record) for record, _ in batch]
            embeddings = embed_batch(
                client,
                args.model,
                payloads,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            upsert_embeddings(conn, batch, embeddings, args.model)
            processed += len(batch)
            logging.info("Processed batch %d/%d (%d/%d rows)", batch_index, total_batches, processed, len(to_process))


if __name__ == "__main__":
    main()

