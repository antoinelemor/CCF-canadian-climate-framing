#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
09_create_sentence_embeddings.py

MAIN OBJECTIVE:
---------------
This script ingests the BGE-M3 sentence embeddings produced by the
companion CCF-media-cascade-detection repository into the CCF_Database
as a pgvector halfvec(1024) column. The embeddings cover every sentence
in CCF_processed_data plus the article title (stored under
sentence_id = 0), enabling semantic search, near-duplicate detection,
and cosine-similarity queries directly from SQL through the pgvector
extension.

The upstream artefacts produced by the BGE-M3 encoder consist of:
- embeddings.npy: a float16 memory-mapped array of shape (N, 1024)
  where N is the total number of embedded units;
- index.pkl: a dictionary that maps each (doc_id, sentence_id) pair
  to the corresponding row index in embeddings.npy.

The vectors are already L2-normalised (encoded with
normalize_embeddings=True in the upstream sentence-transformers call),
so cosine similarity reduces to a dot product and can be retrieved
through the pgvector ``<=>'' operator without any further preparation.

Input dependencies:
- ../CCF-media-cascade-detection/data/embeddings/embeddings.npy
- ../CCF-media-cascade-detection/data/embeddings/index.pkl
  (location overridable via the CCF_EMBEDDINGS_DIR environment variable)
- pgvector >= 0.7 installed and CREATE EXTENSION vector in CCF_Database
  (the script aborts with a clear message otherwise).

Side effects:
- Creates (or replaces) the CCF_sentence_embeddings table with one
  row per (doc_id, sentence_id) and an embedding column of type
  halfvec(1024). The table is then equipped with an HNSW index on the
  cosine-distance operator class so approximate nearest-neighbour
  queries return in milliseconds.

Dependencies:
-------------
- psycopg2
- numpy
- pathlib
- pickle

MAIN FEATURES:
--------------
1) Streaming COPY -- The 9.2 million vectors are inserted via the
   PostgreSQL COPY protocol with a pgvector-compatible textual
   representation. Streaming keeps Python memory bounded to one chunk
   at a time, regardless of corpus size.
2) Faithful storage -- The halfvec(1024) column preserves the float16
   precision of the source file at exactly the same memory cost
   (2 bytes per dimension), so storage is identical to the .npy file.
3) HNSW index for cosine -- After ingestion the script builds an HNSW
   index on the cosine-distance operator class (halfvec_cosine_ops);
   this delivers sub-second top-k retrieval over the full corpus.
4) Resumable -- If the table already exists and contains the expected
   number of rows, the script restarts at the index step rather than
   re-ingesting from scratch.

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################

import io
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import psycopg2

##############################################################################
#                                  CONFIG                                    #
##############################################################################

DB_PARAMS = {
    "host": os.getenv("CCF_DB_HOST", "localhost"),
    "port": int(os.getenv("CCF_DB_PORT", "5432")),
    "dbname": os.getenv("CCF_DB_NAME", "CCF_Database"),
    # Default to the current OS user (e.g. "antoine" on the development
    # workstation); fall back to "postgres" only if neither CCF_DB_USER
    # nor USER is set. This mirrors how psql resolves the role.
    "user": os.getenv("CCF_DB_USER", os.getenv("USER", "postgres")),
    "password": os.getenv("CCF_DB_PASS", ""),
}

# Embeddings live in a sibling repository on the same machine.
EMBED_DIR = Path(
    os.getenv(
        "CCF_EMBEDDINGS_DIR",
        "/Users/antoine/Documents/GitHub/CCF-media-cascade-detection/data/embeddings",
    )
)
EMBEDDINGS_NPY = EMBED_DIR / "embeddings.npy"
INDEX_PKL = EMBED_DIR / "index.pkl"

TABLE_NAME = "CCF_sentence_embeddings"
EMBED_DIM = 1024
COPY_BATCH_SIZE = 50_000  # rows streamed per COPY batch.


##############################################################################
#                              HELPERS                                       #
##############################################################################


def _check_extension(cur) -> None:
    cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector';")
    row = cur.fetchone()
    if row is None:
        print(
            "ERROR: the pgvector extension is not installed in this database.\n"
            "       Run CREATE EXTENSION vector; first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  pgvector version: {row[0]}")


def _create_table(cur) -> None:
    cur.execute(
        f'CREATE TABLE IF NOT EXISTS "{TABLE_NAME}" ('
        '"doc_id"      BIGINT  NOT NULL, '
        '"sentence_id" INTEGER NOT NULL, '
        f'"embedding"  halfvec({EMBED_DIM}) NOT NULL, '
        'PRIMARY KEY (doc_id, sentence_id)'
        ');'
    )


def _format_vector(vec: np.ndarray) -> str:
    """Return the pgvector text representation of a single 1024-d vector.

    pgvector accepts the format ``[v1,v2,...,v1024]`` for halfvec input.
    We use 5 significant digits which is sufficient for float16 fidelity
    (the smallest representable difference is ~1e-3 near values around 1).
    """
    return "[" + ",".join(f"{x:.5f}" for x in vec) + "]"


def _iter_rows(index_map: dict, mmap: np.memmap) -> Iterator[tuple[int, int, str]]:
    # Sort by row index so we read the .npy sequentially -- the OS page cache
    # then serves contiguous chunks rather than random-access reading 18 GB.
    items = sorted(index_map.items(), key=lambda kv: kv[1])
    for (doc_id, sentence_id), row_idx in items:
        vec = mmap[row_idx].astype(np.float32, copy=False)
        yield int(doc_id), int(sentence_id), _format_vector(vec)


##############################################################################
#                              MAIN PROCESS                                  #
##############################################################################


def main() -> None:
    print("=" * 72)
    print("CCF_Database -- sentence embeddings (pgvector halfvec(1024))")
    print("=" * 72)

    for p in (EMBEDDINGS_NPY, INDEX_PKL):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            sys.exit(1)

    print(f"\n[1/4] Connecting and verifying pgvector")
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    cur = conn.cursor()
    _check_extension(cur)

    print(f"[2/4] Loading index.pkl and memory-mapping embeddings.npy")
    with INDEX_PKL.open("rb") as f:
        index_map = pickle.load(f)
    print(f"  index entries : {len(index_map):,}")
    mmap = np.load(EMBEDDINGS_NPY, mmap_mode="r")
    print(f"  array shape   : {mmap.shape}  dtype={mmap.dtype}")
    assert mmap.shape[1] == EMBED_DIM, (
        f"Expected embedding dimension {EMBED_DIM}, got {mmap.shape[1]}"
    )
    assert mmap.shape[0] >= len(index_map), (
        "embeddings.npy has fewer rows than index entries"
    )

    print(f"[3/4] Building {TABLE_NAME} via COPY (this can take ~30 min)")
    _create_table(cur)

    # Detect resume: if the row count matches we skip ingestion.
    cur.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}";')
    existing = cur.fetchone()[0]
    if existing == len(index_map):
        print(
            f"  table already populated ({existing:,} rows); skipping COPY."
        )
    else:
        if existing != 0:
            print(
                f"  partial table detected ({existing:,} rows); truncating."
            )
            cur.execute(f'TRUNCATE "{TABLE_NAME}";')
            conn.commit()

        # Pre-sort once so the .npy is read sequentially across batches.
        print(f"  pre-sorting {len(index_map):,} index entries by row index ...")
        items = sorted(index_map.items(), key=lambda kv: kv[1])
        print(f"  starting COPY ...")
        t0 = time.perf_counter()
        rows_done = 0
        for batch_start in range(0, len(items), COPY_BATCH_SIZE):
            slice_items = items[batch_start: batch_start + COPY_BATCH_SIZE]
            buf = io.StringIO()
            for (doc_id, sentence_id), row_idx in slice_items:
                vec = mmap[row_idx].astype(np.float32, copy=False)
                buf.write(
                    f"{doc_id}\t{sentence_id}\t"
                    f"[{','.join(f'{x:.5f}' for x in vec)}]\n"
                )
            buf.seek(0)
            cur.copy_expert(
                f'COPY "{TABLE_NAME}" (doc_id, sentence_id, embedding) '
                f'FROM STDIN WITH (FORMAT text)',
                buf,
            )
            conn.commit()
            rows_done += len(slice_items)
            elapsed = time.perf_counter() - t0
            rate = rows_done / elapsed if elapsed else 0
            eta_s = (len(items) - rows_done) / rate if rate > 0 else 0
            # Print every 10 batches to avoid log spam.
            if (batch_start // COPY_BATCH_SIZE) % 10 == 0 or rows_done == len(items):
                print(
                    f"  {rows_done:>10,} / {len(items):,} rows  "
                    f"({100*rows_done/len(items):5.1f}%)  "
                    f"elapsed {elapsed:>5.0f}s  ETA {eta_s/60:>5.1f} min",
                    flush=True,
                )

        print(f"  COPY done in {(time.perf_counter()-t0)/60:.1f} min")

    print(f"[4/4] Creating HNSW index on cosine distance + ANALYZE")
    cur.execute(
        f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME.lower()}_hnsw '
        f'ON "{TABLE_NAME}" USING hnsw (embedding halfvec_cosine_ops) '
        'WITH (m = 16, ef_construction = 64);'
    )
    cur.execute(f'ANALYZE "{TABLE_NAME}";')
    conn.commit()

    cur.execute(
        f"SELECT pg_size_pretty(pg_relation_size('public.\"{TABLE_NAME}\"')), "
        f"pg_size_pretty(pg_total_relation_size('public.\"{TABLE_NAME}\"'));"
    )
    tbl_size, total_size = cur.fetchone()
    cur.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}";')
    n = cur.fetchone()[0]
    print(f"\n  table size (data only)   : {tbl_size}")
    print(f"  table size (incl indexes): {total_size}")
    print(f"  total rows               : {n:,}")

    cur.close()
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
