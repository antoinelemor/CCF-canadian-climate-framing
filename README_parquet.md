# CCF Database — Apache Parquet bundle

This bundle ships the six relational tables of the **Canadian Climate Framing (CCF) Database** as standalone Apache Parquet files (ZSTD-compressed). It is the no-database-server alternative to the canonical PostgreSQL edition. The two editions are sister Zenodo deposits:

- **PostgreSQL edition** (canonical): [DOI `10.5281/zenodo.20346364`](https://doi.org/10.5281/zenodo.20346364) — ships `CCF_Database.tar` (37 GB, `pg_dump -Fd` archive with HNSW indexing).
- **Apache Parquet mirror** (this bundle): [DOI `10.5281/zenodo.20346373`](https://doi.org/10.5281/zenodo.20346373) — ships the six `.parquet` files described below.

Both formats share **identical schemas and identical row counts**; the only adaptation in Parquet is that the `halfvec(1024)` embedding column is materialised as a 1024-element `LIST<FLOAT>` and the JSONB entity arrays are serialised as UTF-8 JSON strings.

## Files

| File | Rows | Size |
|---|---|---|
| `CCF_full_data.parquet` | 266,271 | 8.2 MB |
| `CCF_processed_data.parquet` | 9,198,958 | 251 MB |
| `CCF_article_aggregates.parquet` | 266,271 | 12.4 MB |
| `CCF_article_entities.parquet` | 266,271 | 23 MB |
| `CCF_reliability_tiers.parquet` | 65 | 7.9 KB |
| `CCF_sentence_embeddings.parquet` | 9,462,845 | 15.5 GB |

**Total ≈ 15.8 GB**. The companion file `CCF_Database.parquet.sha256` lists the SHA-256 hashes of all six files.

## Loading the data

### pandas

```python
import pandas as pd
full = pd.read_parquet("CCF_full_data.parquet")
agg  = pd.read_parquet("CCF_article_aggregates.parquet")
emb  = pd.read_parquet("CCF_sentence_embeddings.parquet")
```

### polars (lazy, recommended for the embedding table)

```python
import polars as pl
emb = pl.scan_parquet("CCF_sentence_embeddings.parquet")
sample = emb.filter(pl.col("doc_id") == 12345).collect()
```

### DuckDB (in-process SQL)

```python
import duckdb
con = duckdb.connect()
con.execute("CREATE VIEW emb AS SELECT * FROM read_parquet('CCF_sentence_embeddings.parquet')")
con.execute("CREATE VIEW agg AS SELECT * FROM read_parquet('CCF_article_aggregates.parquet')")
con.execute("CREATE VIEW full AS SELECT * FROM read_parquet('CCF_full_data.parquet')")
con.execute("""
  SELECT f.media, f.date, f.title, a.top_frame_prop
  FROM   agg a JOIN full f USING (doc_id)
  WHERE  a.top_frame = 'political_frame'
    AND  EXTRACT(year FROM f.date) = 2024
""").df()
```

### R / arrow

```r
library(arrow)
full <- read_parquet("CCF_full_data.parquet")
agg  <- read_parquet("CCF_article_aggregates.parquet")
```

## Joining the tables

All five non-tier tables share a `doc_id` (BIGINT, 1..266,578) for article-level joins. The `CCF_sentence_embeddings` table also exposes `sentence_id`. The `CCF_reliability_tiers` table joins on `code` (text), matching the column name of each annotation in `CCF_processed_data` and the suffix of each `prop_X` column in `CCF_article_aggregates`.

```python
import duckdb
con = duckdb.connect()
con.execute("""
  WITH agg AS (SELECT * FROM read_parquet('CCF_article_aggregates.parquet')),
       full AS (SELECT * FROM read_parquet('CCF_full_data.parquet')),
       tier AS (SELECT * FROM read_parquet('CCF_reliability_tiers.parquet'))
  SELECT f.title, a.top_frame, t.tier_overall
  FROM   agg a JOIN full f USING (doc_id) JOIN tier t ON t.code = a.top_frame
  WHERE  t.tier_overall = 'A' LIMIT 10
""").df()
```

## Embedding column

The embedding column in `CCF_sentence_embeddings.parquet` is a `LIST<FLOAT>` of length 1024 (the BAAI/bge-m3 multilingual encoder, L2-normalised at encoding time). Cosine similarity reduces to a dot product:

```python
import numpy as np
import pandas as pd
emb = pd.read_parquet("CCF_sentence_embeddings.parquet")
query = np.array(emb["embedding"].iloc[0], dtype=np.float32)
# Brute-force top-10 nearest neighbours
mat = np.vstack(emb["embedding"].values).astype(np.float32)
sims = mat @ query                         # cosine (vectors are L2-normalised)
top10 = np.argsort(-sims)[:10]
emb.iloc[top10][["doc_id", "sentence_id"]]
```

Brute-force scans over 9.46 million vectors take roughly 30 seconds on a modern laptop. For sub-second retrieval, use the PostgreSQL archive (`CCF_Database.tar`) which ships an HNSW index on the `pgvector` `<=>` operator.

## Reproducibility

The Parquet files are produced from a freshly restored PostgreSQL database by `Scripts/Database_creation/enrichment/05_export_to_parquet.py` (bundled in `ccf_code_and_paper.tar.gz` and also archived on OSF, [DOI `10.17605/OSF.IO/Q5W47`](https://doi.org/10.17605/OSF.IO/Q5W47)), which uses DuckDB + the PostgreSQL scanner with a server-side cast of `halfvec(1024)` → `real[]`. Re-running the script on the same database snapshot produces byte-identical Parquet files (modulo the modification time embedded in the Parquet footer).

## Licence

CC-BY 4.0 (see `LICENSE` in this deposit). Raw newspaper text is excluded for copyright reasons; bibliographic coordinates are sufficient to recover the original text via Factiva, Eureka.cc, or ProQuest Canadian Newsstream.
