# Reproducing the CCF manuscript and tables

This document is the complete reproducibility recipe. By the end of it you will have:

1. A working copy of the deposited PostgreSQL database (restored from the Zenodo dump, DOI [`10.5281/zenodo.20667151`](https://doi.org/10.5281/zenodo.20667151)).
2. The full set of canonical CSVs under `Database/Training_data/`.
3. The LaTeX tables under `paper/CCF_Methodology/Results/Outputs/Tables/` and the manuscript figures under `paper/CCF_Methodology/Results/Outputs/Figures/`.
4. A re-compiled `paper/CCF_Methodology/Latex/CCF_Methodology.pdf` and `CCF_Methodology_SI.pdf` that match the deposited PDFs at the content level (modulo embedded PDF compile timestamps).

The chain runs on a single workstation. Reference machine: Apple Mac Studio (M2 Ultra, 128 GB unified memory). All commands are executed **from the deposit root**.

## 0. Prerequisites

- Python 3.11+
- PostgreSQL 16+ with the `pgvector` extension (≥ 0.8.2)
- TeX Live 2024 with `biber`
- Python packages:

  ```bash
  pip install pandas psycopg2-binary numpy matplotlib seaborn krippendorff scikit-learn spacy transformers torch tqdm
  ```

The annotation pipeline (Step 7 below) also needs the language models pulled by `spacy` and `transformers` on first use. These are *not* required to reproduce the tables and the manuscript; only the reporting pipeline (Steps 2–4) and the figures (Step 5) are needed for that.

## 1. Restore the deposited PostgreSQL dump (or read the Parquet bundle)

The CCF Database is distributed on Zenodo in two complementary editions sharing identical schemas: the canonical PostgreSQL edition ([DOI `10.5281/zenodo.20667151`](https://doi.org/10.5281/zenodo.20667151)) and the Apache Parquet mirror ([DOI `10.5281/zenodo.20667154`](https://doi.org/10.5281/zenodo.20667154)).

### Option A — PostgreSQL dump (full schema with HNSW indexing)

The dump is published as a single tarball (`CCF_Database.tar`, ≈ 37 GB) that wraps a `pg_dump` *directory* archive (`-Fd`). Restoring it is a three-step process: untar, create the empty database with the `pgvector` extension, then `pg_restore` from the directory using parallel workers.

```bash
# 1. Download CCF_Database.tar from the Zenodo deposit
#    (DOI 10.5281/zenodo.20667151).
tar -xf CCF_Database.tar           # extracts the CCF_Database_dump/ directory

# 2. Create the target database and load pgvector.
createdb CCF_Database
psql -d CCF_Database -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Restore in parallel (8 workers; adjust -j to your core count).
pg_restore -d CCF_Database --no-owner --no-privileges -j 8 CCF_Database_dump
```

Then sanity-check:

```sql
\dt
-- Expect 6 tables: CCF_full_data, CCF_processed_data, CCF_article_aggregates,
--                  CCF_article_entities, CCF_reliability_tiers, CCF_sentence_embeddings

SELECT COUNT(*) FROM "CCF_full_data";              -- 266,271
SELECT COUNT(*) FROM "CCF_processed_data";         -- 9,198,958
SELECT COUNT(*) FROM "CCF_sentence_embeddings";    -- 9,462,845
SELECT tier_overall, COUNT(*) FROM "CCF_reliability_tiers" GROUP BY tier_overall;
-- A | 27   B | 21   C | 17
```

The five core tables share the same `doc_id` keyspace (1 .. 266,578). Any `JOIN ... USING (doc_id)` returns exactly 266,271 articles.

### Option B — Apache Parquet bundle (no database server required)

```bash
# Download the six *.parquet files from the Zenodo Parquet mirror
# (DOI 10.5281/zenodo.20667154).
```

```python
import pandas as pd
articles  = pd.read_parquet("parquet/CCF_full_data.parquet")           # 266,271 rows
sentences = pd.read_parquet("parquet/CCF_processed_data.parquet")      # 9,198,958 rows
embeds    = pd.read_parquet("parquet/CCF_sentence_embeddings.parquet") # 9,462,845 rows
```

The Parquet bundle is regenerated from a freshly restored PostgreSQL database by
`Scripts/Database_creation/enrichment/05_export_to_parquet.py`; the script
streams each table via DuckDB's `postgres_query()` and writes ZSTD-compressed
Parquet files (one per table). The `halfvec(1024)` embedding column is
materialised as a 1024-element `LIST<FLOAT>` and the JSONB entity arrays are
serialised as UTF-8 JSON strings; the two formats share otherwise identical
schemas.

## 2. Reporting pipeline — regenerate the canonical CSVs

```bash
python Scripts/Annotation/15_normalize_categories.py
python Scripts/Annotation/16_build_normalized_csvs.py
```

These two scripts read everything they need from `Database/Training_data/` (the training-time CSVs, the inter-coder JSONL aggregates, and the per-model `Training_logs/`) and write back into the same directory. They are deterministic and idempotent: running them twice produces byte-for-byte identical output.

Files produced or refreshed:

| Script | Output(s) |
|---|---|
| `15_normalize_categories.py` | `all_best_models_normalized.csv`, `final_annotation_metrics_normalized.csv`, `manual_annotations_metrics_normalized.csv`, `training_database_metrics_normalized.csv` |
| `16_build_normalized_csvs.py` | `per_category_reliability_normalized.csv`, `reliability_tiers.csv`, `training_hyperparameters_normalized.csv`, `training_static_configuration.csv` |

## 3. Reporting pipeline — regenerate the LaTeX tables

```bash
python Scripts/Annotation/17_generate_tables.py
```

This script reads the canonical CSVs from Step 2 **and** the live PostgreSQL database (for Supplementary Table S8, which is computed directly from `CCF_processed_data`). It writes thirteen LaTeX tables under `paper/CCF_Methodology/Results/Outputs/Tables/`:

| Section | Tables |
|---|---|
| Main manuscript | `table_performance_training.tex`, `table_validation_overall.tex`, `table_intercoder_blind.tex` |
| Supplementary | `table_s4_training_metrics.tex`, `table_s5_training_distribution.tex`, `table_s7_test_metrics.tex`, `table_s8_database_distribution.tex`, `table_s9_per_category_reliability.tex`, `table_s10_reliability_tiers.tex`, `table_s11_training_hyperparameters.tex`, `table_s12_data_dictionary.tex` |

The manuscript sources `\input{}` these files directly via relative path `../Results/Outputs/Tables/`.

## 4. Data Overview — regenerate Table 11, Table 12 and Figure 7

```bash
python paper/CCF_Methodology/Results/Scripts/3_data_overview.py
```

This script connects to PostgreSQL and produces:

- `paper/CCF_Methodology/Results/Outputs/Figures/data_overview_frames_lines.png` (yearly per-article frame shares, 1980–2024).
- `paper/CCF_Methodology/Results/Outputs/Tables/table_data_overview_descriptives.tex` (article-level descriptive statistics).
- `paper/CCF_Methodology/Results/Outputs/Tables/table_data_overview_entities.tex` (top-10 named entities per type).

## 5. Manuscript figures

The article-level overview figures and the temporal F1 plot are regenerated by:

```bash
python paper/CCF_Methodology/Results/Scripts/1_overview_plots.py          # articles_by_media / _year / _province
python paper/CCF_Methodology/Results/Scripts/2_temporal_f1_validation.py  # temporal_f1_evolution.png
```

The inter-coder reliability progression figure comes from the annotation pipeline:

```bash
python Scripts/Annotation/14_create_intercoder_progression_plot.py
```

All figures are written under `paper/CCF_Methodology/Results/Outputs/Figures/`. The manuscript `\includegraphics{}` paths point at `Figures/` inside the LaTeX source tree, so figures used by the manuscript are also expected at `paper/CCF_Methodology/Latex/Figures/` (the deposit ships a pre-built copy there for convenience).

## 6. Compile the manuscript and the Supplementary Information

```bash
cd paper/CCF_Methodology/Latex
pdflatex CCF_Methodology.tex
biber    CCF_Methodology
pdflatex CCF_Methodology.tex
pdflatex CCF_Methodology.tex

pdflatex CCF_Methodology_SI.tex
biber    CCF_Methodology_SI
pdflatex CCF_Methodology_SI.tex
pdflatex CCF_Methodology_SI.tex
```

Expected outputs:

- `CCF_Methodology.pdf` (≈ 33 pages, ≈ 2.3 MB)
- `CCF_Methodology_SI.pdf` (≈ 55 pages, ≈ 0.7 MB)

Both are byte-comparable to the deposited PDFs at the level of content; embedded PDF metadata (compile timestamps) is the only legitimate source of divergence.

## 7. (Optional) Full pipeline from scratch

Steps 1–6 cover what is needed to reproduce the *manuscript* from the deposited database. To reproduce the deposited database *itself* from raw articles, the full pipeline is:

```bash
python Scripts/Annotation/01_Preprocess.py                       # two-sentence segmentation
python Scripts/Annotation/02_JSONL.py                            # JSONL for manual annotation
python Scripts/Annotation/03_Manual_annotations.py               # aggregate manual annotations
python Scripts/Annotation/04_JSONL_for_training.py               # train / validation split
python Scripts/Annotation/05_populate_SQL_database.py            # initial PostgreSQL DB
python Scripts/Annotation/06_Training_best_models.py             # 128 BERT / CamemBERT classifiers
python Scripts/Annotation/07_Annotation.py                       # apply models to entire corpus
python Scripts/Annotation/08_NER.py                              # PER / ORG / LOC on every sentence
python Scripts/Annotation/09_create_sentence_embeddings.py       # BAAI/bge-m3 halfvec(1024) + HNSW
python Scripts/Annotation/10_JSONL_for_recheck.py                # stratified validation sample
python Scripts/Annotation/11_Annotation_metrics.py               # precision / recall / F1 vs gold
python Scripts/Annotation/12_Blind_verification.py               # strip labels for blind pass
python Scripts/Annotation/13_Intercoder_reliability.py           # κ / α / AC1 per category
python Scripts/Annotation/14_create_intercoder_progression_plot.py
```

Step 5 requires the raw newspaper text, which is *not* deposited (copyright). Researchers with institutional access to Factiva, Eureka.cc, or ProQuest Canadian Major Dailies can rebuild that input from the bibliographic coordinates in `CCF_full_data`.

## 8. Determinism notes

- The canonical CSVs and the LaTeX tables are produced with explicit sort orders and three-decimal formatting; running the pipeline twice on the same input produces byte-for-byte identical files.
- Supplementary Table S8 is computed live against `CCF_processed_data`; it is deterministic given a frozen database snapshot.
- BAAI/bge-m3 embeddings (Step 9) are deterministic given the source text and the upstream model checkpoint; we ship the resulting vectors in `CCF_sentence_embeddings` to remove any drift from later releases of the encoder.

## 9. Troubleshooting

- **`CREATE EXTENSION vector` fails** — install `pgvector` (https://github.com/pgvector/pgvector). Version ≥ 0.8.2 is required for `halfvec`.
- **`pg_restore` complains about ownership** — pass `--no-owner --no-privileges` as in the snippet above.
- **`pdflatex` reports missing fonts** — your TeX Live distribution is incomplete; install the `texlive-fonts-recommended` and `texlive-fonts-extra` packages (Debian/Ubuntu) or use the upstream TeX Live installer.
- **HNSW index missing after restore** — the dump is intended to ship the HNSW index pre-built. If your restore is missing it, rebuild with:

  ```sql
  CREATE INDEX IF NOT EXISTS idx_ccf_sentence_embeddings_hnsw
    ON "CCF_sentence_embeddings" USING hnsw (embedding halfvec_cosine_ops);
  ```

  Construction takes ≈ 7 h on the reference machine. Querying without the index falls back to a sequential scan and is correct but slow.
