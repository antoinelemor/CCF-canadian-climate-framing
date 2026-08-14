"""
PROJECT
-------
CCF-Canadian-Climate-Framing

TITLE
-----
9_JSONL_for_recheck.py

MAIN OBJECTIVE
--------------
Produce a multilingual JSONL for re-checking model annotations with a sampling
scheme tailored to robust re-training and per-subclass F1 evaluation.

1. Min / Max constraints per label
    MIN_PER_LABEL = 40 → guarantees enough positives (≥ ~20 for each
     train/validation split) to compute meaningful recall/F1.
    MAX_LABEL_PCT = 0.35 → caps very prolific labels, preventing the sample
     from being monopolised by Messenger or Location sentences.
2. Dynamic sample size
    `NB_SENTENCES_TOTAL` is now a lower bound.  The script automatically
     upsizes the draw if the sum of min quotas would otherwise exceed the
     target.
3. Root-inverse weighting
    Row weights ∝ Σ (1/√fᵢ) instead of 1/fᵢ.  This still favours rare labels
     but tempers the boost for extremely sparse ones, yielding smoother
     distributions.
4. Iterative post-processing
    After the weighted draw we add rows for any label below its min and
     trim rows for labels above their max share, always respecting the
     other labels present in multi-label rows.
5. Fully parameterised
    All thresholds (`MIN_PER_LABEL`, `MAX_LABEL_PCT`) and the weighting
     function exponent (`BETA = 0.5` ⇒ 1/√f) are constants at the top of the
     file for quick tuning.

The combination « root-inverse weights + min/max constraints » corresponds to an
optimum allocation under inequality constraints (Kish 1965; Särndal et al.
1992). It approximates Neyman allocation for multi-label strata while enforcing
practical bounds for variance estimation.

Author:
-------
[Anonymous]
"""

from __future__ import annotations

# ───────────────────────── Imports ────────────────────────── #
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as _PGConnection
from psycopg2 import sql
from tqdm.auto import tqdm

# ──────────────────────── Configuration ──────────────────────── #
BASE_DIR = Path(__file__).resolve().parent

# Existing manual JSONL (to exclude)
MANUAL_ANNOT_DIR = (
    BASE_DIR / ".." / ".." / "Database" / "Training_data" / "manual_annotations_JSONL"
)
MANUAL_ANNOT_EN = MANUAL_ANNOT_DIR / "sentences_to_annotate_EN.jsonl"
MANUAL_ANNOT_FR = MANUAL_ANNOT_DIR / "sentences_to_annotate_FR.jsonl"

# Output template
OUTPUT_JSONL = MANUAL_ANNOT_DIR / "sentences_to_recheck_multiling_{ts}.jsonl"

# ---- Sampling hyper-parameters ---- #
NB_SENTENCES_TOTAL = 2_000         # lower bound; may grow to honour MIN_PER_LABEL
MIN_PER_LABEL      = 40            # ≥ this many positives per label (if available)
MAX_LABEL_PCT      = 0.35          # ≤ 35 % of sample per label
BETA               = 0.5           # weight exponent ⇒ 1/f^β (β=0.5 → root-inverse)
RANDOM_STATE       = 42
random.seed(RANDOM_STATE)

# PostgreSQL credentials
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF"),
    "user":     os.getenv("CCF_DB_USER", "postgres"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

# Non-annotation columns kept as metadata
NON_ANNOT_COLS: Set[str] = {
    "language", "sentences", "id_article", "Unnamed: 0", "doc_id",
    "sentence_id", "words_count_updated", "words_count",
}

# ─────────────────────── Helper Utilities ─────────────────────── #

def open_pg(params: Dict[str, Any]) -> _PGConnection:
    try:
        return psycopg2.connect(**params)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"[FATAL] PostgreSQL connection failed: {exc}") from exc


def fetch_df(conn: _PGConnection) -> pd.DataFrame:
    q = sql.SQL("SELECT * FROM {};").format(sql.Identifier(TABLE_NAME)).as_string(conn)
    return pd.read_sql_query(q, conn)


def load_existing(jsonl_path: Path) -> Set[str]:
    if not jsonl_path.exists():
        return set()
    texts: Set[str] = set()
    for line in jsonl_path.read_text("utf-8").splitlines():
        try:
            payload = json.loads(line)
            texts.add(payload.get("text", ""))
        except json.JSONDecodeError:
            continue
    return texts


def detect_labels(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in NON_ANNOT_COLS and pd.api.types.is_numeric_dtype(df[c]) and df[c].sum() > 0
    ]

# ─────────── Core sampling functions ─────────── #

def row_weights(df_lang: pd.DataFrame, ann_cols: List[str]) -> pd.Series:
    freq = df_lang[ann_cols].sum().clip(lower=1)
    inv_freq_beta = 1.0 / (freq ** BETA)  # 1 / f^β
    w = df_lang[ann_cols].dot(inv_freq_beta)
    w = w.mask(w == 0, other=w[w > 0].min())  # rows with no positives → min weight
    return w


def guarantee_min(sample: pd.DataFrame, pool: pd.DataFrame, ann_cols: List[str]) -> pd.DataFrame:
    """Add rows until every label ≥ MIN_PER_LABEL (when possible)."""
    counts = sample[ann_cols].sum()
    lacking = [lab for lab, n in counts.items() if n < MIN_PER_LABEL and pool[lab].sum() > n]
    for lab in lacking:
        need = MIN_PER_LABEL - int(counts[lab])
        add_rows = pool.loc[pool[lab] == 1].sample(
            n=min(need, len(pool.loc[pool[lab] == 1])),
            random_state=random.randint(0, 9999),
        )
        sample = pd.concat([sample, add_rows])
        pool = pool.drop(index=add_rows.index)
        counts[lab] += len(add_rows)
    return sample.reset_index(drop=True), pool


def enforce_max(sample: pd.DataFrame, ann_cols: List[str]) -> pd.DataFrame:
    """Randomly drop rows until every label ≤ MAX_LABEL_PCT of sample."""
    max_allowed = MAX_LABEL_PCT * len(sample)
    counts = Counter({lab: int(sample[lab].sum()) for lab in ann_cols})
    over = [lab for lab, n in counts.items() if n > max_allowed]
    while over:
        lab = over.pop()
        excess = counts[lab] - int(max_allowed)
        # rows eligible for removal: contain *lab* but not any label currently under-represented
        candidates = sample[sample[lab] == 1]
        protect = [l for l, n in counts.items() if n < MIN_PER_LABEL]
        if protect:
            protect_mask = candidates[protect].any(axis=1)
            candidates = candidates.loc[~protect_mask]
        if not len(candidates):
            continue  # cannot trim without hurting other mins
        drop_rows = candidates.sample(n=min(excess, len(candidates)), random_state=random.randint(0, 9999))
        sample = sample.drop(index=drop_rows.index)
        counts[lab] -= len(drop_rows)
        # re-compute max_allowed because sample size changed
        max_allowed = MAX_LABEL_PCT * len(sample)
        # update over list if needed
        over = [l for l, n in counts.items() if n > max_allowed]
    return sample.reset_index(drop=True)


def sample_language(df_lang: pd.DataFrame, ann_cols: List[str], quota: int) -> pd.DataFrame:
    if quota <= 0 or df_lang.empty:
        return df_lang.head(0).copy()

    weights = row_weights(df_lang, ann_cols)
    initial = df_lang.sample(n=quota, weights=weights, random_state=RANDOM_STATE)
    pool = df_lang.drop(index=initial.index)

    # Ensure minimum per label
    initial, pool = guarantee_min(initial, pool, ann_cols)

    # If sample grew beyond quota, randomly drop surplus (without violating mins)
    while len(initial) > quota:
        # identify removable rows (all labels above min)
        counts = initial[ann_cols].sum()
        safe_labels = [lab for lab, n in counts.items() if n > MIN_PER_LABEL]
        removable = initial[initial[safe_labels].any(axis=1)]
        if removable.empty:
            break  # cannot shrink without breaking mins
        to_drop = removable.sample(n=len(initial) - quota, random_state=random.randint(0, 9999))
        initial = initial.drop(index=to_drop.index)

    # Enforce max percentage
    initial = enforce_max(initial, ann_cols)
    return initial

# ─────────────────────── Serialization ─────────────────────── #

def doccano_entry(row: pd.Series, ann_cols: List[str]) -> Dict[str, Any]:
    labels = [c for c in ann_cols if row[c] == 1]
    meta = {
        c: (None if pd.isna(v) else v)
        for c, v in row.items() if c not in ("sentences", *ann_cols)
    }
    return {"text": row["sentences"], "label": labels, "meta": meta}

# ─────────────────────────── Main ──────────────────────────── #

def main() -> None:  # noqa: C901
    # 1 — Load data
    print("[INFO] Connecting to PostgreSQL…")
    with open_pg(DB_PARAMS) as conn:
        df = fetch_df(conn)
    print(f"[INFO] {len(df):,} rows retrieved.")

    # 2 — Detect annotation columns
    ann_cols = detect_labels(df)
    df[ann_cols] = df[ann_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # 3 — Exclude already-annotated sentences
    print("[INFO] Excluding already-annotated sentences…")
    excl_en, excl_fr = load_existing(MANUAL_ANNOT_EN), load_existing(MANUAL_ANNOT_FR)
    before = len(df)
    df = df[~(
        (df["language"] == "EN") & df["sentences"].isin(excl_en) | (df["language"] == "FR") & df["sentences"].isin(excl_fr)
    )]
    print(f"[INFO] {before - len(df):,} sentences excluded. {len(df):,} remain.")

    # 4 — Compute language quotas (initial)
    quota_half = NB_SENTENCES_TOTAL // 2
    en_quota = min(quota_half, len(df[df["language"] == "EN"]))
    fr_quota = min(quota_half, len(df[df["language"] == "FR"]))
    # redistribute if one language lacks sentences
    if en_quota < quota_half:
        fr_quota = min(fr_quota + (quota_half - en_quota), len(df[df["language"] == "FR"]))
    if fr_quota < quota_half:
        en_quota = min(en_quota + (quota_half - fr_quota), len(df[df["language"] == "EN"]))

    # 5 — Adaptive upscale to satisfy MIN_PER_LABEL
    total_labels = len(ann_cols)
    min_total_needed = MIN_PER_LABEL * total_labels * 0.5  # heuristic: multi-label overlap ≈ 50 %
    target_total = max(NB_SENTENCES_TOTAL, int(min_total_needed))
    scale = target_total / (en_quota + fr_quota)
    en_quota, fr_quota = int(en_quota * scale), int(fr_quota * scale)

    print(f"[INFO] Target sample size adjusted to ≈ {en_quota + fr_quota:,} rows (EN {en_quota} / FR {fr_quota}).")

    # 6 — Sample per language
    print("[INFO] Sampling English subset…")
    df_en = df[df["language"] == "EN"].copy()
    sample_en = sample_language(df_en, ann_cols, en_quota)

    print("[INFO] Sampling French subset…")
    df_fr = df[df["language"] == "FR"].copy()
    sample_fr = sample_language(df_fr, ann_cols, fr_quota)

    df_final = pd.concat([sample_en, sample_fr], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)

    # 7 — Stats
    print("[INFO] Final sample size:", len(df_final))
    print("       ├─ EN:", (df_final["language"] == "EN").sum())
    print("       └─ FR:", (df_final["language"] == "FR").sum())

    print("\n[INFO] Coverage per label:")
    for col in ann_cols:
        n = int(df_final[col].sum())
        print(f"• {col:<30} {n:>4} ({n / len(df_final):6.2%})")

    # 8 — Write JSONL
    ts = pd.Timestamp.now(tz="America/Toronto").strftime("%Y_%m_%d_%H%M")
    out_path = OUTPUT_JSONL.with_name(OUTPUT_JSONL.name.format(ts=ts))
    print("\n[INFO] Writing JSONL →", out_path.name)
    with out_path.open("w", encoding="utf-8") as fo:
        for _, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Serialising"):
            fo.write(json.dumps(doccano_entry(row, ann_cols), ensure_ascii=False, separators=(",", ":")) + "\n")

    print("[INFO] Done ✔")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
