"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
9_JSONL_for_recheck.py

MAIN OBJECTIVE:
---------------
This script cross-checks model predictions stored in a PostgreSQL table
(CCF_processed_data) against a manually-annotated JSONL “gold” file,
computes exhaustive binary-classification metrics per label and per
language (EN, FR, ALL), and writes the results to a clean, analysis-ready
CSV file.

DEPENDENCIES:
-------------
- os, json, csv, random, collections.Counter / defaultdict
- typing (Dict, List, Tuple, Any)
- pandas ≥ 2.0
- psycopg2-binary ≥ 2.9

MAIN FEATURES:
--------------
1) Robust PostgreSQL connection
   - Credentials can be overridden via environment variables  
   - Graceful termination with clear fatal error messages

2) Automatic label discovery  
   - Detects every numeric prediction column that is not a metadata
     field, so the script adapts to schema changes without manual edits

3) Gold-standard loader  
   - Reads a JSONL file of manually verified sentences  
   - Explodes the list of labels into a wide binary matrix  
   - Keeps only rows with both doc_id and sentence_id keys

4) Data alignment  
   - Inner join on (doc_id, sentence_id) to ensure one-to-one
     comparison between predictions and gold annotations

5) Per-label confusion matrices  
   - Computes TP, FP, FN, TN ignoring rows with NaN predictions  
   - Derives precision, recall, F1 for class 1 and class 0  
     (named `precision_1`, `recall_0`, `F1_1`, `F1_0`, etc.)

6) Language-specific breakdowns  
   - Metrics are produced for ALL rows, English only, and French only

7) Aggregate scores  
   - Micro, macro, and weighted averages across labels for each language

8) Deterministic, well-formatted output  
   - Fixed column order for easy downstream use  
   - Floats rounded to four decimals; other fields left untouched  
   - Saved to Database/Training_data/final_annotation_metrics.csv

9) Reproducibility & safety  
   - Global `RANDOM_STATE` for any stochastic process  
   - Comprehensive sanity checks with fatal exits on empty merges,
     missing columns, or empty gold data sets

AUTHOR :
--------
Antoine Lemor
"""

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as _PGConnection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF"),
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", "Absitreverentiavero19!"),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

GOLD_JSONL = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "Database",
    "Training_data",
    "manual_annotations_JSONL",
    "sentences_to_recheck_multiling_done.jsonl",
)
OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "Database",
    "Training_data",
    "final_annotation_metrics.csv",
)

# ######## Constants ######## #
RANDOM_STATE = 42
random.seed(RANDOM_STATE)

# Columns that are not annotation labels
NON_LABEL_COLS = {
    "language",
    "sentences",
    "id_article",
    "Unnamed: 0",
    "doc_id",
    "sentence_id",
    "words_count_updated",
    "words_count",
}

##############################################################################
#                          HELPER UTILITIES                                  #
##############################################################################


def open_pg(params: Dict[str, Any]) -> _PGConnection:
    """Open a PostgreSQL connection or exit gracefully."""
    try:
        return psycopg2.connect(**params)
    except Exception as exc:
        raise SystemExit(f"[FATAL] PostgreSQL connection failed: {exc}") from exc


def fetch_predictions(conn: _PGConnection) -> pd.DataFrame:
    """Fetch the full prediction table into a pandas DataFrame."""
    query = sql.SQL("SELECT * FROM {};").format(sql.Identifier(TABLE_NAME)).as_string(
        conn
    )
    return pd.read_sql_query(query, conn)


def detect_label_columns(df: pd.DataFrame) -> List[str]:
    """Return every numeric column that is *not* listed in NON_LABEL_COLS."""
    return [
        col
        for col in df.columns
        if col not in NON_LABEL_COLS
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_gold_jsonl(path: str) -> pd.DataFrame:
    """
    Load the manually verified JSONL into a tidy DataFrame with one binary
    column per label.

    Returns
    -------
    DataFrame with columns:
        doc_id | sentence_id | language | <label_1> ... <label_n>
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as err:  # pragma: no cover
                print(f"[WARN] Skipped malformed line: {err}")
                continue

            meta = entry.get("meta", {}) or {}
            doc_id = meta.get("doc_id")
            sent_id = meta.get("sentence_id")
            language = meta.get("language")
            if doc_id is None or sent_id is None:
                continue  # cannot match without keys
            labels = set(entry.get("label", []))
            records.append(
                {
                    "doc_id": doc_id,
                    "sentence_id": sent_id,
                    "language": language,
                    "labels": labels,
                }
            )

    gold_df = pd.DataFrame.from_records(records)
    if gold_df.empty:
        raise SystemExit("[FATAL] Gold JSONL contains zero usable rows.")

    # Explode label lists → wide binary matrix
    all_labels = sorted({lab for labels in gold_df["labels"] for lab in labels})
    for lab in all_labels:
        gold_df[lab] = gold_df["labels"].apply(lambda s: int(lab in s))

    return gold_df.drop(columns="labels")

##############################################################################
#                           METRICS COMPUTATION                              #  
##############################################################################

def compute_confusion_per_label(
    df: pd.DataFrame, label: str
) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN for `label` given a DataFrame that must contain
    columns `<label>_pred` and `<label>_gold`.

    Rows where the prediction is NaN are *ignored* for this label.
    """
    mask_eval = df[f"{label}_pred"].notna()
    if mask_eval.sum() == 0:
        return 0, 0, 0, 0  # nothing to evaluate

    pred = df.loc[mask_eval, f"{label}_pred"].astype(int)
    gold = df.loc[mask_eval, f"{label}_gold"].astype(int)

    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    return tp, fp, fn, tn


def binary_scores(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Return precision, recall, F1 given TP/FP/FN (standard definitions).
    """
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def class0_scores(tp: int, fp: int, fn: int, tn: int) -> Tuple[float, float, float]:
    """
    Metrics for the negative class (0).  Here the “positive” events are the
    *negatives* of the original label, hence:

    TP₀ ≡ TN₁ ; FP₀ ≡ FN₁ ; FN₀ ≡ FP₁
    """
    tp0, fp0, fn0 = tn, fn, fp
    return binary_scores(tp0, fp0, fn0)


def aggregate_scores(
    per_label: Dict[str, Dict[str, Any]],
    labels: List[str],
    weights: Dict[str, int] | None = None,
) -> Dict[str, float]:
    """
    Compute micro, macro and weighted averages over *labels*.

    Parameters
    ----------
    per_label : dict
        Mapping label → metrics dict produced by `collect_metrics`.
    labels : list[str]
        Which labels to aggregate (can exclude *_sub for the top-level average).
    weights : dict[str, int] or None
        Optional support weights for the weighted average.  If None,
        the gold support is used.
    """
    # micro: sum over numerators / sum over denominators
    sum_tp = sum(per_label[l]["tp"] for l in labels)
    sum_fp = sum(per_label[l]["fp"] for l in labels)
    sum_fn = sum(per_label[l]["fn"] for l in labels)
    micro_p, micro_r, micro_f1 = binary_scores(sum_tp, sum_fp, sum_fn)

    # macro: un-weighted arithmetic mean
    macro_p = sum(per_label[l]["precision_1"] for l in labels) / len(labels)
    macro_r = sum(per_label[l]["recall_1"] for l in labels) / len(labels)
    macro_f1 = sum(per_label[l]["F1_1"] for l in labels) / len(labels)

    # weighted: weighted by support (default) or custom weights
    if weights is None:
        weights = {l: per_label[l]["support_gold"] for l in labels}
    total_w = sum(weights.values()) or 1  # avoid ÷0
    w_p = sum(per_label[l]["precision_1"] * weights[l] for l in labels) / total_w
    w_r = sum(per_label[l]["recall_1"] * weights[l] for l in labels) / total_w
    w_f1 = sum(per_label[l]["F1_1"] * weights[l] for l in labels) / total_w

    return {
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "F1_micro": micro_f1,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "F1_macro": macro_f1,
        "precision_weighted": w_p,
        "recall_weighted": w_r,
        "F1_weighted": w_f1,
    }

##############################################################################
#                         MAIN                                               #
##############################################################################

def main() -> None:  # noqa: C901
    # 1 — Load data
    print("[INFO] Connecting to PostgreSQL…")
    with open_pg(DB_PARAMS) as conn:
        pred_df = fetch_predictions(conn)
    print(f"[INFO] {len(pred_df):,} predicted rows retrieved.")

    print("[INFO] Loading gold annotations…")
    gold_df = load_gold_jsonl(GOLD_JSONL)
    print(f"[INFO] {len(gold_df):,} gold rows loaded.")

    # 2 — Detect label columns in predictions
    label_cols = detect_label_columns(pred_df)
    if not label_cols:
        raise SystemExit("[FATAL] No numeric prediction columns detected.")

    # 3 — Keep only intersection of doc_id + sentence_id
    cols_to_merge = ["doc_id", "sentence_id"]
    merged = pd.merge(
        gold_df, pred_df, on=cols_to_merge, how="inner", suffixes=("_gold", "_pred")
    )
    if merged.empty:
        raise SystemExit("[FATAL] No matching rows between gold and predictions.")
    print(f"[INFO] {len(merged):,} rows matched for evaluation.")

    # 4 — Add gold binary columns (already there) & make sure predictions are int
    for lab in label_cols:
        merged[f"{lab}_pred"] = merged[lab].astype("Int64")  # keep NA
    merged.drop(columns=label_cols, inplace=True)  # tidy up

    # 5 — Evaluate per language
    results: List[Dict[str, Any]] = []
    languages = {
        "ALL": slice(None),
        "EN": merged["language"] == "EN",
        "FR": merged["language"] == "FR",
    }

    for lang_code, mask in languages.items():
        df_lang = merged.loc[mask].copy()
        if df_lang.empty:
            continue
        per_label_metrics: Dict[str, Dict[str, Any]] = {}
        for lab in label_cols:
            tp, fp, fn, tn = compute_confusion_per_label(df_lang, lab)
            prec1, rec1, f1_1 = binary_scores(tp, fp, fn)
            prec0, rec0, f1_0 = class0_scores(tp, fp, fn, tn)
            per_label_metrics[lab] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision_1": prec1,
                "recall_1": rec1,
                "F1_1": f1_1,
                "precision_0": prec0,
                "recall_0": rec0,
                "F1_0": f1_0,
                "support_gold": int(df_lang[f"{lab}_gold"].sum()),
                "support_pred": int(df_lang[f"{lab}_pred"].fillna(0).astype(int).sum()),
                "support_eval": int(df_lang[f"{lab}_pred"].notna().sum()),
            }

            results.append(
                {
                    "label": lab,
                    "language": lang_code,
                    **per_label_metrics[lab],
                }
            )

        # 6 — Aggregated (micro / macro / weighted)
        agg = aggregate_scores(per_label_metrics, label_cols)
        results.append(
            {
                "label": "__aggregate__",
                "language": lang_code,
                **{k: None for k in ("tp", "fp", "fn", "tn")},
                **agg,
            }
        )

    # 7 — Save CSV
    # Ensure deterministic ordering
    field_order = [
        "label",
        "language",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision_1",
        "recall_1",
        "F1_1",
        "precision_0",
        "recall_0",
        "F1_0",
        "support_gold",
        "support_pred",
        "support_eval",
        "precision_micro",
        "recall_micro",
        "F1_micro",
        "precision_macro",
        "recall_macro",
        "F1_macro",
        "precision_weighted",
        "recall_weighted",
        "F1_weighted",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=field_order, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
            )

    print(f"[INFO] Metrics written → {OUTPUT_CSV}")
    print("[INFO] Done ✔")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
