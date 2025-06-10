#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
10_Annotation_metrics.py

MAIN OBJECTIVE:
---------------
This script benchmarks the automatic sentence-level annotations stored in the
PostgreSQL table CCF_processed_data against a hand-labelled “gold” JSONL file.
It produces a fully-formatted CSV summarising precision, recall, and F1-score
for both positive (1) and negative (0) classes, calculated separately for
English (EN), French (FR), and the combined corpus (ALL). For each label and
language, the script also provides micro, macro, and weighted averages
(including a global “ALL” row) in wide format. All metadata fields in the JSONL
('meta') are ignored for the computation except for doc_id, sentence_id, and
language (used strictly to match and filter rows), so that no metadata
(e.g. title, author, date, page_number, etc.) influences the metrics.

Dependencies:
-------------
- csv
- json
- os
- pathlib.Path
- collections.defaultdict
- typing (Any, Dict, List, Tuple, Set)
- pandas ≥ 1.5
- psycopg2 ≥ 2.9
- tqdm ≥ 4.65

MAIN FEATURES:
--------------
1) Robust PostgreSQL connection – pulls every column from CCF_processed_data
   using credentials supplied via environment variables (with sane defaults).
2) Automatic column typing – forces all annotation columns to numeric dtypes
   and downcasts perfect {0, 1, NaN} columns to nullable Int8 for memory economy.
3) Gold-standard loader – parses a multilingual JSONL file and extracts
   labels, language, doc_id, and sentence_id to serve as ground truth.
   All other metadata in the JSON is discarded.
4) Language-aware confusion matrices – tallies TP, FP, FN for each label,
   for both classes (1 and 0) and for each language (EN, FR, ALL).
5) Comprehensive metric computation – derives per-label P/R/F1 plus
   micro-, macro-, and weighted averages, respecting class imbalance.
6) Wide-format CSV export – writes a clean, publication-ready metrics table
   (four-decimal rounding) with separate columns for class 1 and 0 measures
   (precision_1, recall_1, f1_1, etc.) as well as aggregated micro, macro,
   and weighted metrics. Also provides an “ALL” row (label=ALL) to show
   aggregated performance across all labels.
7) Progress & logging – informative console messages and a tqdm progress bar
   track the evaluation pipeline end-to-end.

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as _PGConnection
from tqdm.auto import tqdm

# Adjust this BASE_DIR according to your directory structure
BASE_DIR = Path(__file__).resolve().parent

# PostgreSQL connection parameters (env variables have priority)
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF"),
    "user":     os.getenv("CCF_DB_USER", ""),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}

TABLE_NAME = "CCF_processed_data"

# Path to the directory containing the gold JSONL file
MANUAL_DIR = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "manual_annotations_JSONL").resolve()
GOLD_JSONL = MANUAL_DIR / "all.jsonl"

# Output CSV path
OUTPUT_CSV = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "final_annotation_metrics_v3.csv").resolve()

# Columns we never treat as "annotation" (nor do we measure them):
# We also exclude the typical metadata columns. We keep doc_id and sentence_id
# (and language) strictly to locate and match the data, but we do NOT measure them.
NON_ANNOT_COLS: Set[str] = {
    "language", "sentences", "id_article", "Unnamed: 0", "doc_id",
    "sentence_id", "words_count_updated", "words_count",
    # Exclude any known metadata columns we do not want to evaluate:
    "news_type", "title", "author", "media", "date", "page_number", "ner_entities",
}

##############################################################################
#                          HELPER UTILITIES                                  #
##############################################################################

def open_pg(params: Dict[str, Any]) -> _PGConnection:
    """
    Opens a PostgreSQL connection using the provided parameters.
    """
    try:
        return psycopg2.connect(**params)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"[FATAL] PostgreSQL connection failed: {exc}") from exc

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Given TP, FP, and FN, returns (precision, recall, f1).
    """
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1

##############################################################################
#                          FETCH PREDICTIONS                                 #
##############################################################################

def fetch_predictions(conn: _PGConnection) -> pd.DataFrame:
    """
    Query the full table of model predictions and coerce every annotation
    column to a numeric dtype. If the column is strictly {0, 1, NaN}, it is
    down-cast to pandas’ nullable Int8 for memory economy.

    Returns
    -------
    DataFrame with *all* original columns from the DB.  Annotation columns are
    guaranteed to be numeric (Int8 or float64) with NaNs for missing predictions.
    """
    query = sql.SQL("SELECT * FROM {};").format(
        sql.Identifier(TABLE_NAME)
    ).as_string(conn)

    # Pandas warns because it prefers SQLAlchemy; we can safely ignore here
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

    df = pd.read_sql_query(query, conn)

    # Identify columns for potential annotation
    # (everything that is not explicitly in NON_ANNOT_COLS)
    annot_cols = [
        c for c in df.columns
        if c not in NON_ANNOT_COLS
    ]

    # Attempt to convert each annotation column to numeric
    for col in annot_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # If only {0, 1, NaN}, convert to Int8
        uniq = set(df[col].dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            df[col] = df[col].astype("Int8", copy=False)

    return df

##############################################################################
#                          LOAD GOLD JSONL                                   #
##############################################################################

def load_gold_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Loads the gold-standard JSONL, ignoring all metadata except doc_id,
    sentence_id, and language (needed to match DB rows). `label` is read
    as the gold annotation set.
    """
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fo:
        for ln in fo:
            if not ln.strip():
                continue
            rec = json.loads(ln)

            # We do NOT use meta fields except doc_id, sentence_id, language
            meta = rec.get("meta", {})
            doc_id = meta.get("doc_id")
            sentence_id = meta.get("sentence_id")
            language = meta.get("language")

            # The gold labels are in rec['label'] (a list)
            gold_labels = set(rec.get("label", []))

            entries.append({
                "doc_id":      doc_id,
                "sentence_id": sentence_id,
                "language":    language,
                "gold_labels": gold_labels
            })
    return entries

##############################################################################
#                   DETECT WHICH COLUMNS ARE TRUE LABELS                     #
##############################################################################

def detect_annotation_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns the list of columns in df that can be considered annotation labels,
    excluding any known metadata. We do *not* rely on JSON's meta fields.
    """
    annot_cols = [
        c for c in df.columns
        if c not in NON_ANNOT_COLS
    ]
    return annot_cols

##############################################################################
#                          CONFUSION MATRIX STORAGE                          #
##############################################################################

class ConfusionDict(defaultdict):
    """
    A nested dictionary structure to store TP, FP, FN for:
    (label, class, language).

    Example usage:
        self[(label, cls, lang)] -> {"TP": int, "FP": int, "FN": int}
    """
    def __init__(self):
        super().__init__(lambda: {"TP": 0, "FP": 0, "FN": 0})

    def increment(self, label: str, cls: int, lang: str,
                  tp: int=0, fp: int=0, fn: int=0) -> None:
        key = (label, cls, lang)
        self[key]["TP"] += tp
        self[key]["FP"] += fp
        self[key]["FN"] += fn

    def update_counts(self, label: str, gold: int, pred: int, lang: str) -> None:
        """
        For a given label, gold, and pred (each 0 or 1), increments:
          - confusion for (class=1) if it's about that class
          - confusion for (class=0) if it's about that class
        across the specified language and also for language="ALL".
        """
        # We handle class=1:
        if gold == 1 and pred == 1:
            self.increment(label, 1, lang, tp=1)
            self.increment(label, 1, "ALL", tp=1)
        elif gold == 1 and pred == 0:
            self.increment(label, 1, lang, fn=1)
            self.increment(label, 1, "ALL", fn=1)
        elif gold == 0 and pred == 1:
            self.increment(label, 1, lang, fp=1)
            self.increment(label, 1, "ALL", fp=1)

        # We handle class=0 simultaneously
        gold_0 = 1 - gold
        pred_0 = 1 - pred
        if gold_0 == 1 and pred_0 == 1:
            self.increment(label, 0, lang, tp=1)
            self.increment(label, 0, "ALL", tp=1)
        elif gold_0 == 1 and pred_0 == 0:
            self.increment(label, 0, lang, fn=1)
            self.increment(label, 0, "ALL", fn=1)
        elif gold_0 == 0 and pred_0 == 1:
            self.increment(label, 0, lang, fp=1)
            self.increment(label, 0, "ALL", fp=1)

##############################################################################
#                          MAIN EVALUATION LOGIC                             #
##############################################################################

def main() -> None:
    print("[INFO] Connecting to PostgreSQL…")
    with open_pg(DB_PARAMS) as conn:
        df_pred = fetch_predictions(conn)

    # Identify which columns are actual annotation labels
    annot_cols = detect_annotation_columns(df_pred)
    print(f"[INFO] {len(df_pred):,} prediction rows loaded "
          f"| {len(annot_cols)} potential label columns identified.")

    # Convert df to a lookup dict for easy access:
    # (doc_id, sentence_id) -> row-dict of predictions
    pred_lookup: Dict[Tuple[Any, Any], Dict[str, Any]] = {
        (row["doc_id"], row["sentence_id"]): row
        for row in df_pred.to_dict(orient="records")
    }

    print("[INFO] Loading gold annotations…")
    gold_entries = load_gold_jsonl(GOLD_JSONL)
    print(f"[INFO] {len(gold_entries):,} gold sentences loaded.")

    # Build confusion dictionary
    conf = ConfusionDict()

    # Populate confusion matrix
    for entry in tqdm(gold_entries, desc="Scoring"):
        key = (entry["doc_id"], entry["sentence_id"])
        lang = entry["language"] or "UNK"
        gold_set = entry["gold_labels"]

        if key not in pred_lookup:
            # If we have no predicted row, skip
            continue
        pred_row = pred_lookup[key]

        # For each label column in the DB, compare to gold
        for label in annot_cols:
            pred_val = pred_row.get(label)
            # If DB has no prediction (NaN), skip
            if pd.isna(pred_val):
                continue

            # gold=1 if this label is in the gold set, else 0
            gold_bin = 1 if label in gold_set else 0
            pred_bin = int(pred_val)

            # Update confusion
            conf.update_counts(
                label=label,
                gold=gold_bin,
                pred=pred_bin,
                lang=lang
            )

    print("[INFO] Confusion matrix complete. Building final metrics…")

    # We'll create a structure for all metrics in wide format:
    # For each label and language, we gather:
    #   TP_1, FP_1, FN_1, precision_1, recall_1, f1_1, support_1,
    #   TP_0, FP_0, FN_0, precision_0, recall_0, f1_0, support_0,
    #   precision_micro, recall_micro, f1_micro,
    #   precision_macro, recall_macro, f1_macro,
    #   precision_weighted, recall_weighted, f1_weighted,
    #   support_total
    #
    # Then also we produce a label="ALL" row that aggregates across labels.

    # First, gather a sorted list of labels
    unique_labels = sorted({lab for (lab, _, _) in conf.keys()})
    languages = ("EN", "FR", "ALL")  # We always produce these. If no EN/FR data,
                                    # the confusion is simply zeroes.

    # Build a helper to fetch confusion stats from conf and compute P/R/F
    def get_stats(lab: str, cls: int, lng: str):
        c = conf[(lab, cls, lng)]
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        p, r, f1_ = prf(tp, fp, fn)
        s = tp + fn
        return tp, fp, fn, p, r, f1_, s

    # We accumulate final rows here
    metric_rows: List[Dict[str, Any]] = []

    # Helper to compute micro, macro, weighted for a single (label, lang)
    # across classes {0,1}.
    def compute_averages(label: str, lang: str,
                         row_1: Dict[str, Any],
                         row_0: Dict[str, Any]) -> Dict[str, float]:
        # row_1, row_0 each has: {TP, FP, FN, precision, recall, f1, support}
        # We'll produce a dict for micro, macro, weighted for that label+lang
        tp_micro = row_1["TP"] + row_0["TP"]
        fp_micro = row_1["FP"] + row_0["FP"]
        fn_micro = row_1["FN"] + row_0["FN"]
        p_mi, r_mi, f_mi = prf(tp_micro, fp_micro, fn_micro)

        p_ma = 0.5 * (row_1["precision"] + row_0["precision"])
        r_ma = 0.5 * (row_1["recall"]    + row_0["recall"])
        f_ma = 0.5 * (row_1["f1"]        + row_0["f1"])

        sup_1 = row_1["support"]
        sup_0 = row_0["support"]
        sup_tot = sup_1 + sup_0

        # Weighted average
        if sup_tot > 0:
            w_p = ((row_1["precision"] * sup_1) + (row_0["precision"] * sup_0)) / sup_tot
            w_r = ((row_1["recall"]    * sup_1) + (row_0["recall"]    * sup_0)) / sup_tot
            w_f = ((row_1["f1"]        * sup_1) + (row_0["f1"]        * sup_0)) / sup_tot
        else:
            w_p, w_r, w_f = 0.0, 0.0, 0.0

        return {
            "precision_micro":   p_mi,
            "recall_micro":      r_mi,
            "f1_micro":          f_mi,
            "precision_macro":   p_ma,
            "recall_macro":      r_ma,
            "f1_macro":          f_ma,
            "precision_weighted": w_p,
            "recall_weighted":    w_r,
            "f1_weighted":        w_f,
            "support_total":      sup_tot
        }

    # We'll also aggregate across all labels at the end for label="ALL".
    # So we store partial sums for each label to build the global row.
    # However, user wants also the final "ALL" broken out for class=1 and class=0.
    # We'll gather them in the same confusion dictionary so it’s simpler.
    # Then we do the pivot for "ALL" label at the very end.

    # Step 1: Build rows for each (label, lang)
    for lab in unique_labels:
        for lng in languages:
            # Get class=1 stats
            tp_1, fp_1, fn_1, p_1, r_1, f1_1, s_1 = get_stats(lab, 1, lng)
            # Get class=0 stats
            tp_0, fp_0, fn_0, p_0, r_0, f0_0, s_0 = get_stats(lab, 0, lng)

            # Averages for that label+lang
            avg = compute_averages(
                label=lab, lang=lng,
                row_1={
                    "TP": tp_1, "FP": fp_1, "FN": fn_1,
                    "precision": p_1, "recall": r_1, "f1": f1_1,
                    "support": s_1
                },
                row_0={
                    "TP": tp_0, "FP": fp_0, "FN": fn_0,
                    "precision": p_0, "recall": r_0, "f1": f0_0,
                    "support": s_0
                }
            )

            metric_rows.append({
                "label": lab,
                "language": lng,

                "TP_1": tp_1,
                "FP_1": fp_1,
                "FN_1": fn_1,
                "precision_1": p_1,
                "recall_1": r_1,
                "f1_1": f1_1,
                "support_1": s_1,

                "TP_0": tp_0,
                "FP_0": fp_0,
                "FN_0": fn_0,
                "precision_0": p_0,
                "recall_0": r_0,
                "f1_0": f0_0,
                "support_0": s_0,

                "precision_micro":   avg["precision_micro"],
                "recall_micro":      avg["recall_micro"],
                "f1_micro":          avg["f1_micro"],

                "precision_macro":   avg["precision_macro"],
                "recall_macro":      avg["recall_macro"],
                "f1_macro":          avg["f1_macro"],

                "precision_weighted": avg["precision_weighted"],
                "recall_weighted":    avg["recall_weighted"],
                "f1_weighted":        avg["f1_weighted"],

                "support_total": avg["support_total"],
            })

    # Step 2: Create an overall "ALL" label that sums across every real label,
    # for each class (0,1) and language. We can just sum the confusion from
    # the existing dictionary. Then compute micro/macro/weighted the same way.
    # This is effectively how we produce a "grand total" row.

    # We'll compute a new confusion for label="ALL" by summing across labels
    # but skipping the existing "ALL" label entry to avoid double counting.
    # Then produce a single row per language.
    overall_conf = ConfusionDict()

    for (lab, cls, lng), dct in conf.items():
        if lab == "ALL":
            # We never store confusion for a label literally named "ALL"
            # in the data, so skip if present
            continue
        # We add to label="ALL" in a new structure
        overall_conf.increment(
            "ALL", cls, lng,
            tp=dct["TP"], fp=dct["FP"], fn=dct["FN"]
        )

    # Now produce the final row(s) for label="ALL"
    for lng in languages:
        # fetch stats for class=1
        atp_1 = overall_conf[("ALL", 1, lng)]["TP"]
        afp_1 = overall_conf[("ALL", 1, lng)]["FP"]
        afn_1 = overall_conf[("ALL", 1, lng)]["FN"]
        p_1, r_1, f1_1 = prf(atp_1, afp_1, afn_1)
        s_1 = atp_1 + afn_1

        # fetch stats for class=0
        atp_0 = overall_conf[("ALL", 0, lng)]["TP"]
        afp_0 = overall_conf[("ALL", 0, lng)]["FP"]
        afn_0 = overall_conf[("ALL", 0, lng)]["FN"]
        p_0, r_0, f0_0 = prf(atp_0, afp_0, afn_0)
        s_0 = atp_0 + afn_0

        # compute averages
        all_avg = compute_averages(
            label="ALL", lang=lng,
            row_1={
                "TP": atp_1, "FP": afp_1, "FN": afn_1,
                "precision": p_1, "recall": r_1, "f1": f1_1,
                "support": s_1
            },
            row_0={
                "TP": atp_0, "FP": afp_0, "FN": afn_0,
                "precision": p_0, "recall": r_0, "f1": f0_0,
                "support": s_0
            }
        )

        metric_rows.append({
            "label": "ALL",
            "language": lng,

            "TP_1": atp_1,
            "FP_1": afp_1,
            "FN_1": afn_1,
            "precision_1": p_1,
            "recall_1": r_1,
            "f1_1": f1_1,
            "support_1": s_1,

            "TP_0": atp_0,
            "FP_0": afp_0,
            "FN_0": afn_0,
            "precision_0": p_0,
            "recall_0": r_0,
            "f1_0": f0_0,
            "support_0": s_0,

            "precision_micro":   all_avg["precision_micro"],
            "recall_micro":      all_avg["recall_micro"],
            "f1_micro":          all_avg["f1_micro"],

            "precision_macro":   all_avg["precision_macro"],
            "recall_macro":      all_avg["recall_macro"],
            "f1_macro":          all_avg["f1_macro"],

            "precision_weighted": all_avg["precision_weighted"],
            "recall_weighted":    all_avg["recall_weighted"],
            "f1_weighted":        all_avg["f1_weighted"],

            "support_total": all_avg["support_total"],
        })

    # Step 3: Write out the final CSV in wide format
    # We'll keep 4-decimal rounding for all P/R/F measures
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fo:
        fieldnames = [
            "label", "language",
            "TP_1", "FP_1", "FN_1", "precision_1", "recall_1", "f1_1", "support_1",
            "TP_0", "FP_0", "FN_0", "precision_0", "recall_0", "f1_0", "support_0",
            "precision_micro", "recall_micro", "f1_micro",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "support_total"
        ]
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        for row in metric_rows:
            # Round float fields
            for k in fieldnames:
                val = row[k]
                if isinstance(val, float):
                    row[k] = f"{val:.4f}"
            writer.writerow(row)

    print(f"[INFO] Metrics CSV saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
