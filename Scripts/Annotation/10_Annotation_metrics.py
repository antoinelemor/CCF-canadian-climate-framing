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
English (EN), French (FR) and the combined corpus (ALL). For each label and
language, all metrics for class=1 and class=0 appear side-by-side in the same
row. In addition, it provides micro, macro, and weighted averages (computed
across the two classes 0 and 1) per label–language. Finally, it appends
aggregate metrics (micro, macro, weighted) across all labels for each
language. The final CSV is publication-ready for scientific articles.

Important Notes:
----------------
1) “meta” data in the gold JSONL file is fully ignored for the evaluation,
   except for doc_id, sentence_id, and language. Values such as author, title,
   news_type, etc. are *not* used in the metrics.

2) For each label L found in CCF_processed_data, we treat L=1 if the gold
   annotation set for that sentence includes L, else 0. If a value is null
   in the predictions, it is not counted in the confusion matrix (i.e.,
   we skip that row entirely).

3) The final CSV includes:
   - TP_1, FP_1, FN_1, precision_1, recall_1, f1_1, support_1
   - TP_0, FP_0, FN_0, precision_0, recall_0, f1_0, support_0
   - microPrecision, microRecall, microF1
   - macroPrecision, macroRecall, macroF1
   - weightedPrecision, weightedRecall, weightedF1
   per label–language pair. Additional lines provide the same metrics
   aggregated across all labels for each language (“ALL” row).

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

BASE_DIR = Path(__file__).resolve().parent
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF"),
    "user":     os.getenv("CCF_DB_USER", ""),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

MANUAL_DIR = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "manual_annotations_JSONL").resolve()
GOLD_JSONL = MANUAL_DIR / "all.jsonl"

OUTPUT_CSV = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "final_annotation_metrics_v3.csv").resolve()

##############################################################################
#    We exclude these table columns from the label set, as they are not      #
#    actual binary annotations. doc_id and sentence_id are used only to      #
#    match rows, language is used for grouping, etc.                         #
##############################################################################
NON_ANNOT_COLS: Set[str] = {
    "language", "sentences", "id_article", "Unnamed: 0", "doc_id",
    "sentence_id", "words_count_updated", "words_count",
}

##############################################################################
#                          HELPER UTILITIES                                  #
##############################################################################

def open_pg(params: Dict[str, Any]) -> _PGConnection:
    """
    Open a PostgreSQL connection using given parameters or environment vars.
    """
    try:
        return psycopg2.connect(**params)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"[FATAL] PostgreSQL connection failed: {exc}") from exc


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score from TP, FP, FN counts.
    If denominators are zero, returns 0.0 to avoid ZeroDivisionError.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


##############################################################################
#                          FETCH PREDICTIONS                                 #
##############################################################################

def fetch_predictions(conn: _PGConnection) -> pd.DataFrame:
    """
    Query the full table of model predictions and coerce every annotation
    column to a numeric dtype. If the column is strictly {0, 1, NaN} we
    down-cast to pandas’ nullable Int8; otherwise we keep it as float64.

    Returns
    -------
    A DataFrame with *all* original columns. Annotation columns are
    guaranteed to be numeric (Int8 or float64) with NaNs for missing
    predictions. Non-annotation columns (doc_id, sentence_id, language, etc.)
    are preserved for indexing / grouping but excluded from the label set.
    """
    query = sql.SQL("SELECT * FROM {};").format(
        sql.Identifier(TABLE_NAME)
    ).as_string(conn)

    # pandas warns because it prefers SQLAlchemy; we silence that
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

    df = pd.read_sql_query(query, conn)

    # Identify the annotation columns
    annot_cols = [c for c in df.columns if c not in NON_ANNOT_COLS]

    for col in annot_cols:
        # Force numeric: anything unconvertible → NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # If the column contains only {0, 1, NaN}, cast to Int8
        uniq = set(df[col].dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            df[col] = df[col].astype("Int8", copy=False)

    return df


##############################################################################
#                          LOAD GOLD ANNOTATIONS                             #
##############################################################################

def load_gold_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load the gold JSONL file. We only keep doc_id, sentence_id, and language
    from the 'meta' field. The other 'meta' values (author, title, media, etc.)
    are completely ignored for metrics, as requested.

    Returns
    -------
    A list of dicts with:
        {
          "labels": set([...]),
          "language": <str or None>,
          "doc_id": <int or str>,
          "sentence_id": <int>,
        }
    """
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fo:
        for ln in fo:
            ln = ln.strip()
            if not ln:
                continue

            rec = json.loads(ln)
            meta = rec.get("meta", {})
            # The set of gold labels is stored in rec["label"]
            # We do not consider any meta-based labels.
            entry = {
                "labels":      set(rec.get("label", [])),
                "language":    meta.get("language"),
                "doc_id":      meta.get("doc_id"),
                "sentence_id": meta.get("sentence_id"),
            }
            entries.append(entry)

    return entries


def detect_annotation_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns that are considered potential binary labels
    from the DataFrame. Exclude known non-annotation columns.
    """
    return [c for c in df.columns if c not in NON_ANNOT_COLS]


##############################################################################
#                          CONFUSION STORAGE                                 #
##############################################################################

class ConfusionMatrix:
    """
    A container that stores confusion counts for label–language–class triplets:

        confusion[(label, language)][class] = { "TP": X, "FP": Y, "FN": Z }

    where class ∈ {0, 1}. We then compute metrics from these counts.
    """

    def __init__(self):
        # conf[(label, lang)] -> { 0: {"TP":..., "FP":..., "FN":...},
        #                          1: {"TP":..., "FP":..., "FN":...} }
        self.conf = defaultdict(lambda: {
            0: {"TP": 0, "FP": 0, "FN": 0},
            1: {"TP": 0, "FP": 0, "FN": 0},
        })

    def add(self, label: str, language: str, gold: int, pred: int):
        """
        Update confusion counts for the given label–language.
        'gold' and 'pred' are either 0 or 1. We store the relevant
        increments in self.conf[(label, language)][class].
        """
        # For class=1
        if gold == 1 and pred == 1:
            self.conf[(label, language)][1]["TP"] += 1
        elif gold == 1 and pred == 0:
            self.conf[(label, language)][1]["FN"] += 1
        elif gold == 0 and pred == 1:
            self.conf[(label, language)][1]["FP"] += 1

        # For class=0 (mirroring the same logic)
        # gold=0, pred=0 => True Positive for class=0
        # gold=0, pred=1 => False Negative for class=0
        # gold=1, pred=0 => False Positive for class=0
        if gold == 0 and pred == 0:
            self.conf[(label, language)][0]["TP"] += 1
        elif gold == 0 and pred == 1:
            self.conf[(label, language)][0]["FN"] += 1
        elif gold == 1 and pred == 0:
            self.conf[(label, language)][0]["FP"] += 1

    def keys(self):
        """
        Return all (label, language) pairs that have been populated.
        """
        return self.conf.keys()

    def get_counts(self, label: str, language: str, cls: int) -> Dict[str, int]:
        """
        Retrieve the dictionary {TP, FP, FN} for a given label–language–class.
        """
        return self.conf[(label, language)][cls]


##############################################################################
#                          METRIC AGGREGATION                                #
##############################################################################

def compute_metrics_for_two_classes(
    tp_1: int, fp_1: int, fn_1: int,
    tp_0: int, fp_0: int, fn_0: int
) -> Dict[str, float]:
    """
    Given confusion counts for class=1 and class=0, compute:
      - precision_1, recall_1, f1_1
      - precision_0, recall_0, f1_0
      - microPrecision, microRecall, microF1
      - macroPrecision, macroRecall, macroF1
      - weightedPrecision, weightedRecall, weightedF1

    Returns
    -------
    A dict with 12 keys:
        {
           "precision_1", "recall_1", "f1_1",
           "precision_0", "recall_0", "f1_0",
           "microPrecision", "microRecall", "microF1",
           "macroPrecision", "macroRecall", "macroF1",
           "weightedPrecision", "weightedRecall", "weightedF1"
        }
    """
    p1, r1, f1_1 = prf(tp_1, fp_1, fn_1)
    p0, r0, f1_0 = prf(tp_0, fp_0, fn_0)

    # Micro across the two classes
    total_tp = tp_1 + tp_0
    total_fp = fp_1 + fp_0
    total_fn = fn_1 + fn_0
    micro_p, micro_r, micro_f1 = prf(total_tp, total_fp, total_fn)

    # Macro = average of the two classes
    macro_p = (p1 + p0) / 2
    macro_r = (r1 + r0) / 2
    macro_f1 = (f1_1 + f1_0) / 2

    # Weighted average: weigh each class by its support
    # support for class=1 is (tp_1+fn_1), class=0 is (tp_0+fn_0)
    support_1 = tp_1 + fn_1
    support_0 = tp_0 + fn_0
    total_supp = support_1 + support_0
    if total_supp > 0:
        w_p = (p1 * support_1 + p0 * support_0) / total_supp
        w_r = (r1 * support_1 + r0 * support_0) / total_supp
        w_f1 = (f1_1 * support_1 + f1_0 * support_0) / total_supp
    else:
        w_p = w_r = w_f1 = 0.0

    return {
        "precision_1": p1,
        "recall_1": r1,
        "f1_1": f1_1,
        "precision_0": p0,
        "recall_0": r0,
        "f1_0": f1_0,
        "microPrecision": micro_p,
        "microRecall": micro_r,
        "microF1": micro_f1,
        "macroPrecision": macro_p,
        "macroRecall": macro_r,
        "macroF1": macro_f1,
        "weightedPrecision": w_p,
        "weightedRecall": w_r,
        "weightedF1": w_f1,
    }


##############################################################################
#                          MAIN EVALUATION LOGIC                             #
##############################################################################

def main() -> None:
    print("[INFO] Connecting to PostgreSQL…")
    with open_pg(DB_PARAMS) as conn:
        df_pred = fetch_predictions(conn)

    # Identify the annotation columns
    annot_cols = detect_annotation_columns(df_pred)
    print(f"[INFO] {len(df_pred):,} prediction rows loaded | "
          f"{len(annot_cols)} annotation labels detected.")

    # Build a lookup for predictions:
    # Key: (doc_id, sentence_id) -> row of predicted columns
    pred_lookup: Dict[Tuple[Any, Any], Dict[str, Any]] = {
        (row["doc_id"], row["sentence_id"]): row
        for row in df_pred.to_dict(orient="records")
    }

    print("[INFO] Loading gold annotations…")
    gold_entries = load_gold_jsonl(GOLD_JSONL)
    print(f"[INFO] {len(gold_entries):,} gold sentences loaded.")

    # Initialize confusion structure
    cm = ConfusionMatrix()

    # --------------------------------------------------------------------- #
    #  Iterate through gold sentences and match with predictions. For each  #
    #  label in the predicted columns, check if it's 1 or 0 in gold.        #
    #  If the predicted value is NaN, we skip that row.                     #
    # --------------------------------------------------------------------- #
    for entry in tqdm(gold_entries, desc="Scoring"):
        key = (entry["doc_id"], entry["sentence_id"])
        pred_row = pred_lookup.get(key)
        if pred_row is None:
            # No prediction for this sentence
            continue

        lang = entry["language"] or "ALL"  # If None, fallback to "ALL"
        gold_set = entry["labels"]

        for label in annot_cols:
            pred_val = pred_row.get(label)
            if pd.isna(pred_val):
                # Null predictions are not counted
                continue

            # gold=1 if label is in gold_set, else 0
            gold_bin = 1 if (label in gold_set) else 0
            # pred is 1 or 0 (we cast pred_val to int)
            pred_bin = int(pred_val)

            cm.add(label, lang, gold_bin, pred_bin)

    # --------------------------------------------------------------------- #
    #  Now build final metrics. We want one CSV row per label–language,     #
    #  containing side-by-side columns for class=1 and class=0, plus the    #
    #  micro/macro/weighted average across these two classes.               #
    # --------------------------------------------------------------------- #
    results: List[Dict[str, Any]] = []
    label_language_pairs = sorted(list(cm.keys()))

    # Gather everything into results
    for (label, language) in label_language_pairs:
        # For class=1
        c1 = cm.get_counts(label, language, 1)  # {TP,FP,FN}
        # For class=0
        c0 = cm.get_counts(label, language, 0)  # {TP,FP,FN}

        tp_1, fp_1, fn_1 = c1["TP"], c1["FP"], c1["FN"]
        tp_0, fp_0, fn_0 = c0["TP"], c0["FP"], c0["FN"]

        # Compute metrics
        met = compute_metrics_for_two_classes(tp_1, fp_1, fn_1, tp_0, fp_0, fn_0)

        # Create a row with everything
        row = {
            "label": label,
            "language": language,
            "TP_1": tp_1,
            "FP_1": fp_1,
            "FN_1": fn_1,
            "precision_1": met["precision_1"],
            "recall_1": met["recall_1"],
            "f1_1": met["f1_1"],
            "support_1": tp_1 + fn_1,

            "TP_0": tp_0,
            "FP_0": fp_0,
            "FN_0": fn_0,
            "precision_0": met["precision_0"],
            "recall_0": met["recall_0"],
            "f1_0": met["f1_0"],
            "support_0": tp_0 + fn_0,

            "microPrecision": met["microPrecision"],
            "microRecall": met["microRecall"],
            "microF1": met["microF1"],

            "macroPrecision": met["macroPrecision"],
            "macroRecall": met["macroRecall"],
            "macroF1": met["macroF1"],

            "weightedPrecision": met["weightedPrecision"],
            "weightedRecall": met["weightedRecall"],
            "weightedF1": met["weightedF1"],
        }
        results.append(row)

    # --------------------------------------------------------------------- #
    #  For each language, add an "ALL" row that averages or sums across all  #
    #  labels. We combine the confusion for each label in that language,     #
    #  then compute final metrics.                                          #
    # --------------------------------------------------------------------- #
    def sum_confusions_for_language(lang: str) -> Tuple[int,int,int, int,int,int]:
        """
        Sum the confusion across all labels for the given language.
        Returns (tp_1, fp_1, fn_1, tp_0, fp_0, fn_0).
        """
        tot_tp_1 = tot_fp_1 = tot_fn_1 = 0
        tot_tp_0 = tot_fp_0 = tot_fn_0 = 0
        for (lbl, lng) in cm.keys():
            if lng == lang:
                tot_tp_1 += cm.conf[(lbl, lng)][1]["TP"]
                tot_fp_1 += cm.conf[(lbl, lng)][1]["FP"]
                tot_fn_1 += cm.conf[(lbl, lng)][1]["FN"]
                tot_tp_0 += cm.conf[(lbl, lng)][0]["TP"]
                tot_fp_0 += cm.conf[(lbl, lng)][0]["FP"]
                tot_fn_0 += cm.conf[(lbl, lng)][0]["FN"]
        return (tot_tp_1, tot_fp_1, tot_fn_1, tot_tp_0, tot_fp_0, tot_fn_0)

    # Identify all languages used (the code above can have multiple).
    # We'll also add "ALL" if it exists in the data.
    languages_found = sorted(set(lang for (_, lang) in cm.keys()))

    for lang in languages_found:
        (tp1, fp1, fn1, tp0, fp0, fn0) = sum_confusions_for_language(lang)
        mets = compute_metrics_for_two_classes(tp1, fp1, fn1, tp0, fp0, fn0)

        row_all = {
            "label": "ALL_LABELS",
            "language": lang,
            "TP_1": tp1,
            "FP_1": fp1,
            "FN_1": fn1,
            "precision_1": mets["precision_1"],
            "recall_1": mets["recall_1"],
            "f1_1": mets["f1_1"],
            "support_1": tp1 + fn1,

            "TP_0": tp0,
            "FP_0": fp0,
            "FN_0": fn0,
            "precision_0": mets["precision_0"],
            "recall_0": mets["recall_0"],
            "f1_0": mets["f1_0"],
            "support_0": tp0 + fn0,

            "microPrecision": mets["microPrecision"],
            "microRecall": mets["microRecall"],
            "microF1": mets["microF1"],

            "macroPrecision": mets["macroPrecision"],
            "macroRecall": mets["macroRecall"],
            "macroF1": mets["macroF1"],

            "weightedPrecision": mets["weightedPrecision"],
            "weightedRecall": mets["weightedRecall"],
            "weightedF1": mets["weightedF1"],
        }
        results.append(row_all)

    # --------------------------------------------------------------------- #
    #  Convert results to a DataFrame for nice rounding, then write to CSV.  #
    # --------------------------------------------------------------------- #
    df_res = pd.DataFrame(results)

    # Round floats to four decimals
    float_cols = [
        "precision_1", "recall_1", "f1_1",
        "precision_0", "recall_0", "f1_0",
        "microPrecision", "microRecall", "microF1",
        "macroPrecision", "macroRecall", "macroF1",
        "weightedPrecision", "weightedRecall", "weightedF1"
    ]
    for c in float_cols:
        df_res[c] = df_res[c].apply(lambda x: f"{x:.4f}")

    # Ensure the output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"[INFO] Metrics CSV saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
