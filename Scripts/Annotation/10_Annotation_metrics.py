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
English (EN), French (FR) and the combined corpus (ALL), and complemented with
micro, macro, and weighted averages.

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
   labels, language, doc_id, and sentence_id metadata for each sentence.
4) Language-aware confusion matrices – tallies TP, FP, FN for each label,
   for both classes (1 and 0) and for each language (EN, FR, ALL).
5) Comprehensive metric computation – derives per-label P/R/F1 plus
   micro-, macro-, and weighted averages, respecting class imbalance.
6) Well-formatted CSV export – writes a clean, publication-ready metrics
   table (four-decimal rounding) to
   Database/Training_data/final_annotation_metrics_v3.csv.
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

BASE_DIR = Path(__file__).resolve().parent
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF"),
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", "Absitreverentiavero19!"),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

MANUAL_DIR = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "manual_annotations_JSONL").resolve()
GOLD_JSONL = MANUAL_DIR / "sentences_to_recheck_multiling_2025_06_03_0704.jsonl"

OUTPUT_CSV = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "final_annotation_metrics_v3.csv").resolve()

NON_ANNOT_COLS: Set[str] = {
    "language", "sentences", "id_article", "Unnamed: 0", "doc_id",
    "sentence_id", "words_count_updated", "words_count",
}

##############################################################################
#                          HELPER UTILITIES                                  #
##############################################################################

def open_pg(params: Dict[str, Any]) -> _PGConnection:
    try:
        return psycopg2.connect(**params)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"[FATAL] PostgreSQL connection failed: {exc}") from exc


##############################################################################
#                          FETCH PREDICTIONS                                 #
##############################################################################

def fetch_predictions(conn: _PGConnection) -> pd.DataFrame:
    """
    Query the full table of model predictions and coerce every annotation
    column to a numeric dtype.  If the column is strictly {0, 1, NaN} we
    down-cast to pandas’ nullable Int8; otherwise we keep it as float64.

    Returns
    -------
    DataFrame with *all* original columns.  Annotation columns are
    guaranteed to be numeric (Int8 or float64) with NaNs for missing
    predictions.
    """
    query = sql.SQL("SELECT * FROM {};").format(
        sql.Identifier(TABLE_NAME)
    ).as_string(conn)

    # pandas warns because it prefers SQLAlchemy, on l’ignore
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

    df = pd.read_sql_query(query, conn)

    # Liste des colonnes d’annotation (numériques ou booléennes dans la BD)
    annot_cols = [
        c for c in df.columns
        if c not in NON_ANNOT_COLS
    ]

    for col in annot_cols:
        # Force numérique : tout ce qui n'est pas convertible → NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Si la colonne ne contient que {0.0, 1.0, NaN} on la convertit en Int8
        uniq = set(df[col].dropna().unique().tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            df[col] = df[col].astype("Int8", copy=False)

    return df


def load_gold_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fo:
        for ln in fo:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            meta = rec.get("meta", {})
            entries.append({
                "labels":      set(rec.get("label", [])),
                "language":    meta.get("language"),
                "doc_id":      meta.get("doc_id"),
                "sentence_id": meta.get("sentence_id"),
            })
    return entries


def detect_annotation_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_ANNOT_COLS]


##############################################################################
#                          EVALUATION CORE                                   #
##############################################################################

class Confusion(defaultdict):
    def __init__(self):
        super().__init__(lambda: {"TP": 0, "FP": 0, "FN": 0})

    def _inc(self, key: Tuple[str, int, str], tp=0, fp=0, fn=0):
        self[key]["TP"] += tp
        self[key]["FP"] += fp
        self[key]["FN"] += fn

    def add(self, label: str, gold: int, pred: int, lang: str):
        """
        Update counts for both classes (1 and 0) and for ‘ALL’ language.
        """
        for cls, g, p in ((1, gold, pred), (0, 1 - gold, 1 - pred)):
            if g == 1 and p == 1:
                self._inc((label, cls, lang), tp=1)
                self._inc((label, cls, "ALL"), tp=1)      # FIX
            elif g == 1 and p == 0:
                self._inc((label, cls, lang), fn=1)
                self._inc((label, cls, "ALL"), fn=1)      # FIX
            elif g == 0 and p == 1:
                self._inc((label, cls, lang), fp=1)
                self._inc((label, cls, "ALL"), fp=1)      # FIX
            # TN never used for P/R/F1


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def average(rows: List[Dict[str, Any]], kind: str, lang: str) -> Dict[str, Any]:
    if not rows:
        return {}

    if kind == "micro":
        tp = sum(r["TP"] for r in rows)
        fp = sum(r["FP"] for r in rows)
        fn = sum(r["FN"] for r in rows)
        p, r_, f1 = prf(tp, fp, fn)
    elif kind == "macro":
        p = sum(r["precision"] for r in rows) / len(rows)
        r_ = sum(r["recall"]    for r in rows) / len(rows)
        f1 = sum(r["f1"]        for r in rows) / len(rows)
    elif kind == "weighted":
        tot = sum(r["support"] for r in rows)
        p = sum(r["precision"] * r["support"] for r in rows) / tot
        r_ = sum(r["recall"]    * r["support"] for r in rows) / tot
        f1 = sum(r["f1"]        * r["support"] for r in rows) / tot
    else:
        raise ValueError

    return {
        "label": kind.upper(),
        "class": "",
        "language": lang,
        "TP": "",
        "FP": "",
        "FN": "",
        "precision": p,
        "recall": r_,
        "f1": f1,
        "support": sum(r["support"] for r in rows),
    }

##############################################################################
#                          MAIN                                              #
##############################################################################

def main() -> None:
    print("[INFO] Connecting to PostgreSQL…")
    with open_pg(DB_PARAMS) as conn:
        df_pred = fetch_predictions(conn)

    annot_cols = detect_annotation_columns(df_pred)
    print(f"[INFO] {len(df_pred):,} prediction rows loaded | {len(annot_cols)} labels.")

    # Lookup (doc_id, sentence_id) → dict(row)
    pred_lookup: Dict[Tuple[Any, Any], Dict[str, Any]] = {
        (row["doc_id"], row["sentence_id"]): row
        for row in df_pred.to_dict(orient="records")
    }

    print("[INFO] Loading gold annotations…")
    gold_entries = load_gold_jsonl(GOLD_JSONL)
    print(f"[INFO] {len(gold_entries):,} gold sentences loaded.")

    # ##################### Evaluation loop ##################### #
    conf = Confusion()
    for entry in tqdm(gold_entries, desc="Scoring"):
        key = (entry["doc_id"], entry["sentence_id"])
        pred_row = pred_lookup.get(key)
        if pred_row is None:
            continue

        lang = entry["language"] or "ALL"
        gold_set = entry["labels"]

        for label in annot_cols:
            pred_val = pred_row.get(label)
            if pd.isna(pred_val):
                continue

            conf.add(
                label=label,
                gold=1 if label in gold_set else 0,
                pred=int(pred_val),
                lang=lang,
            )

    # ####### Build metrics table ####### #
    rows: List[Dict[str, Any]] = []
    for lab in annot_cols:                    
        for cls in (1, 0):
            for lg in ("EN", "FR", "ALL"):
                tp = conf[(lab, cls, lg)]["TP"]
                fp = conf[(lab, cls, lg)]["FP"]
                fn = conf[(lab, cls, lg)]["FN"]
                supp = tp + fn
                p, r_, f1 = prf(tp, fp, fn)
                rows.append({
                    "label": lab,
                    "class": cls,
                    "language": lg,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": p,
                    "recall": r_,
                    "f1": f1,
                    "support": supp,
                })

    # ####### Averages ####### #
    for lg in ("EN", "FR", "ALL"):
        base = [r for r in rows if r["language"] == lg and r["class"] == 1]
        for kind in ("micro", "macro", "weighted"):
            rows.append(average(base, kind, lg))

    # ####### Write CSV ####### #
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=[
            "label", "class", "language",
            "TP", "FP", "FN",
            "precision", "recall", "f1", "support",
        ])
        writer.writeheader()
        for r in rows:
            rec = r.copy()
            for k in ("precision", "recall", "f1"):
                if rec[k] != "":
                    rec[k] = f"{rec[k]:.4f}"
            writer.writerow(rec)

    print(f"[INFO] Metrics CSV saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")