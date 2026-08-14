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
[Anonymous]
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
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
    "user":     os.getenv("CCF_DB_USER", "postgres"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}

TABLE_NAME = "CCF_processed_data"

# Path to the directory containing the gold JSONL file
MANUAL_DIR = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "manual_annotations_JSONL").resolve()
GOLD_JSONL = MANUAL_DIR / "all.jsonl"

# Output CSV paths
OUTPUT_CSV = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "final_annotation_metrics.csv").resolve()
TEMPORAL_CSV = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
                "temporal_drift_metrics.csv").resolve()

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

# Label mapping dictionary for better readability
LABEL_MAPPING = {
    # Geographic Focus
    "detect_location": "Canadian Context",
    "Location_Detection": "Canadian Context",
    "detect_location_SUB": "Canadian Context",
    "Canadian Context": "Canadian Context",  # Identity mapping
    
    # Events - Main Detection
    "detect_event": "Event Detection",
    "Event_Detection": "Event Detection",
    "event_Detection": "Event Detection",
    
    # Events - Subcategories
    "event_1": "Extreme Weather Event",
    "event_2": "Meeting/Conference",
    "event_3": "Publication",
    "event_4": "Election",
    "event_5": "Policy Announcement",
    "event_6": "Judiciary Decision",
    "event_7": "Cultural Event",
    "event_8": "Protest",
    "Event_1_SUB": "Extreme Weather Event",
    "Event_2_SUB": "Meeting/Conference",
    "Event_3_SUB": "Publication",
    "Event_4_SUB": "Election",
    "Event_5_SUB": "Policy Announcement",
    "Event_6_SUB": "Judiciary Decision",
    "Event_7_SUB": "Cultural Event",
    "Event_8_SUB": "Protest",
    
    # Messengers - Main Detection
    "detect_messenger": "Messenger Detection",
    "Messenger_Detection": "Messenger Detection",
    "messenger_Detection": "Messenger Detection",
    
    # Messengers - Subcategories
    "messenger_1": "Health Expert",
    "messenger_2": "Economic Expert",
    "messenger_3": "Security Expert",
    "messenger_4": "Legal Expert",
    "messenger_5": "Cultural Expert",
    "messenger_6": "Natural Scientist",
    "messenger_7": "Social Scientist",
    "messenger_8": "Activist",
    "messenger_9": "Public Official",
    "Messenger_1_SUB": "Health Expert",
    "Messenger_2_SUB": "Economic Expert",
    "Messenger_3_SUB": "Security Expert",
    "Messenger_4_SUB": "Legal Expert",
    "Messenger_5_SUB": "Cultural Expert",
    "Messenger_6_SUB": "Natural Scientist",
    "Messenger_7_SUB": "Social Scientist",
    "Messenger_8_SUB": "Activist",
    "Messenger_9_SUB": "Public Official",
    
    # Solutions - Main Detection
    "detect_solutions": "Solutions Detection",
    "Solutions_Detection": "Solutions Detection",
    "solutions_Detection": "Solutions Detection",
    
    # Solutions - Subcategories
    "solutions_1": "Mitigation Strategy",
    "solutions_2": "Adaptation Strategy",
    "Solutions_1_SUB": "Mitigation Strategy",
    "Solutions_2_SUB": "Adaptation Strategy",
    
    # Public Health Frame - Main Detection
    "detect_PBH": "Health Frame Detection",
    "Pbh_Detection": "Health Frame Detection",
    "PBH_Detection": "Health Frame Detection",
    
    # Public Health Frame - Subcategories
    "Pbh_1": "Negative Health Impacts",
    "Pbh_2": "Positive Health Impacts",
    "Pbh_3": "Health Co-benefits",
    "Pbh_4": "Health Sector Footprint",
    "Pbh_1_SUB": "Negative Health Impacts",
    "Pbh_2_SUB": "Positive Health Impacts",
    "Pbh_3_SUB": "Health Co-benefits",
    "Pbh_4_SUB": "Health Sector Footprint",
    "PBH_1_SUB": "Negative Health Impacts",
    "PBH_2_SUB": "Positive Health Impacts",
    "PBH_3_SUB": "Health Co-benefits",
    "PBH_4_SUB": "Health Sector Footprint",
    
    # Economic Frame - Main Detection
    "detect_ECO": "Economic Frame Detection",
    "Eco_Detection": "Economic Frame Detection",
    "ECO_Detection": "Economic Frame Detection",
    
    # Economic Frame - Subcategories
    "Eco_1": "Negative Economic Impacts",
    "Eco_2": "Positive Economic Impacts", 
    "Eco_3": "Costs of Climate Action",
    "Eco_4": "Benefits of Climate Action",
    "Eco_5": "Economic Sector Footprint",
    "Eco_1_SUB": "Negative Economic Impacts",
    "Eco_2_SUB": "Positive Economic Impacts",
    "Eco_3_SUB": "Costs of Climate Action",
    "Eco_4_SUB": "Benefits of Climate Action",
    "Eco_5_SUB": "Economic Sector Footprint",
    "ECO_1_SUB": "Negative Economic Impacts",
    "ECO_2_SUB": "Positive Economic Impacts",
    "ECO_3_SUB": "Costs of Climate Action",
    "ECO_4_SUB": "Benefits of Climate Action",
    "ECO_5_SUB": "Economic Sector Footprint",
    
    # Security Frame - Main Detection
    "detect_SECU": "Security Frame Detection",
    "Secu_Detection": "Security Frame Detection",
    "SECU_Detection": "Security Frame Detection",
    
    # Security Frame - Subcategories
    "Secu_1": "Military Disaster Response",
    "Secu_2": "Military Base Disruption",
    "Secu_3": "Climate-Driven Displacement",
    "Secu_4": "Resource Conflict",
    "Secu_5": "Defense Sector Footprint",
    "Secu_1_SUB": "Military Disaster Response",
    "Secu_2_SUB": "Military Base Disruption",
    "Secu_3_SUB": "Climate-Driven Displacement",
    "Secu_4_SUB": "Resource Conflict",
    "Secu_5_SUB": "Defense Sector Footprint",
    "SECU_1_SUB": "Military Disaster Response",
    "SECU_2_SUB": "Military Base Disruption",
    "SECU_3_SUB": "Climate-Driven Displacement",
    "SECU_4_SUB": "Resource Conflict",
    "SECU_5_SUB": "Defense Sector Footprint",
    
    # Justice Frame - Main Detection
    "detect_JUST": "Justice Frame Detection",
    "Just_Detection": "Justice Frame Detection",
    "JUST_Detection": "Justice Frame Detection",
    
    # Justice Frame - Subcategories
    "Just_1": "Winners & Losers",
    "Just_2": "North-South Responsibility",
    "Just_3": "Unequal Impacts",
    "Just_4": "Unequal Access",
    "Just_5": "Intergenerational Justice",
    "Just_1_SUB": "Winners & Losers",
    "Just_2_SUB": "North-South Responsibility",
    "Just_3_SUB": "Unequal Impacts",
    "Just_4_SUB": "Unequal Access",
    "Just_5_SUB": "Intergenerational Justice",
    "JUST_1_SUB": "Winners & Losers",
    "JUST_2_SUB": "North-South Responsibility",
    "JUST_3_SUB": "Unequal Impacts",
    "JUST_4_SUB": "Unequal Access",
    "JUST_5_SUB": "Intergenerational Justice",
    
    # Cultural Frame - Main Detection
    "detect_CULT": "Cultural Frame Detection",
    "Cult_Detection": "Cultural Frame Detection",
    "CULT_Detection": "Cultural Frame Detection",
    
    # Cultural Frame - Subcategories
    "Cult_1": "Artistic Representation",
    "Cult_2": "Event Disruption",
    "Cult_3": "Loss of Indigenous Practices",
    "Cult_4": "Cultural Sector Footprint",
    "Cult_1_SUB": "Artistic Representation",
    "Cult_2_SUB": "Event Disruption",
    "Cult_3_SUB": "Loss of Indigenous Practices",
    "Cult_4_SUB": "Cultural Sector Footprint",
    "CULT_1_SUB": "Artistic Representation",
    "CULT_2_SUB": "Event Disruption",
    "CULT_3_SUB": "Loss of Indigenous Practices",
    "CULT_4_SUB": "Cultural Sector Footprint",
    
    # Scientific Frame - Main Detection
    "detect_SCI": "Scientific Frame Detection",
    "Sci_Detection": "Scientific Frame Detection",
    "SCI_Detection": "Scientific Frame Detection",
    
    # Scientific Frame - Subcategories
    "Sci_1": "Scientific Controversy",
    "Sci_2": "Discovery & Innovation",
    "Sci_3": "Scientific Uncertainty",
    "Sci_4": "Scientific Certainty",
    "Sci_1_SUB": "Scientific Controversy",
    "Sci_2_SUB": "Discovery & Innovation",
    "Sci_3_SUB": "Scientific Uncertainty",
    "Sci_4_SUB": "Scientific Certainty",
    "SCI_1_SUB": "Scientific Controversy",
    "SCI_2_SUB": "Discovery & Innovation",
    "SCI_3_SUB": "Scientific Uncertainty",
    "SCI_4_SUB": "Scientific Certainty",
    
    # Environmental Frame - Main Detection
    "detect_ENVT": "Environmental Frame Detection",
    "Envt_Detection": "Environmental Frame Detection",
    "ENVT_Detection": "Environmental Frame Detection",
    
    # Environmental Frame - Subcategories
    "Envt_1": "Habitat Loss",
    "Envt_2": "Species Loss",
    "Envt_1_SUB": "Habitat Loss",
    "Envt_2_SUB": "Species Loss",
    "ENVT_1_SUB": "Habitat Loss",
    "ENVT_2_SUB": "Species Loss",
    
    # Political Frame - Main Detection
    "detect_POL": "Political Frame Detection",
    "Pol_Detection": "Political Frame Detection",
    "POL_Detection": "Political Frame Detection",
    
    # Political Frame - Subcategories
    "Pol_1": "Policy Measures",
    "Pol_2": "Political Debate",
    "Pol_3": "Political Positioning",
    "Pol_4": "Public Opinion",
    "Pol_1_SUB": "Policy Measures",
    "Pol_2_SUB": "Political Debate",
    "Pol_3_SUB": "Political Positioning",
    "Pol_4_SUB": "Public Opinion",
    "POL_1_SUB": "Policy Measures",
    "POL_2_SUB": "Political Debate",
    "POL_3_SUB": "Political Positioning",
    "POL_4_SUB": "Public Opinion",
    
    # Emotions and Urgency
    "detect_red": "Urgency/Alarmism",
    "RED_Detection": "Urgency/Alarmism",
    "emotion_pos": "Positive Emotion",
    "emotion_neu": "Neutral Emotion",
    "emotion_neg": "Negative Emotion",
    "Emotion:_Positive": "Positive Emotion",
    "Emotion:_Negative": "Negative Emotion",
    "Emotion:_Neutral": "Neutral Emotion",
    "Emotion_Classification": "Emotion Classification",
    
    # Extreme Weather
    "Extreme_Weather_Detection": "Extreme Weather Mentions",
    "detect_extreme_weather": "Extreme Weather Mentions",
    
    # Named Entities
    "PER": "Person Mentions",
    "ORG": "Organization Mentions",
    "LOC": "Location Mentions"
}

def get_readable_label(label: str) -> str:
    """
    Convert technical label to readable format using the mapping dictionary.
    Returns the original label if no mapping is found.
    """
    return LABEL_MAPPING.get(label, label)

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
    sentence_id, language, and date (needed to match DB rows and temporal analysis).
    `label` is read as the gold annotation set.
    """
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fo:
        for ln in fo:
            if not ln.strip():
                continue
            rec = json.loads(ln)

            # We do NOT use meta fields except doc_id, sentence_id, language, date
            meta = rec.get("meta", {})
            doc_id = meta.get("doc_id")
            sentence_id = meta.get("sentence_id")
            language = meta.get("language")
            date_str = meta.get("date")  # format: "mm-dd-yyyy"

            # The gold labels are in rec['label'] (a list)
            gold_labels = set(rec.get("label", []))

            entries.append({
                "doc_id":      doc_id,
                "sentence_id": sentence_id,
                "language":    language,
                "date":        date_str,
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
#                          TEMPORAL DRIFT ANALYSIS                           #
##############################################################################

def parse_date(date_str: str) -> Tuple[int, str]:
    """
    Parse date string in format "mm-dd-yyyy" and return year and decade.
    Returns (year, decade_str) or (None, None) if parsing fails.
    """
    if not date_str:
        return None, None
    
    try:
        # Parse date in format "mm-dd-yyyy"
        date_obj = datetime.strptime(date_str, "%m-%d-%Y")
        year = date_obj.year
        decade = f"{(year // 10) * 10}s"  # e.g., "1990s", "2000s"
        return year, decade
    except (ValueError, TypeError):
        return None, None

def get_time_period(date_str: str, period_type: str = "decade") -> str:
    """
    Get time period from date string.
    period_type can be "year", "decade", or "5year"
    """
    year, decade = parse_date(date_str)
    
    if year is None:
        return "Unknown"
    
    if period_type == "year":
        return str(year)
    elif period_type == "decade":
        return decade
    elif period_type == "5year":
        # Group into 5-year periods
        period_start = (year // 5) * 5
        period_end = period_start + 4
        return f"{period_start}-{period_end}"
    else:
        return "Unknown"

class TemporalConfusionDict(defaultdict):
    """
    Extended confusion dictionary that also tracks temporal information.
    Structure: (label, class, language, time_period) -> {"TP": int, "FP": int, "FN": int}
    """
    def __init__(self):
        super().__init__(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    def increment(self, label: str, cls: int, lang: str, period: str,
                  tp: int=0, fp: int=0, fn: int=0) -> None:
        key = (label, cls, lang, period)
        self[key]["TP"] += tp
        self[key]["FP"] += fp
        self[key]["FN"] += fn
    
    def update_counts(self, label: str, gold: int, pred: int, lang: str, period: str) -> None:
        """
        Update confusion matrix for temporal analysis.
        """
        # Handle class=1
        if gold == 1 and pred == 1:
            self.increment(label, 1, lang, period, tp=1)
            self.increment(label, 1, "ALL", period, tp=1)
        elif gold == 1 and pred == 0:
            self.increment(label, 1, lang, period, fn=1)
            self.increment(label, 1, "ALL", period, fn=1)
        elif gold == 0 and pred == 1:
            self.increment(label, 1, lang, period, fp=1)
            self.increment(label, 1, "ALL", period, fp=1)
        
        # Handle class=0
        gold_0 = 1 - gold
        pred_0 = 1 - pred
        if gold_0 == 1 and pred_0 == 1:
            self.increment(label, 0, lang, period, tp=1)
            self.increment(label, 0, "ALL", period, tp=1)
        elif gold_0 == 1 and pred_0 == 0:
            self.increment(label, 0, lang, period, fn=1)
            self.increment(label, 0, "ALL", period, fn=1)
        elif gold_0 == 0 and pred_0 == 1:
            self.increment(label, 0, lang, period, fp=1)
            self.increment(label, 0, "ALL", period, fp=1)

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

    # Build confusion dictionaries
    conf = ConfusionDict()
    temporal_conf = TemporalConfusionDict()

    # Process entries directly without parallelization for better performance on small datasets
    print("[INFO] Processing gold annotations...")
    
    for entry in tqdm(gold_entries, desc="Scoring"):
        key = (entry["doc_id"], entry["sentence_id"])
        lang = entry["language"] or "UNK"
        gold_set = entry["gold_labels"]
        date_str = entry.get("date")
        
        if key not in pred_lookup:
            continue
            
        pred_row = pred_lookup[key]
        
        # Get time period for temporal analysis
        decade = get_time_period(date_str, "decade")
        
        # For each label column in the DB, compare to gold
        for label in annot_cols:
            pred_val = pred_row.get(label)
            # If DB has no prediction (NaN), skip
            if pd.isna(pred_val):
                continue
            
            # gold=1 if this label is in the gold set, else 0
            gold_bin = 1 if label in gold_set else 0
            pred_bin = int(pred_val)
            
            # Update regular confusion
            conf.update_counts(
                label=label,
                gold=gold_bin,
                pred=pred_bin,
                lang=lang
            )
            
            # Update temporal confusion for different time periods
            if date_str:  # Only if we have a valid date
                temporal_conf.update_counts(
                    label=label,
                    gold=gold_bin,
                    pred=pred_bin,
                    lang=lang,
                    period=decade
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
                "label": get_readable_label(lab),
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

    # Step 3: Write out the final CSV with improved formatting
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort rows for better presentation: by label (ALL last), then by language
    metric_rows.sort(key=lambda x: (
        x["label"] == "ALL",  # Put ALL at the end
        x["label"],
        {"EN": 0, "FR": 1, "ALL": 2}.get(x["language"], 3)  # Order languages
    ))
    
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fo:
        # Reorganized fieldnames for clearer presentation
        fieldnames = [
            "label", "language",
            # Core metrics
            "f1_macro", "f1_micro", "f1_weighted",
            "precision_macro", "recall_macro",
            "precision_micro", "recall_micro", 
            "precision_weighted", "recall_weighted",
            # Class-specific metrics
            "f1_1", "precision_1", "recall_1", "support_1",
            "f1_0", "precision_0", "recall_0", "support_0",
            # Raw counts (optional, can be removed if too detailed)
            "TP_1", "FP_1", "FN_1",
            "TP_0", "FP_0", "FN_0",
            "support_total"
        ]
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()

        for row in metric_rows:
            # Round float fields to 3 decimals for cleaner presentation
            for k in fieldnames:
                val = row.get(k)
                if isinstance(val, float):
                    row[k] = f"{val:.3f}"
            writer.writerow(row)

    print(f"[INFO] Metrics CSV saved → {OUTPUT_CSV}")
    
    # ========================================================================
    # TEMPORAL DRIFT ANALYSIS
    # ========================================================================
    print("\n[INFO] Computing temporal drift metrics...")
    
    # Get unique time periods and labels from temporal confusion
    unique_periods = sorted(set(period for (_, _, _, period) in temporal_conf.keys()))
    unique_labels_temporal = sorted(set(lab for (lab, _, _, _) in temporal_conf.keys()))
    
    temporal_rows: List[Dict[str, Any]] = []
    
    # For each label and time period, calculate metrics
    for lab in unique_labels_temporal:
        for period in unique_periods:
            if period == "Unknown":
                continue  # Skip entries without valid dates
            
            # Calculate for each language
            for lng in ("EN", "FR", "ALL"):
                # Get stats for class=1
                tp_1 = temporal_conf[(lab, 1, lng, period)]["TP"]
                fp_1 = temporal_conf[(lab, 1, lng, period)]["FP"]
                fn_1 = temporal_conf[(lab, 1, lng, period)]["FN"]
                p_1, r_1, f1_1 = prf(tp_1, fp_1, fn_1)
                s_1 = tp_1 + fn_1
                
                # Get stats for class=0
                tp_0 = temporal_conf[(lab, 0, lng, period)]["TP"]
                fp_0 = temporal_conf[(lab, 0, lng, period)]["FP"]
                fn_0 = temporal_conf[(lab, 0, lng, period)]["FN"]
                p_0, r_0, f0_0 = prf(tp_0, fp_0, fn_0)
                s_0 = tp_0 + fn_0
                
                # Calculate macro F1
                macro_f1 = (f1_1 + f0_0) / 2
                
                # Only add row if there's actual data
                if s_1 + s_0 > 0:
                    temporal_rows.append({
                        "label": get_readable_label(lab),
                        "language": lng,
                        "time_period": period,
                        "precision_1": p_1,
                        "recall_1": r_1,
                        "f1_1": f1_1,
                        "support_1": s_1,
                        "precision_0": p_0,
                        "recall_0": r_0,
                        "f1_0": f0_0,
                        "support_0": s_0,
                        "macro_f1": macro_f1,
                        "total_support": s_1 + s_0
                    })
    
    # Also create aggregated "ALL" label rows for each time period
    overall_temporal_conf = TemporalConfusionDict()
    
    for (lab, cls, lng, period), dct in temporal_conf.items():
        if lab == "ALL":
            continue
        # Add to overall statistics
        overall_temporal_conf.increment(
            "ALL", cls, lng, period,
            tp=dct["TP"], fp=dct["FP"], fn=dct["FN"]
        )
    
    # Add overall rows
    for period in unique_periods:
        if period == "Unknown":
            continue
        
        for lng in ("EN", "FR", "ALL"):
            # Get overall stats for class=1
            atp_1 = overall_temporal_conf[("ALL", 1, lng, period)]["TP"]
            afp_1 = overall_temporal_conf[("ALL", 1, lng, period)]["FP"]
            afn_1 = overall_temporal_conf[("ALL", 1, lng, period)]["FN"]
            p_1, r_1, f1_1 = prf(atp_1, afp_1, afn_1)
            s_1 = atp_1 + afn_1
            
            # Get overall stats for class=0
            atp_0 = overall_temporal_conf[("ALL", 0, lng, period)]["TP"]
            afp_0 = overall_temporal_conf[("ALL", 0, lng, period)]["FP"]
            afn_0 = overall_temporal_conf[("ALL", 0, lng, period)]["FN"]
            p_0, r_0, f0_0 = prf(atp_0, afp_0, afn_0)
            s_0 = atp_0 + afn_0
            
            # Calculate macro F1
            macro_f1 = (f1_1 + f0_0) / 2
            
            if s_1 + s_0 > 0:
                temporal_rows.append({
                    "label": "ALL",
                    "language": lng,
                    "time_period": period,
                    "precision_1": p_1,
                    "recall_1": r_1,
                    "f1_1": f1_1,
                    "support_1": s_1,
                    "precision_0": p_0,
                    "recall_0": r_0,
                    "f1_0": f0_0,
                    "support_0": s_0,
                    "macro_f1": macro_f1,
                    "total_support": s_1 + s_0
                })
    
    # Write temporal drift CSV with improved formatting
    TEMPORAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort for better presentation
    temporal_rows.sort(key=lambda x: (
        x["label"] == "ALL",  # ALL labels at the end
        x["label"],
        x["time_period"],
        {"EN": 0, "FR": 1, "ALL": 2}.get(x["language"], 3)
    ))
    
    with TEMPORAL_CSV.open("w", newline="", encoding="utf-8") as fo:
        fieldnames = [
            "label", "time_period", "language",
            # Primary metric
            "macro_f1",
            # Class-specific metrics
            "f1_1", "precision_1", "recall_1", "support_1",
            "f1_0", "precision_0", "recall_0", "support_0",
            "total_support"
        ]
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in temporal_rows:
            # Round float fields to 3 decimals
            for k in fieldnames:
                val = row.get(k)
                if isinstance(val, float):
                    row[k] = f"{val:.3f}"
            writer.writerow(row)
    
    print(f"[INFO] Temporal drift metrics CSV saved → {TEMPORAL_CSV}")
    print(f"[INFO] Found {len(unique_periods)-1} time periods (excluding 'Unknown')")
    print(f"[INFO] Analyzed {len(unique_labels_temporal)} labels across time")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Find the overall "ALL" metrics for summary
    overall_metrics = [row for row in metric_rows 
                      if row["label"] == "ALL" and row["language"] == "ALL"]
    
    if overall_metrics:
        overall = overall_metrics[0]
        print(f"\nOVERALL PERFORMANCE (All Labels, All Languages):")
        print(f"  • F1 Score (Macro):    {float(overall['f1_macro']):.3f}")
        print(f"  • F1 Score (Micro):    {float(overall['f1_micro']):.3f}")
        print(f"  • F1 Score (Weighted): {float(overall['f1_weighted']):.3f}")
        print(f"  • Precision (Macro):   {float(overall['precision_macro']):.3f}")
        print(f"  • Recall (Macro):      {float(overall['recall_macro']):.3f}")
        print(f"  • Total Support:       {int(overall['support_total']):,}")
    
    # Calculate average performance by language
    en_metrics = [row for row in metric_rows 
                  if row["label"] == "ALL" and row["language"] == "EN"]
    fr_metrics = [row for row in metric_rows 
                  if row["label"] == "ALL" and row["language"] == "FR"]
    
    if en_metrics:
        print(f"\nENGLISH Performance:")
        print(f"  • F1 Score (Macro):    {float(en_metrics[0]['f1_macro']):.3f}")
        print(f"  • F1 Score (Weighted): {float(en_metrics[0]['f1_weighted']):.3f}")
    
    if fr_metrics:
        print(f"\nFRENCH Performance:")
        print(f"  • F1 Score (Macro):    {float(fr_metrics[0]['f1_macro']):.3f}")
        print(f"  • F1 Score (Weighted): {float(fr_metrics[0]['f1_weighted']):.3f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
