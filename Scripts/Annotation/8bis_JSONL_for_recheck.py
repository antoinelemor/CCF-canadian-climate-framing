# -*- coding: utf-8 -*-
"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
8bis_JSONL_for_recheck.py

MAIN OBJECTIVE:
---------------
This script regenerates a JSONL file named 'sentences_to_recheck_multiling_bis.jsonl'
using the exact same sentences that were previously saved in 
'sentences_to_recheck_multiling.jsonl'. However, the updated annotations 
(labels) are fetched from the newly re-annotated CSV 
'CCF.media_processed_texts_annotated.csv'. 

In other words, the text strings remain the same as in the original 
'sentences_to_recheck_multiling.jsonl', but the associated category labels 
(for each sentence) are updated to reflect any new or revised model annotations.

Dependencies:
-------------
- os
- sys
- json
- pandas
- numpy

MAIN FEATURES:
--------------
1) Loads the original JSONL 'sentences_to_recheck_multiling.jsonl' 
   to retrieve the exact set of sentences (and associated metadata if desired).
2) Loads 'CCF.media_processed_texts_annotated.csv', which contains 
   the *current* annotation columns (0 or 1).
3) Identifies which columns in the CSV correspond to annotation categories.
4) For each sentence from the original JSONL, it looks up the corresponding 
   row in the CSV and rebuilds the JSON entry with:
   - "text": the exact sentence
   - "label": all annotation columns (categories) that are == 1
   - "meta": any additional metadata from the CSV (article info, etc.)
5) Writes out these updated entries to 'sentences_to_recheck_multiling_bis.jsonl'.

Author:
-------
Antoine Lemor
"""

import os
import sys
import json
import pandas as pd
import numpy as np

##############################################################################
#                 A. PATH AND FILE DEFINITIONS
##############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The *original* JSONL containing the sentences we want to re-check
OLD_JSONL = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", 
    "sentences_to_recheck_multiling.jsonl"
)

# The *new* JSONL we want to produce
NEW_JSONL = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", 
    "sentences_to_recheck_multiling_bis.jsonl"
)

# The updated annotated CSV (after re-training & re-annotation)
UPDATED_CSV = os.path.join(
    BASE_DIR, "..", "..", "Database", "Database", 
    "CCF.media_processed_texts_annotated.csv"
)

##############################################################################
#          B. UTILITY FUNCTION TO IDENTIFY ANNOTATION COLUMNS
##############################################################################
def get_annotation_columns(df):
    """
    Returns a list of potential annotation columns, excluding typical
    meta columns like 'language', 'sentences', 'id_article', etc.
    We assume annotation columns have numeric 0/1 values.
    """
    exclude_cols = {
        "language", "sentences", "id_article", "Unnamed: 0",
        "doc_ID", "sentence_id", "words_count_updated", "words_count"
    }

    annotation_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # We check if it's numeric and if it contains at least one '1'
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].sum(skipna=True) > 0:
                annotation_cols.append(col)
    return annotation_cols

##############################################################################
#       C. FUNCTION TO BUILD A JSONL ENTRY FROM A CSV ROW
##############################################################################
def build_jsonl_entry(row, annotation_cols):
    """
    Builds a JSONL entry of the form:
    {
      "text": <sentence>,
      "label": [list_of_positive_categories],
      "meta": { "any": "other CSV info" }
    }
    """
    text = row["sentences"]

    # Identify positive categories (==1)
    positive_labels = []
    for col in annotation_cols:
        val = row[col]
        if pd.notna(val) and val == 1:
            positive_labels.append(col)

    # Build meta (all columns except 'sentences' and annotation columns)
    meta = {}
    for c in row.index:
        if c == "sentences" or c in annotation_cols:
            continue
        # Convert NaN to None
        v = row[c]
        if isinstance(v, float) and pd.isna(v):
            meta[c] = None
        else:
            meta[c] = v

    entry = {
        "text": text,
        "label": positive_labels,
        "meta": meta
    }
    return entry

##############################################################################
#                         D. MAIN LOGIC
##############################################################################
def main():
    print("Script started.")  # Debug
    # ----------------------------------------------------------------------
    # 1) Check existence of essential files
    # ----------------------------------------------------------------------
    if not os.path.exists(OLD_JSONL):
        print(f"[ERROR] Original JSONL not found: {OLD_JSONL}")
        sys.exit(1)
    if not os.path.exists(UPDATED_CSV):
        print(f"[ERROR] Updated annotated CSV not found: {UPDATED_CSV}")
        sys.exit(1)

    # ----------------------------------------------------------------------
    # 2) Load the CSV into a DataFrame
    # ----------------------------------------------------------------------
    print("[INFO] Loading updated CSV with re-annotations...")
    df = pd.read_csv(UPDATED_CSV, low_memory=False)
    print(f"[INFO] CSV loaded. {len(df)} rows total.")

    # Convert any annotation columns to integer 0/1 just to be safe
    print("[INFO] Identifying annotation columns...")
    annotation_cols = get_annotation_columns(df)
    print(f"     -> Found {len(annotation_cols)} potential annotation columns.")
    for c in annotation_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # For faster lookup, create a dictionary keyed by sentence text -> row(s)
    # Because the same sentence might appear multiple times in the dataset, 
    # we must decide how to handle duplicates. We'll assume the first match 
    # is enough. Alternatively, we can store them in a list if you suspect
    # duplicates with different metadata in your dataset.
    print("[INFO] Building a dictionary for text-based lookup in the CSV...")
    text_lookup = {}
    for idx, row in df.iterrows():
        sentence = row["sentences"]
        # Optional: store the first occurrence only
        if sentence not in text_lookup:
            text_lookup[sentence] = row

    # ----------------------------------------------------------------------
    # 3) Load the *old* JSONL to get the *exact same* text entries
    # ----------------------------------------------------------------------
    print("[INFO] Loading original JSONL for re-checking:", OLD_JSONL)
    with open(OLD_JSONL, 'r', encoding='utf-8') as f:
        old_entries = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Found {len(old_entries)} entries in the old JSONL.")
    
    # ----------------------------------------------------------------------
    # 4) For each old entry, fetch updated annotation from the CSV
    # ----------------------------------------------------------------------
    new_entries = []
    missing_count = 0
    for e in old_entries:
        txt = e.get("text", "")
        if txt in text_lookup:
            row_data = text_lookup[txt]
            # Build a new entry with updated labels
            new_entry = build_jsonl_entry(row_data, annotation_cols)
            new_entries.append(new_entry)
        else:
            # If not found, we keep the old entry's text but set label=[]
            # or we skip. Let's store it with empty label to highlight missing.
            missing_count += 1
            new_entry = {
                "text": txt,
                "label": [],
                "meta": {"missing_in_csv": True}
            }
            new_entries.append(new_entry)

    print(f"[INFO] Successfully matched {len(new_entries) - missing_count} texts.")
    print(f"[INFO] Could not find {missing_count} texts in the updated CSV. "
          "They will have empty annotations in the new JSONL.")

    # ----------------------------------------------------------------------
    # 5) Write out the new JSONL
    # ----------------------------------------------------------------------
    print("[INFO] Writing new JSONL to:", NEW_JSONL)
    with open(NEW_JSONL, 'w', encoding='utf-8') as out_f:
        for entry in new_entries:
            try:
                line = json.dumps(entry, ensure_ascii=False)
                out_f.write(line + "\n")
            except (TypeError, ValueError) as exc:
                print(f"[ERROR] Could not serialize entry: {entry}\n    Reason: {exc}")

    print("[INFO] Process completed.")
    print("Script ended.")  # Debug


if __name__ == "__main__":
    main()
