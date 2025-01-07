"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7_Produce_JSON_for_Recheck.py

MAIN OBJECTIVE:
-------------------
This script generates a new JSONL dataset to manually revalidate 
the effectiveness of annotations after model processing. It specifically targets 
underrepresented categories to verify the quality of models 
in all scenarios.

Dependencies:
-------------
- os
- json
- random
- pandas
- math

MAIN FEATURES:
----------------------------
1) Reads the annotated database (e.g., CCF.media_processed_texts_annotated.csv).
2) Detects and filters out sentences already used in previous manual 
   annotations (sentences_to_annotate_EN.jsonl, sentences_to_annotate_FR.jsonl).
3) Randomly selects with oversampling (or "weighting") 
   of underrepresented categories to ensure a more equitable coverage of all classes 
   (detection, sub, etc.).
4) Maintains a 50/50 distribution between English and French languages.
5) Produces a multilingual JSONL file where each entry contains:
   - "text": the sentence itself,
   - "label": the list of active categories (==1) for that sentence,
   - "meta": a dictionary containing all article metadata 
             (e.g., title, source, date, etc.).
         
Author : 
--------
Antoine Lemor
"""

import os
import json
import random
import pandas as pd
import math

##############################################################################
#                      A. CONSTANTS AND PATHS
##############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Annotated CSV file output from 6_Annotate.py
ANNOTATED_CSV = os.path.join(
    BASE_DIR, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv"
)

# Directory containing old manual annotations
MANUAL_ANNOTATIONS_DIR = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL"
)

# JSONL files already manually annotated, which we want to exclude
MANUAL_ANNOTATIONS_EN = os.path.join(MANUAL_ANNOTATIONS_DIR, "sentences_to_annotate_EN.jsonl")
MANUAL_ANNOTATIONS_FR = os.path.join(MANUAL_ANNOTATIONS_DIR, "sentences_to_annotate_FR.jsonl")

# Output file (a single multilingual JSONL, 50/50 EN/FR)
OUTPUT_JSONL = os.path.join(
    MANUAL_ANNOTATIONS_DIR, "sentences_to_recheck_multiling.jsonl"
)

# Total number of sentences to annotate (adjustable)
NB_SENTENCES_TOTAL = 1000  # 200 EN + 200 FR, for example


##############################################################################
#                      B. UTILITY FUNCTIONS
##############################################################################
def load_already_annotated_texts(jsonl_path):
    """
    Loads a JSONL file previously used for manual annotations
    and returns the set of 'text' (sentences) it contains, 
    to exclude them.
    """
    if not os.path.exists(jsonl_path):
        return set()

    texts = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                txt = data.get("text", "")
                texts.add(txt)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decoding failed for line: {line}\nError: {e}")
    return texts


def identify_annotation_columns(df):
    """
    Identifies the columns corresponding to annotation categories.
    Typically, we can take all binary columns (0/1) 
    excluding clearly metadata columns (e.g., 'language', 'sentences', etc.).
    Adjust as per the actual CSV structure.
    """
    excluded_cols = {"language", "sentences", "id_article", "Unnamed: 0"}
    annotation_cols = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        # Check if the content is 0/1 or NaN/1, etc.
        # Simple heuristic: numeric type + at least one '1' in it
        if pd.api.types.is_numeric_dtype(df[col]):
            nb_ones = df[col].sum(skipna=True)
            if nb_ones > 0:
                annotation_cols.append(col)

    return annotation_cols


def get_underrepresented_categories(df, annotation_cols, threshold=50):
    """
    Identifies underrepresented categories, for example those 
    with fewer than 'threshold' positive rows (==1).
    Returns a list of these categories.
    """
    underrepresented = []
    for col in annotation_cols:
        nb_positives = df[col].sum(skipna=True)
        if nb_positives < threshold:
            underrepresented.append(col)
    return underrepresented


def build_doccano_jsonl_entry(row, annotation_cols):
    """
    Constructs a Doccano-compliant JSONL entry.
    - "text": the sentence itself,
    - "label": the list of active categories (==1) for that sentence,
    - "meta": a dictionary containing all article metadata
               (e.g., title, source, date, etc.).
               Fields corresponding to non-positive annotations are excluded.
               NaN values are replaced with null.
    """
    text = row["sentences"]

    # 1) Identify positive labels
    active_labels = [col for col in annotation_cols if pd.notna(row[col]) and row[col] == 1]

    # 2) Build the meta dictionary excluding annotations
    meta = {}
    for col in row.index:
        if col == "sentences":
            continue
        if col in annotation_cols:
            continue
        value = row[col]
        # Replace NaN with None
        if isinstance(value, float) and math.isnan(value):
            meta[col] = None
        else:
            meta[col] = value

    # 3) Validate JSON serialization
    try:
        json.dumps(meta)  # Check that 'meta' is serializable
    except (TypeError, ValueError) as e:
        print(f"[ERROR] Meta not serializable for text: {text}\nError: {e}")
        meta = None  # Or handle otherwise as needed

    # 4) Build the JSON entry
    entry = {
        "text": text,
        "label": active_labels,
        "meta": meta
    }
    return entry


##############################################################################
#                    C. MAIN GENERATION FUNCTION
##############################################################################
def main():
    # ----------------------------------------------------------------------
    # 1) Load the annotated CSV
    # ----------------------------------------------------------------------
    print("[INFO] Loading the annotated CSV...")
    if not os.path.exists(ANNOTATED_CSV):
        raise FileNotFoundError(f"File not found: {ANNOTATED_CSV}")

    df = pd.read_csv(ANNOTATED_CSV, low_memory=False)
    print(f" -> {len(df)} rows loaded.")

    # ----------------------------------------------------------------------
    # 2) Identify annotation columns
    # ----------------------------------------------------------------------
    annotation_cols = identify_annotation_columns(df)
    print(f"[INFO] Detected annotation columns: {annotation_cols}")

    # ----------------------------------------------------------------------
    # 3) Exclude sentences already manually annotated
    # ----------------------------------------------------------------------
    print("[INFO] Excluding sentences already manually annotated...")
    en_annotated = load_already_annotated_texts(MANUAL_ANNOTATIONS_EN)
    fr_annotated = load_already_annotated_texts(MANUAL_ANNOTATIONS_FR)

    initial_len = len(df)
    df = df[~(
        ((df["language"] == "EN") & (df["sentences"].isin(en_annotated))) |
        ((df["language"] == "FR") & (df["sentences"].isin(fr_annotated)))
    )].copy()
    print(f" -> {initial_len - len(df)} rows excluded. {len(df)} rows remaining.")

    # ----------------------------------------------------------------------
    # 4) Identify underrepresented categories
    # ----------------------------------------------------------------------
    # For example, we set a threshold of "threshold=50" (adjustable). 
    # We can then oversample (or force the inclusion) 
    # of a larger number of sentences from these categories.
    under_cat = get_underrepresented_categories(df, annotation_cols, threshold=50)
    print(f"[INFO] Underrepresented categories (less than 50 occurrences): {under_cat}")

    # ----------------------------------------------------------------------
    # 5) Sample composition in two steps:
    #    a) Oversample positive sentences for under_cat
    #    b) Complete the rest to reach NB_SENTENCES_TOTAL, 
    #       maintaining 50% EN and 50% FR
    # ----------------------------------------------------------------------

    # a) Oversample underrepresented categories
    #    We retrieve all sentences that have at least 
    #    one underrepresented category = 1 (union).
    #    Then, we can take a certain number at random.
    df_under = df.copy()
    mask_under = False
    for cat in under_cat:
        mask_under |= (df_under[cat] == 1)
    df_candidates_under = df_under[mask_under]

    # To avoid taking only underrepresented categories, we will limit 
    # this selection if it is too large. For example, we can take 
    # min(len(df_candidates_under), 200) if we want to limit to 200 
    # (or a certain ratio).
    random_under_limit = max(50, int(0.5 * NB_SENTENCES_TOTAL))  # 50% of the total, for example.
    if len(df_candidates_under) > random_under_limit:
        df_candidates_under = df_candidates_under.sample(random_under_limit, random_state=42)
    # We now have a first set relatively oversampling the under_cat
    # without exceeding half of the final target sample.

    # b) Complete the sample (in addition to df_candidates_under) 
    #    from the rest of the DF. 
    #    We want NB_SENTENCES_TOTAL - len(df_candidates_under) sentences in total.
    df_rest = df.drop(df_candidates_under.index, errors="ignore")
    nb_rest_needed = NB_SENTENCES_TOTAL - len(df_candidates_under)
    if nb_rest_needed < 0:
        nb_rest_needed = 0

    if len(df_rest) > nb_rest_needed:
        df_rest = df_rest.sample(nb_rest_needed, random_state=42)

    # c) Combine
    df_final = pd.concat([df_candidates_under, df_rest], axis=0)
    # Shuffle the set to avoid any particular order
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 6) Ensure a 50/50 EN/FR Split
    #    - Strategy: Separate into two dataframes (EN and FR) and balance as 
    #      close to 50/50 as possible, randomly.
    # ----------------------------------------------------------------------
    df_en = df_final[df_final["language"] == "EN"].copy()
    df_fr = df_final[df_final["language"] == "FR"].copy()

    # Maximum number per language: NB_SENTENCES_TOTAL // 2 (e.g., 200)
    half_target = NB_SENTENCES_TOTAL // 2

    if len(df_en) > half_target:
        df_en = df_en.sample(half_target, random_state=42)
    if len(df_fr) > half_target:
        df_fr = df_fr.sample(half_target, random_state=42)

    # If one of the two is smaller than half_target, 
    # we can take it in its entirety and adjust the other
    final_en = len(df_en)
    final_fr = len(df_fr)
    # We can leave it as is if we want exactly NB_SENTENCES_TOTAL 
    # or we can accept a slightly smaller total if one of the two 
    # does not reach half_target.
    # Here we choose flexibility: we take everything possible 
    # up to 50/50. 
    # => We readjust the other group to have the same size.
    if final_en < half_target and final_en > 0:
        # Reduce FR to the same number
        df_fr = df_fr.sample(final_en, random_state=42)
    elif final_fr < half_target and final_fr > 0:
        # Reduce EN to the same number
        df_en = df_en.sample(final_fr, random_state=42)

    # Final reconstruction
    df_final_balanced = pd.concat([df_en, df_fr], axis=0)
    # Final shuffle
    df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[INFO] Final sample: {len(df_final_balanced)} rows.")
    print(f"       -> EN: {(df_final_balanced['language'] == 'EN').sum()}")
    print(f"       -> FR: {(df_final_balanced['language'] == 'FR').sum()}")

    # ----------------------------------------------------------------------
    # 7) Produce the JSONL
    # ----------------------------------------------------------------------
    print("[INFO] Writing the final multilingual JSONL...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for idx, row in df_final_balanced.iterrows():
            entry = build_doccano_jsonl_entry(row, annotation_cols)
            try:
                json_line = json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
                out_f.write(json_line + "\n")
            except (TypeError, ValueError) as e:
                print(f"[ERROR] Serialization failed for entry at index {idx}: {e}")

    print(f"[INFO] JSONL file created: {OUTPUT_JSONL}")
    print("[INFO] End of script.")


if __name__ == "__main__":
    main()
