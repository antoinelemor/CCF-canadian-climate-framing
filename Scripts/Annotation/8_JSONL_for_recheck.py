"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
8_Produce_JSON_for_Recheck.py

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
3) Randomly selects with oversampling (or "weighting") **each** underrepresented 
   category to ensure a more equitable coverage of all classes (detection, sub, etc.).
4) Ensures a strict 50/50 distribution between English and French, 
   to reach exactly NB_SENTENCES_TOTAL = 1000 sentences total 
   (500 EN + 500 FR), if enough data is available in each language subset.
   If a category has fewer than 10 total positive rows in the entire CSV, 
   include all those rows in the final sample (for both EN and FR subsets)
5) Produces a multilingual JSONL file where each entry contains:
   - "text": the sentence itself,
   - "label": the list of active categories (==1) for that sentence,
   - "meta": a dictionary containing all article metadata (e.g., title, source, date, etc.).
6) Prints out the proportion of underrepresented categories 
   and the proportion of all categories in the final JSONL sample.

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

# Output file (a single multilingual JSONL, strictly 50/50 EN/FR if possible)
OUTPUT_JSONL = os.path.join(
    MANUAL_ANNOTATIONS_DIR, "sentences_to_recheck_multiling.jsonl"
)

# Total number of sentences to annotate (adjustable)
NB_SENTENCES_TOTAL = 1000  # ideally 500 EN + 500 FR if data allows

# For categories that are underrepresented but have >=10 rows in a given language,
# we sample up to this number from that category. 
PER_CAT_LIMIT = 50

# If a category has fewer than this many positive rows overall, we consider it 
# underrepresented. (Adjust as needed.)
UNDERREP_THRESHOLD = 5000

# If a category has fewer than MIN_POSITIVE_THRESHOLD (i.e., <10) positives 
# in the entire dataset, we include *all* of them unconditionally. 
# This requirement overrides the half-of-n_target logic.
MIN_POSITIVE_THRESHOLD = 10


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
    Identifies columns corresponding to annotation categories (binary 0/1),
    excluding metadata columns like 'language', 'sentences', 'doc_ID', etc.
    Adjust this set as needed based on the CSV structure.
    """
    excluded_cols = {
        "language", "sentences", "id_article", "Unnamed: 0",
        "doc_ID", "sentence_id", "words_count_updated", "words_count"
    }

    annotation_cols = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        # We'll do a numeric cast step below, but let's see if pandas 
        # already recognizes it as numeric
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
    Constructs a Doccano-compliant JSONL entry:
      - "text": the sentence itself,
      - "label": the list of active categories (==1) for that sentence,
      - "meta": a dictionary of non-annotation metadata columns.
    """
    text = row["sentences"]

    # Identify positive labels
    active_labels = []
    for col in annotation_cols:
        val = row[col]
        # We cast to int previously, so val==1 suffices
        if pd.notna(val) and val == 1:
            active_labels.append(col)

    # Build metadata dictionary
    meta = {}
    for col in row.index:
        # Exclude text column and annotation columns
        if col == "sentences" or col in annotation_cols:
            continue
        # Replace NaN with None
        value = row[col]
        if isinstance(value, float) and math.isnan(value):
            meta[col] = None
        else:
            meta[col] = value

    # Validate JSON serialization
    try:
        json.dumps(meta)
    except (TypeError, ValueError) as e:
        print(f"[ERROR] Meta not serializable for text: {text}\nError: {e}")
        meta = None

    return {
        "text": text,
        "label": active_labels,
        "meta": meta
    }


##############################################################################
#    C. SAMPLING FUNCTION (PER-LANGUAGE) WITH GUARANTEED COVERAGE
##############################################################################
def sample_language_subset_with_guaranteed_coverage(
    df_lang,
    annotation_cols,
    n_target,
    under_cat,
    per_cat_limit=PER_CAT_LIMIT,
    random_state=42,
    min_positive_threshold=MIN_POSITIVE_THRESHOLD
):
    """
    Samples exactly n_target rows from df_lang, ensuring each underrepresented
    category is included. Specifically:

    1) For each category in under_cat, we find df_cat = rows with cat==1.
       - If df_cat has < min_positive_threshold (e.g. <10) rows in the entire DB,
         we take them **all** (since the user wants to keep every instance 
         of extremely rare categories).
       - Else, we sample up to 'per_cat_limit' from df_cat. 
         (e.g., if per_cat_limit=50, we sample up to 50 rows for that cat.)
    2) We union all these sets (so we have guaranteed coverage of each cat).
    3) If the union alone is bigger than n_target//2, we downsample it
       to n_target//2, but we **never** remove the extremely rare categories 
       that have < min_positive_threshold. We only remove the "larger" cat 
       rows if needed.
    4) We then fill the remainder from the rest of df_lang (the "normal" rows
       not in the union). This ensures we reach n_target total, or fewer if
       df_lang is too small.

    This approach ensures that if a category has fewer than min_positive_threshold 
    positives, all of them are included and can't be dropped by half-of-n_target 
    logic, guaranteeing they appear in the final sample.

    Returns a DataFrame with up to n_target rows from df_lang.
    """
    if len(df_lang) == 0 or n_target <= 0:
        return df_lang.head(0)  # empty

    # 1) Collect rows for each underrepresented category individually
    cat_dfs_for_union = []
    half_limit = n_target // 2
    # We'll track separate sets for "extremely rare categories" (below threshold) 
    # vs. "normal underrepresented" (above min_positive_threshold).
    extremely_rare_rows = []

    for cat in under_cat:
        if cat not in df_lang.columns:
            continue

        df_cat = df_lang[df_lang[cat] == 1]
        cat_total = len(df_cat)

        # If there are no rows for that cat in this language subset, skip
        if cat_total == 0:
            continue

        # If cat_total < min_positive_threshold, we take them all unconditionally
        if cat_total < min_positive_threshold:
            extremely_rare_rows.append(df_cat)
        else:
            # Otherwise, sample up to per_cat_limit
            if cat_total > per_cat_limit:
                df_cat_sample = df_cat.sample(per_cat_limit, random_state=random_state)
            else:
                df_cat_sample = df_cat
            cat_dfs_for_union.append(df_cat_sample)

    # Combine extremely rare categories first (they are forced in full)
    df_extremely_rare = pd.concat(extremely_rare_rows, axis=0).drop_duplicates()

    # Combine normal under-cat samples
    df_normal_under = pd.concat(cat_dfs_for_union, axis=0).drop_duplicates()

    # Union them
    df_union_under = pd.concat([df_extremely_rare, df_normal_under], axis=0).drop_duplicates()

    # 2) If union alone is bigger than n_target//2, we downsample 
    #    only from df_normal_under, to avoid removing extremely rare categories
    #    that must be included in full. 
    union_size = len(df_union_under)
    if union_size > half_limit:
        # We must see how many we can keep from normal under-cat
        # Keep all extremely rare rows; only reduce the normal under-cat part
        forced_rare_count = len(df_extremely_rare.drop_duplicates())

        # The remaining capacity for "normal under-cat" after forced rare
        # is half_limit - forced_rare_count
        capacity_for_normal = half_limit - forced_rare_count
        if capacity_for_normal < 0:
            capacity_for_normal = 0  # if extremely rare alone surpasses half-limit

        # Now downsample df_normal_under to that capacity
        if capacity_for_normal < len(df_normal_under):
            df_normal_under = df_normal_under.sample(capacity_for_normal, random_state=random_state)

        # Rebuild the union
        df_union_under = pd.concat([df_extremely_rare, df_normal_under], axis=0).drop_duplicates()

    # 3) Fill the remainder from the "normal" part of df_lang
    #    i.e. rows not in df_union_under
    df_lang_rest = df_lang.drop(df_union_under.index, errors="ignore")
    remainder_needed = n_target - len(df_union_under)
    if remainder_needed < 0:
        remainder_needed = 0

    if len(df_lang_rest) > remainder_needed:
        df_lang_rest = df_lang_rest.sample(remainder_needed, random_state=random_state)

    # Combine final result
    df_result = pd.concat([df_union_under, df_lang_rest], axis=0).drop_duplicates()
    # Shuffle
    df_result = df_result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_result


##############################################################################
#                    D. MAIN GENERATION FUNCTION
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
    # 2.1) Convert annotation columns to numeric so that '1' or '1.0' 
    #      are recognized as integer 1
    for col in annotation_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

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
    # e.g., threshold=UNDERREP_THRESHOLD=5000 => categories with <5000 positives
    under_cat = []
    for col in annotation_cols:
        nb_positives = df[col].sum(skipna=True)
        if nb_positives < UNDERREP_THRESHOLD:
            under_cat.append(col)
    print(f"[INFO] Underrepresented categories (less than {UNDERREP_THRESHOLD} occurrences): {under_cat}")

    # ----------------------------------------------------------------------
    # 5) Build separate subsets for EN and FR
    # ----------------------------------------------------------------------
    df_en = df[df["language"] == "EN"].copy()
    df_fr = df[df["language"] == "FR"].copy()

    # ----------------------------------------------------------------------
    # 6) We want NB_SENTENCES_TOTAL = 1000 => 500 EN + 500 FR
    # ----------------------------------------------------------------------
    half_target = NB_SENTENCES_TOTAL // 2  # 500
    en_needed = half_target
    fr_needed = half_target

    print("[INFO] Sampling English subset with guaranteed coverage (and rare-cat inclusion)...")
    sampled_en = sample_language_subset_with_guaranteed_coverage(
        df_lang=df_en,
        annotation_cols=annotation_cols,
        n_target=en_needed,
        under_cat=under_cat,
        per_cat_limit=PER_CAT_LIMIT,   
        random_state=42,
        min_positive_threshold=MIN_POSITIVE_THRESHOLD
    )

    print("[INFO] Sampling French subset with guaranteed coverage (and rare-cat inclusion)...")
    sampled_fr = sample_language_subset_with_guaranteed_coverage(
        df_lang=df_fr,
        annotation_cols=annotation_cols,
        n_target=fr_needed,
        under_cat=under_cat,
        per_cat_limit=PER_CAT_LIMIT,   
        random_state=42,
        min_positive_threshold=MIN_POSITIVE_THRESHOLD
    )

    # Combine and shuffle
    df_final_balanced = pd.concat([sampled_en, sampled_fr], axis=0)
    df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 7) Display final stats and proportions
    # ----------------------------------------------------------------------
    n_final = len(df_final_balanced)
    print(f"[INFO] Final sample: {n_final} rows.")
    n_en = (df_final_balanced["language"] == "EN").sum()
    n_fr = (df_final_balanced["language"] == "FR").sum()
    print(f"       -> EN: {n_en}")
    print(f"       -> FR: {n_fr}")

    # 7a) Show proportion of underrepresented categories in final sample
    if under_cat:
        print("[INFO] Proportions of underrepresented categories in final sample:")
        for cat in under_cat:
            if cat not in df_final_balanced.columns:
                continue
            cat_count = df_final_balanced[cat].sum()
            cat_percent = (cat_count / n_final) * 100 if n_final > 0 else 0
            print(f"    - {cat}: {cat_count} rows ({cat_percent:.2f}%)")

    # 7b) Show proportion of *all* annotation categories in final sample
    print("[INFO] Proportions of all annotation categories in final sample:")
    for col in annotation_cols:
        if col not in df_final_balanced.columns:
            continue
        count_pos = df_final_balanced[col].sum()
        percent_pos = (count_pos / n_final) * 100 if n_final > 0 else 0
        print(f"    - {col}: {count_pos} rows ({percent_pos:.2f}%)")

    # ----------------------------------------------------------------------
    # 8) Write the final multilingual JSONL
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
