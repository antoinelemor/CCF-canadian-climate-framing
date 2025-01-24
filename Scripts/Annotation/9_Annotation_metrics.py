"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
9_Annotation_metrics.py

MAIN OBJECTIVE:
---------------
This script evaluates the performance of the previously generated labels 
by comparing them to manually verified labels. Specifically, it calculates 
precision, recall, and F1-score for each category (label), as well as an 
overall score across all categories. It additionally computes these metrics 
separately for English, for French, and for all languages combined. 
It also reports the distribution (i.e., number of occurrences) of each label 
in the manually verified data.

Dependencies:
-------------
- os
- json
- csv
- pandas
- math
- collections (for default dictionaries)

MAIN FEATURES:
--------------
1) Reads the JSONL file with predicted labels 
   (sentences_to_recheck_multiling.jsonl).
2) Reads the JSONL file with manually verified (gold) labels 
   (sentences_to_recheck_multiling_done.jsonl).
3) For each sentence present in the gold file, compares the predicted labels 
   vs. the gold labels.
4) Computes per-label confusion values (TP, FP, FN), then calculates precision, 
   recall, and F1-score for each label.
5) Performs the same metric computations for:
   - All languages combined,
   - French (FR) only,
   - English (EN) only.
6) Exports all metrics to a CSV file (final_annotation_metrics.csv).

Author:
-------
Antoine Lemor
"""

import os
import json
import csv
import pandas as pd
import math
from collections import defaultdict

##############################################################################
#                      A. CONSTANTS AND PATHS
##############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input predicted JSONL (generated from 8_Produce_JSON_for_Recheck.py)
PREDICTED_JSONL = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL",
    "sentences_to_recheck_multiling_bis.jsonl"
)

# Manually checked (gold) JSONL
GOLD_JSONL = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL",
    "sentences_to_recheck_multiling_done.jsonl"
)

# CSV output containing final metrics
OUTPUT_CSV = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data",
    "final_annotation_metrics_2.csv"
)


##############################################################################
#                      B. UTILITY FUNCTIONS
##############################################################################
def load_jsonl_labels(jsonl_path):
    """
    Loads a JSONL file where each line is a dictionary with:
      - "text": the sentence,
      - "label": the list of active categories,
      - "meta": containing metadata, including "language".
    
    Returns a dictionary: 
        {
          sentence_text: {
              "labels": set([...]),
              "language": "EN" or "FR" (if available)
          },
          ...
        }
    If language is missing, defaults to None.
    """
    data_dict = {}
    if not os.path.exists(jsonl_path):
        print(f"[WARNING] File not found: {jsonl_path}")
        return data_dict
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                text = entry.get("text", "")
                labels = entry.get("label", [])
                meta = entry.get("meta", {})
                language = None
                if meta and isinstance(meta, dict):
                    language = meta.get("language", None)
                
                data_dict[text] = {
                    "labels": set(labels),
                    "language": language
                }
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decoding failed for line:\n{line}\nError: {e}")
    return data_dict


def compute_confusion_stats(predicted_dict, gold_dict):
    """
    Computes confusion statistics (TP, FP, FN) for each label, 
    for each language, and overall.

    Parameters
    ----------
    predicted_dict: dict
        { sentence: { "labels": set([...]), "language": "EN"/"FR"/None } }
    gold_dict: dict
        { sentence: { "labels": set([...]), "language": "EN"/"FR"/None } }

    Returns
    -------
    confusion: dict of the form:
        {
          (label, lang): { "TP": ..., "FP": ..., "FN": ... },
          ...
        }
      where lang can be "ALL", "EN", "FR".
      Also includes an entry for ("ALL_LABELS", lang) to track metrics 
      across all labels combined.
    """
    # Default dictionary for each (label, lang) 
    confusion = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    # We'll store union of all sentences that appear in the gold data 
    # (since some sentences might not have been manually checked).
    gold_sentences = set(gold_dict.keys())

    # For each sentence in gold, retrieve predicted labels (or empty set if missing)
    for sent in gold_sentences:
        gold_labels = gold_dict[sent]["labels"]
        gold_lang = gold_dict[sent]["language"]
        predicted_labels = set()
        predicted_lang = gold_lang  # default if missing

        if sent in predicted_dict:
            predicted_labels = predicted_dict[sent]["labels"]
            predicted_lang = predicted_dict[sent]["language"]
            # If the predicted language is None, fallback to gold language
            if not predicted_lang:
                predicted_lang = gold_lang

        # For each label in the union of gold and predicted
        all_labels = gold_labels.union(predicted_labels)

        for label in all_labels:
            # Build language code
            lang_code = predicted_lang if predicted_lang else gold_lang
            if not lang_code:
                # If no language at all, we won't specifically track EN or FR
                # but we still track "ALL".
                lang_code = "ALL"

            # We also want to track an "ALL" language line
            # so we will compute confusion for (label, "ALL") as well.

            # Ground truth presence
            in_gold = label in gold_labels
            # Predicted presence
            in_predicted = label in predicted_labels

            if in_gold and in_predicted:
                confusion[(label, lang_code)]["TP"] += 1
                confusion[(label, "ALL")]["TP"] += 1
                confusion[("ALL_LABELS", lang_code)]["TP"] += 1
                confusion[("ALL_LABELS", "ALL")]["TP"] += 1
            elif in_gold and not in_predicted:
                confusion[(label, lang_code)]["FN"] += 1
                confusion[(label, "ALL")]["FN"] += 1
                confusion[("ALL_LABELS", lang_code)]["FN"] += 1
                confusion[("ALL_LABELS", "ALL")]["FN"] += 1
            elif not in_gold and in_predicted:
                confusion[(label, lang_code)]["FP"] += 1
                confusion[(label, "ALL")]["FP"] += 1
                confusion[("ALL_LABELS", lang_code)]["FP"] += 1
                confusion[("ALL_LABELS", "ALL")]["FP"] += 1
            else:
                # label not in gold, not in predicted => no change 
                # (TN not used for precision/recall in multi-label context)
                pass

    return confusion


def compute_precision_recall_f1(tp, fp, fn):
    """
    Given TP, FP, and FN, returns precision, recall, and F1 as a tuple.
    If denominators are zero, returns 0.0 for those metrics.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
    """
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def create_metrics_table(confusion_dict):
    """
    Takes the confusion dictionary of the form:
      confusion[(label, lang)] = {"TP": x, "FP": x, "FN": x}
    and computes for each (label, lang): 
      - label,
      - lang,
      - TP,
      - FP,
      - FN,
      - Precision,
      - Recall,
      - F1

    Returns a list of dictionaries, each representing one row.
    """
    rows = []
    for (label, lang), counts in confusion_dict.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
        row = {
            "label": label,
            "language": lang,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        rows.append(row)
    return rows


def add_distribution_info(rows, gold_dict):
    """
    For each row in rows (which contain label-language metrics), 
    add the 'gold_count' field indicating how many times that label 
    appears in the gold data for the given language. 
    Also add 'predicted_count' for how many times that label 
    appears in the predicted data for that language, if needed.

    We rely on the stored dictionary:
       gold_dict[sentence]["labels"] -> set of labels
       gold_dict[sentence]["language"] -> "EN"/"FR"

    We'll create helper mappings:
       gold_count[(label, lang)] = ...
    Similarly for predicted_count[(label, lang)] if desired.
    But the user specifically asked for distribution "for each label". 
    Usually, distribution is how many times the label is present in the 
    gold standard. We can also add predicted distribution if needed.

    We'll do:
       label_dist_gold[(label, lang)] = number of gold occurrences
       label_dist_pred[(label, lang)] = number of predicted occurrences
    Then we enrich the rows with that info.
    """
    # Build distribution maps
    label_dist_gold = defaultdict(int)
    label_dist_pred = defaultdict(int)

    # We can find the predicted labels from the confusion approach or 
    # simply recompute from gold_dict + predicted_dict. 
    # We'll assume each row in the confusion covers gold or predicted 
    # across all sentences, but let's do a direct pass again for clarity.
    # We'll load predicted too, so we need it in the function arguments 
    # (or can do global?). Let's do it separately for generality.
    # Actually let's pass predicted_dict in. 
    pass


def get_label_distributions(gold_dict, predicted_dict):
    """
    Returns two dictionaries:
      gold_distribution[(label, lang)] = count
      pred_distribution[(label, lang)] = count

    Summarizes how many times a label appears for a given language 
    in the gold data and predicted data.
    Also includes (label, "ALL").
    """
    gold_distribution = defaultdict(int)
    pred_distribution = defaultdict(int)

    # 1) gold distribution
    for sent, info in gold_dict.items():
        labels = info["labels"]
        lang = info["language"] if info["language"] else "ALL"
        for label in labels:
            gold_distribution[(label, lang)] += 1
            gold_distribution[(label, "ALL")] += 1
            gold_distribution[("ALL_LABELS", lang)] += 1
            gold_distribution[("ALL_LABELS", "ALL")] += 1

    # 2) predicted distribution
    for sent, info in predicted_dict.items():
        labels = info["labels"]
        lang = info["language"] if info["language"] else "ALL"
        for label in labels:
            pred_distribution[(label, lang)] += 1
            pred_distribution[(label, "ALL")] += 1
            pred_distribution[("ALL_LABELS", lang)] += 1
            pred_distribution[("ALL_LABELS", "ALL")] += 1

    return gold_distribution, pred_distribution


##############################################################################
#                     C. MAIN METRICS CALCULATION
##############################################################################
def main():
    print("[INFO] Loading predicted annotations...")
    predicted_dict = load_jsonl_labels(PREDICTED_JSONL)

    print("[INFO] Loading gold (manually verified) annotations...")
    gold_dict = load_jsonl_labels(GOLD_JSONL)

    print("[INFO] Computing confusion statistics...")
    confusion_dict = compute_confusion_stats(predicted_dict, gold_dict)

    print("[INFO] Building metrics table...")
    metrics_rows = create_metrics_table(confusion_dict)

    # We also want the distribution of labels (in gold and predicted).
    print("[INFO] Computing label distributions...")
    gold_distribution, pred_distribution = get_label_distributions(gold_dict, predicted_dict)

    # Enrich the metrics rows with distribution data
    # (gold_count, predicted_count).
    # We'll match row label-language with distribution keys.
    for row in metrics_rows:
        label = row["label"]
        lang = row["language"]
        # For the distribution keys, if lang is not "EN" or "FR", 
        # it might be "ALL". We'll use exactly that combination.
        row["gold_count"] = gold_distribution.get((label, lang), 0)
        row["predicted_count"] = pred_distribution.get((label, lang), 0)

    print("[INFO] Writing final metrics to CSV...")
    # Let's define field order
    fieldnames = [
        "label", "language", 
        "TP", "FP", "FN", 
        "precision", "recall", "f1", 
        "gold_count", "predicted_count"
    ]
    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({
                "label": row["label"],
                "language": row["language"],
                "TP": row["TP"],
                "FP": row["FP"],
                "FN": row["FN"],
                "precision": f"{row['precision']:.4f}",
                "recall": f"{row['recall']:.4f}",
                "f1": f"{row['f1']:.4f}",
                "gold_count": row["gold_count"],
                "predicted_count": row["predicted_count"]
            })

    print(f"[INFO] Metrics CSV created at: {OUTPUT_CSV}")
    print("[INFO] End of script.")


if __name__ == "__main__":
    main()
