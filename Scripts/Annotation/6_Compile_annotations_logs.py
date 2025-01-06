#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
Canadian-Climate-Framing

TITLE:
------
6_Compile_annotations_logs.py

MAIN OBJECTIVE:
---------------
This script illustrates a more sophisticated and automated version for selecting
the best epoch, using:
1) macro-F1, weighted-F1, and recall of class "1".
2) Cross-validation (k folds): aggregation of metrics by epoch across folds and
   exclusion of epochs if overfit is detected.
3) A final score per epoch = alpha * macro_F1 + beta * weighted_F1 + gamma * recall_class_1.
4) Tie-breaking by test_loss, then train_loss.
5) Appending additional info to the CSV (class 0/1 support, precision/recalls/F1-scores,
   and the train/test distribution from the logs).

Dependencies:
-------------
- Python >= 3.7
- pandas, re, os, numpy

Author:
-------
Antoine Lemor
"""

import os
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
#               Selection parameters and scoring
# --------------------------------------------------------------------------
overfit_threshold = 0.65  # If (test_loss - train_loss) > 0.05 => overfit
alpha = 0.6               # Weight of macro-F1
beta = 0.2                # Weight of weighted-F1
gamma = 3.2               # Weight of recall of class "1"

# Optional: Minimum recall of the positive class to avoid neglecting it
min_positive_recall = 0.0  # e.g., 0.05 for a minimum recall

# --------------------------------------------------------------------------
#    Path definitions (adapt to your directory structure)
# --------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

# Directory where logs are located
log_output_dir = os.path.join(
    base_path, "..", "..", "Database", "Training_data", "Annotation_logs"
)

csv_output_dir = os.path.join(
    base_path, "..", "..", "Database", "Training_data"
)

# Output file
csv_output_path = os.path.join(csv_output_dir, "models_metrics_summary_advanced_2.csv")

# --------------------------------------------------------------------------
#     Automatic log recognition & grouping by model
# --------------------------------------------------------------------------
log_files = [f for f in os.listdir(log_output_dir) if f.endswith("_training_log.txt")]

fold_pattern = re.compile(r"^(.*)_fold(\d+)_training_log\.txt$")
model_to_folds = {}
for lf in log_files:
    match = fold_pattern.match(lf)
    if match:
        model_name, fold_num = match.groups()
        if model_name not in model_to_folds:
            model_to_folds[model_name] = []
        model_to_folds[model_name].append(lf)
    else:
        # Case without fold => treat as a single fold
        model_name_alt = lf.replace("_training_log.txt", "")
        if model_name_alt not in model_to_folds:
            model_to_folds[model_name_alt] = []
        model_to_folds[model_name_alt].append(lf)

# --------------------------------------------------------------------------
#    Utility functions: parsing the classification report
# --------------------------------------------------------------------------
def parse_classification_report(report_str):
    """
    Analyzes a raw classification report and returns a dictionary with 
    overall metrics and per-class metrics such as precision, recall, 
    f1-score, and support.

    Parameters:
    -----------
    report_str : str
        The raw classification report as a single string.

    Returns:
    --------
    dict
        A dictionary containing parsed metrics including accuracy, macro avg, 
        weighted avg, and class-specific metrics.
    """
    lines = [line.strip() for line in report_str.strip().split("\n")]
    report_data = {}
    if len(lines) < 2:
        return report_data

    for line in lines:
        if not line:
            continue
        tokens = line.split()

        # Ex: "accuracy                           0.95       286"
        if tokens[0] == "accuracy":
            try:
                report_data["accuracy"] = float(tokens[-2])
            except:
                pass
            continue

        # Ex : "macro avg" or "weighted avg"
        joined_first_two = " ".join(tokens[:2])
        if joined_first_two in ["macro avg", "weighted avg"]:
            label = joined_first_two
            try:
                precision, recall, f1_score, support = map(float, tokens[2:])
                report_data[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "support": support,
                }
            except:
                pass
            continue

        # Otherwise, assume it's a class (e.g., '0', '1')
        # Check that we have at least 5 tokens => label + precision + recall + f1 + support
        if len(tokens) >= 5:
            label_candidate = tokens[0]
            if label_candidate.isdigit() or label_candidate in ["0","1","2"]:
                try:
                    precision, recall, f1_score, support = map(float, tokens[1:5])
                    report_data[label_candidate] = {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1_score,
                        "support": support,
                    }
                except:
                    pass

    return report_data

# --------------------------------------------------------------------------
#   parse_single_log : extracting metrics + distribution
# --------------------------------------------------------------------------
def parse_single_log(full_log_path):
    """
    Extracts epoch-wise metrics and class distribution for a given training log.

    Parameters:
    -----------
    full_log_path : str
        Full path to the log file.

    Returns:
    --------
    dict
        A dictionary containing 'epochs_data' for each epoch and
        the distribution of classes 0 and 1 in training and testing sets.
    """
    if not os.path.exists(full_log_path) or os.path.getsize(full_log_path) == 0:
        return {}

    with open(full_log_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}

    # ----------------------------------------------------------------------
    # 1) Extracting Train/Test distribution from the log
    # ----------------------------------------------------------------------
    train_class_0_count = 0
    train_class_1_count = 0
    test_class_0_count = 0
    test_class_1_count = 0

    lines = content.split("\n")
    nb_lines = len(lines)
    
    i = 0
    while i < nb_lines:
        line = lines[i]
        # ------------------------------------------------------------------
        # Training distribution
        # ------------------------------------------------------------------
        if "Training label distribution" in line:
            dist_train = {}
            j = i + 1
            # Loop until "Name: count" or end
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                # Look for pattern: "0    29" or "1    26", etc.
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_train[label_found] = count_found
                j += 1
            # Assign
            train_class_0_count = dist_train.get('0', 0)
            train_class_1_count = dist_train.get('1', 0)
            i = j
            continue

        # ------------------------------------------------------------------
        # Validation distribution
        # ------------------------------------------------------------------
        if "Validation label distribution" in line:
            dist_test = {}
            j = i + 1
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_test[label_found] = count_found
                j += 1
            test_class_0_count = dist_test.get('0', 0)
            test_class_1_count = dist_test.get('1', 0)
            i = j
            continue

        i += 1

    # ----------------------------------------------------------------------
    # 2) Extracting metrics by epoch
    # ----------------------------------------------------------------------
    epoch_pattern = re.compile(r"^[=]{4,}\s*Epoch\s+(\d+)\s*/\s*(\d+)\s*[=]{4,}", re.MULTILINE)
    matches = list(epoch_pattern.finditer(content))
    if not matches:
        # No epoch found => return only the distribution
        return {
            "epochs_data": {},
            "train_class_0_count": train_class_0_count,
            "train_class_1_count": train_class_1_count,
            "test_class_0_count": test_class_0_count,
            "test_class_1_count": test_class_1_count,
        }

    epoch_data = {}
    for i, match in enumerate(matches):
        epoch_num = int(match.group(1))
        start_pos = match.end()
        end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(content)
        block = content[start_pos:end_pos]

        # Extract train_loss
        m_train = re.search(r"Average training loss:\s*([\d.]+)", block)
        train_loss = float(m_train.group(1)) if m_train else float("inf")

        # Extract test_loss
        m_test = re.search(r"Average test loss:\s*([\d.]+)", block)
        test_loss = float(m_test.group(1)) if m_test else float("inf")

        # Locate the classification report
        classif_start = re.search(
            r"(?:^|\n)\s*precision\s+recall\s+f1-score\s+support\s*\n", block
        )
        if not classif_start:
            # No report => skip
            continue

        start_idx = classif_start.end()
        classif_content = block[start_idx:].strip()
        next_epoch_marker = re.search(r"^[=]{4,}\s*Epoch\s", classif_content, re.MULTILINE)
        if next_epoch_marker:
            classif_content = classif_content[:next_epoch_marker.start()].strip()

        report_data = parse_classification_report(classif_content)
        if "macro avg" not in report_data:
            continue

        macro_f1 = report_data["macro avg"]["f1-score"]
        weighted_f1 = report_data.get("weighted avg", {}).get("f1-score", 0.0)

        # Extract values for class 1
        class_1_precision = 0.0
        class_1_recall = 0.0
        class_1_f1_score = 0.0
        class_1_support = 0.0
        if "1" in report_data:
            class_1_precision = report_data["1"]["precision"]
            class_1_recall = report_data["1"]["recall"]
            class_1_f1_score = report_data["1"]["f1-score"]
            class_1_support = report_data["1"]["support"]

        # Extract support for class 0
        class_0_support = 0.0
        if "0" in report_data:
            class_0_support = report_data["0"]["support"]

        epoch_data[epoch_num] = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "recall_1": class_1_recall,  # for scoring
            "class_0_support": class_0_support,
            "class_1_support": class_1_support,
            "class_1_precision": class_1_precision,
            "class_1_recall": class_1_recall,
            "class_1_f1_score": class_1_f1_score,
        }

    return {
        "epochs_data": epoch_data,
        "train_class_0_count": train_class_0_count,
        "train_class_1_count": train_class_1_count,
        "test_class_0_count": test_class_0_count,
        "test_class_1_count": test_class_1_count,
    }

# --------------------------------------------------------------------------
#   Aggregation cross-validation: combine metrics of each fold
# --------------------------------------------------------------------------
def aggregate_folds_metrics(list_of_epoch_dicts):
    """
    Aggregates the metrics of multiple folds for each epoch by averaging 
    most metrics and summing support counts.

    Parameters:
    -----------
    list_of_epoch_dicts : list
        A list of dictionaries mapping epoch numbers to their metrics.

    Returns:
    --------
    dict
        A dictionary aggregating the average metrics for each epoch 
        across folds while checking for overfitting.
    """
    if not list_of_epoch_dicts:
        return {}

    # 1) Common epochs across all folds
    common_epochs = set(list_of_epoch_dicts[0].keys())
    for d in list_of_epoch_dicts[1:]:
        common_epochs = common_epochs.intersection(d.keys())

    aggregated = {}
    for ep in sorted(common_epochs):
        macro_f1_vals = []
        weighted_f1_vals = []
        test_loss_vals = []
        train_loss_vals = []
        recall_1_vals = []
        class_1_precision_vals = []
        class_1_f1_vals = []
        class_0_support_total = 0.0
        class_1_support_total = 0.0

        overfit_detected = False

        for fold_dict in list_of_epoch_dicts:
            ep_metrics = fold_dict[ep]

            # Check overfit
            if (ep_metrics["test_loss"] - ep_metrics["train_loss"]) > overfit_threshold:
                overfit_detected = True
                break

            # Accumulate
            macro_f1_vals.append(ep_metrics["macro_f1"])
            weighted_f1_vals.append(ep_metrics["weighted_f1"])
            test_loss_vals.append(ep_metrics["test_loss"])
            train_loss_vals.append(ep_metrics["train_loss"])
            recall_1_vals.append(ep_metrics["recall_1"])

            class_1_precision_vals.append(ep_metrics["class_1_precision"])
            class_1_f1_vals.append(ep_metrics["class_1_f1_score"])

            class_0_support_total += ep_metrics["class_0_support"]
            class_1_support_total += ep_metrics["class_1_support"]

        if overfit_detected:
            # Completely exclude this epoch
            continue

        aggregated[ep] = {
            "macro_f1": np.mean(macro_f1_vals),
            "weighted_f1": np.mean(weighted_f1_vals),
            "test_loss": np.mean(test_loss_vals),
            "train_loss": np.mean(train_loss_vals),
            "recall_1": np.mean(recall_1_vals),
            "class_1_precision": np.mean(class_1_precision_vals),
            "class_1_f1_score": np.mean(class_1_f1_vals),
            "class_0_support": class_0_support_total,
            "class_1_support": class_1_support_total,
        }

    return aggregated

# --------------------------------------------------------------------------
#   Compute a global score (to prioritize the epoch)
# --------------------------------------------------------------------------
def compute_epoch_score(macro_f1, weighted_f1, recall_1):
    """
    Computes a global score for an epoch by combining macro_f1, weighted_f1, 
    and recall_1 according to predefined weights.

    Parameters:
    -----------
    macro_f1 : float
    weighted_f1 : float
    recall_1 : float

    Returns:
    --------
    float
        The computed weighted score.
    """
    return alpha * macro_f1 + beta * weighted_f1 + gamma * recall_1

# --------------------------------------------------------------------------
#   Select the "best" epoch
# --------------------------------------------------------------------------
def select_best_epoch(epochs_metrics):
    """
    Selects the best epoch based on the global score, using test_loss 
    and train_loss for tie-breaking.

    Parameters:
    -----------
    epochs_metrics : dict
        Dictionary where each key is an epoch number and the value 
        is another dict containing its metrics.

    Returns:
    --------
    (int, dict)
        A tuple of (best_epoch_number, best_epoch_data).
    """
    best_epoch = None
    best_data = None
    best_score = -1.0

    for ep, vals in epochs_metrics.items():
        macro_f1 = vals["macro_f1"]
        weighted_f1 = vals["weighted_f1"]
        recall_1 = vals["recall_1"]
        test_loss = vals["test_loss"]
        train_loss = vals["train_loss"]

        if recall_1 < min_positive_recall:
            continue

        current_score = compute_epoch_score(macro_f1, weighted_f1, recall_1)

        if current_score > best_score:
            best_epoch = ep
            best_data = vals
            best_score = current_score
        elif abs(current_score - best_score) < 1e-9:
            # Tie => compare test_loss, then train_loss
            if test_loss < best_data["test_loss"]:
                best_epoch = ep
                best_data = vals
                best_score = current_score
            elif abs(test_loss - best_data["test_loss"]) < 1e-9:
                if train_loss < best_data["train_loss"]:
                    best_epoch = ep
                    best_data = vals
                    best_score = current_score

    if best_epoch is None:
        return None, None
    return best_epoch, best_data

# --------------------------------------------------------------------------
#                           MAIN : Global Loop
# --------------------------------------------------------------------------
final_rows = []

for model_name, fold_logs in model_to_folds.items():
    # 1) Parse each log => store:
    #    - a dict "epochs_data" in a list (for cross-fold aggregation)
    #    - the train/test distribution
    fold_epoch_dicts = []
    train_class_0_counts = []
    train_class_1_counts = []
    test_class_0_counts = []
    test_class_1_counts = []

    for lf in fold_logs:
        path_log = os.path.join(log_output_dir, lf)
        parsed = parse_single_log(path_log)
        # Consider only if there is an "epochs_data"
        if "epochs_data" in parsed and parsed["epochs_data"]:
            fold_epoch_dicts.append(parsed["epochs_data"])
            train_class_0_counts.append(parsed.get("train_class_0_count", 0))
            train_class_1_counts.append(parsed.get("train_class_1_count", 0))
            test_class_0_counts.append(parsed.get("test_class_0_count", 0))
            test_class_1_counts.append(parsed.get("test_class_1_count", 0))

    # If no usable log
    if not fold_epoch_dicts:
        continue

    # 2) Aggregate if multiple folds, otherwise retrieve directly
    if len(fold_epoch_dicts) == 1:
        single_dict = fold_epoch_dicts[0]
        aggregated_metrics = {}
        for ep, metrics in single_dict.items():
            # Check overfit
            if (metrics["test_loss"] - metrics["train_loss"]) > overfit_threshold:
                continue
            aggregated_metrics[ep] = {
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "test_loss": metrics["test_loss"],
                "train_loss": metrics["train_loss"],
                "recall_1": metrics["recall_1"],
                "class_1_precision": metrics["class_1_precision"],
                "class_1_f1_score": metrics["class_1_f1_score"],
                "class_0_support": metrics["class_0_support"],
                "class_1_support": metrics["class_1_support"],
            }
        # Distribution = those of the (only) fold
        total_train_class_0_count = train_class_0_counts[0]
        total_train_class_1_count = train_class_1_counts[0]
        total_test_class_0_count = test_class_0_counts[0]
        total_test_class_1_count = test_class_1_counts[0]

    else:
        # Cross-fold aggregation
        aggregated_metrics = aggregate_folds_metrics(fold_epoch_dicts)
        # Sum the distribution across folds
        total_train_class_0_count = sum(train_class_0_counts)
        total_train_class_1_count = sum(train_class_1_counts)
        total_test_class_0_count = sum(test_class_0_counts)
        total_test_class_1_count = sum(test_class_1_counts)

    if not aggregated_metrics:
        # No valid epoch
        continue

    # 3) Select the best epoch
    best_ep, best_vals = select_best_epoch(aggregated_metrics)
    if best_ep is None:
        continue

    # 4) Create the CSV row
    row = {
        "model_name": model_name,
        "best_epoch": best_ep,
        "score": compute_epoch_score(
            best_vals["macro_f1"],
            best_vals["weighted_f1"],
            best_vals["recall_1"],
        ),
        "macro_f1": best_vals["macro_f1"],
        "weighted_f1": best_vals["weighted_f1"],
        "test_loss": best_vals["test_loss"],
        "train_loss": best_vals["train_loss"],
        "class_1_precision": best_vals["class_1_precision"],
        "class_1_recall": best_vals["recall_1"],  # recall of class 1
        "class_1_f1_score": best_vals["class_1_f1_score"],
        "class_1_support": best_vals["class_1_support"],
        "class_0_support": best_vals["class_0_support"],
        # New fields: train/test distribution
        "train_class_0_count": total_train_class_0_count,
        "train_class_1_count": total_train_class_1_count,
        "test_class_0_count": total_test_class_0_count,
        "test_class_1_count": total_test_class_1_count,
    }
    final_rows.append(row)

# --------------------------------------------------------------------------
#  Export the final DataFrame to CSV
# --------------------------------------------------------------------------
df = pd.DataFrame(final_rows)
df.to_csv(csv_output_path, index=False)
print(f"[END] Advanced metrics summary saved in: {csv_output_path}")
