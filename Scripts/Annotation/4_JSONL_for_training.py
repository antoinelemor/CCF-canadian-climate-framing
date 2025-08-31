"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
4_JSONL_for_training.py

MAIN OBJECTIVE:
-------------------
This script processes manually annotated JSONL sentences, splits them 
into training and validation sets, and exports them as separate JSONL 
files for subsequent model training.

Dependencies:
-------------
- json
- os
- random
- csv
- collections.defaultdict

MAIN FEATURES:
----------------------------
1) Loads label configuration to determine main labels and sub-labels.
2) Creates directory structure for training and validation outputs.
3) Processes manually annotated lines from a JSONL file.
4) Handles main labels, sub-labels, and exception labels.
5) Splits data into training (~80%) and validation (~20%) sets 
   with a guaranteed minimum of 10% positives and 10% negatives in validation.
6) Aggregates annotation counts per label, language, and data split.
7) Exports final JSONL outputs and saves aggregated metrics in a CSV.

Author :
--------
Antoine Lemor
"""

import json
import os
import random
import csv
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import List, Dict, Literal, TypedDict


# Paths (relative)
base_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", "annotated_sentences.jsonl")
label_config_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", "label_config.json")
output_base_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotation_bases")
output_csv_path = os.path.join(os.path.dirname(input_data_path), "training_database_metrics.csv")

# Load label configuration to get annotation labels
with open(label_config_path, 'r', encoding='utf-8') as label_file:
    label_config = json.load(label_file)
    label_names = [label['text'] for label in label_config]

# Define main labels and their corresponding sub-labels automatically
main_labels = {}
exception_labels = [
    "RED Detection", "Location Detection",
    "Emotion: Negative", "Emotion: Neutral", "Emotion: Positive"
]

for label in label_names:
    if label.endswith("Detection") and label not in exception_labels:
        prefix = label.split()[0] if label != "Solutions Detection" else "Solution"
        sub_labels = [sub_label for sub_label in label_names if sub_label.startswith(f"{prefix}_")]
        main_labels[label] = sub_labels

# Create base directory for each annotation type
for label in main_labels.keys() | set(exception_labels):
    label_dir = os.path.join(output_base_dir, label.replace(' ', '_'))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    for data_type in ['train', 'validation']:
        data_type_dir = os.path.join(label_dir, data_type)
        if not os.path.exists(data_type_dir):
            os.makedirs(data_type_dir)
        for language in ['EN', 'FR']:
            lang_dir = os.path.join(data_type_dir, language)
            if not os.path.exists(lang_dir):
                os.makedirs(lang_dir)

# Initialize annotation structures
main_annotations = defaultdict(lambda: defaultdict(list))
sub_annotations = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
exception_annotations = defaultdict(lambda: defaultdict(list))
num_sentences = 0

# Initialize counts dictionary
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'positive': 0, 'negative': 0})))

# Process annotations from the jsonl file
with open(input_data_path, 'r', encoding='utf-8') as data_file:
    for line in data_file:
        data = json.loads(line)
        num_sentences += 1
        labels = data.get('label', [])
        language = data.get('meta', {}).get('language', 'EN')

        # Handle main labels and their sub-labels
        for main_label, sub_labels in main_labels.items():
            # 1.  Traiter l’étiquette principale (comme aujourd’hui)
            is_main_present = main_label in labels
            main_annotations[main_label][language].append(
                {"text": data["text"], "label": int(is_main_present)}
            )
            counts[main_label][language]['total'][
                'positive' if is_main_present else 'negative'
            ] += 1

            # 2.  Traiter *toujours* chaque sous-étiquette
            for sub_label in sub_labels:
                is_sub_present = sub_label in labels
                sub_annotations[main_label][sub_label][language].append(
                    {"text": data["text"], "label": int(is_sub_present)}
                )
                counts[sub_label][language]['total'][
                    'positive' if is_sub_present else 'negative'
                ] += 1


        # Handle exception labels
        for exception_label in exception_labels:
            exception_annotation = {
                "text": data.get("text", ""),
                "label": 1 if exception_label in labels else 0
            }
            exception_annotations[exception_label][language].append(exception_annotation)
            if exception_annotation['label'] == 1:
                counts[exception_label][language]['total']['positive'] += 1
            else:
                counts[exception_label][language]['total']['negative'] += 1

def safe_mkdir(path: str | Path) -> None:
    """
    Create *all* parent directories if they do not exist already.

    Parameters
    ----------
    path : str | Path
        Directory path to create.

    Notes
    -----
    - Silently returns if the directory hierarchy exists.
    - Uses ``Path.mkdir`` with ``parents=True`` and ``exist_ok=True`` for
      atomic-ish behaviour (thread-safe on POSIX).
    """
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)

class SplitDict(TypedDict):
    train: List[dict]
    validation: List[dict]


def split_annotations(
    annotations: List[dict],
    val_ratio: float = 0.20,
    min_val_share: float = 0.10,
) -> SplitDict:
    """
    Split a list of annotations into *train* and *validation* subsets.

    The function guarantees **both** of the following:
    1. Approximately ``val_ratio`` of the data end up in the validation set.
    2. The validation set contains **at least** ``min_val_share`` positives
       **and** negatives if such labels exist.

    Parameters
    ----------
    annotations : List[dict]
        Each dict must contain a boolean-like key ``'label'`` (1 = positive).
    val_ratio : float, default 0.20
        Target share of the validation set (0 < val_ratio < 1).
    min_val_share : float, default 0.10
        Minimum share *per class* required in the validation set.

    Returns
    -------
    SplitDict
        Two keys: ``'train'`` and ``'validation'`` — each a list of annotations.

    Raises
    ------
    ValueError
        If ``val_ratio`` or ``min_val_share`` are outside (0, 1).
    """
    if not 0.0 < val_ratio < 1.0 or not 0.0 < min_val_share < 1.0:
        raise ValueError("val_ratio and min_val_share must be in the open interval (0, 1).")

    if not annotations:
        return {"train": [], "validation": []}

    rng = random.Random(42)  # deterministic shuffling for reproducibility
    rng.shuffle(annotations)

    pos = [a for a in annotations if a["label"] == 1]
    neg = [a for a in annotations if a["label"] == 0]

    # -- compute absolute minima ------------------------------------------------
    min_val_pos = ceil(min_val_share * len(pos)) if pos else 0
    min_val_neg = ceil(min_val_share * len(neg)) if neg else 0

    # -- compute nominal split sizes -------------------------------------------
    val_size = ceil(val_ratio * len(annotations))
    # Ensure class minima are honoured
    val_pos = max(min_val_pos, ceil(val_ratio * len(pos)))
    val_neg = max(min_val_neg, val_size - val_pos)

    # Correct potential overflow (e.g., extremely imbalanced data)
    val_pos = min(val_pos, len(pos))
    val_neg = min(val_neg, len(neg))
    val_size = val_pos + val_neg

    # -- slice ------------------------------------------------------------------
    val_set = pos[:val_pos] + neg[:val_neg]
    train_set = pos[val_pos:] + neg[val_neg:]

    rng.shuffle(train_set)
    rng.shuffle(val_set)

    return {"train": train_set, "validation": val_set}

# Split main annotations and update counts
for label, lang_annotations in main_annotations.items():
    for language, annotations in lang_annotations.items():
        split_data = split_annotations(annotations)
        main_annotations[label][language] = split_data
        for data_type, data_annotations in split_data.items():
            for annotation in data_annotations:
                label_value = annotation['label']
                if label_value == 1:
                    counts[label][language][data_type]['positive'] += 1
                else:
                    counts[label][language][data_type]['negative'] += 1

# Split sub-label annotations and update counts
for main_label, sub_labels in sub_annotations.items():
    for sub_label, lang_annotations in sub_labels.items():
        for language, annotations in lang_annotations.items():
            split_data = split_annotations(annotations)
            sub_annotations[main_label][sub_label][language] = split_data
            for data_type, data_annotations in split_data.items():
                for annotation in data_annotations:
                    label_value = annotation['label']
                    if label_value == 1:
                        counts[sub_label][language][data_type]['positive'] += 1
                    else:
                        counts[sub_label][language][data_type]['negative'] += 1

# Split exception annotations and update counts
for label, lang_annotations in exception_annotations.items():
    for language, annotations in lang_annotations.items():
        split_data = split_annotations(annotations)
        exception_annotations[label][language] = split_data
        for data_type, data_annotations in split_data.items():
            for annotation in data_annotations:
                label_value = annotation['label']
                if label_value == 1:
                    counts[label][language][data_type]['positive'] += 1
                else:
                    counts[label][language][data_type]['negative'] += 1

# Export each annotation base to separate JSONL files for training and validation
# Export main labels
for label, lang_datasets in main_annotations.items():
    for language, datasets in lang_datasets.items():
        for data_type, annotations in datasets.items():
            output_path = os.path.join(
                output_base_dir, label.replace(' ', '_'), data_type, language,
                f"{label.replace(' ', '_')}_{data_type}_{language}.jsonl"
            )
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for annotation in annotations:
                    json.dump(annotation, outfile, ensure_ascii=False)
                    outfile.write('\n')

# Export sub-labels
for main_label, sub_labels in sub_annotations.items():
    for sub_label, lang_annotations in sub_labels.items():
        for language, datasets in lang_annotations.items():
            for data_type, annotations in datasets.items():
                output_path = os.path.join(
                    output_base_dir, main_label.replace(' ', '_'), data_type, language,
                    f"{sub_label.replace(' ', '_')}_{data_type}_{language}.jsonl"
                )
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    for annotation in annotations:
                        json.dump(annotation, outfile, ensure_ascii=False)
                        outfile.write('\n')

# Export exception labels
for label, lang_datasets in exception_annotations.items():
    for language, datasets in lang_datasets.items():
        for data_type, annotations in datasets.items():
            output_path = os.path.join(
                output_base_dir, label.replace(' ', '_'), data_type, language,
                f"{label.replace(' ', '_')}_{data_type}_{language}.jsonl"
            )
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for annotation in annotations:
                    json.dump(annotation, outfile, ensure_ascii=False)
                    outfile.write('\n')

# Write counts to CSV
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'Label',
        'Train_Positive_EN', 'Train_Negative_EN',
        'Validation_Positive_EN', 'Validation_Negative_EN',
        'Train_Positive_FR', 'Train_Negative_FR',
        'Validation_Positive_FR', 'Validation_Negative_FR'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for label in sorted(counts.keys()):
        row = {'Label': label}
        for language in ['EN', 'FR']:
            for data_type in ['train', 'validation']:
                positive = counts[label][language][data_type].get('positive', 0)
                negative = counts[label][language][data_type].get('negative', 0)
                row_key_positive = f"{data_type.capitalize()}_Positive_{language}"
                row_key_negative = f"{data_type.capitalize()}_Negative_{language}"
                row[row_key_positive] = positive
                row[row_key_negative] = negative
        writer.writerow(row)

print(f"Annotation bases have been successfully exported to {output_base_dir}")
print(f"Metrics have been successfully exported to {output_csv_path}")
