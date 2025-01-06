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

# Paths (relative)
base_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", "annotated_sentences.jsonl")
label_config_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_JSONL", "label_config.json")
output_base_dir = "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Training/Annotation_bases"
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
            if main_label in labels:
                main_annotation = {
                    "text": data.get("text", ""),
                    "label": 1
                }
                main_annotations[main_label][language].append(main_annotation)
                counts[main_label][language]['total']['positive'] += 1
                for sub_label in sub_labels:
                    sub_annotation = {
                        "text": data.get("text", ""),
                        "label": 1 if sub_label in labels else 0
                    }
                    sub_annotations[main_label][sub_label][language].append(sub_annotation)
                    if sub_annotation['label'] == 1:
                        counts[sub_label][language]['total']['positive'] += 1
                    else:
                        counts[sub_label][language]['total']['negative'] += 1
            else:
                main_annotation = {
                    "text": data.get("text", ""),
                    "label": 0
                }
                main_annotations[main_label][language].append(main_annotation)
                counts[main_label][language]['total']['negative'] += 1

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

def split_annotations(annotations):
    """
    Divides 'annotations' into a training set (~80%) and a validation set (~20%), 
    ensuring at least 10% positives (label == 1) and 10% negatives (label == 0) in 
    the validation subset.

    Parameters:
    -----------
    annotations (list): A list of annotation dictionaries where each dictionary 
                        contains a 'label' key indicating a positive or 
                        negative annotation.
    
    Returns:
    --------
    dict: 
        A dictionary with two keys: 'train' and 'validation'. Each contains a list 
        of annotations representing the respective subsets.
    """
    if not annotations:
        return {'train': [], 'validation': []}

    # Random shuffle
    random.shuffle(annotations)

    # Separate positive and negative annotations
    positives = [ann for ann in annotations if ann['label'] == 1]
    negatives = [ann for ann in annotations if ann['label'] == 0]

    # Compute minimum validation counts
    min_val_positives = max(1, int(0.1 * len(positives))) if positives else 0
    min_val_negatives = max(1, int(0.1 * len(negatives))) if negatives else 0

    # Compute split indices
    train_positives_count = max(len(positives) - min_val_positives, int(0.8 * len(positives)))
    train_negatives_count = max(len(negatives) - min_val_negatives, int(0.8 * len(negatives)))

    # Split positives
    train_positives = positives[:train_positives_count]
    validation_positives = positives[train_positives_count:]

    # Split negatives
    train_negatives = negatives[:train_negatives_count]
    validation_negatives = negatives[train_negatives_count:]

    # Combine splits
    train_part = train_positives + train_negatives
    validation_part = validation_positives + validation_negatives

    # Final shuffle
    random.shuffle(train_part)
    random.shuffle(validation_part)

    return {
        'train': train_part,
        'validation': validation_part
    }

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
