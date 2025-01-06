"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
3_Manual_annotations.py

MAIN OBJECTIVE:
---------------
This script processes manual annotations from a JSONL file, counts label usage, 
and exports annotation metrics (counts and proportions) per language to CSV.

Dependencies:
-------------
- json
- csv
- os
- collections
- defaultdict

MAIN FEATURES:
--------------
1) Reads and parses manual annotations from a JSONL file.
2) Counts label usage for English and French sentences.
3) Calculates proportions of each label usage.
4) Exports the results to a CSV file for further analysis.

Author:
-------
Antoine Lemor
"""

import json
import csv
import os
from collections import defaultdict

# Paths (relative)
base_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(base_path, "..", "..", "Database", "Training_data", 'manual_annotations_JSONL', "Annotated_sentences.jsonl")
label_config_path = os.path.join(base_path, "..", "..", "Database", "Training_data", 'manual_annotations_JSONL', "label_config.json")
output_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_metrics.csv")

# Load the label configuration, extracting label IDs from the JSON config
with open(label_config_path, 'r', encoding='utf-8') as label_file:
    label_config = json.load(label_file)
    label_names = {label['text']: label['id'] for label in label_config}

# Initialize counters to track annotations and sentence counts
annotation_counts = defaultdict(lambda: {'EN': 0, 'FR': 0})
num_sentences = {'EN': 0, 'FR': 0}

# Read the jsonl file line by line, incrementing counters by language
with open(input_data_path, 'r', encoding='utf-8') as data_file:
    for line in data_file:
        data = json.loads(line)
        language = data.get('meta', {}).get('language', 'EN')
        num_sentences[language] += 1
        # Extract labels directly from 'label' field
        labels = data.get('label', [])
        for label in labels:
            if label in label_names:
                annotation_counts[label][language] += 1

# Calculate proportions of labels for each language
annotation_proportions = {}
for label, counts in annotation_counts.items():
    annotation_proportions[label] = {
        'EN': counts['EN'] / num_sentences['EN'] if num_sentences['EN'] > 0 else 0,
        'FR': counts['FR'] / num_sentences['FR'] if num_sentences['FR'] > 0 else 0
    }

# Create and write results to a CSV file
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Annotation Type', 'Count (EN)', 'Proportion (EN)', 'Count (FR)', 'Proportion (FR)'])
    for label, counts in annotation_counts.items():
        proportion_en = annotation_proportions[label]['EN']
        proportion_fr = annotation_proportions[label]['FR']
        csv_writer.writerow([label, counts['EN'], proportion_en, counts['FR'], proportion_fr])
    # Add total sentences annotated
    csv_writer.writerow(['Total Sentences Annotated (EN)', num_sentences['EN'], '', '', ''])
    csv_writer.writerow(['Total Sentences Annotated (FR)', num_sentences['FR'], '', '', ''])

print(f"Metrics have been successfully exported to {output_csv_path}")