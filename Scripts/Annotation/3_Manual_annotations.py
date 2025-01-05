import json
import csv
import os
from collections import defaultdict

# Paths (relative)
base_path = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(base_path, "..", "..", "Database", "Training_data", 'manual_annotations_JSONL', "Annotated_sentences.jsonl")
label_config_path = os.path.join(base_path, "..", "..", "Database", "Training_data", 'manual_annotations_JSONL', "label_config.json")
output_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "manual_annotations_metrics.csv")

# Load label configuration to get annotation labels
with open(label_config_path, 'r', encoding='utf-8') as label_file:
    label_config = json.load(label_file)
    label_names = {label['text']: label['id'] for label in label_config}

# Initialize counters
annotation_counts = defaultdict(lambda: {'EN': 0, 'FR': 0})
num_sentences = {'EN': 0, 'FR': 0}

# Process annotations from the jsonl file
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

# Calculate total annotations and proportions
annotation_proportions = {}
for label, counts in annotation_counts.items():
    annotation_proportions[label] = {
        'EN': counts['EN'] / num_sentences['EN'] if num_sentences['EN'] > 0 else 0,
        'FR': counts['FR'] / num_sentences['FR'] if num_sentences['FR'] > 0 else 0
    }

# Export metrics to CSV
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