"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
2_JSONL.py

MAIN OBJECTIVE:
---------------
This script loads processed text data from a CSV file, removes duplicates based on doc_ID 
and sentence_id, updates specific date values for certain rows, filters out already used sentences, 
and produces separate JSONL files for French and English samples intended for manual annotation.

Dependencies:
-------------
- os
- pandas
- json
- sklearn.utils.shuffle

MAIN FEATURES:
--------------
1) Loads a CSV file containing processed text data.
2) Removes duplicates and updates specific date entries by doc_ID.
3) Separates French and English sentences and randomly selects a subset for annotation.
4) Produces JSONL files where each record contains:
   - text: the sentence,
   - label: an empty list,
   - meta: metadata (e.g., date, title, doc_ID).

Author:
-------
Antoine Lemor
"""

# Import necessary libraries
import os
import pandas as pd
import json
from sklearn.utils import shuffle

# Relative path to the script directory
script_dir = os.path.dirname(__file__)

# Relative path to the source CSV file
csv_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_processed_texts.csv')

# Path to the existing JSONL file to avoid duplicates
existing_jsonl_path = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Annotated_sentences.jsonl')

# Load existing entries to avoid duplicates
existing_entries = set()
if os.path.exists(existing_jsonl_path):
    with open(existing_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            doc_ID = entry['meta'].get('doc_ID')
            sentence_id = entry['meta'].get('sentence_id')
            if doc_ID and sentence_id:
                existing_entries.add((doc_ID, sentence_id))

# Load the CSV file
df = pd.read_csv(csv_path)

# Replace 'English' with 'EN' in the 'language' column
df['language'] = df['language'].replace('English', 'EN')

# Check that 'language' contains only 'FR' or 'EN'
possible_values = {'FR', 'EN'}
if not df['language'].isin(possible_values).all():
    invalid_rows = df[~df['language'].isin(possible_values)]
    print(f"Warning: Some rows have incorrect 'language' values. Affected rows:\n{invalid_rows}")
else:
    print("The 'language' column contains only valid values ('FR' or 'EN').")

# Filter out already used sentences by checking 'doc_ID' and 'sentence_id'
df_filtered = df[~df.apply(lambda row: (row['doc_ID'], row['sentence_id']) in existing_entries, axis=1)]

# Split the DataFrame into French and English sentences
df_fr = df_filtered[df_filtered['language'] == 'FR']
df_en = df_filtered[df_filtered['language'] == 'EN']

# Shuffle and select 2000 sentences for each language
df_fr = shuffle(df_fr).head(2000)
df_en = shuffle(df_en).head(2000)

def create_jsonl_data(df_subset):
    """
    Generates a list of JSON objects from a DataFrame subset.

    Parameters:
    -----------
    df_subset : pandas.DataFrame
        Subset of rows for which JSON objects will be created.

    Returns:
    --------
    list
        A list of dictionaries formatted for JSONL output, 
        each containing 'text', 'label', and 'meta' keys.
    """
    jsonl_data = []
    for _, row in df_subset.iterrows():
        if pd.notna(row['sentences']) and row['sentences'].strip():  # Check that 'sentences' is not empty
            jsonl_data.append({
                'text': row['sentences'],
                'label': [],  # Empty label column
                'meta': {
                    'media': row['media'] if pd.notna(row['media']) else 'media: not provided',
                    'date': row['date'] if pd.notna(row['date']) else 'date: not provided',
                    'title': row['title'] if pd.notna(row['title']) else 'title: not provided',
                    'page number': row['page_number'] if pd.notna(row['page_number']) else 'page number: not provided',
                    'sentence_id': row['sentence_id'] if pd.notna(row['sentence_id']) else 'sentence_id: not provided',
                    'doc_ID': row['doc_ID'] if pd.notna(row['doc_ID']) else 'doc_ID: not provided',
                    'language': row['language']
                }
            })
        else:
            print(f"Ignored row: '{row}' because 'sentences' is missing or empty.")
    return jsonl_data

# Create JSONL data for French and English
jsonl_data_fr = create_jsonl_data(df_fr)
jsonl_data_en = create_jsonl_data(df_en)

# Paths for the output JSONL files
jsonl_output_path_fr = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Sentences_to_annotate_FR.jsonl')
jsonl_output_path_en = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Sentences_to_annotate_EN.jsonl')

# Write the data to JSONL files
with open(jsonl_output_path_fr, 'w', encoding='utf-8') as jsonl_file_fr:
    for entry in jsonl_data_fr:
        jsonl_file_fr.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open(jsonl_output_path_en, 'w', encoding='utf-8') as jsonl_file_en:
    for entry in jsonl_data_en:
        jsonl_file_en.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"French JSONL file created successfully: {jsonl_output_path_fr}")
print(f"English JSONL file created successfully: {jsonl_output_path_en}")
