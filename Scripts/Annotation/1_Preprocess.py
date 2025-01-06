"""
PROJECT:
-------
CLIMATE.FRAME

TITLE:
------
1_Preprocess.py

MAIN OBJECTIVE:
---------------
Preprocess the media database CSV by loading data, generating sentence contexts, 
counting words, converting and verifying date formats, then saving the processed data.

Dependencies:
-------------
- os
- pandas
- spacy
- datetime
- locale

MAIN FEATURES:
-------------
1) Load and preprocess CSV data.
2) Tokenize text into 2-sentence contexts.
3) Count words and update columns.
4) Validate and convert dates to a standard format.
5) Store processed data into a new CSV file.

Author:
-------
Antoine Lemor
"""

# Import necessary libraries
import os
import pandas as pd
import spacy
from datetime import datetime
import locale

# Relative path to the folder containing the script
script_dir = os.path.dirname(__file__)

# Relative path to the CSV file in the Database folder
csv_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_database.csv')

# Load spaCy models for English and French
nlp_fr = spacy.load('fr_dep_news_trf')
nlp_en = spacy.load('en_core_web_lg')

def count_words(text):
    """
    Count the number of words in the given text.

    Parameters:
    ----------
    text : str
        The text to be analyzed.

    Returns:
    -------
    int
        The number of words in the text.
    """
    return len(text.split())

def tokenize_and_context(text, language):
    """
    Tokenize the text and build two-sentence contexts.

    Parameters:
    ----------
    text : str
        The text to tokenize.
    language : str
        Language code ('FR' or otherwise 'EN').

    Returns:
    -------
    list
        A list of sentence contexts with up to two sentences joined together.
    """
    nlp = nlp_fr if language == 'FR' else nlp_en
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    contexts = []
    for i in range(len(sentences)):
        if i > 0:  # Ensure there is a previous sentence
            context = ' '.join(sentences[i-1:i+1])
            contexts.append(context)
        else:
            contexts.append(sentences[i])
    return contexts if contexts else ['']

# Set locale for date formatting
locale.setlocale(locale.LC_TIME, 'fr_FR' if os.name != 'nt' else 'French_France')

def convert_date(date_str):
    """
    Convert the provided date string to 'YYYY-MM-DD' format if possible.

    Parameters:
    ----------
    date_str : str
        The original date string.

    Returns:
    -------
    str
        The converted date in 'YYYY-MM-DD' format, or 'date : not provided'.
    """
    # List of possible date formats
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d %b %Y', '%d %B %Y']
    
    for date_format in date_formats:
        try:
            date_obj = datetime.strptime(date_str, date_format)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    print(f"Date format error: {date_str}")
    return 'date : not provided'

def verify_dates_format(dates):
    """
    Verify that each date in the given series is correctly formatted as 'YYYY-MM-DD'.

    Parameters:
    ----------
    dates : pd.Series
        A series containing date strings.

    Returns:
    -------
    bool
        True if all dates match the correct format, otherwise False.
    """
    for date in dates:
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print(f"Incorrect date detected: {date}")
            return False
    return True

# Load the CSV file
df = pd.read_csv(csv_path)

# Add the 'words_count_updated' column
df['words_count_updated'] = df['text'].apply(count_words)

# Convert dates
df['date'] = df['date'].apply(convert_date)

# Verify date formats
if verify_dates_format(df['date']):
    print("All dates are in 'yyyy-mm-dd' format.")
else:
    print("Some dates are in an incorrect format.")

# Create a new DataFrame for processed texts
processed_texts = []

doc_id = 0  # Initialize unique document ID
for _, row in df.iterrows():
    doc_id += 1  # Increment ID for each new article
    contexts = tokenize_and_context(row['text'], row['language'])
    for sentence_id, context in enumerate(contexts):
        processed_texts.append({
            'doc_ID': doc_id,
            'sentence_id': sentence_id,
            'news_type': row['news_type'],
            'title': row['title'],
            'author': row['author'],
            'media': row['media'] if pd.notna(row['media']) else 'media : not provided',
            'words_count': row['words_count'],
            'words_count_updated': row['words_count_updated'],
            'date': row['date'],
            'language': row['language'],
            'page_number': row['page_number'],
            'sentences': context
        })

        if sentence_id == 0 and doc_id <= 10:  # Print limit for first documents
            print("Sample entry in processed_texts:", processed_texts[-1])

processed_df = pd.DataFrame(processed_texts)

# Relative path for saving the new DataFrame
output_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_processed_texts.csv')

# Save the new dataframe
processed_df.to_csv(output_path, index=False, header=True)
