"""
PROJECT:
--------
CCF-canadian-climate-framing

TITLE:
------
2_Clean_and_create_database.py

MAIN OBJECTIVE:
---------------
Cleans and prepares the media database by normalizing date formats, 
removing duplicates and short texts, and exporting final statistics.

Dependencies:
-------------
- pandas
- re
- datetime
- locale
- pathlib
- tqdm

MAIN FEATURES:
--------------
1) Identifies date formats and prints samples.
2) Translates and inspects the 'language' column.
3) Cleans database rows by removing very short texts.
4) Removes duplicate rows (title≥90%, date identical, media identical, text≥80%).
5) Normalizes authors and date formatting.
6) Saves the cleaned data and exports it.

Author:
-------
Antoine Lemor
"""

import pandas as pd
import re
from datetime import datetime
import locale
from pathlib import Path
from tqdm import tqdm

# Set the locale for French date format (falls back if not available)
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
except locale.Error:
    print("The locale 'fr_FR.UTF-8' is not available on your system.")
    # You can set another locale or handle the error as needed

# Define the base path relative to the script
script_dir = Path(__file__).resolve().parent  # Script directory
base_dir = script_dir.parent.parent  # Assuming the project structure:
                                     # CLIMATE.FRAME/
                                     # ├── Scripts/
                                     # │   └── Database_creation/
                                     # │       └── 2_Clean_and_create_database.py
                                     # └── Database/
                                     #     └── Database/

# Relative paths to the database and export files
chemin_db = base_dir / 'Database' / 'Database' / 'CCF.media_database.csv'
chemin_export_csv = base_dir / 'Database' / 'Database' / 'Database_media_count.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(chemin_db)


# ------------------------------------------------
# Step 1: Identify date formats
# ------------------------------------------------
def identifier_formats_dates(series_dates):
    """
    Identifies different date formats in a pandas Series.
    
    Parameters:
    -----------
    series_dates : pandas.Series
        Series containing date strings to be tested.

    Returns:
    --------
    tuple
        A tuple with a set of detected formats and a dictionary of examples 
        for each format.
    """
    formats_detectes = set()
    exemples_par_format = {}
    
    # Création d'une barre de progression pour le processus
    valeurs_uniques = series_dates.dropna().unique()
    with tqdm(total=len(valeurs_uniques), desc="Identifying date formats") as pbar:
        for date_str in valeurs_uniques:
            date_str = str(date_str).strip()
            # List of formats to test
            formats_a_tester = [
                '%d %B %Y',    # Example: 25 December 2023
                '%d/%m/%Y',    # Example: 25/12/2023
                '%Y-%m-%d',    # Example: 2023-12-25
                '%m/%d/%Y',    # Example: 12/25/2023
                '%d-%m-%Y',    # Example: 25-12-2023
                '%B %d, %Y',   # Example: December 25, 2023
                '%d %b %Y',    # Example: 25 Dec 2023
                '%d.%m.%Y',    # Example: 25.12.2023
            ]
            for fmt in formats_a_tester:
                try:
                    datetime.strptime(date_str, fmt)
                    formats_detectes.add(fmt)
                    if fmt not in exemples_par_format:
                        exemples_par_format[fmt] = date_str
                    break  # Stop after finding a matching format
                except ValueError:
                    continue
            pbar.update(1)
    
    return formats_detectes, exemples_par_format

formats_dates, exemples_formats = identifier_formats_dates(df['date'])

print("=== Detected date formats ===")
for fmt in formats_dates:
    exemple = exemples_formats.get(fmt, '')
    print(f"Format: {fmt} | Example: {exemple}")


# ------------------------------------------------
# Step 2: Inspect 'language' values
# ------------------------------------------------
valeurs_language = df['language'].dropna().unique()

print("\n=== Unique values in the 'language' column ===")
for valeur in valeurs_language:
    print(valeur)


# ------------------------------------------------
# Step 3: Clean the database
# ------------------------------------------------

# 3.1 Remove rows where 'text' has fewer than 100 words
print("\nFiltering short texts...")
with tqdm(total=len(df), desc="Removing short texts") as pbar:
    condition = []
    for text in df['text']:
        condition.append(len(str(text).split()) >= 100)
        pbar.update(1)
df = df[condition].reset_index(drop=True)

# 3.2 Remove the 'URL' column if it exists
if 'URL' in df.columns:
    df = df.drop(columns=['URL'])


# 3.3 Define similarity functions
def similarite_titres(titre1, titre2):
    """
    Computes similarity between two titles by comparing their words
    (intersection over union).
    
    Parameters:
    -----------
    titre1 : str
        The first title.
    titre2 : str
        The second title.

    Returns:
    --------
    float
        The ratio of the intersection of words to the union of words.
    """
    titre1, titre2 = str(titre1), str(titre2)
    mots_titre1 = set(titre1.lower().split())
    mots_titre2 = set(titre2.lower().split())
    intersection = mots_titre1.intersection(mots_titre2)
    union = mots_titre1.union(mots_titre2)
    return len(intersection) / len(union) if union else 0


def similarite_textes(texte1, texte2):
    """
    Computes similarity between two texts by comparing their words
    (intersection over union).
    
    Parameters:
    -----------
    texte1 : str
        The first text.
    texte2 : str
        The second text.

    Returns:
    --------
    float
        The ratio of the intersection of words to the union of words.
    """
    texte1, texte2 = str(texte1), str(texte2)
    mots_texte1 = set(texte1.lower().split())
    mots_texte2 = set(texte2.lower().split())
    intersection = mots_texte1.intersection(mots_texte2)
    union = mots_texte1.union(mots_texte2)
    return len(intersection) / len(union) if union else 0


# 3.4 Remove duplicates 
#    Condition to consider a row as duplicate of another:
#    - Title similarity ≥ 0.90
#    - Date is exactly the same
#    - Media is the same
#    - Text similarity ≥ 0.80

print("\nRemoving duplicates based on title/date/media/text similarity...")
indices_a_conserver = []
articles_conserves = []  # On stockera ici les informations nécessaires à la comparaison

with tqdm(total=len(df), desc="Checking duplicates") as pbar:
    for idx, row in df.iterrows():
        titre = row['title']
        date = row['date']
        media = row['media']
        texte = row['text']

        # Vérifie si ce nouvel article est un doublon d'un article déjà conservé
        est_doublon = False
        for art in articles_conserves:
            if (similarite_titres(titre, art['title']) >= 0.90 and
                date == art['date'] and
                media == art['media'] and
                similarite_textes(texte, art['text']) >= 0.80):
                est_doublon = True
                break
        
        # Si ce n'est pas un doublon, on le conserve
        if not est_doublon:
            indices_a_conserver.append(idx)
            articles_conserves.append({
                'title': titre,
                'date': date,
                'media': media,
                'text': texte
            })
        pbar.update(1)

df = df.loc[indices_a_conserver].reset_index(drop=True)


# 3.5 Clean the 'author' column
print("\nCleaning 'author' column...")
mots_a_remplacer = [
    'From', ',', 'BY', 'Review by', 'Sources:',
    'Compiled and edited by', 'By', 'by',
    'SPECIAL TO THE STAR', 'Toronto Star'
]
pattern = '|'.join(map(re.escape, mots_a_remplacer))

with tqdm(total=len(df), desc="Cleaning 'author'") as pbar:
    for i in range(len(df)):
        # Supprimer les expressions identifiées
        df.at[i, 'author'] = re.sub(pattern, '', str(df.at[i, 'author']))
        # Supprimer l'auteur si ça contient le mot 'mots'
        if re.search('mots', str(df.at[i, 'author']), flags=re.IGNORECASE):
            df.at[i, 'author'] = ''
        pbar.update(1)


# 3.6 Convert dates to a normalized format
def convertir_date(date_texte):
    """
    Attempt to parse the date_texte according to known formats
    and convert to mm/dd/YYYY. If no format matches, returns the original date.
    """
    formats_a_tester = [
        '%d %B %Y',    # Example: 14 April 2009
        '%Y-%m-%d',    # Example: 1989-03-11
        '%d/%m/%Y',    # If some dates are in DD/MM/YYYY
        '%m/%d/%Y',    # If some dates are in MM/DD/YYYY
    ]
    for fmt in formats_a_tester:
        try:
            return datetime.strptime(date_texte, fmt).strftime('%m/%d/%Y')
        except ValueError:
            continue
    return date_texte  # Return the original if no format matches

print("\nConverting dates to mm/dd/YYYY format...")
with tqdm(total=len(df), desc="Converting dates") as pbar:
    for i in range(len(df)):
        df.at[i, 'date'] = convertir_date(str(df.at[i, 'date']))
        pbar.update(1)


# ------------------------------------------------
# Step 4: Save the changes to the CSV file
# ------------------------------------------------
df.to_csv(chemin_db, index=False)


# ------------------------------------------------
# Step 5: Print and export the number of articles per media
# ------------------------------------------------
def imprimer_et_exporter_nombre_articles_par_media(dataframe):
    articles_par_media = dataframe['media'].value_counts()
    total_articles = articles_par_media.sum()
    
    # Add a row for the total number of articles
    articles_par_media['Total'] = total_articles
    
    # Display the counts
    print("\n=== Number of articles per media ===")
    print(articles_par_media)
    
    # Export the number of articles per media to a CSV file
    articles_par_media.to_csv(chemin_export_csv, header=["Number of articles"])

imprimer_et_exporter_nombre_articles_par_media(df)


print("\nDatabase has been cleaned and saved, and the file has been created.")
