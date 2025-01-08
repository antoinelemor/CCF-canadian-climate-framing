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

MAIN FEATURES:
--------------
1) Identifies date formats and prints samples.
2) Translates and inspects the 'language' column.
3) Cleans database rows by removing very short texts.
4) Removes duplicate titles based on similarity.
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

# Set the locale for French date format (falls back if not available)
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
except locale.Error:
    print("The locale 'fr_FR.UTF-8' is not available on your system.")
    # You can set another locale or handle the error as needed

# Define the base path relative to the script
script_dir = Path(__file__).resolve().parent  # Script directory
base_dir = script_dir.parent.parent  # Assuming the project is structured as follows:
                                    # CLIMATE.FRAME/
                                    # ├── Scripts/
                                    # │   └── Database_creation/
                                    # │       └── 3_Database_creation.py
                                    # └── Database/
                                    #     └── Database/

# Relative paths to the database and export files
chemin_db = base_dir / 'Database' / 'Database' / 'CCF.media_database.csv'
chemin_export_csv = base_dir / 'Database' / 'Database' / 'Database_media_count.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(chemin_db)

# --------------------------------------------
# Step 1: Inspect date formats
# --------------------------------------------

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
    
    for date_str in series_dates.dropna().unique():
        date_str = str(date_str).strip()
        # List of formats to test
        formats_a_tester = [
            '%d %B %Y',    # Example: 25 December 2023
            '%d/%m/%Y',    # Example: 25/12/2023
            '%Y-%m-%d',    # Example: 2023-12-25
            '%m/%d/%Y',    # Example: 12/25/2023
            '%d-%m-%Y',    # Example: 25-12-2023
            '%B %d, %Y',   # Example: December 25, 2023
            '%d %b %Y',    # Example: 25 Dec. 2023
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
    return formats_detectes, exemples_par_format

formats_dates, exemples_formats = identifier_formats_dates(df['date'])

print("=== Detected date formats ===")
for fmt in formats_dates:
    exemple = exemples_formats.get(fmt, '')
    print(f"Format: {fmt} | Example: {exemple}")

# --------------------------------------------
# Step 2: Inspect 'language' values
# --------------------------------------------

valeurs_language = df['language'].dropna().unique()

print("\n=== Unique values in the 'language' column ===")
for valeur in valeurs_language:
    print(valeur)

# --------------------------------------------
# Step 3: Clean the database
# --------------------------------------------

# Remove rows where 'text' has fewer than 100 words
df = df[df['text'].apply(lambda x: len(str(x).split()) >= 100)]

# Remove the 'URL' column
df = df.drop(columns=['URL'])

# Filter based on title similarity
def similarite_titres(titre1, titre2):
    """
    Computes similarity between two titles by comparing their words.
    
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
    mots_titre1, mots_titre2 = set(titre1.lower().split()), set(titre2.lower().split())
    intersection, union = mots_titre1.intersection(mots_titre2), mots_titre1.union(mots_titre2)
    return len(intersection) / len(union) if union else 0

indices_a_conserver = []
titres_conserves = []

for i, titre in enumerate(df['title']):
    if all(similarite_titres(titre, titre_conserve) < 0.90 for titre_conserve in titres_conserves):
        indices_a_conserver.append(i)
        titres_conserves.append(titre)

# Filter the DataFrame to keep only unique rows in terms of similarity
df = df.iloc[indices_a_conserver].reset_index(drop=True)

# Clean the 'author' column
mots_a_remplacer = [
    'From', ',', 'BY', 'Review by', 'Sources:', 
    'Compiled and edited by', 'By', 'by', 
    'SPECIAL TO THE STAR', 'Toronto Star'
]
pattern = '|'.join(map(re.escape, mots_a_remplacer))
df['author'] = df['author'].replace(pattern, '', regex=True)
df.loc[df['author'].str.contains('mots', case=False, na=False), 'author'] = ''

# Convert dates
def convertir_date(date_texte):
    formats_a_tester = [
        '%d %B %Y',    # Example: 14 April 2009
        '%Y-%m-%d',    # Example: 1989-03-11
    ]
    for fmt in formats_a_tester:
        try:
            return datetime.strptime(date_texte, fmt).strftime('%m/%d/%Y')
        except ValueError:
            continue
    return date_texte  # Return the original date if no format matches

df['date'] = df['date'].apply(convertir_date)

# Save the changes to the CSV file
df.to_csv(chemin_db, index=False)

# Function to print and export the number of articles per media
def imprimer_et_exporter_nombre_articles_par_media(df):
    articles_par_media = df['media'].value_counts()
    total_articles = articles_par_media.sum()
    
    # Add a row for the total number of articles
    articles_par_media['Total'] = total_articles
    
    # Display the counts
    print("\n=== Number of articles per media ===")
    print(articles_par_media)
    
    # Export the number of articles per media to a CSV file
    articles_par_media.to_csv(chemin_export_csv, header=["Number of articles"])

# Call the function to display and export the number of articles per media
imprimer_et_exporter_nombre_articles_par_media(df)

print("\nDatabase has been cleaned and saved, and file has been created.")
