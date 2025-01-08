"""
PROJECT:
--------
CCF-canadian-climate-framing

TITLE:
------
1_Database_creation.py

MAIN OBJECTIVE:
---------------
This script merges multiple CSV files (media articles scrapped online) into a single database, 
assigns media names, renames columns to uniform naming, 
and forces language assignment for each entry. 
Finally, it saves the combined DataFrame to a CSV file.

Dependencies:
-------------
- pandas
- re
- unidecode
- pathlib

MAIN FEATURES:
--------------
1) Aggregates multiple CSV files into a single dataset.
2) Maps short media keys to their full names.
3) Enforces the assignment of language based on media name.
4) Exports the final DataFrame to a CSV file.

Author:
-------
Antoine Lemor
"""
import pandas as pd
import re
from unidecode import unidecode
from pathlib import Path

# Adjusted based on script depth
base_dir = Path(__file__).resolve().parent.parent.parent

# Paths to your CSV files relative to the base directory
chemins = {
    'CH': base_dir / 'Database/Raw_data/Calgary_Herald/clean_CH_data/articles_extraits.csv',
    'GM': base_dir / 'Database/Raw_data/Global_and_Mail/clean_GM_data/articles_extraits.csv',
    'TS': base_dir / 'Database/Raw_data/Toronto_Star/clean_TS_data/articles_extraits.csv',
    'NP': base_dir / 'Database/Raw_data/National_Post/clean_NP_data/articles_extraits.csv',
    'LP': base_dir / 'Database/Raw_data/La_Presse_ris/clean_LP_data/articles_extraits.csv',
    'LD': base_dir / 'Database/Raw_data/Le_Devoir/clean_LD_data/articles_extraits.csv',
    'JdM': base_dir / 'Database/Raw_data/Journal_de_Montreal/clean_JDM_data/articles_extraits.csv',
    'LPP': base_dir / 'Database/Raw_data/La_Presse_Plus/clean_LPP_data/articles_extraits.csv',
    'CHH': base_dir / 'Database/Raw_data/Chronicle_Herald/Clean_Chronicle_Herald_data/articles_extraits.csv',
    'VS': base_dir / 'Database/Raw_data/Vancouver_Sun/clean_VS_data/articles_extraits.csv',
    'WHD': base_dir / 'Database/Raw_data/Whitehorse_Daily_Star/Clean_White_Horse_data/articles_extraits.csv',
    'AcN': base_dir / 'Database/Raw_data/Acadie_Nouvelle/clean_Acadie_Nouvelle_data/articles_extraits.csv',
    'LeDr': base_dir / 'Database/Raw_data/Le_Droit/Le_Droit_clean_data/articles_extraits.csv',
    'Tel': base_dir / 'Database/Raw_data/Telegram/Telegram_clean_data/articles_extraits.csv',
    'EdJo': base_dir / 'Database/Raw_data/Edmonton_Journal/ED_clean_data/articles_extraits.csv',
    'MoGa': base_dir / 'Database/Raw_data/Montreal_Gazette/clean_Montreal_Gazette_data/articles_extraits.csv',
    'StPh': base_dir / 'Database/Raw_data/Star_Phoenix/clean_Star_Phoenix_data/articles_extraits.csv',
    'TorSun': base_dir / 'Database/Raw_data/Toronto_Sun/clean_Toronto_Sun_data/articles_extraits.csv',
    'TiCo': base_dir / 'Database/Raw_data/Times_Colonist/Times_Colonist_clean_data/articles_extraits.csv',
    'WiFrPr': base_dir / 'Database/Raw_data/Winnipeg_Free_Press/Winnipeg_Free_Press_clean_data/articles_extraits.csv'
}

# Dictionary mapping keys to their full media names
media_mapping = {
    'CH': 'Calgary Herald',
    'GM': 'Globe and Mail',
    'TS': 'Toronto Star',
    'NP': 'National Post',
    'LP': 'La Presse',
    'LD': 'Le Devoir',
    'JdM': 'Journal de Montreal',
    'LPP': 'La Presse Plus',
    'CHH': 'Chronicle Herald',
    'VS': 'Vancouver Sun',
    'WHD': 'Whitehorse Daily Star',
    'AcN': 'Acadie Nouvelle',
    'LeDr': 'Le Droit',
    'Tel': 'The Telegram',
    'EdJo': 'Edmonton Journal',
    'MoGa': 'Montreal Gazette',
    'StPh': 'Star Phoenix',
    'TorSun': 'Toronto Sun',
    'TiCo': 'Times Colonist',
    'WiFrPr': 'Winnipeg Free Press'
}

# Check that all path keys are in media_mapping
missing_keys = set(chemins.keys()) - set(media_mapping.keys())
if missing_keys:
    print(f"[WARNING] The following keys are not in the media mapping dictionary: {missing_keys}")
    # You can decide to add the missing mappings or handle otherwise
    # For example:
    # media_mapping['NewKey'] = 'Media Name'
    # Or raise an error
    # raise ValueError(f"Missing mappings for keys: {missing_keys}")

# Read CSV files into pandas DataFrames
dataframes = {}
for key, chemin in chemins.items():
    try:
        df = pd.read_csv(chemin)
        # Assign the 'Media' column using the mapping dictionary
        media_nom = media_mapping.get(key, 'Unknown Media')  # 'Unknown Media' if the key is not found
        df['Média'] = media_nom
        dataframes[key] = df
    except FileNotFoundError:
        print(f"[ERROR] The file {chemin} was not found.")
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The file {chemin} is empty.")
    except Exception as e:
        print(f"[ERROR] An error occurred while reading {chemin}: {e}")

# Combine the DataFrames into one
if dataframes:
    df_combiné = pd.concat(dataframes.values(), ignore_index=True)
else:
    print("[INFO] No DataFrame to combine. Check previous errors.")
    exit(1)

# Rename columns according to new specifications
nouveaux_noms = {
    'Type de nouvelle': 'news_type',
    'Titre': 'title',
    'Auteur': 'author',
    'Média': 'media',
    'Nombre de mots': 'words_count',
    'Date': 'date',
    'Langue': 'language',
    'Numéro de page': 'page_number',
    'Texte': 'text'
}

df_combiné.rename(columns=nouveaux_noms, inplace=True)

# --------------------------------------------
# Step: Forced language assignment
# --------------------------------------------

# Define a dictionary mapping media to language (with normalized keys)
mapping_language = {
    'calgary herald': 'EN',
    'globe and mail': 'EN',
    'toronto star': 'EN',
    'national post': 'EN',
    'chronicle herald': 'EN',
    'vancouver sun': 'EN',
    'whitehorse daily star': 'EN',
    'the telegram': 'EN',
    'edmonton journal': 'EN',
    'montreal gazette': 'EN',
    'star phoenix': 'EN',
    'toronto sun': 'EN',
    'times colonist': 'EN',
    'winnipeg free press': 'EN',
    'la presse': 'FR',
    'le devoir': 'FR',
    'journal de montreal': 'FR',         
    'la presse+': 'FR',
    'la presse plus': 'FR',
    'acadie nouvelle': 'FR',
    'le droit': 'FR'
}

# Function to normalize media names
def normalize_media(name):
    """
    Normalizes the media name by converting it to lowercase,
    removing diacritics, and replacing multiple spaces with a single space.
    """
    if pd.isnull(name):
        return ''
    name = unidecode(name.lower().strip())
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with a single one
    return name

# Apply normalization
df_combiné['media_normalized'] = df_combiné['media'].apply(normalize_media)

# Assign language based on the mapping
df_combiné['language'] = df_combiné['media_normalized'].map(mapping_language).fillna('Unknown')

# Optional: Check assignments
print("\n=== Example of language assignment ===")
print(df_combiné[['media', 'language']].drop_duplicates())

# Identify and display unassigned media (optional)
medias_unknown = df_combiné[df_combiné['language'] == 'Unknown']['media'].unique()
if len(medias_unknown) > 0:
    print("\n=== Unassigned media ===")
    for media in medias_unknown:
        print(media)

# Remove the temporary normalization column
df_combiné.drop(columns=['media_normalized'], inplace=True)

# Path to save the combined database (relative path)
chemin_sauvegarde = base_dir / 'Database/Database/CCF.media_database.csv'

# Save the combined DataFrame to a new CSV file
try:
    df_combiné.to_csv(chemin_sauvegarde, index=False)
    print("\nThe combined database has been created and saved here:", chemin_sauvegarde)
except Exception as e:
    print(f"Error saving the file {chemin_sauvegarde}: {e}")
