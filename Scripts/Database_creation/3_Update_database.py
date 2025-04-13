"""
PROJECT:
--------
CCF-canadian-climate-framing

TITLE:
------
3_Update_and_Clean_media_database.py

MAIN OBJECTIVE:
---------------
Update and clean the main media database by importing new articles from three CSV sources:
  - CCF_eureka_data.csv
  - CCF_factiva_data.csv
  - CCF_proquest_data.csv

This script performs the following tasks:
  • Imports and cleans new articles by normalizing media names, languages, and date formats.
  • Combines new articles with the main database.
  • Removes extremely short articles (less than 100 words).
  • Cleans the 'author' column.
  • Normalizes the 'language' column.
  • Adds a unique doc_id to each article for traceability (this is now done before CSV exports).
  • Exports CSV files for:
      - Articles with date formats not matching "mm-dd-yyyy" (date.csv).
      - Articles with unknown media (media_inconnus.csv).
      - Articles with any unknown field (media, author, or date equal to "Inconnu") (inconnu.csv).
      - Articles with more than three words in the 'page_number' field (page_number.csv).
      - Articles with similar titles (>= 90% similarity) within the same media group (doublons.csv).
  • Removes duplicate articles using parallel processing on media groups. Two duplicate tests are used:
      - Full duplicate check (title similarity ≥ 90%, matching date and media, text similarity ≥ 80%).
      - Title-only duplicate check (≥ 90% similarity) for reporting duplicates.
  • Recalculates word counts using spaCy (or falls back to a simple split) in parallel.
  • Updates and exports article counts per media.
  • Saves the updated main media database.

Author:
-------
Antoine Lemor
"""

import pandas as pd
import re
import locale
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# -------------------------------------------------------------------------
# 0) Optional: spaCy usage for accurate word counting
# -------------------------------------------------------------------------
USE_SPACY = True
try:
    import spacy
    # Attempt to load the French and English spaCy language models.
    nlp_fr = spacy.load("fr_dep_news_trf")  # French model
    nlp_en = spacy.load("en_core_web_lg")     # English model
except Exception as e:
    print("[WARNING] spaCy or its models are not available. Falling back to simple split for word count.")
    print("Error:", e)
    USE_SPACY = False

# -------------------------------------------------------------------------
# 1) Define Relative Paths for the Main Database and CSV Outputs
# -------------------------------------------------------------------------
# Determine the script directory and base directory (two levels up)
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent.parent  # Two levels up from Scripts/Database_creation

# Main database and output file paths
main_db_path = base_dir / "Database" / "Database" / "CCF.media_database.csv"
media_count_path = base_dir / "Database" / "Database" / "Database_media_count.csv"
date_csv_path = base_dir / "Database" / "Database" / "date.csv"
media_inconnus_path = base_dir / "Database" / "Database" / "media_inconnus.csv"
doublons_path = base_dir / "Database" / "Database" / "doublons.csv"
inconnu_path = base_dir / "Database" / "Database" / "inconnu.csv"
page_number_path = base_dir / "Database" / "Database" / "page_number.csv"

# CSV files to import
eureka_path = base_dir / "Database" / "Raw_data" / "new_articles" / "CCF_eureka_data.csv"
factiva_path = base_dir / "Database" / "Raw_data" / "new_articles" / "CCF_factiva_data.csv"
proquest_path = base_dir / "Database" / "Raw_data" / "new_articles" / "CCF_proquest_data.csv"

# -------------------------------------------------------------------------
# 2) Mappings for Media and Language Normalization
# -------------------------------------------------------------------------
# Mapping for normalizing media names from different sources
media_map = {
    # EUREKA
    "La Presse+": "La Presse Plus",
    "The Whitehorse Daily Star (YT)": "Whitehorse Daily Star",
    "Toronto Star (ON)": "Toronto Star",
    "Le Journal de Montréal": "Journal de Montreal",
    # FACTIVA
    "Journal de Montréal": "Journal de Montreal",
    "Le Journal de Québec": "Journal de Quebec",  # These will be excluded after mapping
    "Saskatoon Star Phoenix": "Star Phoenix",
    "The Globe and Mail": "Globe and Mail",
    "The Toronto Star": "Toronto Star",
    "The Toronto Sun": "Toronto Sun",
    "Victoria Times Colonist": "Times Colonist",
    # PROQUEST
    "Calgary Herald; Calgary, Alta.": "Calgary Herald",
    "Chronicle - Herald; Halifax, N.S.": "Chronicle Herald",
    "Edmonton Journal; Edmonton, Alta.": "Edmonton Journal",
    "The Globe and Mail; Toronto, Ont.": "Globe and Mail",
    "The Vancouver Sun; Vancouver, B.C.": "Vancouver Sun",
    "Toronto Star; Toronto, Ont.": "Toronto Star",
    "Winnipeg Free Press; Winnipeg, Man.": "Winnipeg Free Press",
}

# Mapping for normalizing language values
language_map = {
    "english": "EN",
    "anglais": "EN",
    "français": "FR",
    "fra": "FR",
    "inconnu": "Inconnu",
}

# Additional mapping to group similar media names together
MEDIA_GROUP_MAP = {
    "Montreal Gazette": [
        "montreal gazette",
        "montrealgazette;montreal, que."
    ],
    "National Post": [
        "national post",
        "national post;don mills, ont.",
        "nationalpost;don mills, ont."
    ],
    "Star Phoenix": [
        "star phoenix",
        "star - phoenix;saskatoon, sask.",
        "star-phoenix;saskatoon, sask."
    ],
    "Chronicle Herald": [
        "chronicle herald",
        "the chronicle herald (halifax, ns)",
        "the chronicle-herald"
    ],
    "The Financial Post": [
        "the financial post",
        "financial post"
    ],
    "The Telegram": [
        "the telegram",
        "the telegram (st. john's)",
        "the telegram (st. john's, nl)"
    ],
    "Le Droit": [
        "le droit",
        "le droit (ottawa, on)"
    ],
    "Whitehorse Daily Star": [
        "whitehorse dailystar",
        "the whitehorse star",
        "the whitehorse daily star (yk)"
    ],
    "Times Colonist": [
        "times colonist",
        "times - colonist;victoria, b.c.",
        "times-colonist;victoria, b.c."
    ],
    "Toronto Star": [
        "toronto star",
        "toronto star (on) (pick)",
        "toronto sun"
    ]
}

# -------------------------------------------------------------------------
# 3) Utility Functions for Normalization and Cleaning
# -------------------------------------------------------------------------
def unify_media_name(media_val: str) -> str:
    """
    Normalize the media name using media_map and MEDIA_GROUP_MAP.

    Parameters:
        media_val (str): The original media name.

    Returns:
        str: The normalized media name.
    """
    if pd.isna(media_val):
        return media_val
    media_val_clean = media_val.strip()
    normalized = re.sub(r"\s+", "", media_val_clean).lower()
    # First, try to match using media_map
    for key, mapped in media_map.items():
        key_normalized = re.sub(r"\s+", "", key).lower()
        if normalized == key_normalized:
            return mapped
    # Then, check if it fits any media group in MEDIA_GROUP_MAP
    for group, synonyms in MEDIA_GROUP_MAP.items():
        for syn in synonyms:
            syn_normalized = re.sub(r"\s+", "", syn).lower()
            if normalized == syn_normalized:
                return group
    return media_val_clean

def unify_language(lang_val: str) -> str:
    """
    Normalize the language value using the language_map.

    Parameters:
        lang_val (str): The original language value.

    Returns:
        str: The normalized language code.
    """
    if not isinstance(lang_val, str):
        return lang_val
    return language_map.get(lang_val.strip().lower(), lang_val.strip())

def clean_language(lang: str) -> str:
    """
    Further clean the language value by mapping variants to canonical codes.

    Parameters:
        lang (str): Original language string.

    Returns:
        str: 'FR' if the language indicates French, 'EN' if it indicates English,
             otherwise returns the original string.
    """
    if not isinstance(lang, str):
        return lang
    lang_clean = lang.strip().lower()
    if lang_clean in ["français", "fra"]:
        return "FR"
    if lang_clean == "anglais":
        return "EN"
    return lang.strip()

# Set locale for date parsing to French
try:
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
except locale.Error:
    print("[INFO] 'fr_FR.UTF-8' locale not available. French date parsing may be limited.")

# ------------------------------ Date Parsing Functions ------------------------------
def parse_date_eureka(date_str: str) -> str:
    """
    Parse dates from the Eureka CSV file and format them as "mm-dd-yyyy".

    Parameters:
        date_str (str): The raw date string.

    Returns:
        str: The formatted date string "mm-dd-yyyy", or the original string if parsing fails.
    """
    if not isinstance(date_str, str):
        return date_str
    date_str = date_str.strip()
    if date_str.lower() == "inconnu":
        return "Inconnu"
    # Handle Eureka numeric dates with trailing .0 (e.g., "20180608.0")
    if re.match(r"^\d{8}\.0$", date_str):
        try:
            dt = datetime.strptime(date_str[:-2], "%Y%m%d")
            return dt.strftime("%m-%d-%Y")
        except Exception:
            pass
    possible_formats = [
        "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d",
    ]
    for fmt in possible_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%m-%d-%Y")
        except ValueError:
            continue
    return date_str

def parse_date_factiva(date_str: str) -> str:
    """
    Parse dates from the Factiva CSV file and format them as "mm-dd-yyyy".

    Parameters:
        date_str (str): The raw date string.

    Returns:
        str: The formatted date string "mm-dd-yyyy", or the original string if parsing fails.
    """
    if not isinstance(date_str, str):
        return date_str
    date_str = date_str.strip()
    if date_str.lower() == "inconnu":
        return "Inconnu"
    possible_formats = [
        "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    ]
    for fmt in possible_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%m-%d-%Y")
        except ValueError:
            continue
    return date_str

def parse_date_proquest(date_str: str) -> str:
    """
    Parse dates from the Proquest CSV file and format them as "mm-dd-yyyy".
    
    Proquest dates are provided in an English format (e.g., 'Feb 18, 1978'). To ensure correct
    parsing even when the global locale is set to French, this function temporarily sets the locale to English.

    Parameters:
        date_str (str): The raw date string.

    Returns:
        str: The formatted date string "mm-dd-yyyy", or the original string if parsing fails.
    """
    if not isinstance(date_str, str):
        return date_str
    dstr = date_str.strip()
    if dstr.lower() == "inconnu":
        return "Inconnu"
    possible_formats = [
        "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d",
    ]
    current_locale = locale.getlocale(locale.LC_TIME)
    try:
        locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    except locale.Error:
        pass
    try:
        for fmt in possible_formats:
            try:
                dt = datetime.strptime(dstr, fmt)
                return dt.strftime("%m-%d-%Y")
            except ValueError:
                continue
    finally:
        try:
            locale.setlocale(locale.LC_TIME, current_locale)
        except Exception:
            pass
    return date_str

def parse_date_main_db(date_str: str) -> str:
    """
    Normalize dates from the main database to the "mm-dd-yyyy" format.

    Parameters:
        date_str (str): The raw date string.

    Returns:
        str: The formatted date string "mm-dd-yyyy", or the original string if parsing fails.
    """
    if not isinstance(date_str, str):
        return date_str
    temp = date_str.strip()
    if temp.lower() == "inconnu":
        return "Inconnu"
    possible_formats = [
        "%m-%d-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y",
    ]
    for fmt in possible_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%m-%d-%Y")
        except ValueError:
            continue
    return date_str

# ------------------------------ Word Counting and Similarity Functions ------------------------------
def count_words(text: str, lang: str) -> int:
    """
    Count the number of words or tokens in a text.
    Uses spaCy if available and the language is recognized; otherwise, falls back to a simple split.

    Parameters:
        text (str): The text to analyze.
        lang (str): The language code ("EN", "FR", etc.).

    Returns:
        int: The number of tokens/words.
    """
    if not isinstance(text, str):
        return 0
    if USE_SPACY:
        lang_clean = lang.strip().upper() if isinstance(lang, str) else "INCONNU"
        try:
            if lang_clean == "FR":
                doc = nlp_fr(text)
            elif lang_clean == "EN":
                doc = nlp_en(text)
            else:
                return len(text.split())
            tokens = [t for t in doc if not t.is_space and not t.is_punct]
            return len(tokens)
        except Exception:
            return len(text.split())
    else:
        return len(text.split())

def title_similarity(t1: str, t2: str) -> float:
    """
    Compute the similarity ratio between two titles based on set overlap.

    Parameters:
        t1 (str): The first title.
        t2 (str): The second title.

    Returns:
        float: A similarity score between 0 and 1.
    """
    set1 = set(str(t1).lower().split())
    set2 = set(str(t2).lower().split())
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def text_similarity(x1: str, x2: str) -> float:
    """
    Compute the similarity ratio between two text bodies based on the overlap of word sets.

    Parameters:
        x1 (str): The first text.
        x2 (str): The second text.

    Returns:
        float: A similarity score between 0 and 1.
    """
    set1 = set(str(x1).lower().split())
    set2 = set(str(x2).lower().split())
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def is_duplicate(row_new: dict, row_keep: dict) -> bool:
    """
    Determine if one article is a duplicate of another based on:
      - Title similarity ≥ 90%
      - Matching date and media
      - Text similarity ≥ 80%

    Parameters:
        row_new (dict): Data of the new article.
        row_keep (dict): Data of an existing article.

    Returns:
        bool: True if the article is considered a duplicate; False otherwise.
    """
    if title_similarity(row_new["title"], row_keep["title"]) < 0.90:
        return False
    if row_new["date"] != row_keep["date"]:
        return False
    if row_new["media"] != row_keep["media"]:
        return False
    if text_similarity(row_new["text"], row_keep["text"]) < 0.80:
        return False
    return True

def clean_author_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'author' column by removing unwanted phrases.

    Parameters:
        df (pd.DataFrame): The DataFrame containing an 'author' column.

    Returns:
        pd.DataFrame: The DataFrame with cleaned 'author' values.
    """
    patterns_to_remove = [
        "From", ",", "BY", "Review by", "Sources:",
        "Compiled and edited by", "By", "by",
        "SPECIAL TO THE STAR", "Toronto Star"
    ]
    combined_pattern = "|".join(map(re.escape, patterns_to_remove))
    df["author"] = df["author"].replace(combined_pattern, "", regex=True)
    df.loc[df["author"].str.contains("mots", case=False, na=False), "author"] = ""
    return df

# ------------------------------ CSV Loading and Cleaning Functions ------------------------------
def load_and_clean_eureka(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the Eureka CSV file by selecting relevant columns, normalizing fields,
    and parsing dates into the "mm-dd-yyyy" format.

    Parameters:
        csv_path (Path): Path to the Eureka CSV file.

    Returns:
        pd.DataFrame: A DataFrame of cleaned Eureka articles.
    """
    print("[INFO] Loading Eureka data...")
    df_eur = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    wanted_cols = ["news_type", "title", "author", "media", "date", "language", "page_number", "text"]
    df_eur = df_eur[[c for c in wanted_cols if c in df_eur.columns]]
    for col in tqdm(df_eur.columns, desc="Normalizing Eureka columns", unit="col"):
        if col == "media":
            df_eur["media"] = df_eur["media"].apply(unify_media_name)
        elif col == "language":
            df_eur["language"] = df_eur["language"].apply(unify_language)
        elif col == "date":
            df_eur["date"] = df_eur["date"].astype(str).apply(parse_date_eureka)
    return df_eur

def load_and_clean_factiva(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the Factiva CSV file by selecting relevant columns, normalizing fields,
    and parsing dates into the "mm-dd-yyyy" format.

    Parameters:
        csv_path (Path): Path to the Factiva CSV file.

    Returns:
        pd.DataFrame: A DataFrame of cleaned Factiva articles.
    """
    print("[INFO] Loading Factiva data...")
    df_fac = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    wanted_cols = ["news_type", "title", "author", "media", "date", "language", "page_number", "text"]
    df_fac = df_fac[[c for c in wanted_cols if c in df_fac.columns]]
    for col in tqdm(df_fac.columns, desc="Normalizing Factiva columns", unit="col"):
        if col == "media":
            df_fac["media"] = df_fac["media"].apply(unify_media_name)
        elif col == "language":
            df_fac["language"] = df_fac["language"].apply(unify_language)
        elif col == "date":
            df_fac["date"] = df_fac["date"].astype(str).apply(parse_date_factiva)
    return df_fac

def load_and_clean_proquest(csv_path: Path) -> pd.DataFrame:
    """
    Load and clean the Proquest CSV file (using ';' as the separator) by selecting relevant columns,
    normalizing fields, and parsing dates into the "mm-dd-yyyy" format.

    Parameters:
        csv_path (Path): Path to the Proquest CSV file.

    Returns:
        pd.DataFrame: A DataFrame of cleaned Proquest articles.
    """
    print("[INFO] Loading Proquest data...")
    df_pro = pd.read_csv(csv_path, sep=";", engine="python", on_bad_lines="skip")
    wanted_cols = ["news_type", "title", "author", "media", "date", "language", "page_number", "text"]
    df_pro = df_pro[[c for c in wanted_cols if c in df_pro.columns]]
    if "text" in df_pro.columns:
        df_pro["text"] = df_pro["text"].astype(str).replace(r'\s*\n\s*', ' ', regex=True)
    for col in tqdm(df_pro.columns, desc="Normalizing Proquest columns", unit="col"):
        if col == "media":
            df_pro["media"] = df_pro["media"].apply(unify_media_name)
        elif col == "language":
            df_pro["language"] = df_pro["language"].apply(unify_language)
        elif col == "date":
            df_pro["date"] = df_pro["date"].astype(str).apply(parse_date_proquest)
    return df_pro

# ------------------------------ Duplicate Removal Functions ------------------------------
def remove_duplicates_in_group(df_group: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Remove duplicate articles within a media group based on:
      - Title similarity ≥ 90%
      - Matching date and media
      - Text similarity ≥ 80%

    Parameters:
        df_group (pd.DataFrame): DataFrame containing articles for a specific media group.

    Returns:
        tuple: (Filtered DataFrame for the group, list of indices removed as duplicates)
    """
    kept_indices = []
    removed_indices = []
    kept_rows = []  # List of dictionaries for kept articles
    data_iter = list(df_group.itertuples(index=True, name=None))
    for row_tuple in data_iter:
        idx = row_tuple[0]
        row_dict = {
            "title": row_tuple[df_group.columns.get_loc("title") + 1],
            "media": row_tuple[df_group.columns.get_loc("media") + 1],
            "date": row_tuple[df_group.columns.get_loc("date") + 1],
            "text": row_tuple[df_group.columns.get_loc("text") + 1],
        }
        duplicate_found = False
        for kept in kept_rows:
            if is_duplicate(row_dict, kept):
                duplicate_found = True
                break
        if duplicate_found:
            removed_indices.append(idx)
        else:
            kept_indices.append(idx)
            kept_rows.append(row_dict)
    df_filtered = df_group.loc[kept_indices].copy()
    return df_filtered, removed_indices

def find_title_duplicates_in_group(df_group: pd.DataFrame) -> list:
    """
    Identify articles within a media group that have at least one other article with a title similarity ≥ 90%.

    Parameters:
        df_group (pd.DataFrame): DataFrame containing articles for a specific media group.

    Returns:
        list: List of row indices identified as title duplicates.
    """
    duplicate_indices = set()
    rows = list(df_group.itertuples(index=True, name=None))
    n = len(rows)
    for i in range(n):
        title_i = rows[i][df_group.columns.get_loc("title") + 1]
        for j in range(i + 1, n):
            title_j = rows[j][df_group.columns.get_loc("title") + 1]
            if title_similarity(title_i, title_j) >= 0.90:
                duplicate_indices.add(rows[i][0])
                duplicate_indices.add(rows[j][0])
    return list(duplicate_indices)

# ------------------------------ Parallel Word Count Functions ------------------------------
def init_spacy():
    """
    Initialize spaCy models in the worker process.
    This ensures that each worker process loads the necessary language models only once.
    """
    global nlp_fr, nlp_en
    if USE_SPACY:
        import spacy
        try:
            nlp_fr = spacy.load("fr_dep_news_trf")
        except Exception as e:
            print("[WARNING] Worker failed to load the French spaCy model:", e)
            nlp_fr = None
        try:
            nlp_en = spacy.load("en_core_web_lg")
        except Exception as e:
            print("[WARNING] Worker failed to load the English spaCy model:", e)
            nlp_en = None

def count_words_wrapper(args: tuple) -> int:
    """
    Wrapper function to count the number of words in a given text with the specified language.
    
    Parameters:
        args (tuple): A tuple containing (text, lang).

    Returns:
        int: The word count.
    """
    text, lang = args
    return count_words(text, lang)

# ------------------------------ Main Processing Function ------------------------------
def main():
    """Main function to update and clean the media database."""
    # -------------------------------------------------------------------------
    # 1) Read the Main Database
    # -------------------------------------------------------------------------
    print("[INFO] Reading main database...")
    df_main = pd.read_csv(main_db_path)
    
    # -------------------------------------------------------------------------
    # 2) Load New Articles from All CSV Sources
    # -------------------------------------------------------------------------
    df_eureka = load_and_clean_eureka(eureka_path)
    df_factiva = load_and_clean_factiva(factiva_path)
    df_proquest = load_and_clean_proquest(proquest_path)
    
    # -------------------------------------------------------------------------
    # 3) Concatenate New Articles and Apply Exclusions
    # -------------------------------------------------------------------------
    print("[INFO] Concatenating new articles...")
    df_new_articles = pd.concat([df_eureka, df_factiva, df_proquest], ignore_index=True)
    # Exclude rows where 'media' is "Agence QMI" (case-insensitive, ignoring extra spaces)
    df_new_articles = df_new_articles[~df_new_articles["media"].str.replace(" ", "").str.lower().eq("agenceqmi")]
    print("[INFO] Excluding 'Journal de Québec' articles (mapped to 'Journal de Quebec')...")
    df_new_articles = df_new_articles[df_new_articles["media"] != "Journal de Quebec"]
    
    # -------------------------------------------------------------------------
    # 4) Normalize Dates in the Main Database and Combine with New Articles
    # -------------------------------------------------------------------------
    print("[INFO] Normalizing dates in the main database...")
    df_main["date"] = df_main["date"].apply(parse_date_main_db)
    # Ensure the 'words_count' column exists (will be recalculated later)
    if "words_count" not in df_main.columns:
        df_main["words_count"] = None
    print("[INFO] Combining main database with new articles...")
    df_combined = pd.concat([df_main, df_new_articles], ignore_index=True)
    
    # -------------------------------------------------------------------------
    # 5) Remove Extremely Short Articles (< 100 words) and Clean the 'author' Column
    # -------------------------------------------------------------------------
    def too_short(text: str) -> bool:
        """
        Determine if an article's text is too short (i.e., less than 100 words).

        Parameters:
            text (str): The article text.

        Returns:
            bool: True if the text is missing or too short, False otherwise.
        """
        return not isinstance(text, str) or len(text.split()) < 100
    
    print("[INFO] Removing articles with very short texts (< 100 words)...")
    df_combined = df_combined[~df_combined["text"].apply(too_short)].copy()
    print("[INFO] Cleaning 'author' column...")
    df_combined = clean_author_column(df_combined)
    
    # -------------------------------------------------------------------------
    # 6) Further Normalize the 'language' Column and Display Unique Values
    # -------------------------------------------------------------------------
    print("[INFO] Further normalizing language column...")
    df_combined["language"] = df_combined["language"].apply(clean_language)
    print("[INFO] Unique language values:")
    print(df_combined["language"].unique())
    print("[INFO] Unique media values:")
    print(df_combined["media"].unique())
    
    # -------------------------------------------------------------------------
    # 6.5) Add Unique doc_id for Article Tracking BEFORE Any CSV Export
    # -------------------------------------------------------------------------
    df_combined.reset_index(drop=True, inplace=True)
    df_combined["doc_id"] = range(1, len(df_combined) + 1)
    
    # -------------------------------------------------------------------------
    # 7) Export Articles with Formatting Issues and Unknown Field Values
    # -------------------------------------------------------------------------
    # Export articles whose date does not match the "mm-dd-yyyy" pattern.
    pattern_date = r'^\d{2}-\d{2}-\d{4}$'
    df_wrong_date = df_combined[~df_combined["date"].astype(str).str.match(pattern_date)]
    df_wrong_date.to_csv(date_csv_path, index=False)
    print(f"[INFO] Exported {len(df_wrong_date)} rows with incorrect date format to {date_csv_path}")
    
    # Export articles where 'media' is 'inconnu' or 'inconnue' (case-insensitive).
    df_media_unknown = df_combined[df_combined["media"].astype(str).str.lower().isin(["inconnu", "inconnue"])]
    df_media_unknown.to_csv(media_inconnus_path, index=False)
    print(f"[INFO] Exported {len(df_media_unknown)} rows with unknown media to {media_inconnus_path}")
    
    # Export articles with any unknown field (media, author, or date equal to "inconnu").
    condition_inconnu = (
        df_combined["media"].str.strip().str.lower().eq("inconnu") |
        df_combined["author"].str.strip().str.lower().eq("inconnu") |
        df_combined["date"].astype(str).str.strip().str.lower().eq("inconnu")
    )
    df_inconnu = df_combined[condition_inconnu].copy()
    df_inconnu.to_csv(inconnu_path, index=False)
    print(f"[INFO] Exported {len(df_inconnu)} rows with unknown values to {inconnu_path}")
    
    # -------------------------------------------------------------------------
    # 8) Duplicate Removal: Using Parallel Processing on Media Groups
    # -------------------------------------------------------------------------
    print("[INFO] Removing duplicate articles in parallel by media groups...")
    groups = list(df_combined.groupby("media"))
    cleaned_groups = []
    all_removed_indices = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(remove_duplicates_in_group, group_df): media for media, group_df in groups}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Processing media groups", unit="group"):
            try:
                df_clean_group, removed_idxs_group = future.result()
                cleaned_groups.append(df_clean_group)
                all_removed_indices.extend(removed_idxs_group)
            except Exception as exc:
                media_key = futures[future]
                print(f"[ERROR] Media group {media_key} generated an exception: {exc}")
    
    df_clean = pd.concat(cleaned_groups).copy()
    print("=== Duplicate Articles Removed ===")
    print(f"Number of removed duplicates: {len(all_removed_indices)}")
    if all_removed_indices:
        df_removed = df_combined.loc[all_removed_indices]
        print("Showing first 10 removed duplicates:")
        print(df_removed[["doc_id", "title", "media", "date"]].head(10))
    
    # Identify title-only duplicates (≥ 90% similarity) per media group for reporting.
    print("[INFO] Identifying title-only duplicates for doublons.csv...")
    title_duplicate_indices = set()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures_dup = {executor.submit(find_title_duplicates_in_group, group_df): media for media, group_df in groups}
        for future in tqdm(concurrent.futures.as_completed(futures_dup), total=len(futures_dup),
                           desc="Finding title duplicates", unit="group"):
            try:
                indices = future.result()
                title_duplicate_indices.update(indices)
            except Exception as exc:
                media_key = futures_dup[future]
                print(f"[ERROR] Media group {media_key} generated an exception in title duplicate search: {exc}")
    
    df_doublons = df_clean.loc[df_clean.index.isin(title_duplicate_indices)].copy()
    print(f"[INFO] Number of articles identified as title duplicates: {len(df_doublons)}")
    df_doublons.to_csv(doublons_path, index=False)
    print(f"[INFO] Title duplicate file saved to: {doublons_path}")
    
    # -------------------------------------------------------------------------
    # 9) Export Articles Where 'page_number' Has More Than Three Words
    # -------------------------------------------------------------------------
    print("[INFO] Extracting records with more than three words in 'page_number'...")
    if "page_number" in df_clean.columns:
        df_page_number = df_clean[
            df_clean["page_number"].astype(str).str.split().apply(len) > 3
        ].copy()
        print(f"[INFO] Number of records with lengthy page_number: {len(df_page_number)}")
        df_page_number.to_csv(page_number_path, index=False)
        print(f"[INFO] Page number file saved to: {page_number_path}")
    else:
        print("[INFO] 'page_number' column not found. Skipping page_number extraction.")
    
    # -------------------------------------------------------------------------
    # 10) Recalculate 'words_count' for Each Article in the Final Dataset
    # -------------------------------------------------------------------------
    print("[INFO] Recalculating 'words_count' for each article in the final dataset...")
    texts_and_langs = list(zip(df_clean["text"].tolist(), df_clean["language"].tolist()))
    with concurrent.futures.ProcessPoolExecutor(initializer=init_spacy) as executor:
        word_counts = list(tqdm(
            executor.map(count_words_wrapper, texts_and_langs),
            total=len(texts_and_langs),
            desc="Counting words (final)", unit="article"
        ))
    df_clean["words_count"] = word_counts
    
    # -------------------------------------------------------------------------
    # 11) Export Article Counts per Media
    # -------------------------------------------------------------------------
    def print_and_save_media_counts(df: pd.DataFrame):
        """
        Compute and export the number of articles per media.

        Parameters:
            df (pd.DataFrame): DataFrame of cleaned articles.
        """
        counts = df["media"].value_counts().sort_index()
        total = counts.sum()
        counts.loc["Total"] = total
        print("\n=== Number of Articles per Media (Updated) ===")
        print(counts)
        counts.to_csv(media_count_path, header=["Number of articles"])
    
    print("[INFO] Updating article counts per media...")
    print_and_save_media_counts(df_clean)
    
    # -------------------------------------------------------------------------
    # 12) Save the Updated Main Media Database
    # -------------------------------------------------------------------------
    print("[INFO] Saving updated media database...")
    df_clean.to_csv(main_db_path, index=False)
    print(f"[INFO] Media database saved to: {main_db_path}")
    
    print("\n[INFO] Update process complete.")

# -------------------------------------------------------------------------
# Main Guard: Ensures the script runs correctly when executed as the main module.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    main()