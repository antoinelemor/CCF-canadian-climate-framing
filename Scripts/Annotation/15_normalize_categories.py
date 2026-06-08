#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
15_normalize_categories.py

MAIN OBJECTIVE:
---------------
This script normalises the annotation category names and codes scattered
across the legacy CSVs (training metrics, test metrics, manual annotation
distribution, train/validation distribution) so that every downstream
consumer sees the same 65-category reference system documented in
Supplementary Table~S3 of the CCF Methodology paper. The script is
deliberately kept side-effect-free with respect to LaTeX outputs: the four
Supplementary Tables that depend on these CSVs (S4 training metrics, S5
training/validation distribution, S7 detailed validation metrics, and S8
database-wide distribution) are produced exclusively by
``17_generate_tables.py``, which imports the helpers defined here.

Input files:
- Database/Training_data/all_best_models.csv
- Database/Training_data/final_annotation_metrics.csv
- Database/Training_data/training_database_metrics.csv
- Database/Training_data/manual_annotations_metrics.csv

Output files:
- Database/Training_data/all_best_models_normalized.csv
- Database/Training_data/final_annotation_metrics_normalized.csv
- Database/Training_data/training_database_metrics_normalized.csv
- Database/Training_data/manual_annotations_metrics_normalized.csv

Public API (imported by 17_generate_tables.py):
- ALL_CATEGORIES, PRIMARY_CATEGORIES, LABEL_TO_NUMBER
- normalize_category, escape_latex, format_code
- get_section_headers
- normalize_training_metrics, normalize_test_metrics,
  normalize_distribution, normalize_training_database
- fetch_database_counts
- DB_PARAMS, PROCESSED_TABLE, DATA_DIR, OUTPUT_DIR

Dependencies:
-------------
- pandas
- pathlib
- psycopg2 (optional; only required by fetch_database_counts)
"""

import os
import pandas as pd
from pathlib import Path

try:
    import psycopg2
except ImportError:
    psycopg2 = None

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "Database" / "Training_data"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "CCF_Methodology" / "Results" / "Outputs" / "Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATABASE CONFIGURATION (CCF_Database for the database-wide distribution table)
# =============================================================================

DB_PARAMS = {
    "host": os.getenv("CCF_DB_HOST", "localhost"),
    "port": int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname": "CCF_Database",
    # Default to the current OS user (e.g. "antoine" on the development
    # workstation); fall back to "postgres" only if neither CCF_DB_USER
    # nor USER is set. This mirrors how psql resolves the role.
    "user": os.getenv("CCF_DB_USER", os.getenv("USER", "postgres")),
    "password": os.getenv("CCF_DB_PASS", ""),
}
PROCESSED_TABLE = "CCF_processed_data"

# =============================================================================
# COMPLETE CATEGORY REFERENCE (65 categories from Supplementary Table S3)
# Format: number -> (official_name, code)
# =============================================================================

ALL_CATEGORIES = {
    # MAIN FRAMES - Economic (1-6)
    1: ("Economic Frame (Primary Category)", "economic_frame"),
    2: ("Negative impacts of climate change on the economy", "eco_neg_impact"),
    3: ("Positive impacts of climate change on the economy", "eco_pos_impact"),
    4: ("Economic disadvantages of climate action", "eco_cost"),
    5: ("Economic benefits of climate action", "eco_benefit"),
    6: ("Carbon footprint of the economic sector", "eco_footprint"),
    # MAIN FRAMES - Health (7-9)
    7: ("Health Frame (Primary Category)", "health_frame"),
    8: ("Negative impacts of climate change on health", "health_neg_impact"),
    9: ("Health co-benefits of climate action", "health_cobenefit"),
    # MAIN FRAMES - Security (10-14)
    10: ("Security Frame (Primary Category)", "security_frame"),
    11: ("Presence of climate refugees", "security_refugees"),
    12: ("Conflict", "security_conflict"),
    13: ("Post-disaster military assistance", "security_military"),
    14: ("Disruption of military operations", "security_disruption"),
    # MAIN FRAMES - Justice (15-20)
    15: ("Justice Frame (Primary Category)", "justice_frame"),
    16: ("Winners and losers of climate action", "justice_winners"),
    17: ("Differentiated responsibility", "justice_responsibility"),
    18: ("Unequal vulnerability to climate change", "justice_vulnerability"),
    19: ("Unequal access to climate action", "justice_access"),
    20: ("Intergenerational justice", "justice_intergen"),
    # MAIN FRAMES - Political (21-25)
    21: ("Political Frame (Primary Category)", "political_frame"),
    22: ("Policy action", "pol_action"),
    23: ("Political debate", "pol_debate"),
    24: ("Political positioning", "pol_position"),
    25: ("Public opinion data", "pol_opinion"),
    # MAIN FRAMES - Scientific (26-30)
    26: ("Scientific Frame (Primary Category)", "scientific_frame"),
    27: ("Scientific debate", "sci_debate"),
    28: ("Popularisation or scientific discovery", "sci_discovery"),
    29: ("Questioning of climate science", "sci_skepticism"),
    30: ("Defense of climate science", "sci_defense"),
    # MAIN FRAMES - Environmental (31-33)
    31: ("Environmental Frame (Primary Category)", "environmental_frame"),
    32: ("Loss of natural environments", "env_habitat"),
    33: ("Loss of fauna and flora", "env_species"),
    # MAIN FRAMES - Cultural (34-38)
    34: ("Cultural Frame (Primary Category)", "cultural_frame"),
    35: ("Artistic representation", "cult_art"),
    36: ("Difficulty to host cultural or sports events", "cult_event_impact"),
    37: ("Loss of indigenous practices", "cult_indigenous"),
    38: ("Carbon footprint of the cultural and sports sectors", "cult_footprint"),
    # PRIMARY CATEGORIES - Actors/Messengers (39-48)
    39: ("Presence of Messengers (Primary Category)", "messenger"),
    40: ("Health expert", "msg_health"),
    41: ("Economic expert", "msg_economic"),
    42: ("Security expert", "msg_security"),
    43: ("Legal expert", "msg_legal"),
    44: ("Cultural or Sport expert", "msg_cultural"),
    45: ("Natural scientist", "msg_scientist"),
    46: ("Social scientist", "msg_social"),
    47: ("Activist", "msg_activist"),
    48: ("Public official", "msg_official"),
    # PRIMARY CATEGORIES - Events (49-57)
    49: ("Presence of Events (Primary Category)", "event"),
    50: ("Extreme meteorological event", "evt_weather"),
    51: ("Meeting", "evt_meeting"),
    52: ("Publication", "evt_publication"),
    53: ("Election", "evt_election"),
    54: ("New policy", "evt_policy"),
    55: ("Judiciary decision", "evt_judiciary"),
    56: ("Cultural or Sports event", "evt_cultural"),
    57: ("Protest", "evt_protest"),
    # PRIMARY CATEGORIES - Solutions (58-60)
    58: ("Presence of Solutions (Primary Category)", "solution"),
    59: ("Mitigation strategy", "sol_mitigation"),
    60: ("Adaptation strategy", "sol_adaptation"),
    # EMOTIONAL TONE (61-63)
    61: ("Emotion: positive", "tone_positive"),
    62: ("Emotion: negative", "tone_negative"),
    63: ("Emotion: neutral", "tone_neutral"),
    # GEOGRAPHIC FOCUS (64)
    64: ("Mention of Canada", "canada"),
    # URGENCY/ALARMISM (65)
    65: ("Urgency to act", "urgency"),
}

# Categories that are PRIMARY (detection) categories - they have
# ``\newline (Primary Category)`` in bold inside the LaTeX cells.
PRIMARY_CATEGORIES = {1, 7, 10, 15, 21, 26, 31, 34, 39, 49, 58}

# =============================================================================
# CSV LABEL MAPPING (legacy CSV codes/labels -> canonical S3 category number)
# =============================================================================

LABEL_TO_NUMBER = {}
for num, (name, code) in ALL_CATEGORIES.items():
    LABEL_TO_NUMBER[name] = num
    LABEL_TO_NUMBER[name.replace(" ", "_")] = num
    LABEL_TO_NUMBER[code] = num

OLD_CSV_TO_NEW_NUMBER = {
    # Economic Frame (1-6)
    "economic_frame": 1,
    "Economic Frame Detection": 1,
    "neg_econ_impacts": 2,
    "Negative Economic Impacts": 2,
    "pos_econ_impacts": 3,
    "Positive Economic Impacts": 3,
    "costs_action": 4,
    "Costs of Climate Action": 4,
    "benefits_action": 5,
    "Benefits of Climate Action": 5,
    "econ_footprint": 6,
    "Economic Sector Footprint": 6,
    # Health Frame (7-9)
    "health_frame": 7,
    "Health Frame Detection": 7,
    "neg_health": 8,
    "Negative Health Impacts": 8,
    "health_cobenefit": 9,
    "Health Co-benefits": 9,
    # Old categories NOT in Supplementary Table S3 - excluded
    "pos_health": None,
    "Positive Health Impacts": None,
    "health_footprint": None,
    "Health Sector Footprint": None,
    # Security Frame (10-14)
    "security_frame": 10,
    "Security Frame Detection": 10,
    "displacement": 11,
    "Climate-Driven Displacement": 11,
    "resource_conflict": 12,
    "Resource Conflict": 12,
    "military_response": 13,
    "Military Disaster Response": 13,
    "military_base": 14,
    "Military Base Disruption": 14,
    "defense_footprint": None,
    "Defense Sector Footprint": None,
    # Justice Frame (15-20)
    "justice_frame": 15,
    "Justice Frame Detection": 15,
    "winners_losers": 16,
    "Winners & Losers": 16,
    "Winners and Losers": 16,
    "north_south": 17,
    "North-South Responsibility": 17,
    "unequal_impacts": 18,
    "Unequal Impacts": 18,
    "unequal_access": 19,
    "Unequal Access": 19,
    "intergenerational": 20,
    "Intergenerational Justice": 20,
    # Political Frame (21-25)
    "political_frame": 21,
    "Political Frame Detection": 21,
    "policy_measures": 22,
    "Policy Measures": 22,
    "political_debate": 23,
    "Political Debate": 23,
    "political_position": 24,
    "Political Positioning": 24,
    "public_opinion": 25,
    "Public Opinion": 25,
    # Scientific Frame (26-30)
    "scientific_frame": 26,
    "Scientific Frame Detection": 26,
    "sci_controversy": 27,
    "Scientific Controversy": 27,
    "discovery_innov": 28,
    "Discovery & Innovation": 28,
    "Discovery and Innovation": 28,
    "sci_uncertainty": 29,
    "Scientific Uncertainty": 29,
    "sci_certainty": 30,
    "Scientific Certainty": 30,
    # Environmental Frame (31-33)
    "environ_frame": 31,
    "Environmental Frame Detection": 31,
    "habitat_loss": 32,
    "Habitat Loss": 32,
    "species_loss": 33,
    "Species Loss": 33,
    # Cultural Frame (34-38)
    "cultural_frame": 34,
    "Cultural Frame Detection": 34,
    "artistic_rep": 35,
    "Artistic Representation": 35,
    "event_disruption": 36,
    "Event Disruption": 36,
    "indigenous_loss": 37,
    "Loss of Indigenous Practices": 37,
    "cultural_footprint": 38,
    "Cultural Sector Footprint": 38,
    # Actors/Messengers (39-48)
    "messenger_detect": 39,
    "Messenger_Detection": 39,
    "Messenger Detection": 39,
    "Actors/Messengers Detection": 39,
    "health_expert": 40,
    "Health Expert": 40,
    "econ_expert": 41,
    "Economic Expert": 41,
    "security_expert": 42,
    "Security Expert": 42,
    "legal_expert": 43,
    "Legal Expert": 43,
    "cultural_expert": 44,
    "Cultural Expert": 44,
    "natural_scientist": 45,
    "Natural Scientist": 45,
    "social_scientist": 46,
    "Social Scientist": 46,
    "activist": 47,
    "Activist": 47,
    "public_official": 48,
    "Public Official": 48,
    # Events (49-57)
    "event_detect": 49,
    "Event Detection": 49,
    "extreme_weather": 50,
    "Extreme Weather Event": 50,
    "meeting_conf": 51,
    "Meeting_Conference": 51,
    "Meeting/Conference": 51,
    "Meeting Conference": 51,
    "publication": 52,
    "Publication": 52,
    "election": 53,
    "Election": 53,
    "policy_announce": 54,
    "Policy Announcement": 54,
    "judiciary": 55,
    "Judiciary Decision": 55,
    "cultural_event": 56,
    "Cultural Event": 56,
    "protest": 57,
    "Protest": 57,
    # Solutions (58-60)
    "solutions_detect": 58,
    "Solutions Detection": 58,
    "Solution_1_SUB": 59,
    "mitigation": 59,
    "Mitigation Strategy": 59,
    "Solution_2_SUB": 60,
    "adaptation": 60,
    "Adaptation Strategy": 60,
    # Emotional Tone (61-63)
    "pos_emotion": 61,
    "Positive Emotion": 61,
    "neg_emotion": 62,
    "Negative Emotion": 62,
    "neutral_emotion": 63,
    "Neutral Emotion": 63,
    # Geographic Focus (64)
    "canadian_context": 64,
    "Canadian Context": 64,
    # Urgency/Alarmism (65)
    "urgency_alarmism": 65,
    "Urgency/Alarmism": 65,
    "Urgency Alarmism": 65,
}

# Merge legacy CSV mappings into the main lookup.
LABEL_TO_NUMBER.update(OLD_CSV_TO_NEW_NUMBER)


def normalize_category(name: str) -> tuple:
    """Normalise a category name to ``(number, official_name, code)``.

    Returns ``(None, None, None)`` for categories explicitly excluded from
    Supplementary Table~S3 (legacy categories that did not survive the
    framework refresh).
    """
    if name in LABEL_TO_NUMBER:
        num = LABEL_TO_NUMBER[name]
        if num is None:
            return (None, None, None)
        return (num, ALL_CATEGORIES[num][0], ALL_CATEGORIES[num][1])

    name_spaces = name.replace("_", " ")
    if name_spaces in LABEL_TO_NUMBER:
        num = LABEL_TO_NUMBER[name_spaces]
        if num is None:
            return (None, None, None)
        return (num, ALL_CATEGORIES[num][0], ALL_CATEGORIES[num][1])

    # Try extracting from a model path format (e.g. ``models/Foo_EN.jsonl``).
    if "models/" in name:
        base = name.split("models/")[1].split(".jsonl")[0]
        for suffix in ["_EN", "_FR"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        if base in LABEL_TO_NUMBER:
            num = LABEL_TO_NUMBER[base]
            if num is None:
                return (None, None, None)
            return (num, ALL_CATEGORIES[num][0], ALL_CATEGORIES[num][1])

    print(f"  WARNING: Category not found in reference: '{name}'")
    return (99, name, name.lower().replace(" ", "_").replace("/", "_"))


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '$': r'\$',
        '_': r'\_',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_code(code: str) -> str:
    """Format a category code for LaTeX ``\\texttt{}`` (escape underscores)."""
    return code.replace('_', r'\_')


# =============================================================================
# NORMALISE TRAINING METRICS (all_best_models.csv)
# =============================================================================

def normalize_training_metrics():
    """Normalise ``all_best_models.csv`` and add ``number``, ``category``, ``code``."""
    print("\n1. Normalising training metrics (all_best_models.csv)...")

    df = pd.read_csv(DATA_DIR / "all_best_models.csv")

    records = []
    for _, row in df.iterrows():
        path = row['saved_model_path']
        base = path.split("models/")[1].split(".jsonl")[0]

        if base.endswith("_EN"):
            lang = "EN"
            cat_name = base[:-3]
        elif base.endswith("_FR"):
            lang = "FR"
            cat_name = base[:-3]
        else:
            continue

        num, official_name, code = normalize_category(cat_name)

        if num is None:
            print(f"   Excluding category not in Supplementary Table S3: {cat_name}")
            continue

        record = {
            'number': num,
            'category': official_name,
            'code': code,
            'language': lang,
            'epoch': row['epoch'],
            'train_loss': row['train_loss'],
            'val_loss': row['val_loss'],
            'precision_0': row['precision_0'],
            'recall_0': row['recall_0'],
            'f1_0': row['f1_0'],
            'support_0': row['support_0'],
            'precision_1': row['precision_1'],
            'recall_1': row['recall_1'],
            'f1_1': row['f1_1'],
            'support_1': row['support_1'],
            'macro_f1': row['macro_f1'],
            'training_phase': row['training_phase']
        }
        records.append(record)

    result = pd.DataFrame(records).sort_values(['number', 'language'])
    result.to_csv(DATA_DIR / "all_best_models_normalized.csv", index=False)
    print(f"   Saved: all_best_models_normalized.csv ({len(result)} rows)")
    return result


# =============================================================================
# NORMALISE TEST METRICS (final_annotation_metrics.csv)
# =============================================================================

def normalize_test_metrics():
    """Normalise ``final_annotation_metrics.csv`` and add ``number``/``category``/``code``."""
    print("\n2. Normalising test metrics (final_annotation_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "final_annotation_metrics.csv")

    records = []
    for _, row in df.iterrows():
        label = row['label']
        lang = row['language']

        if label == "ALL":
            record = {
                'number': 0,
                'category': "GLOBAL SUMMARY",
                'code': "global_summary",
                'language': lang,
            }
        else:
            num, official_name, code = normalize_category(label)
            if num is None:
                print(f"   Excluding category not in Supplementary Table S3: {label}")
                continue
            record = {
                'number': num,
                'category': official_name,
                'code': code,
                'language': lang,
            }

        for col in df.columns:
            if col not in ['label', 'language']:
                record[col] = row[col]
        records.append(record)

    result = pd.DataFrame(records).sort_values(['number', 'language'])
    result.to_csv(DATA_DIR / "final_annotation_metrics_normalized.csv", index=False)
    print(f"   Saved: final_annotation_metrics_normalized.csv ({len(result)} rows)")
    return result


# =============================================================================
# NORMALISE DISTRIBUTION METRICS (manual_annotations_metrics.csv)
# =============================================================================

def normalize_distribution():
    """Normalise ``manual_annotations_metrics.csv`` and add ``number``/``category``/``code``."""
    print("\n3. Normalising distribution metrics (manual_annotations_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "manual_annotations_metrics.csv")

    records = []
    for _, row in df.iterrows():
        cat = row['Annotation Type']

        if 'Total' in str(cat) or pd.isna(cat):
            continue

        num, official_name, code = normalize_category(cat)

        if num is None:
            print(f"   Excluding category not in Supplementary Table S3: {cat}")
            continue

        record = {
            'number': num,
            'category': official_name,
            'code': code,
            'count_en': row['Count (EN)'],
            'proportion_en': row['Proportion (EN)'],
            'count_fr': row['Count (FR)'],
            'proportion_fr': row['Proportion (FR)']
        }
        records.append(record)

    result = pd.DataFrame(records).sort_values('number')
    result.to_csv(DATA_DIR / "manual_annotations_metrics_normalized.csv", index=False)
    print(f"   Saved: manual_annotations_metrics_normalized.csv ({len(result)} rows)")
    return result


# =============================================================================
# NORMALISE TRAINING DATABASE METRICS (training_database_metrics.csv)
# =============================================================================

def normalize_training_database():
    """Normalise ``training_database_metrics.csv`` (train/val distribution)."""
    print("\n4. Normalising training database metrics (training_database_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "training_database_metrics.csv")

    records = []
    for _, row in df.iterrows():
        label = row['Label']

        if pd.isna(label) or 'Total' in str(label):
            continue

        num, official_name, code = normalize_category(label)

        if num is None:
            print(f"   Excluding category not in Supplementary Table S3: {label}")
            continue

        record = {
            'number': num,
            'category': official_name,
            'code': code,
            'train_pos_en': row['Train_Positive_EN'],
            'train_neg_en': row['Train_Negative_EN'],
            'val_pos_en': row['Validation_Positive_EN'],
            'val_neg_en': row['Validation_Negative_EN'],
            'train_pos_fr': row['Train_Positive_FR'],
            'train_neg_fr': row['Train_Negative_FR'],
            'val_pos_fr': row['Validation_Positive_FR'],
            'val_neg_fr': row['Validation_Negative_FR'],
        }
        records.append(record)

    result = pd.DataFrame(records).sort_values('number')
    result.to_csv(DATA_DIR / "training_database_metrics_normalized.csv", index=False)
    print(f"   Saved: training_database_metrics_normalized.csv ({len(result)} rows)")
    return result


# =============================================================================
# SECTION HEADERS FOR LATEX TABLES
# =============================================================================

def get_section_headers(ncols):
    """Return section headers for tables with ``ncols`` columns.

    The headers mirror the canonical structure of Supplementary Table~S3:
    main sections use ``\\cellcolor{gray!10}``, sub-sections use
    ``\\rowcolor{gray!8}`` before an italic header, and each header is
    followed by a ``\\midrule``.
    """
    return {
        1: [f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{MAIN FRAMES}}}} \\\\",
            r"\midrule",
            r"\rowcolor{gray!8}",
            f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Economic Frame}}}} \\\\",
            r"\midrule"],
        7: [r"\midrule",
            r"\rowcolor{gray!8}",
            f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Health Frame}}}} \\\\",
            r"\midrule"],
        10: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Security Frame}}}} \\\\",
             r"\midrule"],
        15: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Justice Frame}}}} \\\\",
             r"\midrule"],
        21: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Political Frame}}}} \\\\",
             r"\midrule"],
        26: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Scientific Frame}}}} \\\\",
             r"\midrule"],
        31: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Environmental Frame}}}} \\\\",
             r"\midrule"],
        34: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Cultural Frame}}}} \\\\",
             r"\midrule"],
        39: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{PRIMARY CATEGORIES}}}} \\\\",
             r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Actors/Messengers}}}} \\\\",
             r"\midrule"],
        49: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Events}}}} \\\\",
             r"\midrule"],
        58: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Solutions}}}} \\\\",
             r"\midrule"],
        61: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{EMOTIONAL TONE}}}} \\\\",
             r"\midrule"],
        64: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{GEOGRAPHIC FOCUS}}}} \\\\",
             r"\midrule"],
        65: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{URGENCY TO ACT}}}} \\\\",
             r"\midrule"],
    }


# =============================================================================
# FETCH DATABASE-WIDE ANNOTATION COUNTS (used by Supplementary Table S8)
# =============================================================================

def fetch_database_counts():
    """Fetch annotation counts from ``CCF_Database`` for Supplementary Table~S8.

    Returns
    -------
    counts : dict[int, dict]
        Mapping ``{category_number: {'en': int, 'fr': int, 'all': int}}``.
    totals : tuple[int, int, int]
        ``(total_en, total_fr, total_all)`` sentence counts in
        ``CCF_processed_data``.
    """
    print("\n   Fetching annotation counts from CCF_Database...")

    if psycopg2 is None:
        print("   ERROR: psycopg2 not available. Cannot connect to database.")
        return None, None

    try:
        conn = psycopg2.connect(
            host=DB_PARAMS["host"],
            port=DB_PARAMS["port"],
            user=DB_PARAMS["user"],
            password=DB_PARAMS["password"],
            dbname=DB_PARAMS["dbname"],
        )
        cur = conn.cursor()

        cur.execute(f"""
            SELECT language, COUNT(*)
            FROM "{PROCESSED_TABLE}"
            GROUP BY language
        """)
        lang_counts = dict(cur.fetchall())
        total_en = lang_counts.get('EN', 0)
        total_fr = lang_counts.get('FR', 0)
        total_all = total_en + total_fr

        print(f"   Total sentences: EN={total_en:,}, FR={total_fr:,}, ALL={total_all:,}")

        # Single-pass aggregation: one full scan of CCF_processed_data,
        # 65 SUM(CASE WHEN col=1 ...) expressions per language, joined
        # in Python. The previous implementation ran 65 sequential
        # COUNT GROUP BY language queries (one per category), which
        # forced one full table scan per category and took 5-10 min
        # on the 9.2-million-row table. The version below produces the
        # same numbers in a single ~30 s scan.
        sum_exprs_en = ", ".join(
            f'SUM(CASE WHEN language=\'EN\' AND "{code}"=1 THEN 1 ELSE 0 END) AS en_{num}'
            for num, (_cat, code) in ALL_CATEGORIES.items()
        )
        sum_exprs_fr = ", ".join(
            f'SUM(CASE WHEN language=\'FR\' AND "{code}"=1 THEN 1 ELSE 0 END) AS fr_{num}'
            for num, (_cat, code) in ALL_CATEGORIES.items()
        )
        query = f'SELECT {sum_exprs_en}, {sum_exprs_fr} FROM "{PROCESSED_TABLE}"'
        try:
            cur.execute(query)
            row = cur.fetchone()
            col_names = [d.name for d in cur.description]
            agg = dict(zip(col_names, row))
            category_counts = {}
            for num in ALL_CATEGORIES:
                count_en = int(agg.get(f"en_{num}") or 0)
                count_fr = int(agg.get(f"fr_{num}") or 0)
                category_counts[num] = {
                    'en': count_en,
                    'fr': count_fr,
                    'all': count_en + count_fr,
                }
        except Exception as e:
            print(f"   WARNING: aggregated query failed ({e}); falling back to per-column COUNTs")
            conn.rollback()
            category_counts = {}
            for num, (_cat, code) in ALL_CATEGORIES.items():
                try:
                    cur.execute(
                        f'SELECT language, COUNT(*) FROM "{PROCESSED_TABLE}" '
                        f'WHERE "{code}"=1 GROUP BY language'
                    )
                    counts = dict(cur.fetchall())
                    count_en = counts.get('EN', 0)
                    count_fr = counts.get('FR', 0)
                    category_counts[num] = {
                        'en': count_en,
                        'fr': count_fr,
                        'all': count_en + count_fr,
                    }
                except Exception as e2:
                    print(f"   WARNING: column '{code}': {e2}")
                    conn.rollback()
                    category_counts[num] = {'en': 0, 'fr': 0, 'all': 0}

        cur.close()
        conn.close()

        print(f"   Successfully fetched counts for {len(category_counts)} categories")
        return category_counts, (total_en, total_fr, total_all)

    except Exception as e:
        print(f"   ERROR: Database connection failed: {e}")
        return None, None


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CCF METHODOLOGY - NORMALISATION OF ANNOTATION CSVS")
    print("=" * 70)

    training_df = normalize_training_metrics()
    test_df = normalize_test_metrics()
    dist_df = normalize_distribution()
    training_db_df = normalize_training_database()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nNormalised CSVs saved to: {DATA_DIR}")
    print(f"  - all_best_models_normalized.csv             ({len(training_df)} rows)")
    print(f"  - final_annotation_metrics_normalized.csv    ({len(test_df)} rows)")
    print(f"  - manual_annotations_metrics_normalized.csv  ({len(dist_df)} rows)")
    print(f"  - training_database_metrics_normalized.csv   ({len(training_db_df)} rows)")
    print("\nSupplementary Tables S4/S5/S7/S8 are rendered from these CSVs by")
    print("  Scripts/Annotation/17_generate_tables.py")


if __name__ == "__main__":
    main()
