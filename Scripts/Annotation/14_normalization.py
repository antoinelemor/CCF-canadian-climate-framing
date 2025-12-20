#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
14_normalization.py

MAIN OBJECTIVE:
---------------
This script normalizes annotation category names and codes across all CSV files
and generates LaTeX tables for the CCF Methodology paper appendix. It ensures
consistency between training metrics, test metrics, and distribution data by
mapping all category labels to a standardized 68-category reference system
matching Table B1 of the paper.

Input files:
- Database/Training_data/all_best_models.csv (training metrics -> Table B2)
- Database/Training_data/final_annotation_metrics.csv (test metrics -> Table B5)
- Database/Training_data/training_database_metrics.csv (distribution -> Table B3)
- Database/Training_data/manual_annotations_metrics.csv (annotation distribution)

Output files:
- Normalized CSVs with number, category, and code columns
- LaTeX longtable files for appendix tables B2, B3, and B5

Dependencies:
-------------
- pandas
- pathlib
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
# DATABASE CONFIGURATION (CCF_Database for Table B6)
# =============================================================================

DB_PARAMS = {
    "host": os.getenv("CCF_DB_HOST", "localhost"),
    "port": int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname": "CCF_Database",
    "user": os.getenv("CCF_DB_USER", "postgres"),
    "password": os.getenv("CCF_DB_PASS", ""),
}
PROCESSED_TABLE = "CCF_processed_data"

# =============================================================================
# COMPLETE CATEGORY REFERENCE (All 65 categories from Table B1 in CCF_metho_2.tex)
# Format: number -> (official_name, code)
# =============================================================================

ALL_CATEGORIES = {
    # THEMATIC FRAMES - Economic (1-6)
    1: ("Economic Frame (Primary Category)", "economic_frame"),
    2: ("Negative impacts of climate change on the economy", "eco_neg_impact"),
    3: ("Positive impacts of climate change on the economy", "eco_pos_impact"),
    4: ("Economic disadvantages of climate action", "eco_cost"),
    5: ("Economic benefits of climate action", "eco_benefit"),
    6: ("Carbon footprint of the economic sector", "eco_footprint"),
    # THEMATIC FRAMES - Health (7-9)
    7: ("Health Frame (Primary Category)", "health_frame"),
    8: ("Negative impacts of climate change on health", "health_neg_impact"),
    9: ("Health co-benefits of climate action", "health_cobenefit"),
    # THEMATIC FRAMES - Security (10-14)
    10: ("Security Frame (Primary Category)", "security_frame"),
    11: ("Presence of climate refugees", "security_refugees"),
    12: ("Conflict", "security_conflict"),
    13: ("Post-disaster military assistance", "security_military"),
    14: ("Disruption of military operations", "security_disruption"),
    # THEMATIC FRAMES - Justice (15-20)
    15: ("Justice Frame (Primary Category)", "justice_frame"),
    16: ("Winners and losers of climate action", "justice_winners"),
    17: ("Differentiated responsibility", "justice_responsibility"),
    18: ("Unequal vulnerability to climate change", "justice_vulnerability"),
    19: ("Unequal access to climate action", "justice_access"),
    20: ("Intergenerational justice", "justice_intergen"),
    # THEMATIC FRAMES - Political (21-25)
    21: ("Political Frame (Primary Category)", "political_frame"),
    22: ("Policy action", "pol_action"),
    23: ("Political debate", "pol_debate"),
    24: ("Political positioning", "pol_position"),
    25: ("Public opinion data", "pol_opinion"),
    # THEMATIC FRAMES - Scientific (26-30)
    26: ("Scientific Frame (Primary Category)", "scientific_frame"),
    27: ("Scientific debate", "sci_debate"),
    28: ("Popularisation or scientific discovery", "sci_discovery"),
    29: ("Questioning of climate science", "sci_skepticism"),
    30: ("Defense of climate science", "sci_defense"),
    # THEMATIC FRAMES - Environmental (31-33)
    31: ("Environmental Frame (Primary Category)", "environmental_frame"),
    32: ("Loss of natural environments", "env_habitat"),
    33: ("Loss of fauna and flora", "env_species"),
    # THEMATIC FRAMES - Cultural (34-38)
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

# =============================================================================
# CSV LABEL MAPPING (OLD CSV codes/labels -> NEW category number from Table B1)
# The CSVs use older naming conventions that must be mapped to Table B1 numbers
# =============================================================================

LABEL_TO_NUMBER = {}
for num, (name, code) in ALL_CATEGORIES.items():
    # Add various label formats from Table B1
    LABEL_TO_NUMBER[name] = num
    LABEL_TO_NUMBER[name.replace(" ", "_")] = num
    LABEL_TO_NUMBER[code] = num

# =============================================================================
# MAPPING OLD CSV CODES TO NEW TABLE B1 NUMBERS
# The old CSVs use different codes/names - this maps them to correct B1 numbers
# =============================================================================

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
    # Health Frame (7-9) - NOTE: old CSV has 5 categories, B1 has 3
    "health_frame": 7,
    "Health Frame Detection": 7,
    "neg_health": 8,
    "Negative Health Impacts": 8,
    "health_cobenefit": 9,
    "Health Co-benefits": 9,
    # Old categories NOT in Table B1 - map to closest or exclude
    "pos_health": None,  # Positive Health Impacts - not in Table B1
    "Positive Health Impacts": None,
    "health_footprint": None,  # Health Sector Footprint - not in Table B1
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
    "defense_footprint": None,  # Defense Sector Footprint - not in Table B1
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

# Merge old CSV mappings into main lookup
LABEL_TO_NUMBER.update(OLD_CSV_TO_NEW_NUMBER)


def normalize_category(name: str) -> tuple:
    """
    Normalize a category name to (number, official_name, code).
    Returns (None, None, None) for categories excluded from Table B1.
    """
    # Direct lookup
    if name in LABEL_TO_NUMBER:
        num = LABEL_TO_NUMBER[name]
        if num is None:
            # Category explicitly excluded from Table B1
            return (None, None, None)
        return (num, ALL_CATEGORIES[num][0], ALL_CATEGORIES[num][1])

    # Try with underscores replaced by spaces
    name_spaces = name.replace("_", " ")
    if name_spaces in LABEL_TO_NUMBER:
        num = LABEL_TO_NUMBER[name_spaces]
        if num is None:
            return (None, None, None)
        return (num, ALL_CATEGORIES[num][0], ALL_CATEGORIES[num][1])

    # Try extracting from model path format
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

    # Not found
    print(f"  WARNING: Category not found in reference: '{name}'")
    return (99, name, name.lower().replace(" ", "_").replace("/", "_"))


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
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
    """Format code for LaTeX \texttt{} command (escape underscores)."""
    return code.replace('_', r'\_')


# =============================================================================
# NORMALIZE TRAINING METRICS (all_best_models.csv)
# =============================================================================

def normalize_training_metrics():
    """Normalize all_best_models.csv and add number, category, code columns."""
    print("\n1. Normalizing training metrics (all_best_models.csv)...")

    df = pd.read_csv(DATA_DIR / "all_best_models.csv")

    records = []
    for _, row in df.iterrows():
        path = row['saved_model_path']
        base = path.split("models/")[1].split(".jsonl")[0]

        # Determine language
        if base.endswith("_EN"):
            lang = "EN"
            cat_name = base[:-3]
        elif base.endswith("_FR"):
            lang = "FR"
            cat_name = base[:-3]
        else:
            continue

        num, official_name, code = normalize_category(cat_name)

        # Skip categories excluded from Table B1
        if num is None:
            print(f"   Excluding category not in Table B1: {cat_name}")
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
# NORMALIZE TEST METRICS (final_annotation_metrics.csv)
# =============================================================================

def normalize_test_metrics():
    """Normalize final_annotation_metrics.csv and add number, category, code columns."""
    print("\n2. Normalizing test metrics (final_annotation_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "final_annotation_metrics.csv")

    records = []
    for _, row in df.iterrows():
        label = row['label']
        lang = row['language']

        # Handle global summary rows (ALL,EN; ALL,FR; ALL,ALL)
        if label == "ALL":
            record = {
                'number': 0,  # 0 for global summaries
                'category': "GLOBAL SUMMARY",
                'code': "global_summary",
                'language': lang,
            }
        else:
            num, official_name, code = normalize_category(label)
            # Skip categories excluded from Table B1
            if num is None:
                print(f"   Excluding category not in Table B1: {label}")
                continue
            record = {
                'number': num,
                'category': official_name,
                'code': code,
                'language': lang,  # Includes "EN", "FR", and "ALL" (aggregate)
            }

        # Add all other columns
        for col in df.columns:
            if col not in ['label', 'language']:
                record[col] = row[col]
        records.append(record)

    result = pd.DataFrame(records).sort_values(['number', 'language'])
    result.to_csv(DATA_DIR / "final_annotation_metrics_normalized.csv", index=False)
    print(f"   Saved: final_annotation_metrics_normalized.csv ({len(result)} rows)")
    return result


# =============================================================================
# NORMALIZE DISTRIBUTION METRICS (manual_annotations_metrics.csv)
# =============================================================================

def normalize_distribution():
    """Normalize manual_annotations_metrics.csv and add number, category, code columns."""
    print("\n3. Normalizing distribution metrics (manual_annotations_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "manual_annotations_metrics.csv")

    records = []
    for _, row in df.iterrows():
        cat = row['Annotation Type']

        if 'Total' in str(cat) or pd.isna(cat):
            continue

        num, official_name, code = normalize_category(cat)

        # Skip categories excluded from Table B1
        if num is None:
            print(f"   Excluding category not in Table B1: {cat}")
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
# NORMALIZE TRAINING DATABASE METRICS (training_database_metrics.csv) -> Table B3
# =============================================================================

def normalize_training_database():
    """Normalize training_database_metrics.csv (Table B3: Train/Val distribution)."""
    print("\n4. Normalizing training database metrics (training_database_metrics.csv)...")

    df = pd.read_csv(DATA_DIR / "training_database_metrics.csv")

    records = []
    for _, row in df.iterrows():
        label = row['Label']

        if pd.isna(label) or 'Total' in str(label):
            continue

        num, official_name, code = normalize_category(label)

        # Skip categories excluded from Table B1
        if num is None:
            print(f"   Excluding category not in Table B1: {label}")
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
    """Return section headers for tables with ncols columns (65 categories from Table B1).

    Format matches CCF_metho_2.tex exactly:
    - Main sections use \cellcolor{gray!10} for headers
    - Sub-sections use \rowcolor{gray!8} before italic header
    - \midrule after each sub-section header
    """
    return {
        # THEMATIC FRAMES start at 1
        1: [f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{THEMATIC FRAMES}}}} \\\\",
            r"\midrule",
            r"\rowcolor{gray!8}",
            f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Economic Frame}}}} \\\\",
            r"\midrule"],
        # Health Frame starts at 7
        7: [r"\midrule",
            r"\rowcolor{gray!8}",
            f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Health Frame}}}} \\\\",
            r"\midrule"],
        # Security Frame starts at 10
        10: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Security Frame}}}} \\\\",
             r"\midrule"],
        # Justice Frame starts at 15
        15: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Justice Frame}}}} \\\\",
             r"\midrule"],
        # Political Frame starts at 21
        21: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Political Frame}}}} \\\\",
             r"\midrule"],
        # Scientific Frame starts at 26
        26: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Scientific Frame}}}} \\\\",
             r"\midrule"],
        # Environmental Frame starts at 31
        31: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Environmental Frame}}}} \\\\",
             r"\midrule"],
        # Cultural Frame starts at 34
        34: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Cultural Frame}}}} \\\\",
             r"\midrule"],
        # PRIMARY CATEGORIES - Actors/Messengers starts at 39
        39: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{PRIMARY CATEGORIES}}}} \\\\",
             r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Actors/Messengers}}}} \\\\",
             r"\midrule"],
        # Events starts at 49
        49: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Events}}}} \\\\",
             r"\midrule"],
        # Solutions starts at 58
        58: [r"\midrule",
             r"\rowcolor{gray!8}",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Solutions}}}} \\\\",
             r"\midrule"],
        # EMOTIONAL TONE starts at 61
        61: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{EMOTIONAL TONE}}}} \\\\",
             r"\midrule"],
        # GEOGRAPHIC FOCUS at 64
        64: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{GEOGRAPHIC FOCUS}}}} \\\\",
             r"\midrule"],
        # URGENCY/ALARMISM at 65
        65: [r"\midrule",
             f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!10}}\\textbf{{URGENCY TO ACT}}}} \\\\",
             r"\midrule"],
    }


# Categories that are PRIMARY (detection) categories - they have \newline (Primary Category) in bold
PRIMARY_CATEGORIES = {1, 7, 10, 15, 21, 26, 31, 34, 39, 49, 58}


# =============================================================================
# GENERATE LATEX TABLE B2: Training Performance Metrics
# =============================================================================

def generate_table_b2(df):
    """Generate LaTeX for Table B2: Complete training performance metrics.

    Format matches existing paper table with Code column added.
    9 columns: # | Category | Code | F1(1) EN | F1(1) FR | F1(0) EN | F1(0) FR | Macro EN | Macro FR
    """
    print("\n5. Generating Table B2 (training metrics)...")

    # Pivot to wide format (EN/FR columns)
    pivot = df.pivot(
        index=['number', 'category', 'code'],
        columns='language',
        values=['f1_1', 'f1_0', 'macro_f1']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Create dict for lookup
    data_dict = {}
    for _, row in pivot.iterrows():
        data_dict[int(row['number'])] = row

    ncols = 9
    section_headers = get_section_headers(ncols)

    lines = []
    # Column order: # | Category | Code | metrics...
    # p{0.4cm} for #, p{4.0cm} for category, p{3.0cm} for code (wider)
    lines.append(r"\begin{longtable}{p{0.4cm}p{4.0cm}p{3.0cm}rrrrrr}")
    lines.append(r"\caption{Complete model training performance metrics for all annotation categories}")
    lines.append(r"\label{tab:complete_training_metrics}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & \multicolumn{2}{c}{\textbf{F1 (Class 0)}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\tablename\\ \\thetable\\ -- \\textit{{Continued from previous page}}}} \\\\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & \multicolumn{2}{c}{\textbf{F1 (Class 0)}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    def fmt(val):
        if pd.isna(val):
            return "--"
        return f"{val:.3f}"

    # Generate rows for all 65 categories (Table B1)
    for num in range(1, 66):
        # Add section headers
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = ALL_CATEGORIES[num]

        if num in data_dict:
            row = data_dict[num]
            f1_1_en = fmt(row.get('f1_1_EN'))
            f1_1_fr = fmt(row.get('f1_1_FR'))
            f1_0_en = fmt(row.get('f1_0_EN'))
            f1_0_fr = fmt(row.get('f1_0_FR'))
            macro_en = fmt(row.get('macro_f1_EN'))
            macro_fr = fmt(row.get('macro_f1_FR'))
        else:
            # Category has no training data
            f1_1_en = f1_1_fr = f1_0_en = f1_0_fr = macro_en = macro_fr = "--"

        # Format category name: Primary Categories are bold with \newline (Primary Category)
        if num in PRIMARY_CATEGORIES:
            # Extract base name (before "(Primary Category)")
            base_name = cat_name.replace(" (Primary Category)", "")
            cat_formatted = f"\\textbf{{{escape_latex(base_name)}\\newline (Primary Category)}}"
        else:
            cat_formatted = escape_latex(cat_name)

        # Order: # | Category | Code | metrics
        lines.append(f"{num} & {cat_formatted} & \\texttt{{{format_code(code)}}} & {f1_1_en} & {f1_1_fr} & {f1_0_en} & {f1_0_fr} & {macro_en} & {macro_fr} \\\\")

        # Add \cmidrule after Primary Categories
        if num in PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Add average rows
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textbf{{AVERAGE PERFORMANCE METRICS}}}} \\\\")

    # Calculate averages from actual data
    en_f1_1 = df[df['language'] == 'EN']['f1_1'].mean()
    fr_f1_1 = df[df['language'] == 'FR']['f1_1'].mean()
    en_f1_0 = df[df['language'] == 'EN']['f1_0'].mean()
    fr_f1_0 = df[df['language'] == 'FR']['f1_0'].mean()
    en_macro = df[df['language'] == 'EN']['macro_f1'].mean()
    fr_macro = df[df['language'] == 'FR']['macro_f1'].mean()

    lines.append(f"& & \\textbf{{English Average}} & {en_f1_1:.3f} & -- & {en_f1_0:.3f} & -- & {en_macro:.3f} & -- \\\\")
    lines.append(f"& & \\textbf{{French Average}} & -- & {fr_f1_1:.3f} & -- & {fr_f1_0:.3f} & -- & {fr_macro:.3f} \\\\")
    lines.append(f"& & \\textbf{{Overall Average}} & \\multicolumn{{2}}{{c}}{{{(en_f1_1+fr_f1_1)/2:.3f}}} & \\multicolumn{{2}}{{c}}{{{(en_f1_0+fr_f1_0)/2:.3f}}} & \\multicolumn{{2}}{{c}}{{{(en_macro+fr_macro)/2:.3f}}} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textbf{{TOTAL: 65 ANNOTATION CATEGORIES}}}} \\\\")

    # Count categories with models
    n_total = df['number'].nunique()
    n_excluded = 65 - n_total

    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textit{{({n_total} categories with at least one model; {n_excluded} categories entirely excluded*)}}}} \\\\")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    (OUTPUT_DIR / "table_b2_training.tex").write_text(output)
    print(f"   Saved: table_b2_training.tex")
    return output


# =============================================================================
# GENERATE LATEX TABLE B3: Training/Validation Distribution
# =============================================================================

def generate_table_b3(df):
    """Generate LaTeX for Table B3: Training and validation dataset distribution.

    Format matches existing paper table with Code column added.
    11 columns: # | Category | Code | Train Pos EN | Train Neg EN | Val Pos EN | Val Neg EN | ...FR
    """
    print("\n6. Generating Table B3 (training/validation distribution)...")

    # Create dict for lookup
    data_dict = {}
    for _, row in df.iterrows():
        data_dict[int(row['number'])] = row

    ncols = 11
    section_headers = get_section_headers(ncols)

    lines = []
    # Column order: # | Category | Code | metrics...
    # p{0.5cm} for #, p{4.0cm} for category, p{3.0cm} for code (wider)
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.0cm}p{3.0cm}rrrrrrrr}")
    lines.append(r"\caption{Training and validation dataset distribution across all annotation categories}")
    lines.append(r"\label{tab:training_distribution}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{3}{*}{\textbf{\#}} & \multirow{3}{*}{\textbf{Category}} & \multirow{3}{*}{\textbf{Code}} & \multicolumn{4}{c}{\textbf{English}} & \multicolumn{4}{c}{\textbf{French}} \\")
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(r"& & & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    lines.append(r"& & & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\tablename\\ \\thetable\\ -- \\textit{{Continued from previous page}}}} \\\\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{3}{*}{\textbf{\#}} & \multirow{3}{*}{\textbf{Category}} & \multirow{3}{*}{\textbf{Code}} & \multicolumn{4}{c}{\textbf{English}} & \multicolumn{4}{c}{\textbf{French}} \\")
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(r"& & & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    lines.append(r"& & & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    def fmt_int(val):
        if pd.isna(val):
            return "0"
        return str(int(val))

    # Generate rows for all 65 categories (Table B1)
    for num in range(1, 66):
        # Add section headers
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = ALL_CATEGORIES[num]

        if num in data_dict:
            row = data_dict[num]
            tr_pos_en = fmt_int(row['train_pos_en'])
            tr_neg_en = fmt_int(row['train_neg_en'])
            val_pos_en = fmt_int(row['val_pos_en'])
            val_neg_en = fmt_int(row['val_neg_en'])
            tr_pos_fr = fmt_int(row['train_pos_fr'])
            tr_neg_fr = fmt_int(row['train_neg_fr'])
            val_pos_fr = fmt_int(row['val_pos_fr'])
            val_neg_fr = fmt_int(row['val_neg_fr'])
        else:
            tr_pos_en = tr_neg_en = val_pos_en = val_neg_en = "0"
            tr_pos_fr = tr_neg_fr = val_pos_fr = val_neg_fr = "0"

        # Format category name: Primary Categories are bold with \newline (Primary Category)
        if num in PRIMARY_CATEGORIES:
            base_name = cat_name.replace(" (Primary Category)", "")
            cat_formatted = f"\\textbf{{{escape_latex(base_name)}\\newline (Primary Category)}}"
        else:
            cat_formatted = escape_latex(cat_name)

        # Order: # | Category | Code | metrics
        lines.append(f"{num} & {cat_formatted} & \\texttt{{{format_code(code)}}} & {tr_pos_en} & {tr_neg_en} & {val_pos_en} & {val_neg_en} & {tr_pos_fr} & {tr_neg_fr} & {val_pos_fr} & {val_neg_fr} \\\\")

        # Add \cmidrule after Primary Categories
        if num in PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    (OUTPUT_DIR / "table_b3_training_distribution.tex").write_text(output)
    print(f"   Saved: table_b3_training_distribution.tex")
    return output


# =============================================================================
# GENERATE LATEX TABLE B5: Test Performance Metrics (Detailed Validation)
# =============================================================================

def generate_table_b5(df):
    """Generate LaTeX for Table B5: Detailed validation performance metrics.

    Format matches existing paper table (detailed_validation_metrics) with Code column added.
    15 columns: # | Category | Code | F1 Macro (EN/FR/ALL) | F1 Micro (EN/FR/ALL) | F1 Weighted (EN/FR/ALL) | Support (EN/FR/ALL)
    """
    print("\n7. Generating Table B5 (detailed validation metrics)...")

    # Filter out global summary rows (number=0) for per-category pivot
    df_categories = df[df['number'] > 0].copy()

    # Pivot to wide format
    pivot = df_categories.pivot(
        index=['number', 'category', 'code'],
        columns='language',
        values=['f1_macro', 'f1_micro', 'f1_weighted', 'support_1']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Create dict for lookup
    data_dict = {}
    for _, row in pivot.iterrows():
        data_dict[int(row['number'])] = row

    # Get global summary data for OVERALL PERFORMANCE row
    global_data = df[df['number'] == 0]

    ncols = 15
    section_headers = get_section_headers(ncols)

    lines = []
    # Column order: # | Category | Code | metrics...
    # p{0.4cm} for #, p{4.0cm} for category, p{3.0cm} for code (wider)
    lines.append(r"\begin{longtable}{p{0.4cm}p{4.0cm}p{3.0cm}cccccccccccc}")
    lines.append(r"\caption{Detailed validation performance metrics for trained models (65 categories)}")
    lines.append(r"\label{tab:detailed_validation_metrics}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{F1 Macro}} & \multicolumn{3}{c}{\textbf{F1 Micro}} & \multicolumn{3}{c}{\textbf{F1 Weighted}} & \multicolumn{3}{c}{\textbf{Support}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\tablename\\ \\thetable\\ -- \\textit{{Continued from previous page}}}} \\\\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{F1 Macro}} & \multicolumn{3}{c}{\textbf{F1 Micro}} & \multicolumn{3}{c}{\textbf{F1 Weighted}} & \multicolumn{3}{c}{\textbf{Support}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    def fmt(val):
        if pd.isna(val):
            return "--"
        return f"{val:.3f}"

    def fmt_int(val):
        if pd.isna(val):
            return "--"
        return str(int(val))

    # Generate rows for all 65 categories (Table B1)
    for num in range(1, 66):
        # Add section headers
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = ALL_CATEGORIES[num]

        if num in data_dict:
            row = data_dict[num]
            f1_macro_en = fmt(row.get('f1_macro_EN'))
            f1_macro_fr = fmt(row.get('f1_macro_FR'))
            f1_macro_all = fmt(row.get('f1_macro_ALL'))
            f1_micro_en = fmt(row.get('f1_micro_EN'))
            f1_micro_fr = fmt(row.get('f1_micro_FR'))
            f1_micro_all = fmt(row.get('f1_micro_ALL'))
            f1_weighted_en = fmt(row.get('f1_weighted_EN'))
            f1_weighted_fr = fmt(row.get('f1_weighted_FR'))
            f1_weighted_all = fmt(row.get('f1_weighted_ALL'))
            support_en = fmt_int(row.get('support_1_EN'))
            support_fr = fmt_int(row.get('support_1_FR'))
            support_all = fmt_int(row.get('support_1_ALL'))
        else:
            f1_macro_en = f1_macro_fr = f1_macro_all = "--"
            f1_micro_en = f1_micro_fr = f1_micro_all = "--"
            f1_weighted_en = f1_weighted_fr = f1_weighted_all = "--"
            support_en = support_fr = support_all = "--"

        # Format category name: Primary Categories are bold with \newline (Primary Category)
        if num in PRIMARY_CATEGORIES:
            base_name = cat_name.replace(" (Primary Category)", "")
            cat_formatted = f"\\textbf{{{escape_latex(base_name)}\\newline (Primary Category)}}"
        else:
            cat_formatted = escape_latex(cat_name)

        # Order: # | Category | Code | metrics
        lines.append(f"{num} & {cat_formatted} & \\texttt{{{format_code(code)}}} & {f1_macro_en} & {f1_macro_fr} & {f1_macro_all} & {f1_micro_en} & {f1_micro_fr} & {f1_micro_all} & {f1_weighted_en} & {f1_weighted_fr} & {f1_weighted_all} & {support_en} & {support_fr} & {support_all} \\\\")

        # Add \cmidrule after Primary Categories
        if num in PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Add OVERALL PERFORMANCE row from global summary
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textbf{{OVERALL PERFORMANCE}}}} \\\\")

    if not global_data.empty:
        en_row = global_data[global_data['language'] == 'EN'].iloc[0] if len(global_data[global_data['language'] == 'EN']) > 0 else None
        fr_row = global_data[global_data['language'] == 'FR'].iloc[0] if len(global_data[global_data['language'] == 'FR']) > 0 else None
        all_row = global_data[global_data['language'] == 'ALL'].iloc[0] if len(global_data[global_data['language'] == 'ALL']) > 0 else None

        f1_macro_en = fmt(en_row['f1_macro']) if en_row is not None else "--"
        f1_macro_fr = fmt(fr_row['f1_macro']) if fr_row is not None else "--"
        f1_macro_all = fmt(all_row['f1_macro']) if all_row is not None else "--"
        f1_micro_en = fmt(en_row['f1_micro']) if en_row is not None else "--"
        f1_micro_fr = fmt(fr_row['f1_micro']) if fr_row is not None else "--"
        f1_micro_all = fmt(all_row['f1_micro']) if all_row is not None else "--"
        f1_weighted_en = fmt(en_row['f1_weighted']) if en_row is not None else "--"
        f1_weighted_fr = fmt(fr_row['f1_weighted']) if fr_row is not None else "--"
        f1_weighted_all = fmt(all_row['f1_weighted']) if all_row is not None else "--"
        support_en = fmt_int(en_row['support_1']) if en_row is not None else "--"
        support_fr = fmt_int(fr_row['support_1']) if fr_row is not None else "--"
        support_all = fmt_int(all_row['support_1']) if all_row is not None else "--"

        lines.append(f"& & \\textbf{{All Categories}} & \\textbf{{{f1_macro_en}}} & \\textbf{{{f1_macro_fr}}} & \\textbf{{{f1_macro_all}}} & \\textbf{{{f1_micro_en}}} & \\textbf{{{f1_micro_fr}}} & \\textbf{{{f1_micro_all}}} & \\textbf{{{f1_weighted_en}}} & \\textbf{{{f1_weighted_fr}}} & \\textbf{{{f1_weighted_all}}} & \\textbf{{{support_en}}} & \\textbf{{{support_fr}}} & \\textbf{{{support_all}}} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    (OUTPUT_DIR / "table_b5_test.tex").write_text(output)
    print(f"   Saved: table_b5_test.tex")
    return output


# =============================================================================
# GENERATE LATEX TABLE B6: Database-wide Distribution (from CCF_Database)
# =============================================================================

def fetch_database_counts():
    """
    Fetch annotation counts from CCF_Database for Table B6.

    Returns:
        dict: {category_number: {'en': count, 'fr': count, 'all': count}}
        tuple: (total_en, total_fr, total_all) sentence counts
    """
    print("\n8. Fetching annotation counts from CCF_Database...")

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

        # Get total sentence counts by language
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

        # Get counts for each annotation category
        category_counts = {}

        for num, (cat_name, code) in ALL_CATEGORIES.items():
            # Query counts for this annotation column (columns are INTEGER 0/1)
            try:
                cur.execute(f"""
                    SELECT language, COUNT(*)
                    FROM "{PROCESSED_TABLE}"
                    WHERE "{code}" = 1
                    GROUP BY language
                """)
                counts = dict(cur.fetchall())
                count_en = counts.get('EN', 0)
                count_fr = counts.get('FR', 0)
                category_counts[num] = {
                    'en': count_en,
                    'fr': count_fr,
                    'all': count_en + count_fr
                }
            except Exception as e:
                print(f"   WARNING: Could not query column '{code}': {e}")
                # Rollback to continue with next query
                conn.rollback()
                category_counts[num] = {'en': 0, 'fr': 0, 'all': 0}

        cur.close()
        conn.close()

        print(f"   Successfully fetched counts for {len(category_counts)} categories")
        return category_counts, (total_en, total_fr, total_all)

    except Exception as e:
        print(f"   ERROR: Database connection failed: {e}")
        return None, None


def generate_table_b6():
    """Generate LaTeX for Table B6: Database-wide distribution from CCF_Database.

    Format matches existing Table B6 exactly:
    8 columns: # | Category | Count EN | Count FR | Count ALL | Proportion EN | Proportion FR | Proportion ALL
    """
    print("\n9. Generating Table B6 (database-wide distribution)...")

    # Fetch data from database
    category_counts, totals = fetch_database_counts()

    if category_counts is None:
        print("   ERROR: Could not fetch database counts. Table B6 not generated.")
        return None

    total_en, total_fr, total_all = totals

    ncols = 9  # Added Code column
    section_headers = get_section_headers(ncols)

    lines = []
    # Column order: # | Category | Code | Count EN | Count FR | Count ALL | Prop EN | Prop FR | Prop ALL
    # p{0.5cm} for #, p{4.0cm} for category, p{2.5cm} for code (wider)
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.0cm}p{2.5cm}rrrrrr}")
    lines.append(r"\caption{Database-wide distribution of annotation categories across 9.2 million sentences}")
    lines.append(r"\label{tab:database_proportions}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{Count}} & \multicolumn{3}{c}{\textbf{Proportion (\%)}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\tablename\\ \\thetable\\ -- \\textit{{Continued from previous page}}}} \\\\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{Count}} & \multicolumn{3}{c}{\textbf{Proportion (\%)}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    def fmt_count(val):
        """Format count with thousands separator."""
        return f"{int(val):,}".replace(",", ",")

    def fmt_prop(count, total):
        """Format proportion as percentage with 2 decimals."""
        if total == 0:
            return "0.00"
        return f"{100 * count / total:.2f}"

    # Generate rows for all 65 categories
    for num in range(1, 66):
        # Add section headers
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = ALL_CATEGORIES[num]
        counts = category_counts.get(num, {'en': 0, 'fr': 0, 'all': 0})

        count_en = counts['en']
        count_fr = counts['fr']
        count_all = counts['all']

        prop_en = fmt_prop(count_en, total_en)
        prop_fr = fmt_prop(count_fr, total_fr)
        prop_all = fmt_prop(count_all, total_all)

        # Format category name: Primary Categories are bold with \newline (Primary Category)
        if num in PRIMARY_CATEGORIES:
            base_name = cat_name.replace(" (Primary Category)", "")
            cat_formatted = f"\\textbf{{{escape_latex(base_name)}\\newline (Primary Category)}}"
        else:
            cat_formatted = escape_latex(cat_name)

        # Order: # | Category | Code | counts | proportions
        lines.append(f"{num} & {cat_formatted} & \\texttt{{{format_code(code)}}} & {fmt_count(count_en)} & {fmt_count(count_fr)} & {fmt_count(count_all)} & {prop_en} & {prop_fr} & {prop_all} \\\\")

        # Add \cmidrule after Primary Categories
        if num in PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Add total row
    lines.append(r"\bottomrule")
    lines.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{Total Sentences}}}} & \\textbf{{{fmt_count(total_en)}}} & \\textbf{{{fmt_count(total_fr)}}} & \\textbf{{{fmt_count(total_all)}}} & \\textbf{{100.00}} & \\textbf{{100.00}} & \\textbf{{100.00}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    (OUTPUT_DIR / "table_b6_database_distribution.tex").write_text(output)
    print(f"   Saved: table_b6_database_distribution.tex")
    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CCF METHODOLOGY - NORMALIZATION AND TABLE GENERATION")
    print("=" * 70)

    # Normalize CSVs
    training_df = normalize_training_metrics()
    test_df = normalize_test_metrics()
    dist_df = normalize_distribution()
    training_db_df = normalize_training_database()

    # Generate LaTeX tables
    generate_table_b2(training_df)
    generate_table_b3(training_db_df)  # Use training_database_metrics for B3
    generate_table_b5(test_df)
    generate_table_b6()  # Database-wide distribution from CCF_Database

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nNormalized CSVs saved to: {DATA_DIR}")
    print(f"  - all_best_models_normalized.csv ({len(training_df)} rows)")
    print(f"  - final_annotation_metrics_normalized.csv ({len(test_df)} rows)")
    print(f"  - manual_annotations_metrics_normalized.csv ({len(dist_df)} rows)")
    print(f"  - training_database_metrics_normalized.csv ({len(training_db_df)} rows)")
    print(f"\nLaTeX tables saved to: {OUTPUT_DIR}")
    print("  - table_b2_training.tex (Training F1 metrics with Code column)")
    print("  - table_b3_training_distribution.tex (Train/Val Pos/Neg distribution with Code column)")
    print("  - table_b5_test.tex (Gold standard test metrics with Code column)")
    print("  - table_b6_database_distribution.tex (Database-wide distribution from CCF_Database)")
    print("\nTo integrate in LaTeX document:")
    print("  \\input{../Results/Outputs/Tables/table_b2_training.tex}")
    print("  \\input{../Results/Outputs/Tables/table_b3_training_distribution.tex}")
    print("  \\input{../Results/Outputs/Tables/table_b5_test.tex}")
    print("  \\input{../Results/Outputs/Tables/table_b6_database_distribution.tex}")


if __name__ == "__main__":
    main()
