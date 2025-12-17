#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables for CCF Methodology paper appendix.
This script normalizes category names and generates tables directly from CSV data.
"""

import pandas as pd
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/antoine/Documents/GitHub/CCF-canadian-climate-framing/Database/Training_data")
OUTPUT_DIR = SCRIPT_DIR.parent / "Outputs" / "Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CATEGORY MAPPING: CSV name -> (Official Name, Code, Number)
# Based on exact LaTeX table numbering (B2/B3 format with all categories)
# =============================================================================

CATEGORY_MAPPING = {
    # THEMATIC FRAMES - Economic (1-6)
    "Economic_Frame_Detection": ("Economic Frame Detection", "economic_frame", 1),
    "Economic Frame Detection": ("Economic Frame Detection", "economic_frame", 1),
    "Negative_Economic_Impacts": ("Negative Economic Impacts", "neg_econ_impacts", 2),
    "Negative Economic Impacts": ("Negative Economic Impacts", "neg_econ_impacts", 2),
    "Positive_Economic_Impacts": ("Positive Economic Impacts", "pos_econ_impacts", 3),
    "Positive Economic Impacts": ("Positive Economic Impacts", "pos_econ_impacts", 3),
    "Costs_of_Climate_Action": ("Costs of Climate Action", "costs_action", 4),
    "Costs of Climate Action": ("Costs of Climate Action", "costs_action", 4),
    "Benefits_of_Climate_Action": ("Benefits of Climate Action", "benefits_action", 5),
    "Benefits of Climate Action": ("Benefits of Climate Action", "benefits_action", 5),
    "Economic_Sector_Footprint": ("Economic Sector Footprint", "econ_footprint", 6),
    "Economic Sector Footprint": ("Economic Sector Footprint", "econ_footprint", 6),

    # THEMATIC FRAMES - Health (7-11)
    "Health_Frame_Detection": ("Health Frame Detection", "health_frame", 7),
    "Health Frame Detection": ("Health Frame Detection", "health_frame", 7),
    "Negative_Health_Impacts": ("Negative Health Impacts", "neg_health", 8),
    "Negative Health Impacts": ("Negative Health Impacts", "neg_health", 8),
    "Positive_Health_Impacts": ("Positive Health Impacts", "pos_health", 9),
    "Positive Health Impacts": ("Positive Health Impacts", "pos_health", 9),
    "Health_Co-benefits": ("Health Co-benefits", "health_cobenefit", 10),
    "Health Co-benefits": ("Health Co-benefits", "health_cobenefit", 10),
    "Health_Sector_Footprint": ("Health Sector Footprint", "health_footprint", 11),
    "Health Sector Footprint": ("Health Sector Footprint", "health_footprint", 11),

    # THEMATIC FRAMES - Security (12-17)
    "Security_Frame_Detection": ("Security Frame Detection", "security_frame", 12),
    "Security Frame Detection": ("Security Frame Detection", "security_frame", 12),
    "Military_Disaster_Response": ("Military Disaster Response", "military_response", 13),
    "Military Disaster Response": ("Military Disaster Response", "military_response", 13),
    "Military_Base_Disruption": ("Military Base Disruption", "military_base", 14),
    "Military Base Disruption": ("Military Base Disruption", "military_base", 14),
    "Climate-Driven_Displacement": ("Climate-Driven Displacement", "displacement", 15),
    "Climate-Driven Displacement": ("Climate-Driven Displacement", "displacement", 15),
    "Resource_Conflict": ("Resource Conflict", "resource_conflict", 16),
    "Resource Conflict": ("Resource Conflict", "resource_conflict", 16),
    "Defense_Sector_Footprint": ("Defense Sector Footprint", "defense_footprint", 17),
    "Defense Sector Footprint": ("Defense Sector Footprint", "defense_footprint", 17),

    # THEMATIC FRAMES - Justice (18-23)
    "Justice_Frame_Detection": ("Justice Frame Detection", "justice_frame", 18),
    "Justice Frame Detection": ("Justice Frame Detection", "justice_frame", 18),
    "Winners_and_Losers": ("Winners & Losers", "winners_losers", 19),
    "Winners and Losers": ("Winners & Losers", "winners_losers", 19),
    "Winners & Losers": ("Winners & Losers", "winners_losers", 19),
    "North-South_Responsibility": ("North-South Responsibility", "north_south", 20),
    "North-South Responsibility": ("North-South Responsibility", "north_south", 20),
    "Unequal_Impacts": ("Unequal Impacts", "unequal_impacts", 21),
    "Unequal Impacts": ("Unequal Impacts", "unequal_impacts", 21),
    "Unequal_Access": ("Unequal Access", "unequal_access", 22),
    "Unequal Access": ("Unequal Access", "unequal_access", 22),
    "Intergenerational_Justice": ("Intergenerational Justice", "intergenerational", 23),
    "Intergenerational Justice": ("Intergenerational Justice", "intergenerational", 23),

    # THEMATIC FRAMES - Political (24-28)
    "Political_Frame_Detection": ("Political Frame Detection", "political_frame", 24),
    "Political Frame Detection": ("Political Frame Detection", "political_frame", 24),
    "Policy_Measures": ("Policy Measures", "policy_measures", 25),
    "Policy Measures": ("Policy Measures", "policy_measures", 25),
    "Political_Debate": ("Political Debate", "political_debate", 26),
    "Political Debate": ("Political Debate", "political_debate", 26),
    "Political_Positioning": ("Political Positioning", "political_position", 27),
    "Political Positioning": ("Political Positioning", "political_position", 27),
    "Public_Opinion": ("Public Opinion", "public_opinion", 28),
    "Public Opinion": ("Public Opinion", "public_opinion", 28),

    # THEMATIC FRAMES - Scientific (29-33)
    "Scientific_Frame_Detection": ("Scientific Frame Detection", "scientific_frame", 29),
    "Scientific Frame Detection": ("Scientific Frame Detection", "scientific_frame", 29),
    "Scientific_Controversy": ("Scientific Controversy", "sci_controversy", 30),
    "Scientific Controversy": ("Scientific Controversy", "sci_controversy", 30),
    "Discovery_and_Innovation": ("Discovery & Innovation", "discovery_innov", 31),
    "Discovery and Innovation": ("Discovery & Innovation", "discovery_innov", 31),
    "Discovery & Innovation": ("Discovery & Innovation", "discovery_innov", 31),
    "Scientific_Uncertainty": ("Scientific Uncertainty", "sci_uncertainty", 32),
    "Scientific Uncertainty": ("Scientific Uncertainty", "sci_uncertainty", 32),
    "Scientific_Certainty": ("Scientific Certainty", "sci_certainty", 33),
    "Scientific Certainty": ("Scientific Certainty", "sci_certainty", 33),

    # THEMATIC FRAMES - Environmental (34-36)
    "Environmental_Frame_Detection": ("Environmental Frame Detection", "environ_frame", 34),
    "Environmental Frame Detection": ("Environmental Frame Detection", "environ_frame", 34),
    "Habitat_Loss": ("Habitat Loss", "habitat_loss", 35),
    "Habitat Loss": ("Habitat Loss", "habitat_loss", 35),
    "Species_Loss": ("Species Loss", "species_loss", 36),
    "Species Loss": ("Species Loss", "species_loss", 36),

    # THEMATIC FRAMES - Cultural (37-41)
    "Cultural_Frame_Detection": ("Cultural Frame Detection", "cultural_frame", 37),
    "Cultural Frame Detection": ("Cultural Frame Detection", "cultural_frame", 37),
    "Artistic_Representation": ("Artistic Representation", "artistic_rep", 38),
    "Artistic Representation": ("Artistic Representation", "artistic_rep", 38),
    "Event_Disruption": ("Event Disruption", "event_disruption", 39),
    "Event Disruption": ("Event Disruption", "event_disruption", 39),
    "Loss_of_Indigenous_Practices": ("Loss of Indigenous Practices", "indigenous_loss", 40),
    "Loss of Indigenous Practices": ("Loss of Indigenous Practices", "indigenous_loss", 40),
    "Cultural_Sector_Footprint": ("Cultural Sector Footprint", "cultural_footprint", 41),
    "Cultural Sector Footprint": ("Cultural Sector Footprint", "cultural_footprint", 41),

    # PRIMARY CATEGORIES - Messengers (42-51)
    "Messenger_Detection": ("Actors/Messengers Detection", "messenger_detect", 42),
    "Messenger Detection": ("Actors/Messengers Detection", "messenger_detect", 42),
    "Health_Expert": ("Health Expert", "health_expert", 43),
    "Health Expert": ("Health Expert", "health_expert", 43),
    "Economic_Expert": ("Economic Expert", "econ_expert", 44),
    "Economic Expert": ("Economic Expert", "econ_expert", 44),
    "Security_Expert": ("Security Expert", "security_expert", 45),
    "Security Expert": ("Security Expert", "security_expert", 45),
    "Legal_Expert": ("Legal Expert", "legal_expert", 46),
    "Legal Expert": ("Legal Expert", "legal_expert", 46),
    "Cultural_Expert": ("Cultural Expert", "cultural_expert", 47),
    "Cultural Expert": ("Cultural Expert", "cultural_expert", 47),
    "Natural_Scientist": ("Natural Scientist", "natural_scientist", 48),
    "Natural Scientist": ("Natural Scientist", "natural_scientist", 48),
    "Social_Scientist": ("Social Scientist", "social_scientist", 49),
    "Social Scientist": ("Social Scientist", "social_scientist", 49),
    "Activist": ("Activist", "activist", 50),
    "Public_Official": ("Public Official", "public_official", 51),
    "Public Official": ("Public Official", "public_official", 51),

    # PRIMARY CATEGORIES - Events (52-60)
    "Event_Detection": ("Event Detection", "event_detect", 52),
    "Event Detection": ("Event Detection", "event_detect", 52),
    "Extreme_Weather_Event": ("Extreme Weather Event", "extreme_weather", 53),
    "Extreme Weather Event": ("Extreme Weather Event", "extreme_weather", 53),
    "Meeting_Conference": ("Meeting/Conference", "meeting_conf", 54),
    "Meeting/Conference": ("Meeting/Conference", "meeting_conf", 54),
    "Meeting Conference": ("Meeting/Conference", "meeting_conf", 54),
    "Publication": ("Publication", "publication", 55),
    "Election": ("Election", "election", 56),
    "Policy_Announcement": ("Policy Announcement", "policy_announce", 57),
    "Policy Announcement": ("Policy Announcement", "policy_announce", 57),
    "Judiciary_Decision": ("Judiciary Decision", "judiciary", 58),
    "Judiciary Decision": ("Judiciary Decision", "judiciary", 58),
    "Cultural_Event": ("Cultural Event", "cultural_event", 59),
    "Cultural Event": ("Cultural Event", "cultural_event", 59),
    "Protest": ("Protest", "protest", 60),

    # PRIMARY CATEGORIES - Solutions (61-63)
    "Solutions_Detection": ("Solutions Detection", "solutions_detect", 61),
    "Solutions Detection": ("Solutions Detection", "solutions_detect", 61),
    "Mitigation_Strategy": ("Mitigation Strategy", "mitigation", 62),
    "Mitigation Strategy": ("Mitigation Strategy", "mitigation", 62),
    "Solution_1_SUB": ("Mitigation Strategy", "mitigation", 62),
    "Adaptation_Strategy": ("Adaptation Strategy", "adaptation", 63),
    "Adaptation Strategy": ("Adaptation Strategy", "adaptation", 63),
    "Solution_2_SUB": ("Adaptation Strategy", "adaptation", 63),

    # EMOTIONAL TONE (64-66)
    "Positive_Emotion": ("Positive Emotion", "pos_emotion", 64),
    "Positive Emotion": ("Positive Emotion", "pos_emotion", 64),
    "Negative_Emotion": ("Negative Emotion", "neg_emotion", 65),
    "Negative Emotion": ("Negative Emotion", "neg_emotion", 65),
    "Neutral_Emotion": ("Neutral Emotion", "neutral_emotion", 66),
    "Neutral Emotion": ("Neutral Emotion", "neutral_emotion", 66),

    # GEOGRAPHIC FOCUS (67)
    "Canadian_Context": ("Canadian Context", "canadian_context", 67),
    "Canadian Context": ("Canadian Context", "canadian_context", 67),

    # URGENCY/ALARMISM (68)
    "Urgency_Alarmism": ("Urgency/Alarmism", "urgency_alarmism", 68),
    "Urgency/Alarmism": ("Urgency/Alarmism", "urgency_alarmism", 68),
    "Urgency Alarmism": ("Urgency/Alarmism", "urgency_alarmism", 68),
}


def normalize_category_name(name: str) -> tuple:
    """Return (official_name, code, number) for a category name."""
    # Try direct lookup first
    if name in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[name]

    # Try with underscores replaced by spaces
    name_spaces = name.replace("_", " ")
    if name_spaces in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[name_spaces]

    # Try extracting from model path
    if "models/" in name:
        base = name.split("models/")[1].split(".jsonl")[0]
        # Remove language suffix
        for suffix in ["_EN", "_FR"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        if base in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[base]

    # Return original if not found
    print(f"Warning: Category not found in mapping: {name}")
    return (name, name.lower().replace(" ", "_"), 99)


def load_and_normalize_training_metrics():
    """Load and normalize all_best_models.csv - keeping ALL original columns"""
    df = pd.read_csv(DATA_DIR / "all_best_models.csv")

    # Extract category and language from saved_model_path
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

        official_name, code, num = normalize_category_name(cat_name)

        # Keep ALL original columns
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

    result = pd.DataFrame(records)
    result = result.sort_values(['number', 'language'])
    return result


def load_and_normalize_test_metrics():
    """Load and normalize final_annotation_metrics.csv - keeping ALL original columns"""
    df = pd.read_csv(DATA_DIR / "final_annotation_metrics.csv")

    records = []
    for _, row in df.iterrows():
        label = row['label']
        lang = row['language']

        if label == "ALL" or lang == "ALL":
            continue

        official_name, code, num = normalize_category_name(label)

        # Keep ALL original columns
        record = {
            'number': num,
            'category': official_name,
            'code': code,
            'language': lang,
        }
        # Add all other columns from original
        for col in df.columns:
            if col not in ['label', 'language']:
                record[col] = row[col]
        records.append(record)

    result = pd.DataFrame(records)
    result = result.sort_values(['number', 'language'])
    return result


def load_and_normalize_distribution():
    """Load and normalize manual_annotations_metrics.csv - keeping ALL original columns"""
    df = pd.read_csv(DATA_DIR / "manual_annotations_metrics.csv")

    records = []
    for _, row in df.iterrows():
        cat = row['Annotation Type']

        if 'Total' in str(cat):
            continue

        official_name, code, num = normalize_category_name(cat)

        # Keep ALL original columns with normalized names
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

    result = pd.DataFrame(records)
    result = result.sort_values('number')
    return result


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '#': r'\#',
        '$': r'\$',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_table_b2_training(df):
    """Generate LaTeX for Table B2: Training performance metrics"""

    # Pivot to wide format
    pivot = df.pivot(index=['number', 'category', 'code'], columns='language', values=['f1_1', 'f1_0', 'macro_f1'])
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index().sort_values('number')

    lines = []
    lines.append(r"\begin{longtable}{p{0.4cm}p{3.5cm}p{2.5cm}rrrrrr}")
    lines.append(r"\caption{Complete model training performance metrics for all annotation categories}")
    lines.append(r"\label{tab:complete_training_metrics}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & \multicolumn{2}{c}{\textbf{F1 (Class 0)}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{9}{c}{\tablename\ \thetable\ -- \textit{Continued from previous page}} \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & \multicolumn{2}{c}{\textbf{F1 (Class 0)}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in pivot.iterrows():
        num = int(row['number'])
        cat = escape_latex(row['category'])
        code = row['code']

        # Format values
        def fmt(val):
            if pd.isna(val):
                return "--"
            return f"{val:.3f}"

        f1_1_en = fmt(row.get('f1_1_EN'))
        f1_1_fr = fmt(row.get('f1_1_FR'))
        f1_0_en = fmt(row.get('f1_0_EN'))
        f1_0_fr = fmt(row.get('f1_0_FR'))
        macro_en = fmt(row.get('macro_f1_EN'))
        macro_fr = fmt(row.get('macro_f1_FR'))

        lines.append(f"{num} & {cat} & \\texttt{{{code}}} & {f1_1_en} & {f1_1_fr} & {f1_0_en} & {f1_0_fr} & {macro_en} & {macro_fr} \\\\")

    lines.append(r"\end{longtable}")

    return "\n".join(lines)


def generate_table_b5_test(df):
    """Generate LaTeX for Table B5: Test performance metrics"""

    # Pivot to wide format
    pivot = df.pivot(index=['number', 'category', 'code'], columns='language',
                     values=['f1_macro', 'f1_1', 'precision_1', 'recall_1', 'support_1'])
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index().sort_values('number')

    lines = []
    lines.append(r"\begin{longtable}{p{0.4cm}p{3.5cm}p{2.2cm}rrrrrrrr}")
    lines.append(r"\caption{Gold standard test performance metrics for all annotation categories}")
    lines.append(r"\label{tab:gold_test_metrics}")
    lines.append(r" \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{Macro F1}} & \multicolumn{2}{c}{\textbf{F1 (Pos)}} & \multicolumn{2}{c}{\textbf{Precision}} & \multicolumn{2}{c}{\textbf{Recall}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{11}{c}{\tablename\ \thetable\ -- \textit{Continued from previous page}} \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{Macro F1}} & \multicolumn{2}{c}{\textbf{F1 (Pos)}} & \multicolumn{2}{c}{\textbf{Precision}} & \multicolumn{2}{c}{\textbf{Recall}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{r}{\textit{Continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in pivot.iterrows():
        num = int(row['number'])
        cat = escape_latex(row['category'])
        code = row['code']

        def fmt(val):
            if pd.isna(val):
                return "--"
            return f"{val:.3f}"

        macro_en = fmt(row.get('f1_macro_EN'))
        macro_fr = fmt(row.get('f1_macro_FR'))
        f1_en = fmt(row.get('f1_1_EN'))
        f1_fr = fmt(row.get('f1_1_FR'))
        prec_en = fmt(row.get('precision_1_EN'))
        prec_fr = fmt(row.get('precision_1_FR'))
        rec_en = fmt(row.get('recall_1_EN'))
        rec_fr = fmt(row.get('recall_1_FR'))

        lines.append(f"{num} & {cat} & \\texttt{{{code}}} & {macro_en} & {macro_fr} & {f1_en} & {f1_fr} & {prec_en} & {prec_fr} & {rec_en} & {rec_fr} \\\\")

    lines.append(r"\end{longtable}")

    return "\n".join(lines)


def main():
    print("Loading and normalizing data...")

    # Load normalized data
    training_df = load_and_normalize_training_metrics()
    test_df = load_and_normalize_test_metrics()
    dist_df = load_and_normalize_distribution()

    # Save normalized CSVs
    training_df.to_csv(DATA_DIR / "all_best_models_normalized.csv", index=False)
    test_df.to_csv(DATA_DIR / "final_annotation_metrics_normalized.csv", index=False)
    dist_df.to_csv(DATA_DIR / "manual_annotations_metrics_normalized.csv", index=False)

    print(f"Saved normalized CSVs to {DATA_DIR}")

    # Generate LaTeX tables
    table_b2 = generate_table_b2_training(training_df)
    table_b5 = generate_table_b5_test(test_df)

    # Save LaTeX files
    (OUTPUT_DIR / "table_b2_training.tex").write_text(table_b2)
    (OUTPUT_DIR / "table_b5_test.tex").write_text(table_b5)

    print(f"Saved LaTeX tables to {OUTPUT_DIR}")

    # Print summary
    print(f"\nTraining metrics: {len(training_df)} rows")
    print(f"Test metrics: {len(test_df)} rows")
    print(f"Distribution: {len(dist_df)} rows")


if __name__ == "__main__":
    main()
