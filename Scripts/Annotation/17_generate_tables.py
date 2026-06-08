#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
17_generate_tables.py

MAIN OBJECTIVE:
---------------
This script is the single Python entry point for every numerical table
of the CCF Methodology paper that is reproducibly generated from the
canonical CSVs. No other Python file writes the eleven ``.tex``
artefacts that feed the main manuscript and the Supplementary
Information.

Three main-manuscript tables are produced:

  - table_performance_training.tex      : compact training-metric
    summary for the eleven primary detection categories plus the
    "Mention of Canada" and "Urgency to act" rows, ending with a
    Combined (ALL) average row.
  - table_validation_overall.tex        : three-row F1 macro / micro /
    weighted summary on the ALL,EN / ALL,FR / ALL,ALL rows of the
    gold standard.
  - table_intercoder_blind.tex          : single-row kappa / Gwet AC1
    / Krippendorff alpha table for the blind coding phase
    (sentences 601-1000).

Eight Supplementary Information tables are produced:

  S4  - Complete training performance metrics (per-category F1).
  S5  - Training / validation dataset distribution (positive and
        negative counts in each split, by language).
  S7  - Detailed validation performance metrics on the 1,000-sentence
        gold standard.
  S8  - Database-wide distribution of annotation categories across
        the 9.2-million-sentence corpus.
  S9  - Per-category inter-coder reliability metrics (full sample and
        blind phase: Cohen's kappa, Krippendorff alpha, Gwet AC1).
  S10 - Reliability tier assignment (Tier A / B / C with
        prevalence-induced reliability deflation and exclusion flags).
  S11 - Per-model training provenance (best epoch, training phase,
        validation macro F1, positive-class support).
  S12 - Complete data dictionary of the enriched CCF Database.

Tables S1 (literature review), S2 (newspapers), S3 (complete
framework), and S6 (NER performance) are not produced here; their
LaTeX source lives inline in ``CCF_Methodology_SI.tex``.

Section/header conventions reuse ``get_section_headers()`` and
``ALL_CATEGORIES`` from ``15_normalize_categories.py``, so the
styling of every Supplementary table is consistent with the prose
tables A1, A2, and S3 that live verbatim in the SI document.

Input files:
- Database/Training_data/training_hyperparameters_normalized.csv
- Database/Training_data/final_annotation_metrics.csv
- Database/Training_data/manual_annotations_JSONL/intercoder_reliability_4_before_after_600.csv
- Database/Training_data/all_best_models_normalized.csv
- Database/Training_data/final_annotation_metrics_normalized.csv
- Database/Training_data/training_database_metrics_normalized.csv
- Database/Training_data/per_category_reliability_normalized.csv
- Database/Training_data/reliability_tiers.csv
- Scripts/Annotation/15_normalize_categories.py
  (ALL_CATEGORIES, PRIMARY_CATEGORIES, LABEL_TO_NUMBER,
  get_section_headers, escape_latex, format_code,
  normalize_training_metrics, normalize_test_metrics,
  normalize_training_database, fetch_database_counts).

Output files:
- paper/CCF_Methodology/Results/Outputs/Tables/
      table_performance_training.tex
      table_validation_overall.tex
      table_intercoder_blind.tex
      table_s4_training_metrics.tex
      table_s5_training_distribution.tex
      table_s7_test_metrics.tex
      table_s8_database_distribution.tex
      table_s9_per_category_reliability.tex
      table_s10_reliability_tiers.tex
      table_s11_training_hyperparameters.tex
      table_s12_data_dictionary.tex

Dependencies:
-------------
- pandas
- pathlib
- importlib

MAIN FEATURES:
--------------
1) Single entry point - ``main()`` runs the eleven generators in
   sequence (main tables first, then SI S4, S5, S7, S8, S9, S10, S11,
   S12) and prints a one-line status per file. No Python script other
   than this one writes any of the eleven tables.
2) Exact reproduction - Every numerical value is read directly from
   a canonical CSV (or, for S8, pulled live from the deposited
   PostgreSQL database via ``norm.fetch_database_counts``); no value
   is hard-coded.
3) Strict consistency with the paper - The script reuses the exact
   styling helpers from ``15_normalize_categories.py`` (section
   headers, primary-category bolding, LaTeX escaping, code formatting)
   so that all SI tables share a uniform appearance.
4) Deterministic output - All SI tables iterate over the 65 canonical
   Supplementary Table S3 numbers in order; categories absent from
   the underlying CSV are rendered with ``--'' placeholders so the
   table is always complete.
5) Self-contained labels - Each SI longtable embeds a
   ``\refstepcounter{table}`` and a ``\label{}`` that uses the SI
   numbering, so cross-document references resolve without the SI
   builder having to inject any counter manipulation.

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################

import importlib.util
import sys
from pathlib import Path
from typing import List

import pandas as pd

##############################################################################
#                                  PATHS                                     #
##############################################################################

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "Database" / "Training_data"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "paper"
    / "CCF_Methodology"
    / "Results"
    / "Outputs"
    / "Tables"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main-manuscript inputs.
INPUT_TRAINING = DATA_DIR / "training_hyperparameters_normalized.csv"
INPUT_VALIDATION = DATA_DIR / "final_annotation_metrics.csv"
INPUT_INTERCODER = (
    DATA_DIR / "manual_annotations_JSONL" / "intercoder_reliability_4_before_after_600.csv"
)

# Supplementary-Information inputs.
INPUT_RELIABILITY = DATA_DIR / "per_category_reliability_normalized.csv"
INPUT_TIERS = DATA_DIR / "reliability_tiers.csv"
INPUT_HYPERPARAMS = DATA_DIR / "training_hyperparameters_normalized.csv"

# Main-manuscript outputs.
OUT_PERFORMANCE = OUTPUT_DIR / "table_performance_training.tex"
OUT_VALIDATION = OUTPUT_DIR / "table_validation_overall.tex"
OUT_INTERCODER = OUTPUT_DIR / "table_intercoder_blind.tex"

# Supplementary-Information outputs.
OUTPUT_S4 = OUTPUT_DIR / "table_s4_training_metrics.tex"
OUTPUT_S5 = OUTPUT_DIR / "table_s5_training_distribution.tex"
OUTPUT_S7 = OUTPUT_DIR / "table_s7_test_metrics.tex"
OUTPUT_S8 = OUTPUT_DIR / "table_s8_database_distribution.tex"
OUTPUT_S9 = OUTPUT_DIR / "table_s9_per_category_reliability.tex"
OUTPUT_S10 = OUTPUT_DIR / "table_s10_reliability_tiers.tex"
OUTPUT_S11 = OUTPUT_DIR / "table_s11_training_hyperparameters.tex"
OUTPUT_S12 = OUTPUT_DIR / "table_s12_data_dictionary.tex"


##############################################################################
#               DYNAMIC IMPORT OF 15_normalize_categories.py                 #
##############################################################################


def _load_normalization_module():
    """Load 15_normalize_categories.py and return its module object."""
    spec_path = SCRIPT_DIR / "15_normalize_categories.py"
    spec = importlib.util.spec_from_file_location("ccf_normalization", spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load normalisation module at {spec_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ccf_normalization"] = module
    spec.loader.exec_module(module)
    return module


##############################################################################
#                              HELPERS                                       #
##############################################################################


def _fmt_float(value, decimals: int = 3) -> str:
    """Format a float to ``decimals`` digits or ``--`` if missing."""
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _fmt_int(value) -> str:
    if pd.isna(value):
        return "--"
    return str(int(value))


def _fmt_bool(value) -> str:
    if pd.isna(value):
        return "--"
    return "Yes" if bool(value) else ""


def _fmt_category_cell(num: int, name: str, primary_set: set, escape_latex) -> str:
    """Render the Category column, bolding primary categories."""
    if num in primary_set:
        base_name = name.replace(" (Primary Category)", "")
        return f"\\textbf{{{escape_latex(base_name)}\\newline (Primary Category)}}"
    return escape_latex(name)


##############################################################################
#       Table 3: Training performance, primary detection categories          #
##############################################################################

PRIMARY_DETECTION_ROWS = [
    # (code in training CSV, display label, group)
    ("economic_frame",      "Economic Frame",      "main"),
    ("health_frame",        "Health Frame",        "main"),
    ("security_frame",      "Security Frame",      "main"),
    ("justice_frame",       "Justice Frame",       "main"),
    ("political_frame",     "Political Frame",     "main"),
    ("scientific_frame",    "Scientific Frame",    "main"),
    ("environmental_frame", "Environmental Frame", "main"),
    ("cultural_frame",      "Cultural Frame",      "main"),
    ("messenger",           "Presence of Messengers", "other"),
    ("event",               "Presence of Events",    "other"),
    ("solution",            "Presence of Solutions", "other"),
    ("canada",              "Mention of Canada",     "other"),
    ("urgency",             "Urgency to act",        "other"),
]


def generate_table_performance() -> None:
    """Render the training-metric table for the primary detection categories."""
    df = pd.read_csv(INPUT_TRAINING)

    def _row(code: str, label: str) -> str:
        en = df[(df["code"] == code) & (df["language"] == "EN")]
        fr = df[(df["code"] == code) & (df["language"] == "FR")]
        if en.empty or fr.empty:
            raise RuntimeError(f"Missing training metrics for {code}")
        e = en.iloc[0]
        f = fr.iloc[0]
        return (
            f"{label} & {e['best_f1_1']:.3f} & {f['best_f1_1']:.3f} & "
            f"{e['best_f1_0']:.3f} & {f['best_f1_0']:.3f} & "
            f"{e['best_macro_f1']:.3f} & {f['best_macro_f1']:.3f} \\\\"
        )

    # Per-language averages across the 13 primary detection rows shown in
    # the table. These match the "Overall Average*" line, whose footnote
    # marker (*) refers to a clarification placed in the main manuscript
    # ("Overall average across active categories shown in this table").
    codes = [c for c, _l, _g in PRIMARY_DETECTION_ROWS]
    en_subset = df[(df["code"].isin(codes)) & (df["language"] == "EN")]
    fr_subset = df[(df["code"].isin(codes)) & (df["language"] == "FR")]
    en_avg_f1_1 = en_subset["best_f1_1"].mean()
    fr_avg_f1_1 = fr_subset["best_f1_1"].mean()
    en_avg_f1_0 = en_subset["best_f1_0"].mean()
    fr_avg_f1_0 = fr_subset["best_f1_0"].mean()
    en_avg_macro = en_subset["best_macro_f1"].mean()
    fr_avg_macro = fr_subset["best_macro_f1"].mean()

    # Combined (ALL) -- macro across the entire set of trained models
    # (one row per model in the training_hyperparameters table; both EN
    # and FR rows are pooled). This matches the headline "0.826 macro F1
    # during the training phase" cited in the Methods section.
    en_all = df[df["language"] == "EN"]
    fr_all = df[df["language"] == "FR"]
    all_subset = pd.concat([en_all, fr_all])
    comb_f1_1 = all_subset["best_f1_1"].mean()
    comb_f1_0 = all_subset["best_f1_0"].mean()
    comb_macro = all_subset["best_macro_f1"].mean()

    lines = [
        r"\begin{tabular}{l|rr|rr|rr}",
        r"\toprule",
        r"\rowcolor{gray!10}",
        r"\textbf{Category} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & "
        r"\multicolumn{2}{c}{\textbf{F1 (Class 0)}} & "
        r"\multicolumn{2}{c}{\textbf{Macro F1}} \\",
        r"\cline{2-7}",
        r"\rowcolor{gray!10}",
        r"& \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & "
        r"\textbf{EN} & \textbf{FR} \\",
        r"\midrule",
        r"\rowcolor{gray!10}",
        r"\multicolumn{7}{l}{\textit{Main Frames}} \\",
        r"\midrule",
    ]
    for code, label, group in PRIMARY_DETECTION_ROWS:
        if group == "other" and not any(
            ll == "Presence of Messengers" for ll in [label]
        ):
            pass
        if group == "main":
            lines.append(_row(code, label))
    lines.extend([
        r"\midrule",
        r"\rowcolor{gray!10}",
        r"\multicolumn{7}{l}{\textit{Other Primary Categories: Actors, Events and Solutions}} \\",
        r"\midrule",
    ])
    for code, label, group in PRIMARY_DETECTION_ROWS:
        if group == "other":
            lines.append(_row(code, label))
    lines.extend([
        r"\midrule",
        r"\rowcolor{gray!15}",
        (
            f"\\textbf{{Overall Average}}* & \\textbf{{{en_avg_f1_1:.3f}}} & "
            f"\\textbf{{{fr_avg_f1_1:.3f}}} & \\textbf{{{en_avg_f1_0:.3f}}} & "
            f"\\textbf{{{fr_avg_f1_0:.3f}}} & \\textbf{{{en_avg_macro:.3f}}} & "
            f"\\textbf{{{fr_avg_macro:.3f}}} \\\\"
        ),
        r"\rowcolor{gray!15}",
        (
            f"\\textbf{{Combined (ALL)}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{comb_f1_1:.3f}}}}} & "
            f"\\multicolumn{{2}}{{c}}{{\\textbf{{{comb_f1_0:.3f}}}}} & "
            f"\\multicolumn{{2}}{{c}}{{\\textbf{{{comb_macro:.3f}}}}} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ])
    OUT_PERFORMANCE.write_text("\n".join(lines))
    print(f"  wrote: {OUT_PERFORMANCE.relative_to(PROJECT_ROOT)}")


##############################################################################
#       Table 4: Overall validation performance metrics                      #
##############################################################################


def generate_table_validation_overall() -> None:
    """Render the EN / FR / ALL validation table (F1 macro / micro / weighted)."""
    df = pd.read_csv(INPUT_VALIDATION)
    rows: list[tuple[str, str, float, float, float]] = []
    for lang_csv, label in [("EN", "English (EN)"), ("FR", "French (FR)"), ("ALL", "Combined (ALL)")]:
        r = df[(df["label"] == "ALL") & (df["language"] == lang_csv)]
        if r.empty:
            raise RuntimeError(f"Missing ALL row for language {lang_csv}")
        r = r.iloc[0]
        rows.append((label, lang_csv, r["f1_macro"], r["f1_micro"], r["f1_weighted"]))

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Language} & \textbf{F1 Macro} & \textbf{F1 Micro} & "
        r"\textbf{F1 Weighted} \\",
        r"\midrule",
    ]
    en_row = rows[0]
    fr_row = rows[1]
    all_row = rows[2]
    lines.append(f"{en_row[0]} & {en_row[2]:.3f} & {en_row[3]:.3f} & {en_row[4]:.3f} \\\\")
    lines.append(f"{fr_row[0]} & {fr_row[2]:.3f} & {fr_row[3]:.3f} & {fr_row[4]:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{{all_row[0]}}} & \\textbf{{{all_row[2]:.3f}}} & "
        f"\\textbf{{{all_row[3]:.3f}}} & \\textbf{{{all_row[4]:.3f}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT_VALIDATION.write_text("\n".join(lines))
    print(f"  wrote: {OUT_VALIDATION.relative_to(PROJECT_ROOT)}")


##############################################################################
#       Inter-coder table for the blind coding phase                         #
##############################################################################


def generate_table_intercoder_blind() -> None:
    """Render the single-row blind-phase \\kappa / Gwet AC1 / Krippendorff
    \\alpha table from the canonical CSV."""
    df = pd.read_csv(INPUT_INTERCODER)
    row = df[df["label"] == "OVERALL"]
    if row.empty:
        raise RuntimeError("OVERALL row missing from intercoder before/after CSV")
    r = row.iloc[0]
    after_n = int(r["after_n"])
    kappa = float(r["after_kappa"])
    gwet = float(r["after_gwet"])
    kripp = float(r["after_kripp"])

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & Cohen's $\kappa$ & Gwet's AC1 & Krippendorff's $\alpha$ \\",
        r"\midrule",
        f"Blind coding phase (601--1000, $n={after_n}$) & {kappa:.3f} & {gwet:.3f} & {kripp:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    OUT_INTERCODER.write_text("\n".join(lines))
    print(f"  wrote: {OUT_INTERCODER.relative_to(PROJECT_ROOT)}")


##############################################################################
#               TABLE S4 - Complete training performance metrics             #
##############################################################################


def generate_table_s4(norm, training_df) -> str:
    """Render LaTeX for Supplementary Table~S4 (training F1 metrics).

    Nine-column layout: # | Category | Code | F1(1) EN | F1(1) FR |
    F1(0) EN | F1(0) FR | Macro EN | Macro FR.
    """
    print("\n  generating Supplementary Table S4 (training metrics)...")

    # Pivot to wide format (EN/FR side by side).
    pivot = training_df.pivot(
        index=['number', 'category', 'code'],
        columns='language',
        values=['f1_1', 'f1_0', 'macro_f1']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    data_dict = {int(row['number']): row for _, row in pivot.iterrows()}

    ncols = 9
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # No \caption inside the longtable: the "Supplementary Table S4."
    # title is already written above the landscape block by
    # CCF_Methodology_SI.tex. \refstepcounter advances
    # the table counter so \ref{tab:training_metrics_s4} resolves
    # correctly.
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:training_metrics_s4}")
    lines.append(r"\begin{longtable}{p{0.4cm}p{4.0cm}p{3.0cm}rrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{2}{c}{\textbf{F1 (Class 1)}} & \multicolumn{2}{c}{\textbf{F1 (Class 0)}} & \multicolumn{2}{c}{\textbf{Macro F1}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} & \textbf{EN} & \textbf{FR} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{Supplementary Table~S4 -- Continued from previous page}}}} \\\\")
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

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )

        if num in data_dict:
            row = data_dict[num]
            f1_1_en = fmt(row.get('f1_1_EN'))
            f1_1_fr = fmt(row.get('f1_1_FR'))
            f1_0_en = fmt(row.get('f1_0_EN'))
            f1_0_fr = fmt(row.get('f1_0_FR'))
            macro_en = fmt(row.get('macro_f1_EN'))
            macro_fr = fmt(row.get('macro_f1_FR'))
        else:
            f1_1_en = f1_1_fr = f1_0_en = f1_0_fr = macro_en = macro_fr = "--"

        lines.append(
            f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
            f"{f1_1_en} & {f1_1_fr} & {f1_0_en} & {f1_0_fr} & "
            f"{macro_en} & {macro_fr} \\\\"
        )

        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Aggregate averages.
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textbf{{AVERAGE PERFORMANCE METRICS}}}} \\\\")
    en_f1_1 = training_df[training_df['language'] == 'EN']['f1_1'].mean()
    fr_f1_1 = training_df[training_df['language'] == 'FR']['f1_1'].mean()
    en_f1_0 = training_df[training_df['language'] == 'EN']['f1_0'].mean()
    fr_f1_0 = training_df[training_df['language'] == 'FR']['f1_0'].mean()
    en_macro = training_df[training_df['language'] == 'EN']['macro_f1'].mean()
    fr_macro = training_df[training_df['language'] == 'FR']['macro_f1'].mean()
    lines.append(f"& & \\textbf{{English Average}} & {en_f1_1:.3f} & -- & {en_f1_0:.3f} & -- & {en_macro:.3f} & -- \\\\")
    lines.append(f"& & \\textbf{{French Average}} & -- & {fr_f1_1:.3f} & -- & {fr_f1_0:.3f} & -- & {fr_macro:.3f} \\\\")
    lines.append(
        f"& & \\textbf{{Overall Average}} & "
        f"\\multicolumn{{2}}{{c}}{{{(en_f1_1+fr_f1_1)/2:.3f}}} & "
        f"\\multicolumn{{2}}{{c}}{{{(en_f1_0+fr_f1_0)/2:.3f}}} & "
        f"\\multicolumn{{2}}{{c}}{{{(en_macro+fr_macro)/2:.3f}}} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textbf{{TOTAL: 65 ANNOTATION CATEGORIES}}}} \\\\")
    n_total = training_df['number'].nunique()
    n_excluded = 65 - n_total
    lines.append(
        f"\\multicolumn{{{ncols}}}{{l}}{{\\cellcolor{{gray!20}}\\textit{{("
        f"{n_total} categories with at least one model; {n_excluded} categories "
        f"entirely excluded*)}}}} \\\\"
    )
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    OUTPUT_S4.write_text(output)
    print(f"  wrote: {OUTPUT_S4.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#       TABLE S5 - Training / validation dataset distribution                #
##############################################################################


def generate_table_s5(norm, training_db_df) -> str:
    """Render LaTeX for Supplementary Table~S5 (train/val distribution).

    Eleven-column layout: # | Category | Code |
    Train Pos/Neg EN | Val Pos/Neg EN | Train Pos/Neg FR | Val Pos/Neg FR.
    """
    print("\n  generating Supplementary Table S5 (training/validation distribution)...")

    data_dict = {int(row['number']): row for _, row in training_db_df.iterrows()}

    ncols = 11
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # No \caption: the "Supplementary Table S5." title is written
    # above the landscape block by the SI builder.
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:training_distribution_s5}")
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.0cm}p{3.0cm}rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{3}{*}{\textbf{\#}} & \multirow{3}{*}{\textbf{Category}} & \multirow{3}{*}{\textbf{Code}} & \multicolumn{4}{c}{\textbf{English}} & \multicolumn{4}{c}{\textbf{French}} \\")
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(r"& & & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} & \multicolumn{2}{c}{\textbf{Training}} & \multicolumn{2}{c}{\textbf{Validation}} \\")
    lines.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    lines.append(r"& & & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} & \textbf{Pos} & \textbf{Neg} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{Supplementary Table~S5 -- Continued from previous page}}}} \\\\")
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

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )

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

        lines.append(
            f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
            f"{tr_pos_en} & {tr_neg_en} & {val_pos_en} & {val_neg_en} & "
            f"{tr_pos_fr} & {tr_neg_fr} & {val_pos_fr} & {val_neg_fr} \\\\"
        )

        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    OUTPUT_S5.write_text(output)
    print(f"  wrote: {OUTPUT_S5.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#         TABLE S7 - Detailed validation performance on gold standard        #
##############################################################################


def generate_table_s7(norm, test_df) -> str:
    """Render LaTeX for Supplementary Table~S7 (detailed validation metrics).

    Fifteen-column layout: # | Category | Code |
    F1 macro (EN/FR/ALL) | F1 micro (EN/FR/ALL) |
    F1 weighted (EN/FR/ALL) | Support (EN/FR/ALL).
    """
    print("\n  generating Supplementary Table S7 (detailed validation metrics)...")

    # Filter out global summary rows (number=0) for per-category pivot.
    df_categories = test_df[test_df['number'] > 0].copy()

    pivot = df_categories.pivot(
        index=['number', 'category', 'code'],
        columns='language',
        values=['f1_macro', 'f1_micro', 'f1_weighted', 'support_1']
    )
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    data_dict = {int(row['number']): row for _, row in pivot.iterrows()}

    # Global summary data for the OVERALL PERFORMANCE row.
    global_data = test_df[test_df['number'] == 0]

    ncols = 15
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # No \caption: the "Supplementary Table S7." title is written
    # above the landscape block by the SI builder.
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:test_metrics_s7}")
    lines.append(r"\begin{longtable}{p{0.4cm}p{4.0cm}p{3.0cm}cccccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{F1 Macro}} & \multicolumn{3}{c}{\textbf{F1 Micro}} & \multicolumn{3}{c}{\textbf{F1 Weighted}} & \multicolumn{3}{c}{\textbf{Support}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} \cmidrule(lr){13-15}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{Supplementary Table~S7 -- Continued from previous page}}}} \\\\")
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

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )

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

        lines.append(
            f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
            f"{f1_macro_en} & {f1_macro_fr} & {f1_macro_all} & "
            f"{f1_micro_en} & {f1_micro_fr} & {f1_micro_all} & "
            f"{f1_weighted_en} & {f1_weighted_fr} & {f1_weighted_all} & "
            f"{support_en} & {support_fr} & {support_all} \\\\"
        )

        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

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

        lines.append(
            f"& & \\textbf{{All Categories}} & "
            f"\\textbf{{{f1_macro_en}}} & \\textbf{{{f1_macro_fr}}} & \\textbf{{{f1_macro_all}}} & "
            f"\\textbf{{{f1_micro_en}}} & \\textbf{{{f1_micro_fr}}} & \\textbf{{{f1_micro_all}}} & "
            f"\\textbf{{{f1_weighted_en}}} & \\textbf{{{f1_weighted_fr}}} & \\textbf{{{f1_weighted_all}}} & "
            f"\\textbf{{{support_en}}} & \\textbf{{{support_fr}}} & \\textbf{{{support_all}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    OUTPUT_S7.write_text(output)
    print(f"  wrote: {OUTPUT_S7.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#           TABLE S8 - Database-wide distribution (from CCF_Database)        #
##############################################################################


def generate_table_s8(norm) -> str:
    """Render LaTeX for Supplementary Table~S8 (database-wide distribution).

    Nine-column layout: # | Category | Code |
    Count (EN/FR/ALL) | Proportion (EN/FR/ALL).
    Counts are pulled live from ``CCF_Database`` by
    ``norm.fetch_database_counts()``.
    """
    print("\n  generating Supplementary Table S8 (database-wide distribution)...")

    category_counts, totals = norm.fetch_database_counts()

    if category_counts is None:
        print("   ERROR: Could not fetch database counts. Table S8 not generated.")
        return None

    total_en, total_fr, total_all = totals

    ncols = 9
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # No \caption: the "Supplementary Table S8." title is written
    # above the landscape block by the SI builder.
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:database_distribution_s8}")
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.0cm}p{2.5cm}rrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & \multirow{2}{*}{\textbf{Code}} & \multicolumn{3}{c}{\textbf{Count}} & \multicolumn{3}{c}{\textbf{Proportion (\%)}} \\")
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}")
    lines.append(r"& & & \textbf{EN} & \textbf{FR} & \textbf{ALL} & \textbf{EN} & \textbf{FR} & \textbf{ALL} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\textit{{Supplementary Table~S8 -- Continued from previous page}}}} \\\\")
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
        return f"{int(val):,}"

    def fmt_prop(count, total):
        if total == 0:
            return "0.00"
        return f"{100 * count / total:.2f}"

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)

        cat_name, code = norm.ALL_CATEGORIES[num]
        counts = category_counts.get(num, {'en': 0, 'fr': 0, 'all': 0})

        count_en = counts['en']
        count_fr = counts['fr']
        count_all = counts['all']

        prop_en = fmt_prop(count_en, total_en)
        prop_fr = fmt_prop(count_fr, total_fr)
        prop_all = fmt_prop(count_all, total_all)

        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )

        lines.append(
            f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
            f"{fmt_count(count_en)} & {fmt_count(count_fr)} & {fmt_count(count_all)} & "
            f"{prop_en} & {prop_fr} & {prop_all} \\\\"
        )

        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    lines.append(r"\bottomrule")
    lines.append(
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Total Sentences}}}} & "
        f"\\textbf{{{fmt_count(total_en)}}} & \\textbf{{{fmt_count(total_fr)}}} & "
        f"\\textbf{{{fmt_count(total_all)}}} & "
        f"\\textbf{{100.00}} & \\textbf{{100.00}} & \\textbf{{100.00}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")

    output = "\n".join(lines)
    OUTPUT_S8.write_text(output)
    print(f"  wrote: {OUTPUT_S8.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#                  TABLE S9 - Per-category reliability                       #
##############################################################################


def generate_table_s9(norm) -> str:
    """Render LaTeX for Supplementary Table S9 (per-category reliability)."""
    df = pd.read_csv(INPUT_RELIABILITY)
    data_by_number = {int(r["number"]): r for _, r in df.iterrows()}

    ncols = 9  # # | Category | Code | k_full | k_blind | α_blind | AC1_blind | %ag_blind | n_blind
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # ------------------------------------------------------------------
    # No \caption inside the longtable: see generate_table_s11 for the
    # rationale (longtable's caption builder reads \hsize at construction
    # time, when it equals the cell width rather than the page width,
    # which produces an extraneous title line and shifts the alignment).
    # The "Supplementary Table S9." title is injected above the
    # landscape block by CCF_Methodology_SI.tex;
    # \refstepcounter advances the table counter so
    # \ref{tab:per_category_reliability} resolves and "Continued ..."
    # footers print the right number.
    # ------------------------------------------------------------------
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:per_category_reliability}")
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.4cm}p{3.0cm}rrrrrr}")
    header_main = (
        r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & "
        r"\multirow{2}{*}{\textbf{Code}} & \textbf{Cohen's $\kappa$} & "
        r"\multicolumn{4}{c}{\textbf{Blind phase (sentences 601--1000)}} & "
        r"\multirow{2}{*}{\textbf{$n$}} \\"
    )
    header_sub = (
        r"& & & \textbf{(full)} & \textbf{$\kappa$} & "
        r"\textbf{Krippendorff $\alpha$} & \textbf{Gwet AC1} & "
        r"\textbf{\% agree} & \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){5-8}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{c}}"
        r"{\textit{Supplementary Table~S9 -- Continued from previous page}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){5-8}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)
        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )
        if num in data_by_number:
            r = data_by_number[num]
            lines.append(
                f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
                f"{_fmt_float(r.get('cohens_kappa_full'))} & "
                f"{_fmt_float(r.get('cohens_kappa_blind_phase'))} & "
                f"{_fmt_float(r.get('krippendorff_alpha_blind_phase'))} & "
                f"{_fmt_float(r.get('gwet_ac1_blind_phase'))} & "
                f"{_fmt_float(r.get('percent_agree_blind_phase'))} & "
                f"{_fmt_int(r.get('n_blind_phase'))} \\\\"
            )
        else:
            lines.append(
                f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} "
                "& -- & -- & -- & -- & -- & -- \\\\"
            )
        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    lines.append(r"\end{longtable}")
    output = "\n".join(lines)
    OUTPUT_S9.write_text(output)
    print(f"  wrote: {OUTPUT_S9.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#                  TABLE S10 - Reliability tiers                             #
##############################################################################


def generate_table_s10(norm) -> str:
    """Render LaTeX for Supplementary Table S10 (reliability tiers)."""
    df = pd.read_csv(INPUT_TIERS)
    data_by_number = {int(r["number"]): r for _, r in df.iterrows()}

    ncols = 11  # # | Category | Code | f1_all | f1_en | f1_fr | k_blind | tier_o | tier_en | tier_fr | flag
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # ------------------------------------------------------------------
    # Layout rationale
    # ------------------------------------------------------------------
    # 22 cm landscape text block. p-column totals:
    # 0.5 + 4.2 + 2.8 + 3*1.1 + 1.3 + 3*0.9 + 1.6 = 13.8 cm. The Flag
    # column is 1.6 cm wide (compact since we now use the acronym
    # "PIRD*" instead of the full phrase; the definition is given in
    # the table footer).
    #
    # No \caption inside the longtable: see generate_table_s11 for the
    # rationale. The "Supplementary Table S10." title is injected above
    # the landscape block by CCF_Methodology_SI.tex and
    # the table counter is advanced manually via \refstepcounter so the
    # \label resolves correctly from cross-document \ref.
    # ------------------------------------------------------------------
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:reliability_tiers}")
    lines.append(
        r"\begin{longtable}{p{0.5cm}p{4.2cm}p{2.8cm}"
        r">{\centering\arraybackslash}p{1.1cm}>{\centering\arraybackslash}p{1.1cm}"
        r">{\centering\arraybackslash}p{1.1cm}>{\centering\arraybackslash}p{1.3cm}"
        r">{\centering\arraybackslash}p{0.9cm}>{\centering\arraybackslash}p{0.9cm}"
        r">{\centering\arraybackslash}p{0.9cm}>{\centering\arraybackslash}p{1.6cm}}"
    )
    header_main = (
        r"\multirow{2}{*}{\textbf{\#}} & \multirow{2}{*}{\textbf{Category}} & "
        r"\multirow{2}{*}{\textbf{Code}} & "
        r"\multicolumn{3}{p{3.3cm}}{\centering\textbf{Validation $F_1$ macro}} & "
        r"\textbf{Blind} & "
        r"\multicolumn{3}{p{2.7cm}}{\centering\textbf{Tier}} & "
        r"\multirow{2}{*}{\textbf{Flag}} \\"
    )
    header_sub = (
        r"& & & \textbf{All} & \textbf{EN} & \textbf{FR} & "
        r"\textbf{$\kappa$} & \textbf{All} & \textbf{EN} & \textbf{FR} & \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){8-10}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{c}}"
        r"{\textit{Supplementary Table~S10 -- Continued from previous page}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){4-6} \cmidrule(lr){8-10}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)
        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )
        r = data_by_number.get(num)
        if r is None:
            lines.append(
                f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} "
                "& -- & -- & -- & -- & -- & -- & -- & -- \\\\"
            )
        else:
            flag_parts: list[str] = []
            if bool(r.get("paradox_kappa")):
                # Compact acronym; the definition is provided in the
                # table footer (see PIRD note).
                flag_parts.append("PIRD*")
            reason = r.get("exclusion_reason")
            if isinstance(reason, str) and reason.strip():
                flag_parts.append("excluded")
            flag_cell = ", ".join(flag_parts) if flag_parts else ""
            lines.append(
                f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
                f"{_fmt_float(r.get('f1_all'))} & "
                f"{_fmt_float(r.get('f1_en'))} & "
                f"{_fmt_float(r.get('f1_fr'))} & "
                f"{_fmt_float(r.get('cohens_kappa_blind_phase'))} & "
                f"\\textbf{{{r.get('tier_overall')}}} & "
                f"{r.get('tier_en')} & {r.get('tier_fr')} & "
                f"{flag_cell} \\\\"
            )
        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Aggregate counts at the bottom.
    counts_overall = df["tier_overall"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    counts_en = df["tier_en"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    counts_fr = df["tier_fr"].value_counts().reindex(["A", "B", "C"]).fillna(0).astype(int)
    n_paradox = int(df["paradox_kappa"].sum())
    n_excl = int(
        df["exclusion_reason"].fillna("").astype(str).str.len().gt(0).sum()
    )
    # Wrap the tier-counts summary in a width-bounded \multicolumn{N}{p{X}}
    # so the long sentence breaks across lines instead of stretching the
    # whole alignment to its natural width.
    lines.append(
        f"\\multicolumn{{{ncols}}}{{p{{\\dimexpr\\linewidth-2\\tabcolsep\\relax}}}}"
        f"{{\\cellcolor{{gray!20}}\\textbf{{TIER COUNTS:}} "
        f"All: A={counts_overall['A']} / B={counts_overall['B']} / C={counts_overall['C']}; "
        f"EN: A={counts_en['A']} / B={counts_en['B']} / C={counts_en['C']}; "
        f"FR: A={counts_fr['A']} / B={counts_fr['B']} / C={counts_fr['C']}; "
        f"PIRD*: {n_paradox}; excluded from training: {n_excl}.}} \\\\"
    )
    lines.append(r"\end{longtable}")
    output = "\n".join(lines)
    OUTPUT_S10.write_text(output)
    print(f"  wrote: {OUTPUT_S10.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#                  TABLE S11 - Training hyperparameters                      #
##############################################################################


def generate_table_s11(norm) -> str:
    """Render LaTeX for Supplementary Table S11 (per-model training provenance)."""
    df = pd.read_csv(INPUT_HYPERPARAMS)
    # Build a wide dataframe with one row per Table B1 number, columns EN/FR side-by-side.
    en = df[df["language"] == "EN"].set_index("number")
    fr = df[df["language"] == "FR"].set_index("number")

    ncols = 11
    section_headers = norm.get_section_headers(ncols)

    lines: List[str] = []
    # ------------------------------------------------------------------
    # Layout rationale (validated visually, 2026-05-21 — S5-style)
    # ------------------------------------------------------------------
    # Same approach as Supplementary Table S5: use right-aligned `r`
    # columns for the numeric fields so each cell takes only the
    # natural width of its longest value, and let longtable produce a
    # naturally-narrow table that is centred horizontally on the page.
    #
    #   #         p{0.5cm}
    #   Category  p{4.0cm}
    #   Code      p{3.0cm}
    #   Ep./Ph./F_1/n^+  r  (auto-width, 8 columns total)
    #
    # No \caption inside the longtable: see S10 generator.
    # ------------------------------------------------------------------
    lines.append(r"\refstepcounter{table}")
    lines.append(r"\label{tab:training_hyperparameters}")
    lines.append(r"\begin{longtable}{p{0.5cm}p{4.0cm}p{3.0cm}rrrrrrrr}")
    header_main = (
        r"\textbf{\#} & \textbf{Category} & \textbf{Code} & "
        r"\multicolumn{4}{c}{\textbf{English (BERT)}} & "
        r"\multicolumn{4}{c}{\textbf{French (CamemBERT)}} \\"
    )
    header_sub = (
        r"& & & \textbf{Epoch} & \textbf{Training phase} & \textbf{$F_1$} & "
        r"\textbf{$n^{+}$} & \textbf{Epoch} & \textbf{Training phase} & "
        r"\textbf{$F_1$} & \textbf{$n^{+}$} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{c}}"
        r"{\textit{Supplementary Table~S11 -- Continued from previous page}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header_main)
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    lines.append(header_sub)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{ncols}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    def _phase_letter(p) -> str:
        if not isinstance(p, str):
            return "--"
        if p.startswith("reinforced"):
            return "r"
        if p == "normal":
            return "n"
        return p[:1]

    for num in range(1, 66):
        if num in section_headers:
            for header in section_headers[num]:
                lines.append(header)
        cat_name, code = norm.ALL_CATEGORIES[num]
        cell_cat = _fmt_category_cell(
            num, cat_name, norm.PRIMARY_CATEGORIES, norm.escape_latex
        )
        en_row = en.loc[num] if num in en.index else None
        fr_row = fr.loc[num] if num in fr.index else None

        def _row_cells(r):
            if r is None:
                return ("--", "--", "--", "--")
            return (
                _fmt_int(r.get("best_epoch")),
                _phase_letter(r.get("best_training_phase")),
                _fmt_float(r.get("best_macro_f1")),
                _fmt_int(r.get("support_1")),
            )

        en_cells = _row_cells(en_row)
        fr_cells = _row_cells(fr_row)
        lines.append(
            f"{num} & {cell_cat} & \\texttt{{{norm.format_code(code)}}} & "
            f"{en_cells[0]} & {en_cells[1]} & {en_cells[2]} & {en_cells[3]} & "
            f"{fr_cells[0]} & {fr_cells[1]} & {fr_cells[2]} & {fr_cells[3]} \\\\"
        )
        if num in norm.PRIMARY_CATEGORIES:
            lines.append(f"\\cmidrule(lr){{1-{ncols}}}")

    # Footer aggregates.
    n_total = len(df)
    n_reinf = int(df["reinforcement_triggered"].sum())
    n_best_r = int(
        df["best_training_phase"].fillna("").astype(str).str.startswith("reinforced").sum()
    )
    # Summary row pinned to a fixed-width p-box (rather than \linewidth)
    # so it does NOT stretch the longtable to full page width. 12 cm
    # matches the natural width of the longtable (0.5 + 4.0 + 3.0 + 8
    # narrow numeric columns ≈ 11-12 cm) so the cell aligns with the
    # body above without forcing the surrounding alignment to widen.
    lines.append(
        f"\\multicolumn{{{ncols}}}{{p{{12cm}}}}"
        f"{{\\cellcolor{{gray!20}}\\textbf{{Summary:}} "
        f"{n_total} trained models. Reinforcement triggered for {n_reinf} models; "
        f"best epoch retained from the reinforced phase for {n_best_r} models. "
        f"Static training configuration (optimiser, learning rate, batch size, "
        f"random seed, hardware) is reported in the Methods section.}} \\\\"
    )
    lines.append(r"\end{longtable}")
    output = "\n".join(lines)
    OUTPUT_S11.write_text(output)
    print(f"  wrote: {OUTPUT_S11.relative_to(PROJECT_ROOT)} ({len(output):,} chars)")
    return output


##############################################################################
#                  TABLE S12 - Data dictionary                               #
##############################################################################

# The data dictionary is a hand-curated description of every column of the
# six tables in the deposited CCF Database, plus the pgvector extension.
# It is stable across CCF releases and does not depend on the live database
# at generation time.
TABLE_S12_BODY = r"""\setlength{\tabcolsep}{4pt}
% No \caption inside the longtable: see generate_table_s11 for the
% rationale (longtable's caption builder reads \hsize at construction
% time, when it equals the cell width rather than the page width). The
% "Supplementary Table S12." title is injected above the landscape
% block by CCF_Methodology_SI.tex; \refstepcounter
% advances the table counter so that \ref{tab:data_dictionary} prints
% the right number.
\refstepcounter{table}
\label{tab:data_dictionary}
\begin{longtable}{p{3.6cm}p{2.4cm}p{16.5cm}}
\toprule
\textbf{Column} & \textbf{Type} & \textbf{Definition} \\
\midrule
\endfirsthead
\multicolumn{3}{c}{\textit{Supplementary Table~S12 -- Continued from previous page}} \\
\toprule
\textbf{Column} & \textbf{Type} & \textbf{Definition} \\
\midrule
\endhead
\midrule
\multicolumn{3}{r}{\textit{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_full\_data} \emph{(article-level metadata, 266{,}271 rows)}} \\
\midrule
\texttt{doc\_id}           & BIGINT (PK)   & Unique article identifier. \\
\texttt{news\_type}        & TEXT          & Editorial category (news, opinion, editorial, etc.). \\
\texttt{title}             & TEXT          & Article headline. \\
\texttt{author}            & TEXT          & Author byline when available. \\
\texttt{media}             & TEXT          & Newspaper name. \\
\texttt{words\_count}      & BIGINT        & Total word count of the article. \\
\texttt{date}              & DATE          & Publication date. \\
\texttt{language}          & TEXT          & ``EN'' or ``FR''. \\
\texttt{page\_number}      & TEXT          & Print page (free text, e.g.~``A1'' or ``B3''). \\
\midrule

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_processed\_data} \emph{(sentence-level annotations, 9{,}198{,}958 rows)}} \\
\midrule
\texttt{doc\_id}             & BIGINT       & Foreign key to CCF\_full\_data. \\
\texttt{sentence\_id}        & BIGINT       & Sentence position within the article (starting at 1; 0 reserved for title in the embeddings table). \\
\multicolumn{3}{l}{\textit{Ten article-level metadata columns are duplicated here for query convenience (news\_type, title, author, media, words\_count, date, language, page\_number).}} \\
\midrule
\textbf{65 annotation columns} & INTEGER (0/1, nullable) & One binary flag per category from Supplementary Table~S3. Sub-categories are NULL when their parent primary detection is negative. \\
\texttt{ner\_entities}       & TEXT (JSON)  & Per-sentence NER output as \texttt{\{"PER": [..], "ORG": [..], "LOC": [..]\}}. \\
\midrule

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_article\_aggregates} \emph{(article-level rollup, 266{,}271 rows)}} \\
\midrule
\texttt{doc\_id}                  & BIGINT (PK)   & Foreign key to CCF\_full\_data. \\
\texttt{n\_sentences}             & INTEGER       & Total number of two-sentence analytical units in the article. \\
\texttt{n\_sentences\_en}         & INTEGER       & Subset in English. \\
\texttt{n\_sentences\_fr}         & INTEGER       & Subset in French. \\
\textbf{65 \texttt{prop\_X} columns} & DOUBLE PRECISION ($[0, 1]$) & Article-level proportion of sentences positive for category $X$ (sum / n\_sentences, with NULL sub-categories coalesced to 0). \\
\texttt{top\_frame}               & TEXT          & Main frame with the highest \texttt{prop\_X}, or NULL for articles with no positive frame. \\
\texttt{top\_frame\_prop}         & DOUBLE PRECISION & Proportion of the top frame. \\
\texttt{entropy\_frames}          & DOUBLE PRECISION & Shannon entropy (natural base) of the eight main-frame proportions; 0 when the article has no positive frame. \\
\texttt{n\_unique\_messenger\_subs} & SMALLINT     & Number of distinct messenger sub-categories present in the article (0--9). \\
\texttt{n\_unique\_event\_subs}    & SMALLINT      & Same for event sub-categories (0--8). \\
\texttt{n\_unique\_solution\_subs} & SMALLINT      & Same for solution sub-categories (0--2). \\
\midrule

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_article\_entities} \emph{(NER rollup, 266{,}271 rows)}} \\
\midrule
\texttt{doc\_id}                & BIGINT (PK)   & Foreign key to CCF\_full\_data. \\
\texttt{entities\_per}          & JSONB         & Deduplicated array of persons mentioned in the article. \\
\texttt{entities\_org}          & JSONB         & Deduplicated array of organisations. \\
\texttt{entities\_loc}          & JSONB         & Deduplicated array of locations. \\
\texttt{n\_unique\_per}         & INTEGER       & Cardinality of \texttt{entities\_per}. \\
\texttt{n\_unique\_org}         & INTEGER       & Cardinality of \texttt{entities\_org}. \\
\texttt{n\_unique\_loc}         & INTEGER       & Cardinality of \texttt{entities\_loc}. \\
\texttt{n\_sentences\_with\_per} & INTEGER      & Number of sentences with at least one PER entity. \\
\texttt{n\_sentences\_with\_org} & INTEGER      & Same for ORG. \\
\texttt{n\_sentences\_with\_loc} & INTEGER      & Same for LOC. \\
\midrule

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_reliability\_tiers} \emph{(per-category quality lookup, 65 rows)}} \\
\midrule
\texttt{number}                       & SMALLINT (PK)     & Table B1 number (1--65). \\
\texttt{category}                     & TEXT              & Readable category name. \\
\texttt{code}                         & TEXT (UNIQUE)     & Column code as used in CCF\_processed\_data. \\
\texttt{tier\_overall}                & CHAR(1)           & Tier on the combined corpus: ``A'', ``B'', or ``C''. \\
\texttt{tier\_en}                     & CHAR(1)           & Tier on the English subset. \\
\texttt{tier\_fr}                     & CHAR(1)           & Tier on the French subset. \\
\texttt{paradox\_kappa}               & BOOLEAN           & True if the category exhibits prevalence-induced reliability deflation, i.e.\ $F_1 \geq 0.70$ and $\kappa < 0.40$. The column name is kept for backwards compatibility with the original schema. \\
\texttt{exclusion\_reason}            & TEXT              & ``insufficient\_training\_data'' for the two categories that could not be trained, NULL otherwise. \\
\texttt{f1\_macro\_all}, \texttt{f1\_macro\_en}, \texttt{f1\_macro\_fr} & DOUBLE PRECISION & Validation macro $F_1$ score on the gold standard. \\
\texttt{support\_all}, \texttt{support\_en}, \texttt{support\_fr}      & INTEGER          & Number of positive (class 1) validation examples. \\
\texttt{cohens\_kappa\_blind\_phase}, \texttt{gwet\_ac1\_blind\_phase}, \texttt{krippendorff\_alpha\_blind\_phase} & DOUBLE PRECISION & Inter-coder agreement on the blind phase (sentences 601--1000). \\
\texttt{n\_blind\_phase}              & INTEGER           & Sentence count of the blind phase. \\
\texttt{prev\_coder1\_full}, \texttt{prev\_coder2\_full} & DOUBLE PRECISION & Prevalence (proportion of positives) per coder on the full 1{,}000-sentence sample. \\
\midrule

\multicolumn{3}{l}{\cellcolor{gray!12}\textbf{Table CCF\_sentence\_embeddings} \emph{(BGE-M3 dense vectors, 9{,}462{,}845 rows)}} \\
\midrule
\texttt{doc\_id}     & BIGINT (PK, with \texttt{sentence\_id}) & Foreign key to CCF\_full\_data. \\
\texttt{sentence\_id} & INTEGER (PK)   & Sentence position within the article. \texttt{sentence\_id} = 0 is reserved for the article title. \\
\texttt{embedding}   & \texttt{halfvec(1024)} & L2-normalised BGE-M3 vector. \texttt{pgvector} cosine distance via \texttt{<=>}; HNSW index on cosine ops. \\

\end{longtable}
"""


def generate_table_s12() -> str:
    """Write Supplementary Table S12 (data dictionary of the enriched DB).

    The body is a hand-curated description of every column of every
    relation in the deposited CCF Database. It does not depend on the
    live database at generation time.
    """
    OUTPUT_S12.write_text(TABLE_S12_BODY)
    print(f"  wrote: {OUTPUT_S12.relative_to(PROJECT_ROOT)} ({len(TABLE_S12_BODY):,} chars)")
    return TABLE_S12_BODY


##############################################################################
#                                  MAIN                                      #
##############################################################################


def main() -> None:
    print("=" * 72)
    print("CCF reproducible tables (main manuscript + Supplementary S4--S12)")
    print("=" * 72)

    # Main-manuscript input checks.
    for p in (INPUT_TRAINING, INPUT_VALIDATION, INPUT_INTERCODER):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            sys.exit(1)
        print(f"  input : {p.relative_to(PROJECT_ROOT)}")

    # Supplementary-Information input checks.
    for path in (INPUT_RELIABILITY, INPUT_TIERS, INPUT_HYPERPARAMS):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        print(f"  input : {path.relative_to(PROJECT_ROOT)}")

    # ---- Main-manuscript tables (Table 3, Table 4, inter-coder) ----
    print("\n[1/3] Rendering main-manuscript tables:")
    generate_table_performance()
    generate_table_validation_overall()
    generate_table_intercoder_blind()

    # ---- CSV normalisation for S4/S5/S7/S8 ----
    # Normalisation is re-run here so the script can be executed
    # standalone after a fresh checkout (the normalised CSVs are tracked
    # in git but kept in sync with the source CSVs by calling
    # normalize_* deterministically).
    norm = _load_normalization_module()
    print("\n[2/3] Re-running CSV normalisation (delegated to script 14):")
    training_df = norm.normalize_training_metrics()
    test_df = norm.normalize_test_metrics()
    training_db_df = norm.normalize_training_database()

    # ---- Supplementary-Information tables S4--S12 ----
    print("\n[3/3] Rendering Supplementary Information tables S4--S12:")
    generate_table_s4(norm, training_df)
    generate_table_s5(norm, training_db_df)
    generate_table_s7(norm, test_df)
    generate_table_s8(norm)
    generate_table_s9(norm)
    generate_table_s10(norm)
    generate_table_s11(norm)
    generate_table_s12()
    print("\nDone.")


if __name__ == "__main__":
    main()
