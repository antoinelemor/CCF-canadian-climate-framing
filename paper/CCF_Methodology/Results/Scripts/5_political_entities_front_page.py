#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

PAPER:
------
CCF_Methodology

TITLE:
------
5_political_entities_front_page.py

MAIN OBJECTIVE:
---------------
Extract and visualize the 50 most mentioned persons in Canadian climate
coverage (2024), demonstrating the CCF database's Named Entity Recognition
capabilities independent of any specific framing variable.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- sqlalchemy
- config_db (local module)

MAIN FEATURES:
-------------
1) Load NER entities from all 2024 sentences
2) Extract and normalize person names from NER_PER field
3) Count occurrences across all articles
4) Visualize top 50 most mentioned persons
5) Generate publication-quality figure

Author:
-------
Antoine Lemor
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

# Output directories
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "Outputs" / "Figures"
STATS_DIR = SCRIPT_DIR.parent / "Outputs" / "Stats"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Number of top entities to display
TOP_N = 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_person_name(name: str) -> str:
    """Normalize a person's name to group variants."""
    # Remove extra spaces and capitalize
    name = ' '.join(name.split()).title()

    # Handle comma-separated names (keep first)
    if ',' in name:
        name = name.split(',')[0].strip()

    # Handle concatenated names (keep first two words)
    name_parts = name.split()
    if len(name_parts) > 3:
        name = ' '.join(name_parts[:2])

    # Known name variants mapping (canonical name -> list of variants)
    replacements = {
        'Justin Trudeau': ['J. Trudeau', 'Prime Minister Trudeau', 'PM Trudeau', 'Trudeau'],
        'Pierre Poilievre': ['P. Poilievre', 'Poilievre'],
        'Stephen Harper': ['S. Harper', 'Prime Minister Harper', 'PM Harper', 'Harper'],
        'Donald Trump': ['D. Trump', 'President Trump', 'Trump'],
        'Joe Biden': ['J. Biden', 'President Biden', 'Biden'],
        'Kamala Harris': ['K. Harris', 'Vice President Harris', 'Harris'],
        'Doug Ford': ['D. Ford', 'Premier Ford', 'Ford'],
        'François Legault': ['F. Legault', 'Legault'],
        'Jason Kenney': ['J. Kenney', 'Kenney'],
        'Rachel Notley': ['R. Notley', 'Notley'],
        'Danielle Smith': ['D. Smith', 'Premier Smith', 'Smith'],
        'Steven Guilbeault': ['S. Guilbeault', 'Guilbeault'],
        'Catherine McKenna': ['C. Mckenna', 'Catherine Mckenna', 'Mckenna', 'McKenna'],
        'Mark Carney': ['M. Carney', 'Carney'],
        'Jagmeet Singh': ['J. Singh', 'Singh'],
        'Elizabeth May': ['E. May', 'May'],
        'John Rustad': ['J. Rustad', 'Rustad'],
        'David Eby': ['D. Eby', 'Eby'],
        'Chrystia Freeland': ['C. Freeland', 'Freeland'],
    }

    # Check exact match first
    name_lower = name.lower()
    for canonical, variants in replacements.items():
        variants_lower = [v.lower() for v in variants]
        if name_lower in variants_lower or name_lower == canonical.lower():
            return canonical

    return name


def extract_persons_from_ner(ner_value) -> List[str]:
    """Extract person names from NER JSON field."""
    if pd.isna(ner_value) or ner_value in ('', '{}'):
        return []

    try:
        if isinstance(ner_value, str):
            data = json.loads(ner_value)
            if isinstance(data, dict) and 'PER' in data:
                persons = data.get('PER', [])
                return [normalize_person_name(p) for p in persons if p and isinstance(p, str)]
    except (json.JSONDecodeError, TypeError):
        pass

    return []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load 2024 NER data from the CCF database."""
    print("=" * 70)
    print("TOP 50 MOST MENTIONED PERSONS IN CLIMATE COVERAGE (2024)")
    print("=" * 70)
    print("\n[1/4] Loading NER data from CCF_Database...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )

    sql = text(f"""
        SELECT doc_id, ner_entities
        FROM "{TABLE_NAME}"
        WHERE EXTRACT(YEAR FROM date) = 2024
          AND ner_entities IS NOT NULL
          AND ner_entities != '{{}}'
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    print(f"    → Loaded {len(df):,} sentences with NER data")
    print(f"    → From {df['doc_id'].nunique():,} articles")

    return df


# =============================================================================
# ENTITY EXTRACTION AND COUNTING
# =============================================================================

def extract_and_count_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all persons and count their mentions."""
    print("\n[2/4] Extracting person entities...")

    # Extract persons from each sentence
    df['persons'] = df['ner_entities'].apply(extract_persons_from_ner)

    # Flatten all persons across all sentences
    all_persons = []
    for persons_list in df['persons']:
        all_persons.extend(persons_list)

    # Count occurrences
    person_counts = Counter(all_persons)

    print(f"    → Found {len(person_counts):,} unique persons")
    print(f"    → Total person mentions: {sum(person_counts.values()):,}")

    # Get top N
    top_persons = person_counts.most_common(TOP_N)

    # Create DataFrame
    stats_df = pd.DataFrame(top_persons, columns=['person', 'mentions'])
    stats_df['rank'] = range(1, len(stats_df) + 1)

    # Calculate percentage of total
    total_mentions = sum(person_counts.values())
    stats_df['percentage'] = (stats_df['mentions'] / total_mentions * 100).round(2)

    return stats_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figure(stats_df: pd.DataFrame) -> None:
    """Create publication-quality horizontal bar chart."""
    print("\n[3/4] Creating publication figure...")

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(12, 14))
    fig.patch.set_facecolor('white')

    # Y positions
    y_pos = np.arange(len(stats_df))

    # Color gradient based on mentions (log scale for better distribution)
    log_mentions = np.log10(stats_df['mentions'])
    norm_mentions = (log_mentions - log_mentions.min()) / (log_mentions.max() - log_mentions.min())
    colors = plt.cm.Blues(0.3 + 0.6 * norm_mentions)

    # Create horizontal bar chart
    bars = ax.barh(
        y_pos,
        stats_df['mentions'],
        color=colors,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )

    # Add mention counts
    for bar, row in zip(bars, stats_df.itertuples()):
        width = bar.get_width()
        ax.text(
            width + stats_df['mentions'].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{row.mentions:,}',
            ha='left', va='center',
            fontsize=8, fontweight='bold', color='#333333'
        )

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats_df['person'], fontsize=9)
    ax.invert_yaxis()  # Highest at top

    ax.set_xlabel('Number of Mentions', fontweight='bold')
    ax.set_ylabel('Person', fontweight='bold')

    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # X-axis limits
    ax.set_xlim(0, stats_df['mentions'].max() * 1.12)

    # =========================================================================
    # FINALIZE
    # =========================================================================
    plt.tight_layout()

    # Save
    output_path = FIGURES_DIR / "political_debate_entities_front_page_2024.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    → Figure saved: {output_path}")


# =============================================================================
# STATISTICAL OUTPUT
# =============================================================================

def save_statistics(stats_df: pd.DataFrame) -> None:
    """Save statistical summaries."""
    print("\n[4/4] Saving statistics...")

    # Save top 50 to CSV
    output_path = STATS_DIR / "top_50_persons_2024.csv"
    stats_df.to_csv(output_path, index=False)
    print(f"    → Statistics saved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("TOP 10 MOST MENTIONED PERSONS")
    print("=" * 70)
    print(f"{'Rank':<6} {'Person':<30} {'Mentions':>12} {'%':>8}")
    print("-" * 70)

    for row in stats_df.head(10).itertuples():
        print(f"{row.rank:<6} {row.person:<30} {row.mentions:>12,} {row.percentage:>7.2f}%")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""

    # Load data
    df = load_data()

    # Extract and count persons
    stats_df = extract_and_count_persons(df)

    # Create visualization
    create_publication_figure(stats_df)

    # Save statistics
    save_statistics(stats_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
