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
6b_political_debate_entities_2024.py

MAIN OBJECTIVE:
---------------
Analyze political entities (persons) from sentences where pol_debate = 1
(political debate framing) for year 2024 only. Calculate front page placement
probability for most frequent entities and create visualization.

This generates image.png for the paper.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- sqlalchemy
- json

MAIN FEATURES:
-------------
1) Load political debate sentences from CCF_Database (2024 only)
2) Extract named entities (PER) from these sentences
3) Calculate front page probability for most frequent entities
4) Compute Wilson confidence intervals
5) Create horizontal bar chart visualization

Author:
-------
Antoine Lemor
"""

import os
import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from typing import Dict, Any

# ============================================================================
# 1. Connection parameters - CCF_Database (publication database)
# ============================================================================
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
}
TABLE_NAME = "CCF_processed_data"

# Output directory
OUT_DIR = Path(__file__).resolve().parents[1] / "Outputs" / "Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regex to detect front page
FP_REGEX = re.compile(r"""(?ix)
    (front)
  | ^[A-Z]?1$
  | ^1$
""")

# ============================================================================
# 2. Functions for name normalization and intelligent merging
# ============================================================================
def basic_normalize_name(name: str) -> str:
    """Basic normalization: clean whitespace, title case, remove trailing parts"""
    name = ' '.join(name.split())
    name = name.title()

    # Remove parts after comma (titles, etc.)
    if ',' in name:
        name = name.split(',')[0].strip()

    # Remove French titles M./Mme/Mlle that are common in Canadian French media
    # These are titles (Monsieur/Madame), not initials
    french_titles = ['M.', 'Mme', 'Mme.', 'Mlle', 'Mlle.', 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']
    name_parts = name.split()
    if name_parts and name_parts[0] in french_titles:
        name_parts = name_parts[1:]
        name = ' '.join(name_parts)

    # Remove double periods (formatting errors like "Schreiner. Schreiner")
    name = name.replace('. ', ' ').replace('  ', ' ').strip()

    # Remove duplicate consecutive words (e.g., "Schreiner Schreiner" -> "Schreiner")
    name_parts = name.split()
    cleaned_parts = []
    for part in name_parts:
        if not cleaned_parts or part.lower() != cleaned_parts[-1].lower():
            cleaned_parts.append(part)
    name_parts = cleaned_parts

    # Limit to 3 words max (first name + middle + last)
    if len(name_parts) > 3:
        name_parts = name_parts[:3]

    name = ' '.join(name_parts)
    return name


def get_name_words(name: str) -> list:
    """Extract all words from a name, normalized to lowercase, preserving order"""
    words = name.split()
    # Normalize: lowercase, remove periods
    return [w.lower().rstrip('.') for w in words if len(w.rstrip('.')) > 0]


def get_surname(name: str) -> str:
    """Get the last word of a name (likely the surname)"""
    words = get_name_words(name)
    return words[-1] if words else ""


def should_merge(short_name: str, long_name: str) -> bool:
    """
    Determine if short_name should be merged into long_name.

    Rules:
    1. Don't merge into names with generational suffixes (Jr, Sr, II, III, etc.)
       because these are different people (e.g., Donald Trump ≠ Donald Trump Jr)
    2. Don't merge into names with more than 3 words (likely NER errors)
    3. Single-word names: only merge if it matches the SURNAME (last word) of long_name
       This prevents "Pierre" from matching "Pierre Poilievre" (Pierre is not the surname)
       But allows "Carney" to match "Mark Carney"
    4. Multi-word names: merge if all words appear in long_name

    This is conservative to avoid merging different people who share a first name.
    """
    short_words = get_name_words(short_name)
    long_words = get_name_words(long_name)

    if not short_words or not long_words:
        return False

    # Don't merge into names with generational suffixes (different people!)
    generational_suffixes = {'jr', 'jr.', 'sr', 'sr.', 'junior', 'senior', 'ii', 'iii', 'iv', 'v'}
    if long_words[-1] in generational_suffixes:
        return False

    # Don't merge into names with more than 2 words (standard: first name + last name)
    # This avoids merging into NER errors like "Justin Trudeau Sarah"
    if len(long_words) > 2:
        return False

    # Don't merge very short words (likely initials or abbreviations)
    if len(short_words) == 1 and len(short_words[0]) <= 2:
        return False

    # Single-word name: must match the surname (last word) of the longer name
    if len(short_words) == 1:
        surname = long_words[-1]  # Last word is likely the surname
        return short_words[0] == surname

    # Multi-word name: all words must appear in the longer name
    long_words_set = set(long_words)
    for sw in short_words:
        # Check exact match
        if sw in long_words_set:
            continue
        # Check if it's a single-letter initial matching start of a word
        if len(sw) == 1:
            if any(lw.startswith(sw) for lw in long_words):
                continue
        return False

    return True


def intelligent_merge_names(name_counts: dict) -> dict:
    """
    Intelligently merge name variants by finding names where one is a subset of another.

    Conservative rules:
    - Single-word names only merge if they match the SURNAME (last word)
      E.g., "Carney" -> "Mark Carney" (Carney is the surname)
      But NOT: "Pierre" -> "Pierre Poilievre" (Pierre is the first name)
    - Multi-word names merge if all words appear in the longer name
      E.g., "M. Carney" -> "Mark Carney"

    Args:
        name_counts: Dictionary of {name: count}

    Returns:
        Dictionary mapping original names to canonical (longest) names
    """
    names = list(name_counts.keys())

    # For each name, find if there's a longer name that contains it
    name_mapping = {}

    for name in names:
        # Find all names that could be the "full version" of this name
        candidates = []
        for other_name in names:
            if other_name != name and len(other_name) > len(name):
                if should_merge(name, other_name):
                    candidates.append(other_name)

        if candidates:
            # Choose the shortest candidate (most specific match)
            # E.g., if "Carney" matches both "Mark Carney" and "Mark Carney Jr.", prefer "Mark Carney"
            best_match = min(candidates, key=len)
            name_mapping[name] = best_match
        else:
            name_mapping[name] = name

    # Resolve chains: if A -> B and B -> C, then A -> C
    def resolve(n):
        if name_mapping[n] == n:
            return n
        return resolve(name_mapping[n])

    for name in names:
        name_mapping[name] = resolve(name)

    return name_mapping


def apply_name_mapping(persons_list: list, mapping: dict) -> list:
    """Apply name mapping to a list of persons"""
    return [mapping.get(p, p) for p in persons_list]

# ============================================================================
# 3. Main function
# ============================================================================
def analyze_political_debate_entities():
    """
    Analyze front page probability for persons in political debate sentences (2024)
    """

    print("=" * 70)
    print("POLITICAL DEBATE ENTITIES - FRONT PAGE ANALYSIS (2024)")
    print("=" * 70)

    # Database connection
    print("\n[1/5] Connecting to CCF_Database...")
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )

    # SQL query - pol_debate = 1 for year 2024
    # Date is proper DATE type in CCF_Database
    sql = text(f"""
    SELECT doc_id, page_number, pol_debate, ner_entities, date
    FROM "{TABLE_NAME}"
    WHERE pol_debate = 1
    AND EXTRACT(YEAR FROM date) = 2024
    """)

    print("[2/5] Loading sentences with political debate framing (2024)...")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    if df.empty:
        print("  No data found with pol_debate = 1 for 2024")
        return

    print(f"  Sentences with political debate framing in 2024: {len(df):,}")

    # ============================================================================
    # 4. Entity extraction and aggregation by article
    # ============================================================================
    print("[3/5] Extracting person entities...")

    # Front page detection
    df["front_page"] = df["page_number"].apply(
        lambda x: bool(FP_REGEX.search(str(x).strip())) if x else False
    ).astype(int)

    def extract_persons(ner_value):
        """Extract person names from ner_entities field"""
        if pd.isna(ner_value) or ner_value == '' or ner_value == '{}':
            return []

        try:
            if isinstance(ner_value, str):
                data = json.loads(ner_value)
                if isinstance(data, dict) and 'PER' in data:
                    persons = data.get('PER', [])
                    # Apply basic normalization only (intelligent merge comes later)
                    return [basic_normalize_name(p) for p in persons if p and isinstance(p, str)]
        except:
            pass

        return []

    df['persons'] = df['ner_entities'].apply(extract_persons)

    # Aggregate by article
    article_data = []
    for doc_id, group in df.groupby('doc_id'):
        all_persons = []
        for persons_list in group['persons']:
            all_persons.extend(persons_list)

        front_page = group['front_page'].max()

        article_data.append({
            'doc_id': doc_id,
            'front_page': front_page,
            'persons': all_persons
        })

    articles_df = pd.DataFrame(article_data)

    print(f"  Articles with political debate framing: {len(articles_df):,}")
    print(f"  Articles on front page: {articles_df['front_page'].sum():,} ({articles_df['front_page'].mean():.1%})")

    # ============================================================================
    # 5. Entity counting with intelligent name merging
    # ============================================================================
    print("[4/5] Calculating front page probabilities with intelligent name merging...")

    # First pass: count all persons with basic normalization
    all_persons_flat = []
    for persons_list in articles_df['persons']:
        all_persons_flat.extend(persons_list)

    person_counts_raw = Counter(all_persons_flat)

    # Keep entities with at least some mentions for merging analysis
    MIN_MENTIONS_FOR_MERGE = 5  # Lower threshold for merge detection
    persons_for_merge = {person: count for person, count in person_counts_raw.items()
                        if count >= MIN_MENTIONS_FOR_MERGE}

    # Apply intelligent merging (subset-based: "Carney" -> "Mark Carney")
    print("  Applying intelligent name merging (subset-based)...")
    name_mapping = intelligent_merge_names(persons_for_merge)

    # Show merged groups (for debugging/transparency)
    merged_groups = {}
    for original, canonical in name_mapping.items():
        if original != canonical:
            if canonical not in merged_groups:
                merged_groups[canonical] = []
            merged_groups[canonical].append(original)

    if merged_groups:
        print(f"  Merged {sum(len(v) for v in merged_groups.values())} name variants into {len(merged_groups)} canonical names:")
        for canonical, variants in sorted(merged_groups.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"    {canonical} <- {', '.join(variants)}")

    # Apply mapping to articles
    articles_df['persons_merged'] = articles_df['persons'].apply(
        lambda x: apply_name_mapping(x, name_mapping)
    )

    # Recount with merged names
    all_persons_merged = []
    for persons_list in articles_df['persons_merged']:
        all_persons_merged.extend(persons_list)

    person_counts = Counter(all_persons_merged)

    # Keep only entities mentioned at least 50 times (after merging)
    MIN_MENTIONS = 50
    frequent_persons = {person: count for person, count in person_counts.items()
                       if count >= MIN_MENTIONS}

    print(f"  Entities with >= {MIN_MENTIONS} mentions (after merging): {len(frequent_persons)}")

    # Calculate front page probability for each entity (using merged names)
    entity_stats = []

    for person in frequent_persons:
        articles_with_person = articles_df[
            articles_df['persons_merged'].apply(lambda x: person in x)
        ]

        n_articles = len(articles_with_person)
        n_front_page = articles_with_person['front_page'].sum()

        if n_articles > 0:
            probability = n_front_page / n_articles

            # Wilson score confidence interval (more accurate for proportions)
            z = 1.96
            denominator = 1 + z**2 / n_articles
            center = (probability + z**2 / (2 * n_articles)) / denominator
            margin = z * np.sqrt((probability * (1 - probability) + z**2 / (4 * n_articles)) / n_articles) / denominator
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)

            entity_stats.append({
                'entity': person,
                'n_mentions': frequent_persons[person],
                'n_articles': n_articles,
                'n_front_page': n_front_page,
                'probability': probability,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

    stats_df = pd.DataFrame(entity_stats).sort_values('probability', ascending=False)
    top_entities = stats_df.head(50)

    # ============================================================================
    # 6. Visualization
    # ============================================================================
    print("[5/5] Creating chart...")

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    fig, ax = plt.subplots(figsize=(14, 18))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    y_pos = np.arange(len(top_entities))

    # Calculate error bars
    errors = np.array([
        top_entities['probability'] - top_entities['ci_lower'],
        top_entities['ci_upper'] - top_entities['probability']
    ])

    # Colors based on probability
    colors = plt.cm.RdYlGn_r(top_entities['probability'] / top_entities['probability'].max())

    bars = ax.barh(
        y_pos,
        top_entities['probability'],
        xerr=errors,
        capsize=4,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
        error_kw={'linewidth': 1.2, 'ecolor': 'black', 'alpha': 0.5}
    )

    # Baseline
    baseline = articles_df['front_page'].mean()
    ax.axvline(x=baseline, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Baseline: {baseline:.3f}')

    # Labels
    ax.set_ylabel('Person Entity', fontweight='bold')
    ax.set_xlabel('P(Front Page | Political Debate Mention)', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_entities['entity'])
    ax.invert_yaxis()

    # Add values
    for bar, row in zip(bars, top_entities.itertuples()):
        ax.text(row.ci_upper + 0.005, bar.get_y() + bar.get_height()/2.,
                f'{row.probability:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')

        ax.text(0.98, bar.get_y() + bar.get_height()/2.,
                f'n={row.n_articles}',
                transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8,
                color='gray', style='italic')

    ax.grid(True, axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right')

    ax.set_xlim(0, min(1, top_entities['ci_upper'].max() * 1.15))

    # Bottom note
    fig.text(
        0.5, 0.01,
        f"Note: Only entities mentioned in at least {MIN_MENTIONS} political debate sentences are shown. "
        "n = number of articles mentioning the entity.\n"
        "Error bars represent 95% Wilson confidence intervals. Analysis for year 2024 only.",
        ha='center',
        fontsize=9,
        style='italic',
        wrap=True
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, right=0.92)

    # Save figure
    output_path = OUT_DIR / "political_debate_entities_front_page_2024.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n  Chart saved: {output_path}")

    # ============================================================================
    # 7. Display results
    # ============================================================================
    print("\n" + "=" * 90)
    print("TOP 50 ENTITIES (FRONT PAGE PROBABILITY)")
    print("=" * 90)
    print(f"{'Entity':<30} {'Articles':>10} {'Front Page':>10} {'Probability':>12} {'95% CI':>20}")
    print("-" * 90)

    for _, row in top_entities.iterrows():
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        print(f"{row['entity']:<30} {row['n_articles']:>10} {row['n_front_page']:>10} "
              f"{row['probability']:>12.3f} {ci_str:>20}")

    print(f"\nBaseline (all political debate articles): {baseline:.3f}")
    print("Analysis completed successfully!")

# ============================================================================
# Execution
# ============================================================================
if __name__ == "__main__":
    analyze_political_debate_entities()
