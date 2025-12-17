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
7_science_acceptance_maps.py

MAIN OBJECTIVE:
---------------
Create choropleth maps comparing scientific skepticism indicators by province:
1) Scientific skepticism proportion (among scientific sentences)
2) Trudeau effect on scientific skepticism (regression coefficient)
3) Poilievre effect on scientific skepticism (regression coefficient)

Dependencies:
-------------
- pandas
- numpy
- geopandas
- matplotlib
- sqlalchemy
- statsmodels
- config_db (local module)

MAIN FEATURES:
-------------
1) Load sci_skepticism data from PostgreSQL
2) Calculate average proportions by province (Map A)
3) Calculate leader mention effects on skepticism by province (Maps B & C)
4) Run OLS regression for each leader-province combination (Sep 2022 - Dec 2024)
5) Create side-by-side choropleth maps with diverging color schemes

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
from typing import Dict, Any, List
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sqlalchemy import create_engine, text
import statsmodels.api as sm
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

# Reference files
SHAPEFILE = "/Users/antoine/Documents/GitHub/CCF-canadian-climate-framing/Database/Database/CAN_shp/lpr_000b21a_e.shp"
MEDIA_CSV = "/Users/antoine/Documents/GitHub/CCF-canadian-climate-framing/Database/Database/Canadian_Media_Articles_by_Province.csv"

# Output directory
OUT_DIR = Path(__file__).resolve().parents[1] / "Outputs" / "Figures"
STATS_DIR = Path(__file__).resolve().parents[1] / "Outputs" / "Stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Leader identification patterns (case-insensitive)
TRUDEAU_PATTERNS: List[str] = [
    r'\bjustin\s+trudeau\b',
    r'\bj\.\s*trudeau\b',
    r'\btrudeau\b',
    r'\bpm\s+trudeau\b',
    r'\bprime\s+minister\s+trudeau\b',
]

POILIEVRE_PATTERNS: List[str] = [
    r'\bpierre\s+poilievre\b',
    r'\bp\.\s*poilievre\b',
    r'\bpoilievre\b',
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_patterns(text: str, patterns: List[str]) -> bool:
    """Check if any pattern matches in the given text."""
    if not text or pd.isna(text):
        return False
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def extract_persons_from_ner(ner_value) -> List[str]:
    """Extract person entities from NER JSON field."""
    if pd.isna(ner_value) or ner_value in ('', '{}'):
        return []
    try:
        if isinstance(ner_value, str):
            data = json.loads(ner_value)
            if isinstance(data, dict):
                return data.get('PER', [])
    except (json.JSONDecodeError, TypeError):
        pass
    return []


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scientific_skepticism_data() -> pd.DataFrame:
    """Load data for scientific skepticism map (Map A)."""
    print("\n[1/6] Loading scientific skepticism data...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )

    # Load scientific skepticism data (among scientific sentences)
    sql = text(f"""
        SELECT doc_id, media, scientific_frame, sci_skepticism
        FROM "{TABLE_NAME}"
        WHERE scientific_frame = 1
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    print(f"    -> Loaded {len(df):,} scientific sentences")
    return df


def load_leader_skepticism_data() -> pd.DataFrame:
    """Load data for leader effect on skepticism maps - Sep 2022 to Dec 2024."""
    print("\n[2/6] Loading leader skepticism data (Sep 2022 - Dec 2024)...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )

    # Data from September 10, 2022 (Poilievre became Conservative leader) to end of 2024
    sql = text(f"""
        SELECT doc_id, sentence_id, media, sci_skepticism, ner_entities
        FROM "{TABLE_NAME}"
        WHERE date >= '2022-09-10' AND date <= '2024-12-31'
    """)

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    engine.dispose()

    print(f"    -> Loaded {len(df):,} sentences from 2023-2024")
    return df


def identify_leader_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Identify Trudeau and Poilievre mentions in each sentence."""
    print("    -> Identifying leader mentions...")

    def check_leader_patterns(ner_val, patterns):
        if pd.isna(ner_val) or ner_val in ('', '{}'):
            return False
        try:
            if isinstance(ner_val, str):
                data = json.loads(ner_val)
                if isinstance(data, dict):
                    persons = data.get('PER', [])
                    text = ' '.join(persons).lower() if persons else ''
                    return check_patterns(text, patterns)
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    df['mentions_trudeau'] = df['ner_entities'].apply(
        lambda x: check_leader_patterns(x, TRUDEAU_PATTERNS)
    ).astype(int)
    df['mentions_poilievre'] = df['ner_entities'].apply(
        lambda x: check_leader_patterns(x, POILIEVRE_PATTERNS)
    ).astype(int)

    n_trudeau = df['mentions_trudeau'].sum()
    n_poilievre = df['mentions_poilievre'].sum()
    print(f"    -> Trudeau mentions: {n_trudeau:,} sentences")
    print(f"    -> Poilievre mentions: {n_poilievre:,} sentences")

    return df


# =============================================================================
# PROVINCE MAPPING
# =============================================================================

def map_media_to_provinces(df: pd.DataFrame) -> pd.DataFrame:
    """Map media outlets to provinces."""
    media_df = pd.read_csv(MEDIA_CSV, dtype=str)
    media_df['media'] = media_df['media'].str.strip().str.lower()
    media_df['province'] = media_df['region'].str.strip()

    # Exclude national media
    media_df = media_df[media_df['province'].str.lower() != 'national']
    media_df = media_df[['media', 'province']]

    # Normalize media names in data
    df['media'] = df['media'].astype(str).str.strip().str.lower()

    # Merge
    df = df.merge(media_df, on='media', how='left')
    df = df.dropna(subset=['province'])

    return df


# =============================================================================
# CALCULATIONS
# =============================================================================

def calculate_scientific_skepticism_by_province(df: pd.DataFrame) -> pd.Series:
    """Calculate proportion of scientific skepticism by province."""
    print("\n[3/6] Calculating scientific skepticism by province...")

    df = map_media_to_provinces(df)

    # Average per article first
    article_means = df.groupby(['doc_id', 'province'])['sci_skepticism'].mean().reset_index()

    # Then average by province
    province_means = article_means.groupby('province')['sci_skepticism'].mean()

    print(f"    -> {len(province_means)} provinces with data")
    return province_means


def calculate_leader_effects_by_province(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate Trudeau and Poilievre effects on skepticism by province using regression."""
    print("\n[4/7] Calculating leader effects by province...")

    df = identify_leader_mentions(df)
    df = map_media_to_provinces(df)

    # Aggregate to article level
    articles = df.groupby(['doc_id', 'province']).agg({
        'mentions_trudeau': ['max', 'mean'],
        'mentions_poilievre': ['max', 'mean'],
        'sci_skepticism': 'mean',
        'sentence_id': 'count'
    })
    articles.columns = [
        'trudeau_present', 'trudeau_intensity',
        'poilievre_present', 'poilievre_intensity',
        'sci_skepticism', 'n_sentences'
    ]
    articles = articles.reset_index()

    results = {}

    for leader in ['trudeau', 'poilievre']:
        print(f"\n    {leader.upper()} effect:")
        leader_results = []

        for province in articles['province'].unique():
            prov_data = articles[articles['province'] == province].copy()
            prov_data = prov_data.dropna(subset=['sci_skepticism', f'{leader}_intensity'])

            n_with_leader = (prov_data[f'{leader}_present'] == 1).sum()
            n_total = len(prov_data)

            if n_with_leader >= 10 and n_total >= 50:
                try:
                    y = prov_data['sci_skepticism'].values
                    X = prov_data[f'{leader}_intensity'].values.reshape(-1, 1)
                    X = sm.add_constant(X)

                    model = sm.OLS(y, X)
                    fit = model.fit(cov_type='HC1')

                    leader_results.append({
                        'province': province,
                        'coefficient': fit.params[1],
                        'std_error': fit.bse[1],
                        'p_value': fit.pvalues[1],
                        'n_articles': n_total,
                        'n_with_leader': n_with_leader,
                        'significant': fit.pvalues[1] < 0.05
                    })

                    sig = '***' if fit.pvalues[1] < 0.001 else '**' if fit.pvalues[1] < 0.01 else '*' if fit.pvalues[1] < 0.05 else ''
                    print(f"      {province:25s}: B = {fit.params[1]:+.3f} (n={n_total}, mentions={n_with_leader}, p={fit.pvalues[1]:.3f}) {sig}")

                except Exception as e:
                    print(f"      {province:25s}: Regression failed - {str(e)[:30]}")
            else:
                print(f"      {province:25s}: Insufficient data (n={n_total}, mentions={n_with_leader})")

        results[leader] = pd.DataFrame(leader_results)

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_triple_maps(
    skepticism_by_province: pd.Series,
    leader_effects: Dict[str, pd.DataFrame]
) -> None:
    """Create three choropleth maps: skepticism proportion, Trudeau effect, Poilievre effect."""
    print("\n[5/7] Creating maps...")

    # Load shapefile
    gdf_canada = gpd.read_file(SHAPEFILE)
    gdf_canada['province_norm'] = gdf_canada['PRENAME'].str.lower().str.strip()

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('white')

    # =========================================================================
    # MAP A: SCIENTIFIC SKEPTICISM BY PROVINCE
    # =========================================================================
    ax1 = axes[0]

    # Prepare data
    skepticism_data = skepticism_by_province.reset_index()
    skepticism_data.columns = ['province', 'sci_skepticism']
    skepticism_data['province_norm'] = skepticism_data['province'].str.lower().str.strip()

    gdf_skepticism = gdf_canada.merge(skepticism_data, on='province_norm', how='left')

    # Colormap - red for skepticism
    cmap_skepticism = LinearSegmentedColormap.from_list(
        'skepticism',
        ['#FFF5F0', '#FCBBA1', '#FB6A4A', '#CB181D', '#67000D']
    )

    vmin_s = gdf_skepticism['sci_skepticism'].min()
    vmax_s = gdf_skepticism['sci_skepticism'].max()

    has_data_s = gdf_skepticism[gdf_skepticism['sci_skepticism'].notnull()]
    no_data_s = gdf_skepticism[gdf_skepticism['sci_skepticism'].isnull()]

    if not has_data_s.empty:
        has_data_s.plot(
            column='sci_skepticism',
            ax=ax1,
            cmap=cmap_skepticism,
            vmin=vmin_s,
            vmax=vmax_s,
            edgecolor='black',
            linewidth=0.8,
            legend=True,
            legend_kwds={
                'label': 'Proportion',
                'orientation': 'horizontal',
                'shrink': 0.7,
                'pad': 0.02,
                'format': '%.2f'
            }
        )

    if not no_data_s.empty:
        no_data_s.plot(ax=ax1, facecolor='lightgrey', edgecolor='black', hatch='///', alpha=0.6)

    # Add values
    for _, row in has_data_s.iterrows():
        if pd.notnull(row['sci_skepticism']):
            rep_pt = row['geometry'].representative_point()
            x_offset, y_offset = 0, 0
            if row['PRENAME'] == 'Prince Edward Island':
                x_offset, y_offset = 2, -0.5
            elif row['PRENAME'] == 'Nova Scotia':
                x_offset = 2
            elif row['PRENAME'] == 'New Brunswick':
                y_offset = -0.8

            ax1.annotate(
                f"{row['sci_skepticism']*100:.1f}%",
                xy=(rep_pt.x + x_offset, rep_pt.y + y_offset),
                ha='center', va='center',
                fontsize=9, color='black', weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9)
            )

    ax1.axis('off')
    ax1.set_title('A. Scientific Skepticism by Province\n(all years)', fontweight='bold', fontsize=12, pad=10)

    # =========================================================================
    # HELPER FUNCTION FOR LEADER EFFECT MAPS
    # =========================================================================
    def plot_leader_effect_map(ax, leader_data, title, leader_color):
        """Plot a leader effect map on given axis."""
        # Prepare data
        effect_data = leader_data.copy()
        effect_data['province_norm'] = effect_data['province'].str.lower().str.strip()

        gdf_effect = gdf_canada.merge(effect_data, on='province_norm', how='left')

        # Diverging colormap centered on leader color
        if leader_color == 'red':
            cmap_effect = LinearSegmentedColormap.from_list(
                'effect',
                ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#EF8A62', '#B2182B']
            )
        else:  # blue
            cmap_effect = LinearSegmentedColormap.from_list(
                'effect',
                ['#B2182B', '#EF8A62', '#FDDBC7', '#F7F7F7', '#D1E5F0', '#67A9CF', '#2166AC']
            )

        # Center colormap at zero
        valid_coefs = gdf_effect['coefficient'].dropna()
        if len(valid_coefs) > 0:
            vmax_e = max(abs(valid_coefs.min()), abs(valid_coefs.max()))
            vmin_e = -vmax_e
            norm = TwoSlopeNorm(vmin=vmin_e, vcenter=0, vmax=vmax_e)
        else:
            norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        has_data = gdf_effect[gdf_effect['coefficient'].notnull()]
        no_data = gdf_effect[gdf_effect['coefficient'].isnull()]

        if not has_data.empty:
            has_data.plot(
                column='coefficient',
                ax=ax,
                cmap=cmap_effect,
                norm=norm,
                edgecolor='black',
                linewidth=0.8,
                legend=True,
                legend_kwds={
                    'label': 'B coefficient',
                    'orientation': 'horizontal',
                    'shrink': 0.7,
                    'pad': 0.02,
                    'format': '%.2f'
                }
            )

        if not no_data.empty:
            no_data.plot(ax=ax, facecolor='lightgrey', edgecolor='black', hatch='///', alpha=0.6)

        # Add values with significance markers
        for _, row in has_data.iterrows():
            if pd.notnull(row['coefficient']):
                rep_pt = row['geometry'].representative_point()
                x_offset, y_offset = 0, 0
                if row['PRENAME'] == 'Prince Edward Island':
                    x_offset, y_offset = 2, -0.5
                elif row['PRENAME'] == 'Nova Scotia':
                    x_offset = 2
                elif row['PRENAME'] == 'New Brunswick':
                    y_offset = -0.8

                # Significance marker
                sig = ''
                if pd.notnull(row.get('p_value')):
                    if row['p_value'] < 0.001:
                        sig = '***'
                    elif row['p_value'] < 0.01:
                        sig = '**'
                    elif row['p_value'] < 0.05:
                        sig = '*'

                ax.annotate(
                    f"{row['coefficient']:+.2f}{sig}",
                    xy=(rep_pt.x + x_offset, rep_pt.y + y_offset),
                    ha='center', va='center',
                    fontsize=9, color='black', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.9)
                )

        ax.axis('off')
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

    # =========================================================================
    # MAP B: TRUDEAU EFFECT ON SKEPTICISM
    # =========================================================================
    plot_leader_effect_map(
        axes[1],
        leader_effects['trudeau'],
        'B. Trudeau Effect on Skepticism\n(Sep 2022 - Dec 2024)',
        'red'
    )

    # =========================================================================
    # MAP C: POILIEVRE EFFECT ON SKEPTICISM
    # =========================================================================
    plot_leader_effect_map(
        axes[2],
        leader_effects['poilievre'],
        'C. Poilievre Effect on Skepticism\n(Sep 2022 - Dec 2024)',
        'blue'
    )

    # =========================================================================
    # FINALIZE
    # =========================================================================
    plt.tight_layout()

    # Save
    output_path = OUT_DIR / "science_acceptance_maps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    -> Map saved: {output_path}")


def save_statistics(
    skepticism_by_province: pd.Series,
    leader_effects: Dict[str, pd.DataFrame]
) -> None:
    """Save statistical summaries."""
    print("\n[6/7] Saving statistics...")

    # Scientific skepticism statistics
    skepticism_df = skepticism_by_province.reset_index()
    skepticism_df.columns = ['province', 'sci_skepticism_proportion']
    skepticism_df = skepticism_df.sort_values('sci_skepticism_proportion', ascending=False)
    skepticism_path = STATS_DIR / "scientific_skepticism_by_province.csv"
    skepticism_df.to_csv(skepticism_path, index=False)
    print(f"    -> Scientific skepticism stats: {skepticism_path}")

    # Leader effect statistics
    for leader, effect_df in leader_effects.items():
        if not effect_df.empty:
            effect_path = STATS_DIR / f"{leader}_effect_by_province.csv"
            effect_df.sort_values('coefficient', ascending=False).to_csv(effect_path, index=False)
            print(f"    -> {leader.capitalize()} effect stats: {effect_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""
    print("=" * 70)
    print("SCIENTIFIC SKEPTICISM MAPS BY PROVINCE")
    print("=" * 70)

    # Load and process data for Map A (scientific skepticism)
    df_skepticism = load_scientific_skepticism_data()
    skepticism_by_province = calculate_scientific_skepticism_by_province(df_skepticism)

    # Load and process data for Maps B & C (leader effects on skepticism)
    df_leaders = load_leader_skepticism_data()
    leader_effects = calculate_leader_effects_by_province(df_leaders)

    # Create visualization
    create_triple_maps(skepticism_by_province, leader_effects)

    # Save statistics
    save_statistics(skepticism_by_province, leader_effects)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
