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
4_temporal_frames_evolution.py

MAIN OBJECTIVE:
---------------
Generate temporal evolution plot showing how climate frames change over time.
Apply LOESS smoothing for better visualization and include confidence intervals.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- seaborn
- sqlalchemy
- statsmodels (for LOESS smoothing)
- config_db (local module)

MAIN FEATURES:
-------------
1) Load frame detection data from PostgreSQL
2) Calculate monthly aggregates for each frame
3) Apply LOESS smoothing with bootstrap confidence intervals
4) Create publication-quality temporal visualization
5) Add legend and watermark

Author:
-------
Antoine Lemor
"""

import os
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.nonparametric.smoothers_lowess import lowess
from pathlib import Path

# ============================================================================
# 1. Paramètres de connexion - CCF_Database (publication database)
# ============================================================================
DB_PARAMS: Dict[str, Any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",  # Publication database with renamed columns
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
TABLE_NAME = "CCF_processed_data"

# Dossier de sortie
OUT_DIR = Path(__file__).resolve().parents[1] / "Outputs" / "Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. Mapping des frames (new column names from CCF_Database)
# ============================================================================
FRAME_MAPPING = {
    'economic_frame': 'Economic',
    'health_frame': 'Health',
    'security_frame': 'Security',
    'justice_frame': 'Justice',
    'political_frame': 'Political',
    'scientific_frame': 'Scientific',
    'environmental_frame': 'Environmental',
    'cultural_frame': 'Cultural'
}

# Couleurs personnalisées pour les frames (palette plus harmonieuse et thématique)
FRAME_COLORS = {
    'Economic': '#2E7D32',      # Dark Green (money/economy)
    'Health': '#D32F2F',        # Red (health/medical)
    'Security': '#8D6E63',      # Light Brown (security/defense)
    'Justice': '#9C27B0',       # Purple (justice/law)
    'Political': '#000000',     # Black (politics)
    'Scientific': '#0288D1',    # Light Blue (science/research)
    'Environmental': '#4CAF50', # Green (environment/nature)
    'Cultural': '#FF6F00'       # Orange (culture/arts)
}

# ============================================================================
# 3. Main function
# ============================================================================
def create_temporal_frames_plot():
    """
    Generate temporal frame chart with LOESS smoothing
    """
    
    print(" Generating temporal frames chart...")
    
    # Database connection
    print(" Connecting to PostgreSQL...")
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )
    
    # Columns to retrieve
    frame_cols = list(FRAME_MAPPING.keys())
    cols_str = ', '.join([f'"{col}"' for col in frame_cols])
    
    # SQL query - CCF_Database has proper DATE type
    sql = f"""
    SELECT doc_id, date, {cols_str}
    FROM "{TABLE_NAME}"
    WHERE date IS NOT NULL
    """

    print(" Loading data...")
    df = pd.read_sql(sql, engine)
    engine.dispose()

    if df.empty:
        print(" No data found")
        return

    # Date conversion (already converted in SQL, just ensure pandas datetime)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Calculate average proportions per article
    print(" Calculating proportions per article...")
    article_means = df.groupby('doc_id')[frame_cols].mean()
    article_means.reset_index(inplace=True)
    
    # Merge with dates
    doc_dates = df[['doc_id', 'date']].drop_duplicates('doc_id')
    article_means = article_means.merge(doc_dates, on='doc_id', how='left')
    
    # Monthly aggregation
    print(" Monthly aggregation...")
    article_means['year_month'] = article_means['date'].dt.to_period('M')
    
    monthly_stats = (
        article_means
        .groupby('year_month')[frame_cols]
        .mean()
    )
    monthly_stats.index = monthly_stats.index.to_timestamp()
    monthly_stats = monthly_stats.sort_index()
    
    # Apply LOESS smoothing for clean visualization
    print(" Applying LOESS smoothing...")
    smoothed_data = {}

    for col in frame_cols:
        # Prepare data for LOESS
        y = monthly_stats[col].values
        x = np.arange(len(y))

        # Apply LOESS with fraction 0.15 for strong smoothing (cleaner trends)
        smoothed = lowess(y, x, frac=0.15, return_sorted=False)

        frame_name = FRAME_MAPPING[col]
        smoothed_data[frame_name] = smoothed
    
    # ============================================================================
    # 4. Creating chart
    # ============================================================================
    print(" Creating chart...")
    
    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # White background with subtle grid
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    # Plot smoothed curves (clean presentation without confidence intervals)
    for frame_name, values in smoothed_data.items():
        ax.plot(monthly_stats.index, values,
                label=frame_name,
                color=FRAME_COLORS[frame_name],
                linewidth=2.5,
                alpha=0.9,
                solid_capstyle='round',
                zorder=10)
    
    # Configuration des axes
    ax.set_xlabel('Year', fontsize=15, fontweight='bold', color='#333333')
    ax.set_ylabel('Average Proportion of Sentences per Article', fontsize=15, fontweight='bold', color='#333333')
    # Title removed as requested - figure already has caption in LaTeX
    
    # Format de l'axe des x (années)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    
    # Rotation des labels de l'axe x
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=12)
    
    # Grille améliorée
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Légende en bas avec disposition horizontale
    legend = ax.legend(title='Thematic Frames',
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.15),
                       ncol=4,
                       frameon=True,
                       fancybox=True,
                       shadow=True,
                       borderpad=1.5,
                       columnspacing=2,
                       fontsize=12,
                       title_fontsize=14,
                       edgecolor='#CCCCCC',
                       facecolor='white',
                       framealpha=0.95)
    legend.get_title().set_fontweight('bold')
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save
    output_path = OUT_DIR / "temporal_frames_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Chart saved: {output_path}")

# ============================================================================
# 5. Execution
# ============================================================================
if __name__ == "__main__":
    create_temporal_frames_plot()