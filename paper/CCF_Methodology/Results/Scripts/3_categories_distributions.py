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
3_categories_distributions.py

MAIN OBJECTIVE:
---------------
Create distribution plots for annotation categories in the CCF database.
Generate bar charts showing the prevalence of different actors, solutions, and events.

Dependencies:
-------------
- pandas
- matplotlib
- seaborn
- sqlalchemy
- config_db (local module)

MAIN FEATURES:
-------------
1) Connect to PostgreSQL database
2) Calculate proportions for each category
3) Generate individual distribution charts (commented out)
4) Create combined distribution visualization
5) Apply publication-quality styling

Author:
-------
Antoine Lemor
"""

import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
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
# 2. Définition des catégories (new column names from CCF_Database)
# ============================================================================

# Acteurs/Messengers (canonical names from Table B1)
ACTORS_MAPPING = {
    'messenger': 'Messenger (Primary Category)',
    'msg_health': 'Health expert',
    'msg_economic': 'Economic expert',
    'msg_security': 'Security expert',
    'msg_legal': 'Legal expert',
    'msg_cultural': 'Cultural/Sport expert',
    'msg_scientist': 'Natural scientist',
    'msg_social': 'Social scientist',
    'msg_activist': 'Activist',
    'msg_official': 'Public official'
}

# Solutions (canonical names from Table B1)
SOLUTIONS_MAPPING = {
    'solution': 'Solution (Primary Category)',
    'sol_mitigation': 'Mitigation strategy',
    'sol_adaptation': 'Adaptation strategy'
}

# Events (canonical names from Table B1)
EVENTS_MAPPING = {
    'event': 'Event (Primary Category)',
    'evt_weather': 'Extreme meteorological event',
    'evt_meeting': 'Meeting/Conference',
    'evt_publication': 'Publication',
    'evt_election': 'Election',
    'evt_policy': 'Policy announcement',
    'evt_judiciary': 'Judiciary decision',
    'evt_cultural': 'Cultural/Sports event',
    'evt_protest': 'Protest'
}

# Couleurs pour chaque catégorie principale
CATEGORY_COLORS = {
    'actors': '#2E86AB',    # Blue
    'solutions': '#A23B72',  # Purple
    'events': '#F18F01'      # Orange
}

# ============================================================================
# 3. Fonction pour créer un graphique à barres
# ============================================================================
def create_bar_chart(data: pd.DataFrame, 
                     columns: List[str], 
                     labels: Dict[str, str], 
                     title: str, 
                     color: str, 
                     output_name: str):
    """
    Create a horizontal bar chart for a given category
    """
    
    # Calculate average proportions
    means = {}
    for col in columns:
        if col in data.columns:
            # Average proportion of sentences containing this category
            mean_val = data[col].mean()
            label = labels.get(col, col)
            means[label] = mean_val * 100  # Convert to percentage
    
    # Sort by descending value
    sorted_means = dict(sorted(means.items(), key=lambda x: x[1], reverse=True))
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Style configuration
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')
    
    # Separate Primary Category from subcategories
    primary_data = {k: v for k, v in sorted_means.items() if 'Primary Category' in k}
    sub_data = {k: v for k, v in sorted_means.items() if 'Primary Category' not in k}
    
    # Sort subcategories by descending value
    sub_data_sorted = dict(sorted(sub_data.items(), key=lambda x: x[1], reverse=True))
    
    # Combine with Primary Category first
    if primary_data:
        all_categories = list(primary_data.keys()) + list(sub_data_sorted.keys())
        all_values = list(primary_data.values()) + list(sub_data_sorted.values())
    else:
        all_categories = list(sub_data_sorted.keys())
        all_values = list(sub_data_sorted.values())
    
    # Create bars with differentiated colors
    colors = []
    y_positions = []
    current_y = len(all_categories) - 1
    
    for i, cat in enumerate(all_categories):
        if 'Primary Category' in cat:
            colors.append(color)  # Main color for Primary Category
            y_positions.append(current_y)
            current_y -= 1
        else:
            # Lighter color for subcategories
            import matplotlib.colors as mcolors
            rgba = mcolors.to_rgba(color)
            lighter_color = (*rgba[:3], 0.7)  # 70% opacity
            colors.append(lighter_color)
            y_positions.append(current_y)
            current_y -= 1
    
    bars = ax.barh(range(len(all_categories)), all_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Set y-axis labels
    ax.set_yticks(range(len(all_categories)))
    ax.set_yticklabels(all_categories)
    
    # Add dotted line after Primary Category
    if primary_data:
        ax.axhline(y=len(all_categories) - len(primary_data) - 0.5, 
                   color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add values on ALL bars
    for bar, value in zip(bars, all_values):
        # Show all values, even small ones
        if value > 0:
            ax.text(value + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}%',
                   ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Configure axes
    ax.set_xlabel('Average Proportion of Sentences (%)', fontsize=12, fontweight='bold')
    # Title removed as requested - figure already has caption in LaTeX
    
    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save
    output_path = OUT_DIR / f"{output_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Chart saved: {output_path}")
    
    return sorted_means

# ============================================================================
# 4. Main function
# ============================================================================
def create_all_distributions():
    """
    Generate all distribution charts
    """
    
    print(" Generating distribution charts...")
    
    # Database connection
    print(" Connecting to PostgreSQL...")
    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}",
        connect_args={"options": DB_PARAMS['options']}
    )
    
    # Columns to retrieve
    all_cols = (list(ACTORS_MAPPING.keys()) + 
                list(SOLUTIONS_MAPPING.keys()) + 
                list(EVENTS_MAPPING.keys()))
    
    # Check which columns exist in the database
    check_sql = f"""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = '{TABLE_NAME}'
    """
    existing_cols_df = pd.read_sql(check_sql, engine)
    existing_cols = existing_cols_df['column_name'].tolist()
    
    # Filter existing columns
    cols_to_fetch = [col for col in all_cols if col in existing_cols]
    
    if not cols_to_fetch:
        print(" No columns found in database")
        engine.dispose()
        return
    
    # Build SQL query
    cols_str = ', '.join([f'"{col}"' for col in cols_to_fetch])
    sql = f"""
    SELECT {cols_str}
    FROM "{TABLE_NAME}"
    """
    
    print(" Loading data...")
    df = pd.read_sql(sql, engine)
    engine.dispose()
    
    if df.empty:
        print(" No data found")
        return
    
    print(f" Data loaded: {len(df)} rows")
    
    # ============================================================================
    # 5. Create charts
    # ============================================================================
    
    # Individual distribution charts commented out - not used in LaTeX
    # These charts are generated in the combined_distributions figure below
    
    # # 1. Graphique des Acteurs
    # print("\n Creating Actors chart...")
    # actors_cols = [col for col in ACTORS_MAPPING.keys() if col in df.columns]
    # if actors_cols:
    #     actors_stats = create_bar_chart(
    #         df, 
    #         actors_cols,
    #         ACTORS_MAPPING,
    #         'Distribution of Actors/Messengers in Climate Coverage',
    #         CATEGORY_COLORS['actors'],
    #         'actors_distribution'
    #     )
    #     
    #     print("\nTop 5 Actors:")
    #     for actor, pct in list(actors_stats.items())[:5]:
    #         print(f"  - {actor}: {pct:.2f}%")
    # 
    # # 2. Graphique des Solutions
    # print("\n Creating Solutions chart...")
    # solutions_cols = [col for col in SOLUTIONS_MAPPING.keys() if col in df.columns]
    # if solutions_cols:
    #     solutions_stats = create_bar_chart(
    #         df,
    #         solutions_cols,
    #         SOLUTIONS_MAPPING,
    #         'Distribution of Climate Solutions in Media Coverage',
    #         CATEGORY_COLORS['solutions'],
    #         'solutions_distribution'
    #     )
    #     
    #     print("\nTop 5 Solutions:")
    #     for solution, pct in list(solutions_stats.items())[:5]:
    #         print(f"  - {solution}: {pct:.2f}%")
    # 
    # # 3. Graphique des Événements
    # print("\n Creating Events chart...")
    # events_cols = [col for col in EVENTS_MAPPING.keys() if col in df.columns]
    # if events_cols:
    #     events_stats = create_bar_chart(
    #         df,
    #         events_cols,
    #         EVENTS_MAPPING,
    #         'Distribution of Climate Events in Media Coverage',
    #         CATEGORY_COLORS['events'],
    #         'events_distribution'
    #     )
    #     
    #     print("\nTop 5 Events:")
    #     for event, pct in list(events_stats.items())[:5]:
    #         print(f"  - {event}: {pct:.2f}%")
    
    # ============================================================================
    # 6. Create combined chart
    # ============================================================================
    print("\n Creating combined chart...")
    
    # Define columns for each category
    actors_cols = [col for col in ACTORS_MAPPING.keys() if col in df.columns]
    solutions_cols = [col for col in SOLUTIONS_MAPPING.keys() if col in df.columns]
    events_cols = [col for col in EVENTS_MAPPING.keys() if col in df.columns]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    # Title removed as requested - figure already has caption in LaTeX
    
    # Configuration for each subplot
    for ax, (category, mapping, color, cols) in zip(
        axes,
        [('Messengers', ACTORS_MAPPING, CATEGORY_COLORS['actors'], actors_cols),
         ('Solutions', SOLUTIONS_MAPPING, CATEGORY_COLORS['solutions'], solutions_cols),
         ('Events', EVENTS_MAPPING, CATEGORY_COLORS['events'], events_cols)]
    ):
        if cols:
            # Calculate averages for this chart (include ALL categories)
            means = {}
            detection_val = None

            # Primary category columns (no underscore = primary)
            primary_cols = ['messenger', 'solution', 'event']

            for col in cols:
                if col in df.columns:
                    mean_val = df[col].mean() * 100
                    label = mapping.get(col, col).replace('_', ' ')

                    if col in primary_cols:
                        detection_val = (label, mean_val)
                    else:
                        means[label] = mean_val
            
            # Sort subcategories by descending value
            sorted_means = dict(sorted(means.items(), key=lambda x: x[1], reverse=True))
            
            if sorted_means or detection_val:
                # Prepare data with Primary Category separated
                if detection_val:
                    categories = [detection_val[0]] + list(sorted_means.keys())
                    values = [detection_val[1]] + list(sorted_means.values())
                else:
                    categories = list(sorted_means.keys())
                    values = list(sorted_means.values())
                
                # Create colors
                bar_colors = []
                for i, cat in enumerate(categories):
                    if 'Primary Category' in cat:
                        bar_colors.append(color)
                    else:
                        import matplotlib.colors as mcolors
                        rgba = mcolors.to_rgba(color)
                        bar_colors.append((*rgba[:3], 0.7))
                
                bars = ax.bar(range(len(categories)), values, color=bar_colors, 
                             edgecolor='black', linewidth=0.5)
                
                # Ajouter ligne verticale pointillée après Primary Category
                if detection_val:
                    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Proportion (%)', fontsize=11)
                # Title removed as requested - figure already has caption in LaTeX
                ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Add values on ALL bars
                for bar, value in zip(bars, values):
                    if value > 0:  # Show all non-zero values
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save combined chart
    output_path = OUT_DIR / "combined_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Combined chart saved: {output_path}")

# ============================================================================
# 7. Execution
# ============================================================================
if __name__ == "__main__":
    create_all_distributions()