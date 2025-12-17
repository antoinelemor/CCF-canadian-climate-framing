#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
13_create_intercoder_progression_plot.py

MAIN OBJECTIVE:
---------------
This script generates a figure showing the progression of inter-coder
reliability metrics across 1000 annotated sentences. 

Dependencies:
-------------
- pandas >= 1.5
- matplotlib >= 3.5
- numpy >= 1.20
- seaborn >= 0.12
- pathlib (standard library)

Key Features:
-------------
1) Loads inter-coder progression data from CSV file
2) Plots Cohen's Kappa and Krippendorff's Alpha progression over 1000 sentences
3) Calculates and displays mean values for training vs blind coding phases
4) Fits linear trend lines to visualize learning effects
5) Highlights the transition point at sentence 600
6) Exports figure for inclusion in methodology paper

Author:
-------
Antoine Lemor
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

##############################################################################
#                               SETUP & CONFIG                               #
##############################################################################

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / ".." / "Database" / "Training_data" / "manual_annotations_JSONL"
OUTPUT_DIR = BASE_DIR / ".." / ".." / "paper" / "CCF_Methodology" / "Results" / "Outputs" / "Figures"

# Read the progression data
progression_file = DATA_DIR / "intercoder_reliability_3_learning_progression.csv"
df_progression = pd.read_csv(progression_file)

##############################################################################
#                          CREATE SINGLE FIGURE                              #
##############################################################################

# Create single figure with Cohen's Kappa and Krippendorff's Alpha
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Cohen's Kappa
ax.plot(df_progression['n_sentences'], df_progression['cohens_kappa'],
        marker='o', linewidth=2.5, markersize=8, color='#2E7D32',
        label="Cohen's κ", zorder=3)

# Plot Krippendorff's Alpha
ax.plot(df_progression['n_sentences'], df_progression['krippendorff_alpha'],
        marker='s', linewidth=2.5, markersize=7, color='#1565C0',
        label="Krippendorff's α", zorder=3)

# Split data for before and after 600
before_600 = df_progression[df_progression['n_sentences'] <= 600]
after_600 = df_progression[df_progression['n_sentences'] > 600]

# Calculate mean values for each phase
kappa_mean_before = before_600['cohens_kappa'].mean()
kappa_mean_after = after_600['cohens_kappa'].mean()
alpha_mean_before = before_600['krippendorff_alpha'].mean()
alpha_mean_after = after_600['krippendorff_alpha'].mean()

# Fit trend lines for Cohen's Kappa
z1_kappa = np.polyfit(before_600['n_sentences'], before_600['cohens_kappa'], 1)
p1_kappa = np.poly1d(z1_kappa)
ax.plot(before_600['n_sentences'], p1_kappa(before_600['n_sentences']),
        '--', color='#2E7D32', alpha=0.5, linewidth=2,
        label=f'κ mean (0-600): {kappa_mean_before:.3f}')

z2_kappa = np.polyfit(after_600['n_sentences'], after_600['cohens_kappa'], 1)
p2_kappa = np.poly1d(z2_kappa)
ax.plot(after_600['n_sentences'], p2_kappa(after_600['n_sentences']),
        '--', color='#2E7D32', alpha=0.7, linewidth=2,
        label=f'κ mean (600+): {kappa_mean_after:.3f}')

# Fit trend lines for Krippendorff's Alpha
z1_alpha = np.polyfit(before_600['n_sentences'], before_600['krippendorff_alpha'], 1)
p1_alpha = np.poly1d(z1_alpha)
ax.plot(before_600['n_sentences'], p1_alpha(before_600['n_sentences']),
        '--', color='#1565C0', alpha=0.5, linewidth=2,
        label=f'α mean (0-600): {alpha_mean_before:.3f}')

z2_alpha = np.polyfit(after_600['n_sentences'], after_600['krippendorff_alpha'], 1)
p2_alpha = np.poly1d(z2_alpha)
ax.plot(after_600['n_sentences'], p2_alpha(after_600['n_sentences']),
        '--', color='#1565C0', alpha=0.7, linewidth=2,
        label=f'α mean (600+): {alpha_mean_after:.3f}')

# Add vertical line at 600
ax.axvline(x=600, color='red', linestyle='--', alpha=0.7, linewidth=2)

# Shade regions
ax.axvspan(0, 600, alpha=0.08, color='blue')
ax.axvspan(600, 1000, alpha=0.08, color='green')

# Add region labels
ax.text(300, 0.72, 'TRAINING', ha='center', va='center',
        fontsize=11, color='blue', alpha=0.6, fontweight='bold')
ax.text(800, 0.72, 'BLIND CODING', ha='center', va='center',
        fontsize=11, color='green', alpha=0.6, fontweight='bold')

# Add annotations for key values
ax.annotate(f'κ = {before_600.iloc[-1]["cohens_kappa"]:.3f}',
            xy=(600, before_600.iloc[-1]["cohens_kappa"]),
            xytext=(480, before_600.iloc[-1]["cohens_kappa"] - 0.0),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', alpha=0.5),
            fontsize=9)

ax.annotate(f'κ = {df_progression.iloc[-1]["cohens_kappa"]:.3f}',
            xy=(1000, df_progression.iloc[-1]["cohens_kappa"]),
            xytext=(880, df_progression.iloc[-1]["cohens_kappa"] + 0.015),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', alpha=0.5),
            fontsize=9)

# Styling
ax.set_xlabel('Number of Sentences Coded', fontsize=13, fontweight='bold')
ax.set_ylabel('Agreement Score', fontsize=13, fontweight='bold')
# No title in figure - will be added in LaTeX
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

# Position legend below the plot area to avoid overlap with curves
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True,
          fancybox=True, shadow=True, fontsize=9, ncol=3)

ax.set_xlim(-25, 1025)
ax.set_ylim(0.50, 0.75)

# Set x-axis ticks every 100 sentences
ax.set_xticks(range(0, 1100, 100))

# Add minor gridlines
ax.grid(True, which='minor', alpha=0.1, linestyle='-', linewidth=0.3)
ax.minorticks_on()

plt.tight_layout()

# Save the figure
output_path = OUTPUT_DIR / "intercoder_progression.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved to: {output_path}")

# Show the plot
plt.show()