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
8_frames_front_page_probability.py

MAIN OBJECTIVE:
---------------
Analyze how frame intensity affects the probability of front page placement.
This analysis uses logistic regression to model the relationship between
the proportion of sentences containing each frame and the likelihood of
an article appearing on the front page.

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- sqlalchemy
- statsmodels
- config_db (local module)

MAIN FEATURES:
-------------
1) Calculate frame intensity (proportion of sentences with each frame per article)
2) Fit logistic regression models: P(front_page) ~ frame_intensity
3) Compute marginal effects curves with confidence bands
4) Visualize dose-response relationships for each frame

Author:
-------
Antoine Lemor
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection parameters
DB_PARAMS: Dict[str, any] = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
}
TABLE_NAME = "CCF_processed_data"

# Output directories
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent / "Outputs" / "Figures"
STATS_DIR = SCRIPT_DIR.parent / "Outputs" / "Stats"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Frame mapping
FRAME_COLUMNS: Dict[str, str] = {
    'economic_frame':     'Economic',
    'political_frame':    'Political',
    'scientific_frame':   'Scientific',
    'environmental_frame': 'Environmental',
    'justice_frame':      'Justice',
    'health_frame':       'Health',
    'security_frame':     'Security',
    'cultural_frame':     'Cultural',
}

# Color palette for frames (colorblind-friendly)
FRAME_COLORS: Dict[str, str] = {
    'Economic':      '#E69F00',  # Orange
    'Political':     '#D55E00',  # Vermillion
    'Scientific':    '#0072B2',  # Blue
    'Environmental': '#009E73',  # Bluish green
    'Justice':       '#CC79A7',  # Reddish purple
    'Health':        '#56B4E9',  # Sky blue
    'Security':      '#F0E442',  # Yellow
    'Cultural':      '#999999',  # Gray
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load article-level data with frame intensities."""
    print("=" * 70)
    print("FRAME INTENSITY AND FRONT PAGE PLACEMENT ANALYSIS")
    print("=" * 70)
    print("\n[1/4] Loading data from CCF_Database...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )

    # Build frame columns for aggregation
    frame_cols = list(FRAME_COLUMNS.keys())
    frame_agg = ', '.join([f'AVG("{col}"::int) AS "{col}_intensity"' for col in frame_cols])
    frame_presence = ', '.join([f'MAX("{col}"::int) AS "{col}_present"' for col in frame_cols])

    query = text(f"""
        WITH article_data AS (
            SELECT
                doc_id,
                MAX(CASE
                    WHEN LOWER(page_number::text) LIKE '%front%' THEN 1
                    WHEN page_number::text ~ '^[A-Z]?1$' THEN 1
                    WHEN page_number::text = '1' THEN 1
                    ELSE 0
                END) AS front_page,
                COUNT(*) AS n_sentences,
                {frame_agg},
                {frame_presence}
            FROM "{TABLE_NAME}"
            GROUP BY doc_id
        )
        SELECT * FROM article_data
        WHERE n_sentences >= 3
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    engine.dispose()

    n_articles = len(df)
    n_front_page = df['front_page'].sum()
    baseline = df['front_page'].mean()

    print(f"    → {n_articles:,} articles loaded")
    print(f"    → {n_front_page:,} on front page ({baseline:.1%})")

    return df


# =============================================================================
# LOGISTIC REGRESSION MODELS
# =============================================================================

def fit_logistic_models(df: pd.DataFrame) -> Dict:
    """Fit logistic regression models for each frame."""
    print("\n[2/4] Fitting logistic regression models...")

    results = {}

    for frame_col, frame_name in FRAME_COLUMNS.items():
        intensity_col = f'{frame_col}_intensity'

        # Filter to articles with non-zero frame presence for meaningful variation
        # Use all articles but the model captures the effect of intensity
        y = df['front_page'].values
        X = df[intensity_col].values.reshape(-1, 1)
        X = sm.add_constant(X)

        # Fit logistic regression
        model = sm.Logit(y, X)
        try:
            fit = model.fit(disp=0, method='bfgs', maxiter=100)

            # Calculate marginal effect at mean
            mean_intensity = df[intensity_col].mean()
            prob_at_mean = fit.predict([1, mean_intensity])[0]
            marginal_effect = fit.params[1] * prob_at_mean * (1 - prob_at_mean)

            results[frame_name] = {
                'model': fit,
                'intercept': fit.params[0],
                'coefficient': fit.params[1],
                'coef_se': fit.bse[1],
                'coef_pvalue': fit.pvalues[1],
                'marginal_effect': marginal_effect,
                'pseudo_r2': fit.prsquared,
                'nobs': fit.nobs,
                'converged': True
            }

            sig = '***' if fit.pvalues[1] < 0.001 else '**' if fit.pvalues[1] < 0.01 else '*' if fit.pvalues[1] < 0.05 else ''
            print(f"    {frame_name:15s}: β = {fit.params[1]:+.3f} (SE={fit.bse[1]:.3f}), p={fit.pvalues[1]:.4f} {sig}")

        except Exception as e:
            print(f"    {frame_name:15s}: Model failed - {str(e)[:50]}")
            results[frame_name] = {'converged': False}

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figure(df: pd.DataFrame, model_results: Dict) -> None:
    """Create publication-quality visualization with marginal effects curves."""
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
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    _, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # =========================================================================
    # PANEL A: MARGINAL EFFECTS CURVES
    # =========================================================================
    ax1 = axes[0]

    # X-axis: frame intensity (0% to 100%)
    x_range = np.linspace(0, 1, 100)

    # Baseline probability
    baseline = df['front_page'].mean()

    # Plot curves for each frame
    for frame_col, frame_name in FRAME_COLUMNS.items():
        if frame_name not in model_results or not model_results[frame_name]['converged']:
            continue

        res = model_results[frame_name]
        color = FRAME_COLORS[frame_name]

        # Calculate predicted probabilities
        X_pred = np.column_stack([np.ones_like(x_range), x_range])
        y_pred = 1 / (1 + np.exp(-(res['intercept'] + res['coefficient'] * x_range)))

        # Calculate confidence bands using delta method
        var_pred = np.sum((X_pred @ res['model'].cov_params()) * X_pred, axis=1)
        se_logit = np.sqrt(var_pred)

        logit_pred = res['intercept'] + res['coefficient'] * x_range
        y_lower = 1 / (1 + np.exp(-(logit_pred - 1.96 * se_logit)))
        y_upper = 1 / (1 + np.exp(-(logit_pred + 1.96 * se_logit)))

        # Line width based on significance
        lw = 2.5 if res['coef_pvalue'] < 0.05 else 1.5
        alpha_line = 0.9 if res['coef_pvalue'] < 0.05 else 0.4

        # Plot curve
        ax1.plot(
            x_range * 100, y_pred * 100,
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=frame_name
        )

        # Confidence band
        ax1.fill_between(
            x_range * 100, y_lower * 100, y_upper * 100,
            color=color, alpha=0.08
        )

    # Baseline reference line
    ax1.axhline(y=baseline * 100, color='black', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Baseline ({baseline:.1%})')

    # Formatting
    ax1.set_xlabel('Frame Intensity (% of article sentences)', fontweight='bold')
    ax1.set_ylabel('Predicted Front Page Probability (%)', fontweight='bold')
    ax1.set_title('A. Marginal Effects of Frame Intensity on Editorial Prominence',
                  fontweight='bold', loc='left', pad=10)

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, None)

    # Grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Legend
    ax1.legend(
        loc='upper left',
        frameon=True,
        fancybox=False,
        edgecolor='gray',
        framealpha=0.95,
        ncol=2
    )

    # Model info
    ax1.text(
        0.98, 0.02,
        'Logistic regression\n95% confidence bands\nThick lines: p < 0.05',
        transform=ax1.transAxes,
        fontsize=8, ha='right', va='bottom',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    # =========================================================================
    # PANEL B: COEFFICIENT COMPARISON
    # =========================================================================
    ax2 = axes[1]

    # Prepare data for coefficient plot
    coef_data = []
    for frame_name in FRAME_COLUMNS.values():
        if frame_name not in model_results or not model_results[frame_name]['converged']:
            continue
        res = model_results[frame_name]
        coef_data.append({
            'frame': frame_name,
            'coefficient': res['coefficient'],
            'ci_low': res['coefficient'] - 1.96 * res['coef_se'],
            'ci_high': res['coefficient'] + 1.96 * res['coef_se'],
            'pvalue': res['coef_pvalue'],
            'color': FRAME_COLORS[frame_name]
        })

    coef_df = pd.DataFrame(coef_data).sort_values('coefficient', ascending=True)

    # Plot horizontal coefficient plot
    y_pos = np.arange(len(coef_df))

    for i, row in enumerate(coef_df.itertuples()):
        # Determine color based on significance and direction
        if row.pvalue >= 0.05:
            bar_color = '#CCCCCC'  # Gray for non-significant
        else:
            bar_color = row.color

        # Plot bar
        ax2.barh(i, row.coefficient, height=0.7, color=bar_color, alpha=0.85,
                edgecolor='white', linewidth=1.5)

        # Error bar
        ax2.errorbar(row.coefficient, i, xerr=[[row.coefficient - row.ci_low],
                     [row.ci_high - row.coefficient]],
                     fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)

        # Significance stars
        if row.pvalue < 0.001:
            sig = '***'
        elif row.pvalue < 0.01:
            sig = '**'
        elif row.pvalue < 0.05:
            sig = '*'
        else:
            sig = ''

        # Annotation
        x_text = row.ci_high + 0.05
        ax2.text(x_text, i, f'{row.coefficient:+.2f} {sig}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Reference line at zero
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, zorder=1)

    # Formatting
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(coef_df['frame'], fontweight='bold')
    ax2.set_xlabel('Logistic Regression Coefficient (β)', fontweight='bold')
    ax2.set_title('B. Effect Size Comparison Across Frames',
                  fontweight='bold', loc='left', pad=10)

    # Grid
    ax2.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    # Expand x-axis for annotations
    x_max = coef_df['ci_high'].max() + 0.4
    x_min = coef_df['ci_low'].min() - 0.2
    ax2.set_xlim(x_min, x_max)

    # Info box
    n_obs = model_results[list(FRAME_COLUMNS.values())[0]]['nobs']
    ax2.text(
        0.98, 0.02,
        f'n = {int(n_obs):,} articles\n'
        '*** p<0.001, ** p<0.01, * p<0.05\n'
        'Gray bars: not significant',
        transform=ax2.transAxes,
        fontsize=8, ha='right', va='bottom',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    # =========================================================================
    # FIGURE FINALIZATION
    # =========================================================================
    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "frames_front_page_probability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    → Figure saved: {output_path}")


# =============================================================================
# STATISTICAL OUTPUT
# =============================================================================

def save_statistics(model_results: Dict) -> None:
    """Export statistical summaries."""
    print("\n[4/4] Saving statistical summaries...")

    # Regression results
    records = []
    for frame_name, res in model_results.items():
        if not res['converged']:
            continue

        records.append({
            'frame': frame_name,
            'coefficient': res['coefficient'],
            'std_error': res['coef_se'],
            'p_value': res['coef_pvalue'],
            'ci_95_low': res['coefficient'] - 1.96 * res['coef_se'],
            'ci_95_high': res['coefficient'] + 1.96 * res['coef_se'],
            'marginal_effect': res['marginal_effect'],
            'pseudo_r2': res['pseudo_r2'],
            'n_obs': res['nobs']
        })

    results_df = pd.DataFrame(records).sort_values('coefficient', ascending=False)
    output_path = STATS_DIR / "frames_front_page_regression.csv"
    results_df.to_csv(output_path, index=False)
    print(f"    → Regression results: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("REGRESSION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Frame':<15} {'β':>10} {'SE':>10} {'p-value':>12} {'Sig':>6}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        print(f"{row['frame']:<15} {row['coefficient']:>+10.3f} {row['std_error']:>10.3f} "
              f"{row['p_value']:>12.4f} {sig:>6}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""

    # Load data
    df = load_data()

    # Fit logistic regression models
    model_results = fit_logistic_models(df)

    # Create visualization
    create_publication_figure(df, model_results)

    # Save statistics
    save_statistics(model_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
