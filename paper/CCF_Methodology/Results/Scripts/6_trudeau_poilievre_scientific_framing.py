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
6_trudeau_poilievre_scientific_framing.py

MAIN OBJECTIVE:
---------------
Analyze the statistical effect of Justin Trudeau and Pierre Poilievre mentions
on scientific skepticism framing (sci_skepticism) in Canadian climate media
coverage (September 2022 - December 2024). This analysis uses OLS regression
to test the hypothesis: "Does mentioning a political leader increase or
decrease scientific skepticism?"

Dependencies:
-------------
- pandas
- numpy
- matplotlib
- scipy
- sqlalchemy
- statsmodels
- config_db (local module)

MAIN FEATURES:
-------------
1) Extract leader mentions from NER entities using pattern matching
2) Calculate article-level metrics for leader mention intensity
3) Run OLS regression models for scientific skepticism
4) Visualize marginal effects curves with confidence bands
5) Compare aggregate proportions between leader groups

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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

# Publication-quality color palette
COLORS: Dict[str, str] = {
    'trudeau':    '#C41E3A',   # Deep red (Liberal)
    'poilievre':  '#1E4D8C',   # Deep blue (Conservative)
}

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


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    return (
        np.percentile(bootstrap_means, 100 * alpha / 2),
        np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    )


# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load data from September 10, 2022 to end of 2024 from the CCF database."""
    print("=" * 70)
    print("TRUDEAU VS POILIEVRE: SCIENTIFIC SKEPTICISM EFFECTS (Sep 2022 - Dec 2024)")
    print("=" * 70)
    print("\n[1/5] Loading data from CCF_Database...")

    engine = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}"
        f"@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )

    # Data from September 10, 2022 (Poilievre became Conservative leader) to end of 2024
    query = text(f"""
        SELECT
            doc_id, sentence_id, date, media,
            scientific_frame, sci_skepticism,
            ner_entities
        FROM "{TABLE_NAME}"
        WHERE date >= '2022-09-10' AND date <= '2024-12-31'
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    engine.dispose()

    print(f"    -> Loaded {len(df):,} sentences from {df['doc_id'].nunique():,} articles")
    return df


def identify_leaders(df: pd.DataFrame) -> pd.DataFrame:
    """Identify leader mentions in each sentence."""
    print("\n[2/5] Identifying leader mentions...")

    def check_leader(ner_val, patterns):
        persons = extract_persons_from_ner(ner_val)
        text = ' '.join(persons).lower() if persons else ''
        return check_patterns(text, patterns)

    df['mentions_trudeau'] = df['ner_entities'].apply(
        lambda x: check_leader(x, TRUDEAU_PATTERNS)
    ).astype(int)

    df['mentions_poilievre'] = df['ner_entities'].apply(
        lambda x: check_leader(x, POILIEVRE_PATTERNS)
    ).astype(int)

    n_trudeau = df['mentions_trudeau'].sum()
    n_poilievre = df['mentions_poilievre'].sum()

    print(f"    -> Trudeau mentions:   {n_trudeau:,} sentences")
    print(f"    -> Poilievre mentions: {n_poilievre:,} sentences")

    return df


def aggregate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level data to article level with intensity metrics."""
    print("\n[3/5] Aggregating to article level...")

    # Calculate both presence and intensity (proportion of sentences)
    articles = df.groupby('doc_id').agg({
        'date': 'first',
        'media': 'first',
        'mentions_trudeau': ['max', 'mean', 'sum'],
        'mentions_poilievre': ['max', 'mean', 'sum'],
        'scientific_frame': 'mean',
        'sci_skepticism': 'mean',
        'sentence_id': 'count'
    })

    # Flatten column names
    articles.columns = [
        'date', 'media',
        'trudeau_present', 'trudeau_intensity', 'trudeau_count',
        'poilievre_present', 'poilievre_intensity', 'poilievre_count',
        'scientific_frame', 'sci_skepticism',
        'n_sentences'
    ]
    articles = articles.reset_index()

    # Summary statistics
    n_trudeau = (articles['trudeau_present'] == 1).sum()
    n_poilievre = (articles['poilievre_present'] == 1).sum()
    n_both = ((articles['trudeau_present'] == 1) & (articles['poilievre_present'] == 1)).sum()

    print(f"    -> Articles mentioning Trudeau:   {n_trudeau:,} ({n_trudeau/len(articles)*100:.1f}%)")
    print(f"    -> Articles mentioning Poilievre: {n_poilievre:,} ({n_poilievre/len(articles)*100:.1f}%)")
    print(f"    -> Articles mentioning both:      {n_both:,} ({n_both/len(articles)*100:.1f}%)")

    return articles


def run_regression_models(articles: pd.DataFrame) -> Dict:
    """Run OLS regression models for each leader on scientific skepticism."""
    print("\n[4/5] Running regression models...")

    results = {}
    leaders = ['trudeau', 'poilievre']

    for leader in leaders:
        # Prepare data - filter to articles mentioning the leader
        leader_articles = articles[articles[f'{leader}_present'] == 1].copy()

        # Drop rows with NaN in outcome or predictor
        clean_data = leader_articles.dropna(subset=['sci_skepticism', f'{leader}_intensity'])

        y = clean_data['sci_skepticism'].values
        X = clean_data[f'{leader}_intensity'].values.reshape(-1, 1)
        X = sm.add_constant(X)

        # Fit OLS model with robust standard errors
        model = sm.OLS(y, X)
        fit = model.fit(cov_type='HC1')

        results[leader] = {
            'model': fit,
            'intercept': fit.params[0],
            'slope': fit.params[1],
            'intercept_se': fit.bse[0],
            'slope_se': fit.bse[1],
            'slope_pvalue': fit.pvalues[1],
            'conf_int': fit.conf_int(),
            'rsquared': fit.rsquared,
            'nobs': fit.nobs,
        }

    # Print summary
    print("\n    Regression Results Summary:")
    print("    " + "-" * 60)
    print(f"    {'Leader':<12} {'Outcome':<25} {'B':>10} {'SE':>10} {'p':>10}")
    print("    " + "-" * 60)

    for leader in leaders:
        res = results[leader]
        sig = '***' if res['slope_pvalue'] < 0.001 else '**' if res['slope_pvalue'] < 0.01 else '*' if res['slope_pvalue'] < 0.05 else ''
        print(f"    {leader.capitalize():<12} {'Scientific Skepticism':<25} "
              f"{res['slope']:>+10.4f} {res['slope_se']:>10.4f} {res['slope_pvalue']:>10.4f} {sig}")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_publication_figure(
    articles: pd.DataFrame,
    regression_results: Dict
) -> None:
    """Create publication-quality visualization with marginal effects curves."""
    print("\n[5/5] Creating publication figure...")

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

    # X-axis: leader mention intensity (0 to max observed)
    x_range = np.linspace(0, 0.5, 100)  # 0% to 50% of sentences

    # Plot curves for each leader
    for leader in ['trudeau', 'poilievre']:
        res = regression_results[leader]
        color = COLORS[leader]

        # Calculate predicted values
        y_pred = res['intercept'] + res['slope'] * x_range

        # Calculate confidence band using delta method
        X_pred = np.column_stack([np.ones_like(x_range), x_range])
        var_pred = np.sum((X_pred @ res['model'].cov_params()) * X_pred, axis=1)
        se_pred = np.sqrt(var_pred)

        y_lower = y_pred - 1.96 * se_pred
        y_upper = y_pred + 1.96 * se_pred

        # Determine line width based on significance
        lw = 2.5 if res['slope_pvalue'] < 0.05 else 1.5
        alpha_line = 0.9 if res['slope_pvalue'] < 0.05 else 0.5

        # Plot line
        ax1.plot(
            x_range * 100, y_pred * 100,
            color=color,
            linewidth=lw,
            alpha=alpha_line,
            label=leader.capitalize()
        )

        # Confidence band
        ax1.fill_between(
            x_range * 100, y_lower * 100, y_upper * 100,
            color=color, alpha=0.15
        )

        # Add significance marker at end of line
        if res['slope_pvalue'] < 0.05:
            sig = '***' if res['slope_pvalue'] < 0.001 else '**' if res['slope_pvalue'] < 0.01 else '*'
            ax1.annotate(
                sig,
                xy=(x_range[-1] * 100, y_pred[-1] * 100),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color=color
            )

    ax1.legend(
        loc='upper left',
        frameon=True,
        fancybox=False,
        edgecolor='gray',
        framealpha=0.95
    )

    # Formatting
    ax1.set_xlabel('Leader Mention Intensity\n(% of article sentences)', fontweight='bold')
    ax1.set_ylabel('Predicted Scientific Skepticism (%)', fontweight='bold')
    ax1.set_title('A. Marginal Effects of Leader Mention Intensity',
                  fontweight='bold', loc='left', pad=10)

    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, None)

    # Grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Model info
    ax1.text(
        0.98, 0.02,
        'OLS with HC1 robust SE\n95% confidence bands\n*** p<0.001, ** p<0.01, * p<0.05',
        transform=ax1.transAxes,
        fontsize=8, ha='right', va='bottom',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    # =========================================================================
    # PANEL B: AGGREGATE COMPARISON WITH CONFIDENCE INTERVALS
    # =========================================================================
    ax2 = axes[1]

    categories = ['Trudeau', 'Poilievre']
    bar_width = 0.5
    x_pos = np.arange(len(categories))

    # Calculate statistics for each group
    stats_data = {}
    for cat in categories:
        col = f'{cat.lower()}_present'
        cat_articles = articles[articles[col] == 1]
        values = cat_articles['sci_skepticism'].dropna().values
        mean_val = np.mean(values)
        ci_low, ci_high = bootstrap_ci(values, n_bootstrap=1000)

        stats_data[cat] = {
            'mean': mean_val,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n': len(cat_articles)
        }

    # Plot bars
    cat_colors = [COLORS['trudeau'], COLORS['poilievre']]
    means = [stats_data[cat]['mean'] * 100 for cat in categories]
    errors_low = [(stats_data[cat]['mean'] - stats_data[cat]['ci_low']) * 100 for cat in categories]
    errors_high = [(stats_data[cat]['ci_high'] - stats_data[cat]['mean']) * 100 for cat in categories]

    ax2.bar(
        x_pos, means, bar_width,
        color=cat_colors,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.85
    )

    ax2.errorbar(
        x_pos, means,
        yerr=[errors_low, errors_high],
        fmt='none',
        color='black',
        capsize=8,
        capthick=1.5,
        linewidth=1.5
    )

    # Statistical significance annotation (Mann-Whitney U test)
    trudeau_vals = articles[articles['trudeau_present'] == 1]['sci_skepticism'].dropna()
    poilievre_vals = articles[articles['poilievre_present'] == 1]['sci_skepticism'].dropna()

    _, p_val = stats.mannwhitneyu(
        trudeau_vals, poilievre_vals, alternative='two-sided'
    )

    # Significance stars
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    # Add bracket and significance
    y_max = max(stats_data['Trudeau']['ci_high'], stats_data['Poilievre']['ci_high']) * 100
    bracket_height = y_max + 3

    ax2.plot(
        [0, 0, 1, 1],
        [bracket_height, bracket_height + 1.5, bracket_height + 1.5, bracket_height],
        color='black', linewidth=1
    )
    ax2.text(
        0.5, bracket_height + 2, sig,
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

    # Formatting
    ax2.set_ylabel('Scientific Skepticism (%)', fontweight='bold')
    ax2.set_title('B. Aggregate Comparison by Leader',
                  fontweight='bold', loc='left', pad=10)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, fontweight='bold')

    # Grid
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylim(bottom=0)

    # Sample size annotation
    n_trudeau = stats_data['Trudeau']['n']
    n_poilievre = stats_data['Poilievre']['n']

    ax2.text(
        0.98, 0.02,
        f'n(Trudeau) = {n_trudeau:,}\nn(Poilievre) = {n_poilievre:,}\nMann-Whitney p = {p_val:.3f}',
        transform=ax2.transAxes,
        fontsize=8,
        ha='right', va='bottom',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.9)
    )

    # =========================================================================
    # FIGURE FINALIZATION
    # =========================================================================
    plt.tight_layout()

    # Save figure
    output_path = FIGURES_DIR / "trudeau_poilievre_scientific_framing.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"    -> Figure saved: {output_path}")


# =============================================================================
# STATISTICAL OUTPUT
# =============================================================================

def save_statistics(
    articles: pd.DataFrame,
    regression_results: Dict
) -> None:
    """Export statistical summaries to CSV files."""
    print("\n[OUTPUT] Saving statistical summaries...")

    # Regression results
    reg_records = []
    leaders = ['trudeau', 'poilievre']

    for leader in leaders:
        res = regression_results[leader]
        ci = res['conf_int']

        reg_records.append({
            'leader': leader.capitalize(),
            'outcome': 'sci_skepticism',
            'intercept': res['intercept'],
            'slope': res['slope'],
            'slope_se': res['slope_se'],
            'slope_pvalue': res['slope_pvalue'],
            'slope_ci_low': ci[1, 0],
            'slope_ci_high': ci[1, 1],
            'r_squared': res['rsquared'],
            'n_obs': res['nobs']
        })

    reg_df = pd.DataFrame(reg_records)
    reg_path = STATS_DIR / "trudeau_poilievre_regression_results.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"    -> Regression results: {reg_path}")

    # Aggregate statistics
    categories = ['Trudeau', 'Poilievre']
    agg_records = []

    for cat in categories:
        col = f'{cat.lower()}_present'
        cat_articles = articles[articles[col] == 1]
        values = cat_articles['sci_skepticism'].dropna().values
        ci_low, ci_high = bootstrap_ci(values, n_bootstrap=1000)

        agg_records.append({
            'leader': cat,
            'variable': 'sci_skepticism',
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'n_articles': len(cat_articles)
        })

    # Mann-Whitney test
    trudeau_vals = articles[articles['trudeau_present'] == 1]['sci_skepticism'].dropna()
    poilievre_vals = articles[articles['poilievre_present'] == 1]['sci_skepticism'].dropna()
    stat, p_val = stats.mannwhitneyu(trudeau_vals, poilievre_vals)

    agg_records.append({
        'leader': 'Comparison',
        'variable': 'sci_skepticism',
        'mann_whitney_U': stat,
        'p_value': p_val
    })

    agg_df = pd.DataFrame(agg_records)
    agg_path = STATS_DIR / "trudeau_poilievre_aggregate_stats.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"    -> Aggregate stats: {agg_path}")

    # Model summaries as text
    for leader in leaders:
        summary_path = STATS_DIR / f"trudeau_poilievre_{leader}_sci_skepticism_model.txt"
        with open(summary_path, 'w') as f:
            try:
                f.write(str(regression_results[leader]['model'].summary()))
            except Exception:
                res = regression_results[leader]
                f.write(f"OLS Regression Results\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Leader: {leader.capitalize()}\n")
                f.write(f"Outcome: sci_skepticism\n")
                f.write(f"N observations: {int(res['nobs'])}\n")
                f.write(f"R-squared: {res['rsquared']:.4f}\n")
                f.write(f"\nCoefficients:\n")
                f.write(f"  Intercept: {res['intercept']:.4f} (SE: {res['intercept_se']:.4f})\n")
                f.write(f"  Slope: {res['slope']:.4f} (SE: {res['slope_se']:.4f}, p: {res['slope_pvalue']:.4f})\n")
    print(f"    -> Model summaries saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Main execution pipeline."""

    # Load and process data
    df = load_data()
    df = identify_leaders(df)
    articles = aggregate_articles(df)

    # Run regression models
    regression_results = run_regression_models(articles)

    # Create visualization
    create_publication_figure(articles, regression_results)

    # Save statistics
    save_statistics(articles, regression_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
