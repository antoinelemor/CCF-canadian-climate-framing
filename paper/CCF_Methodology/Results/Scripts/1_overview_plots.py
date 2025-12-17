#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

PAPER:
------
CCF_Methodology

TITLE:
------
1_overview_plots.py

MAIN OBJECTIVE:
---------------
Generate overview plots showing the distribution of articles in the CCF database:
1) Distribution of article counts by media outlet (top 20)
2) Temporal distribution of articles by year
3) Geographic distribution of articles by province

Dependencies:
-------------
- matplotlib
- pandas
- psycopg2 or CSV fallback
- geopandas (for province map)
- config_db (local module)

MAIN FEATURES:
-------------
1) Create publication-quality bar charts for media outlets
2) Generate temporal trend visualization
3) Create choropleth map of provincial distribution
4) Support both database and CSV data sources

Author:
-------
Antoine Lemor
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings
import csv
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image  # Only for PNG to PDF conversion

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    # Optional high-level connector provided by the project (if present)
    from db_connector import DatabaseConnector  # type: ignore
except Exception:
    DatabaseConnector = None  # optional

try:
    import psycopg2
except Exception:
    psycopg2 = None  # fallback to CSV if not present


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "Results" / "Outputs" / "Figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Database configuration - CCF_Database (publication database)
DB_PARAMS = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   "CCF_Database",  # Publication database with proper DATE type
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
    "options":  "-c client_min_messages=warning",
}
OVERVIEW_TABLE = "CCF_full_data"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CSV_FALLBACK = PROJECT_ROOT / "Database" / "Database" / "CCF.media_database.csv"
MEDIA_PROVINCE_CSV = PROJECT_ROOT / "Database" / "Database" / "Canadian_Media_Articles_by_Province.csv"
SHAPEFILE_PATH = PROJECT_ROOT / "Database" / "Database" / "CAN_shp" / "lpr_000b21a_e.shp"


def add_watermark_and_logo(ax, fig):
    """Placeholder - watermarks removed for publication."""
    pass


def fetch_full_data_counts():
    """Return (media_counts, year_counts, province_counts, national_count, national_media, lang_counts, total_sentences) from DB or CSV.

    Priority: PostgreSQL → CSV fallback.
    """
    # Style tuning for publication-quality figures
    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'axes.edgecolor': '#333333',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

    # Try DB first
    # 1) Preferred: DatabaseConnector if available
    if DatabaseConnector is not None and pd is not None:
        try:
            connector = DatabaseConnector()
            with connector.get_connection() as conn:  # SQLAlchemy connection
                # Media counts
                media_df = pd.read_sql(
                    f"SELECT media, COUNT(*) AS n FROM {OVERVIEW_TABLE} GROUP BY media",
                    conn,
                )
                # Year counts with safe casts from potential text dates
                year_df = pd.read_sql(
                    f"""
                    SELECT EXTRACT(YEAR FROM (
                        CASE WHEN date ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$' THEN date::date
                             WHEN date ~ '^[0-9]{{4}}/[0-9]{{2}}/[0-9]{{2}}$' THEN to_date(date, 'YYYY/MM/DD')
                             WHEN date ~ '^[0-9]{{4}}$' THEN to_date(date||'-01-01','YYYY-MM-DD')
                             WHEN date ~ '^[0-9]{{2}}-[0-9]{{2}}-[0-9]{{4}}$' THEN to_date(date, 'MM-DD-YYYY')
                             ELSE NULL END
                    )) AS yr, COUNT(*) AS n
                    FROM {OVERVIEW_TABLE}
                    GROUP BY yr
                    ORDER BY yr
                    """,
                    conn,
                )
            media_counts = Counter({(m or '').strip(): int(n) for m, n in media_df.values if m})
            year_counts = {int(y): int(n) for y, n in year_df.values if pd.notna(y) and int(y) < 2025}
            
            # Get sentence count from CCF_processed_data table
            try:
                sentence_df = pd.read_sql("SELECT COUNT(*) AS n FROM ccf_processed_data", conn)
                total_sentences = int(sentence_df['n'].iloc[0])
            except:
                # Fallback to counting from annotated file
                total_sentences = 0
                
            # Province counts and language counts not available from DB, return empty
            province_counts = Counter()
            national_count = 0
            national_media = []
            lang_counts = {'EN': 0, 'FR': 0}
            return media_counts, year_counts, province_counts, national_count, national_media, lang_counts, total_sentences
        except Exception:
            pass  # silently fall through to psycopg2 / CSV

    # 2) psycopg2 direct
    if psycopg2 is not None:
        try:
            conn = psycopg2.connect(
                host=DB_PARAMS["host"],
                port=DB_PARAMS["port"],
                user=DB_PARAMS["user"],
                password=DB_PARAMS["password"],
                dbname=DB_PARAMS["dbname"],
            )
            cur = conn.cursor()
            cur.execute(f'SELECT media, COUNT(*) FROM "{OVERVIEW_TABLE}" GROUP BY media;')
            media_rows = cur.fetchall()
            # Year counts - date is proper DATE type in CCF_Database
            cur.execute(
                f"""
                SELECT EXTRACT(YEAR FROM date) AS yr, COUNT(*)
                FROM "{OVERVIEW_TABLE}"
                WHERE date IS NOT NULL
                GROUP BY yr
                ORDER BY yr
                """
            )
            year_rows = cur.fetchall()
            
            # Get sentence count from CCF_processed_data table
            try:
                cur.execute('SELECT COUNT(*) FROM "CCF_processed_data";')
                total_sentences = cur.fetchone()[0]
            except:
                # Fallback to counting from annotated file
                total_sentences = 0

            # Get actual language counts from database (before closing connection)
            try:
                cur.execute(f'SELECT language, COUNT(*) FROM "{OVERVIEW_TABLE}" GROUP BY language;')
                lang_rows = cur.fetchall()
                lang_counts = {lang: int(count) for lang, count in lang_rows if lang in ('EN', 'FR')}
            except Exception:
                lang_counts = None  # Will be estimated below

            cur.close()
            conn.close()
            media_counts = Counter({(m or '').strip(): int(n) for m, n in media_rows if m})
            year_counts = {int(y): int(n) for y, n in year_rows if y is not None and int(y) < 2025}

            # Load media to province mapping to calculate province counts from media data
            media_to_province = {}
            media_is_national = set()
            national_media_names = []
            if MEDIA_PROVINCE_CSV.exists():
                with open(MEDIA_PROVINCE_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        media_orig = (row.get('media') or '').strip()
                        media_lower = media_orig.lower()
                        region = (row.get('region') or '').strip()
                        if media_orig and region:
                            if region.lower() == 'national':
                                media_is_national.add(media_lower)
                                if media_orig not in national_media_names:
                                    national_media_names.append(media_orig)
                            else:
                                media_to_province[media_lower] = region

            # Calculate province counts from media counts
            province_counts = Counter()
            national_count = 0
            for media, count in media_counts.items():
                media_lower = media.lower()
                if media_lower in media_to_province:
                    province_counts[media_to_province[media_lower]] += count
                elif media_lower in media_is_national:
                    national_count += count

            # Fallback to estimation if language counts were not retrieved from DB
            if lang_counts is None:
                french_media = {'le devoir', 'la presse', 'le soleil', 'le droit', 'le journal de montréal', 'le journal de québec'}
                lang_counts = {'EN': 0, 'FR': 0}
                for media, count in media_counts.items():
                    if media.lower() in french_media:
                        lang_counts['FR'] += count
                    else:
                        lang_counts['EN'] += count
            
            return media_counts, year_counts, province_counts, national_count, national_media_names, lang_counts, total_sentences
        except Exception as exc:
            warnings.warn(f"DB connection failed, falling back to CSV: {exc}")

    # Fallback to CSV
    if not CSV_FALLBACK.exists():
        raise SystemExit(f"CSV fallback not found: {CSV_FALLBACK}")
    
    # Load media to province mapping from correct CSV
    media_to_province = {}
    media_is_national = set()
    national_media_names = []
    if MEDIA_PROVINCE_CSV.exists():
        with open(MEDIA_PROVINCE_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                media = (row.get('media') or '').strip().lower()
                media_original = (row.get('media') or '').strip()  # Keep original case for display
                region = (row.get('region') or '').strip()
                if media and region:
                    if region.lower() == 'national':
                        media_is_national.add(media)
                        if media_original not in national_media_names:
                            national_media_names.append(media_original)
                    else:
                        # Direct mapping from CSV - region IS the province name
                        media_to_province[media] = region
    
    media_counts: Counter[str] = Counter()
    year_counts: dict[int, int] = defaultdict(int)
    province_counts: Counter[str] = Counter()
    national_count = 0
    lang_counts = {'EN': 0, 'FR': 0}
    
    with open(CSV_FALLBACK, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            media = (row.get("media") or "").strip()
            if media:
                media_counts[media] += 1
                # Count by province or national
                media_lower = media.lower()
                if media_lower in media_to_province:
                    province_counts[media_to_province[media_lower]] += 1
                elif media_lower in media_is_national:
                    national_count += 1
            
            # Count by language
            language = (row.get("language") or "").strip().upper()
            if language in ['EN', 'FR']:
                lang_counts[language] += 1
                    
            date_str = (row.get("date") or "").strip()
            year = None
            # Try different date patterns
            if '-' in date_str:
                parts = date_str.split('-')
                if len(parts) == 3:
                    # Check for MM-DD-YYYY format
                    if len(parts[2]) == 4 and parts[2].isdigit():
                        year = int(parts[2])
                    # Check for YYYY-MM-DD format
                    elif len(parts[0]) == 4 and parts[0].isdigit():
                        year = int(parts[0])
            elif '/' in date_str:
                parts = date_str.split('/')
                if len(parts) == 3:
                    # Check for MM/DD/YYYY or DD/MM/YYYY format
                    if len(parts[2]) == 4 and parts[2].isdigit():
                        year = int(parts[2])
                    # Check for YYYY/MM/DD format
                    elif len(parts[0]) == 4 and parts[0].isdigit():
                        year = int(parts[0])
            elif len(date_str) == 4 and date_str.isdigit():
                # Just a year
                year = int(date_str)
            if year is not None and 1900 <= year < 2025:  # Sanity check and exclude 2025
                year_counts[year] += 1
    
    # Count sentences from processed texts CSV if available
    total_sentences = 0
    processed_csv = ROOT / "Database" / "Database" / "CCF.media_processed_texts_annotated.csv"
    if processed_csv.exists():
        try:
            with open(processed_csv, 'r', encoding='utf-8') as f:
                total_sentences = sum(1 for _ in f) - 1  # Subtract header line
        except:
            pass
    
    return media_counts, dict(sorted(year_counts.items())), province_counts, national_count, national_media_names, lang_counts, total_sentences


def plot_articles_by_media(media_counts: Counter[str], outpath: Path, total_sentences: int = 0, top_n: int = 20) -> None:
    top = sorted(media_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [m for m, _ in top][::-1]
    values = [n for _, n in top][::-1]
    
    fig, ax = plt.subplots(figsize=(15, 9))
    
    # Create color gradient from dark to light blue
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(values)))
    
    bars = ax.barh(range(len(labels)), values, color=colors[::-1], edgecolor='#333333', linewidth=0.5)
    
    # Customize axes
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Articles', fontweight='semibold', fontsize=12)
    ax.set_ylabel('Media Outlet', fontweight='semibold', fontsize=12)
    # Title removed as requested - figure already has caption in LaTeX
    
    # Add total counts in bottom right
    total_articles = sum(media_counts.values())
    stats_text = f'n_articles={total_articles:,}\nn_sentences={total_sentences:,}'
    ax.text(0.98, 0.02, stats_text, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    # Add value labels
    xmax = max(values) if values else 0
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(v + xmax * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{v:,}', va='center', ha='left', fontsize=9, color='#333333')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlim(0, xmax * 1.1)
    
    # Add subtle grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add watermark and logo
    add_watermark_and_logo(ax, fig)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_articles_by_year(year_counts: dict[int, int], outpath: Path, total_sentences: int = 0) -> None:
    if not year_counts:
        print("Warning: No year data to plot")
        return
        
    years = sorted(y for y in year_counts.keys() if y is not None)
    values = [year_counts[y] for y in years]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line with markers
    ax.plot(years, values, 
            color='#1f77b4', 
            linewidth=2.5, 
            marker='o', 
            markersize=6, 
            markerfacecolor='white', 
            markeredgecolor='#1f77b4', 
            markeredgewidth=2,
            label='Number of Articles')
    
    # Add shaded area under the line
    ax.fill_between(years, values, alpha=0.2, color='#1f77b4')
    
    # Customize axes
    ax.set_xlabel('Year', fontweight='semibold', fontsize=12)
    ax.set_ylabel('Number of Articles', fontweight='semibold', fontsize=12)
    # Title removed as requested - figure already has caption in LaTeX
    
    # Add total counts in top right
    total_articles = sum(year_counts.values())
    stats_text = f'n_articles={total_articles:,}\nn_sentences={total_sentences:,}'
    ax.text(0.98, 0.98, stats_text, 
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    # Format axes
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    # Set x-axis limits with padding
    if years:
        ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
        
        # Ensure integer years on x-axis
        if len(years) <= 20:
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=45, ha='right')
        else:
            # For many years, let matplotlib choose sensible ticks
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=15))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Remove peak annotation - keeping variables for potential future use
    if values:
        max_val = max(values)
        min_val = min(values)
        max_year = years[values.index(max_val)]
        min_year = years[values.index(min_val)]
    
    # Add watermark and logo
    add_watermark_and_logo(ax, fig)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_articles_by_province(province_counts: Counter[str], national_count: int, national_media: list, lang_counts: dict, outpath: Path, total_sentences: int = 0) -> None:
    """Create a choropleth map of articles by Canadian province."""
    print("  [Province Map] Starting...")

    if gpd is None:
        print("Warning: geopandas not available, skipping map generation")
        return

    if not SHAPEFILE_PATH.exists():
        print(f"Warning: Shapefile not found at {SHAPEFILE_PATH}, skipping map generation")
        return

    print("  [Province Map] Loading shapefile...")
    canada_gdf = gpd.read_file(SHAPEFILE_PATH)

    print("  [Province Map] Simplifying geometries (step 1)...")
    canada_gdf['geometry'] = canada_gdf['geometry'].simplify(20.0, preserve_topology=False)

    print("  [Province Map] Dissolving by province...")
    province_name_column = 'PRENAME'
    provinces_gdf = canada_gdf.dissolve(by=province_name_column).reset_index()

    print("  [Province Map] Simplifying geometries (step 2)...")
    provinces_gdf['geometry'] = provinces_gdf['geometry'].simplify(50.0, preserve_topology=False)
    
    # Calculate total articles (including national)
    total_articles = sum(province_counts.values()) + national_count

    print("  [Province Map] Adding article counts...")
    provinces_gdf['article_count'] = provinces_gdf[province_name_column].map(province_counts).fillna(0)
    provinces_gdf['percentage'] = (provinces_gdf['article_count'] / total_articles * 100) if total_articles > 0 else 0

    print("  [Province Map] Creating figure...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot the map
    provinces_gdf.plot(
        column='percentage',
        cmap='Blues',
        linewidth=0.5,
        ax=ax,
        edgecolor='#333333',
        legend=True,
        legend_kwds={
            'label': 'Percentage of Articles (%)',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.05
        },
        missing_kwds={
            'color': 'lightgrey',
            'label': 'No data'
        }
    )
    
    # Add province labels with counts
    for idx, row in provinces_gdf.iterrows():
        if row['article_count'] > 0:
            centroid = row.geometry.centroid
            percentage = row['percentage']
            count = int(row['article_count'])
            
            # Only show labels for provinces with data
            ax.annotate(
                f"{row[province_name_column]}\n{percentage:.1f}%\n({count:,})",
                xy=(centroid.x, centroid.y),
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )
    
    # Customize the plot with lower title
    # Title removed as requested - figure already has caption in LaTeX
    ax.axis('off')
    
    # Calculate language percentages
    total_lang_articles = lang_counts.get("EN", 0) + lang_counts.get("FR", 0)
    en_pct = (lang_counts.get("EN", 0) / total_lang_articles * 100) if total_lang_articles > 0 else 0
    fr_pct = (lang_counts.get("FR", 0) / total_lang_articles * 100) if total_lang_articles > 0 else 0
    
    # Add statistics box with language counts and percentages - improved aesthetics
    stats_text = f'Articles: {total_articles:,}\nSentences: {total_sentences:,}\n\nLanguage Distribution:\nEnglish: {lang_counts.get("EN", 0):,} ({en_pct:.1f}%)\nFrench: {lang_counts.get("FR", 0):,} ({fr_pct:.1f}%)'
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='left',
            fontfamily='sans-serif',
            bbox=dict(
                boxstyle='round,pad=0.7',
                facecolor='white',
                alpha=0.95,
                edgecolor='#666666',
                linewidth=1.2
            ))
    
    # Add National articles label with media names - improved aesthetics
    if national_count > 0:
        national_percentage = (national_count / total_articles * 100) if total_articles > 0 else 0
        # Limit the number of media shown to avoid overcrowding
        media_list = national_media[:5]  # Show first 5
        if len(national_media) > 5:
            media_list.append(f'... and {len(national_media)-5} more')
        media_text = '\n'.join(media_list)
        national_text = f'National Media\n{national_percentage:.1f}% ({national_count:,} articles)\n\nOutlets:\n{media_text}'
        ax.text(0.98, 0.02, national_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right',
                fontfamily='sans-serif',
                bbox=dict(
                    boxstyle='round,pad=0.7',
                    facecolor='white',
                    alpha=0.95,
                    edgecolor='#666666',
                    linewidth=1.2
                ))
    
    plt.tight_layout()
    
    # Save as PNG directly
    plt.savefig(outpath, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    print(f"Map saved: {outpath}")


def main() -> None:
    print("=" * 60)
    print("GENERATING OVERVIEW PLOTS")
    print("=" * 60)

    print("\n[1/4] Fetching data from database...")
    media_counts, year_counts, province_counts, national_count, national_media, lang_counts, total_sentences = fetch_full_data_counts()
    print(f"  Found {len(media_counts)} media outlets, {len(year_counts)} years")
    print(f"  Total sentences: {total_sentences:,}")

    out_media = OUTDIR / "articles_by_media.png"
    out_year = OUTDIR / "articles_by_year.png"
    out_province = OUTDIR / "articles_by_province.png"

    print("\n[2/4] Creating media distribution plot...")
    plot_articles_by_media(media_counts, out_media, total_sentences)
    print(f"  Saved: {out_media}")

    print("\n[3/4] Creating year distribution plot...")
    plot_articles_by_year(year_counts, out_year, total_sentences)
    print(f"  Saved: {out_year}")

    print("\n[4/4] Creating province map...")
    plot_articles_by_province(province_counts, national_count, national_media, lang_counts, out_province, total_sentences)
    print(f"  Saved: {out_province}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
