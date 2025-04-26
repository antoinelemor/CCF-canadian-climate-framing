"""
Project:
--------
CCF-canadian-climate-framing

Title:
------
4_database_graph.py

Main Objective:
---------------
Creates three scientific and aesthetic plots (with all labels in English):
1) Total articles per year across the entire database.
2) Number of articles per media outlet.
3) Number of articles grouped by region (based on a separate CSV that maps media outlets to regions).

The plots are saved in the "Database/Database" directory.

Author:
-------
Antoine Lemor 
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns

# Apply a scientific and aesthetic theme using Seaborn.
# --- Modification: Changed font to 'DejaVu Serif' for availability,
# and set axes background to white for clear grid visibility.
sns.set_theme(
    style="whitegrid",   
    context="talk",      
    font="DejaVu Serif", 
    rc={
        "axes.edgecolor": "dimgray",
        "axes.facecolor": "white",      
        "figure.facecolor": "white",    
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,          
    }
)

# Further update parameters for visual coherence
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 12,
    "figure.titlesize": 12,
})

def main():
    """
    Main function that generates three improved plots based on the media database:

    1. Total articles per year:
       - Filters out rows with unknown dates.
       - Converts date strings into datetime objects and extracts the year as an integer.
       - Ensures all years in the range (min to max) are present, even if some have zero articles.
       - Groups articles by year and creates a vertical bar chart with wider bars.
       - The x-axis tick labels have a slightly reduced font size.
       - Bar annotations are rotated vertically and positioned with a small offset.
       - The overall margins (both internal and external) are reduced.
       - Displays the total count in the top-right corner as 'n='.
       - Adds a legend text in the bottom-left corner: 'Canadian-climate-framing (CCF) Project'.

    2. Articles per media:
       - Groups the articles by media outlet and counts them.
       - Creates a vertical bar chart using the same color scheme.
       - Bar annotations are rotated vertically and positioned with a small offset inside the plot.
       - The y-limit is set with a smaller expansion factor so that a margin exists inside the axes.
       - External margins are reduced so that the plot fills more of the image.
       - Displays the total count in the top-right corner as 'n='.
       - Adds a legend text in the bottom-left corner: 'Canadian-climate-framing (CCF) Project'.

    3. Articles per region:
       - Reads a separate CSV ('Canadian_Media_Articles_by_Province.csv') that maps each media to a region.
       - Merges the main database with this region mapping on 'media'.
       - Groups the merged data by 'region' and counts the total number of articles per region.
       - Creates a vertical bar chart following the same style and margins as the media plot.
       - Bar annotations are rotated vertically and positioned with a small offset.
       - Displays the total count in the top-right corner as 'n='.
       - Adds a legend text in the bottom-left corner: 'Canadian-climate-framing (CCF) Project'.

    All plots are saved as PNG files in the designated database directory.
    """

    # Color used for the bars in all plots
    bar_color = "cornflowerblue"

    # Define relative paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    db_dir = base_dir / "Database" / "Database"

    # Path to the main database CSV
    main_db_path = db_dir / "CCF.media_database.csv"

    # Path to the region CSV (maps media to region)
    region_db_path = db_dir / "Canadian_Media_Articles_by_Province.csv"

    # Read the main database
    tqdm.write("[INFO] Reading main database...")
    df = pd.read_csv(main_db_path)

    # -------------------------------
    # Plot 1: Total articles per year
    # -------------------------------
    tqdm.write("[INFO] Processing: Total articles per year")
    df_valid = df[df["date"] != "Inconnu"].copy()

    # Convert date strings to datetime objects and extract the year
    df_valid["year"] = pd.to_datetime(
        df_valid["date"],
        format="%m-%d-%Y",
        errors="coerce"
    ).dt.year

    # Determine the full range of years (min to max) to include all years
    min_year = int(df_valid["year"].min())
    max_year = int(df_valid["year"].max())

    # Group by year (reindexed to ensure missing years have zero articles)
    all_years = range(min_year, max_year + 1)
    articles_per_year = df_valid.groupby("year").size().reindex(all_years, fill_value=0)

    # Increase the figure width to provide more space for the x-axis
    plt.figure(figsize=(12, 8))
    years = articles_per_year.index
    values_year = articles_per_year.values

    # --- Modification for total articles per year:
    # Increase bar width to 0.8 (to enlarge the already small columns)
    bars_year = plt.bar(years, values_year, color=bar_color, edgecolor="black", width=0.8)

    # Title, axes labels, and ticks (x-tick labels with reduced font size)
    plt.title("Total articles per year talking about climate in the CCF Database")
    plt.xlabel("Year")
    plt.ylabel("Number of articles")
    plt.xticks(years, rotation=45, ha="right", fontsize=9)

    # Set y-axis limit with a modest margin and reduce horizontal margins
    plt.ylim(0, values_year.max() * 1.15)
    plt.xlim(min(years) - 1, max(years) + 1)

    # Annotate each bar vertically above the bar with a slight offset
    for rect, count in zip(bars_year, values_year):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        plt.text(x, y + (0.02 * values_year.max()), f"{int(count)}",
                 rotation=90, ha="center", va="bottom", color="black", fontsize=10)

    # Annotate the total count in the top-right corner
    total_articles_year = int(values_year.sum())
    ax_year = plt.gca()
    ax_year.text(
        0.98, 0.98, f"n={total_articles_year}",
        transform=ax_year.transAxes,
        ha="right", va="top", fontsize=9, color="black"
    )

    # Add legend text at the bottom-left corner
    ax_year.text(
        0.0, -0.16, "Canadian-climate-framing (CCF) Project",
        transform=ax_year.transAxes,
        ha="left", va="top", fontsize=9
    )

    # Reduce the external border spacing and increase the bottom margin for x-axis labels
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.25)

    # Save the figure for total articles per year
    year_plot_path = db_dir / "articles_per_year.png"
    plt.savefig(year_plot_path, dpi=300)
    tqdm.write(f"[INFO] 'Articles per year' plot saved to: {year_plot_path}")
    plt.close()

    # -------------------------------
    # Plot 2: Articles per media
    # -------------------------------
    tqdm.write("[INFO] Processing: Articles per media")
    articles_per_media = df["media"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    media_indices = articles_per_media.index
    values_media = articles_per_media.values

    bars_media = plt.bar(
        media_indices,
        values_media,
        color=bar_color,
        edgecolor="black",
        width=0.7
    )

    plt.title("Articles per media")
    plt.xlabel("Media")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, values_media.max() * 1.15)

    # Annotate each media bar with its count (vertical placement)
    for rect, count in zip(bars_media, values_media):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        plt.text(x, y + (0.02 * values_media.max()), f"{int(count)}",
                 rotation=90, ha="center", va="bottom", color="black", fontsize=10)

    # Annotate the total count in the top-right corner
    total_articles_media = int(values_media.sum())
    ax_media = plt.gca()
    ax_media.text(
        0.98, 0.98, f"n={total_articles_media}",
        transform=ax_media.transAxes,
        ha="right", va="top", fontsize=10, color="black"
    )

    # Add legend text at the bottom-left corner
    ax_media.text(
        0.0, -0.38, "Canadian-climate-framing (CCF) Project",
        transform=ax_media.transAxes,
        ha="left", va="top", fontsize=9
    )

    # Adjust margins to make sure media names are fully visible
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.30)

    # Save the figure for articles per media
    media_plot_path = db_dir / "articles_per_media.png"
    plt.savefig(media_plot_path, dpi=300)
    tqdm.write(f"[INFO] 'Articles per media' plot saved to: {media_plot_path}")
    plt.close()

    # -------------------------------
    # Plot 3: Articles per region
    # -------------------------------
    tqdm.write("[INFO] Processing: Articles per region")

    # Read the CSV mapping media to region
    df_region_map = pd.read_csv(region_db_path)

    # Merge with main DB to attach 'region' info to each article
    df_merged = pd.merge(df, df_region_map, on="media", how="left")
    df_merged["region"] = df_merged["region"].fillna("Unknown")

    # Group by region and count the articles
    articles_per_region = df_merged.groupby("region").size().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    region_indices = articles_per_region.index
    values_region = articles_per_region.values

    bars_region = plt.bar(
        region_indices,
        values_region,
        color=bar_color,
        edgecolor="black",
        width=0.7
    )

    plt.title("Articles per region")
    plt.xlabel("Region")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, values_region.max() * 1.15)

    # Annotate each region bar with its count (vertical placement)
    for rect, count in zip(bars_region, values_region):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        plt.text(x, y + (0.02 * values_region.max()), f"{int(count)}",
                 rotation=90, ha="center", va="bottom", color="black", fontsize=10)

    # Annotate the total count in the top-right corner
    total_articles_region = int(values_region.sum())
    ax_region = plt.gca()
    ax_region.text(
        0.98, 0.98, f"n={total_articles_region}",
        transform=ax_region.transAxes,
        ha="right", va="top", fontsize=10, color="black"
    )

    # Add legend text at the bottom-left corner
    ax_region.text(
        0.0, -0.12, "Canadian-climate-framing (CCF) Project",
        transform=ax_region.transAxes,
        ha="left", va="top", fontsize=9
    )

    # Reduce external margins so the graph occupies more of the image area
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

    # Save the figure for articles per region
    region_plot_path = db_dir / "articles_per_region.png"
    plt.savefig(region_plot_path, dpi=300)
    tqdm.write(f"[INFO] 'Articles per region' plot saved to: {region_plot_path}")
    plt.close()

    tqdm.write("[INFO] All plots created and saved successfully.")


if __name__ == "__main__":
    main()