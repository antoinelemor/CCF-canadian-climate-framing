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
3_data_overview.py

MAIN OBJECTIVE:
---------------
Generate the descriptive artefacts that populate the Data Overview
section of the methodology paper. The script is intentionally
descriptive (not analytical): it only summarises what is in the
deposited CCF Database, with no causal or inferential modelling.

It produces:
  1) A heatmap of the average per-article share of each of the eight
     main frames, year by year (1990--2024).
  2) Three LaTeX tables of the top-10 named entities by type
     (PER / ORG / LOC), generated from CCF_article_entities.
  3) A LaTeX table that summarises article-level descriptive
     statistics (words per article, sentences per article, share of
     each main frame, share of each primary category, top
     frame distribution).

All outputs are written to:
  paper/CCF_Methodology/Results/Outputs/Figures/data_overview_frames_heatmap.png
  paper/CCF_Methodology/Results/Outputs/Tables/table_data_overview_descriptives.tex
  paper/CCF_Methodology/Results/Outputs/Tables/table_data_overview_entities.tex

Author:
-------
Antoine Lemor
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import psycopg2


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "Results" / "Outputs" / "Figures"
TAB_DIR = ROOT / "Results" / "Outputs" / "Tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

DB_PARAMS = {
    "host":     os.getenv("CCF_DB_HOST", "localhost"),
    "port":     int(os.getenv("CCF_DB_PORT", 5432)),
    "dbname":   os.getenv("CCF_DB_NAME", "CCF_Database"),
    "user":     os.getenv("CCF_DB_USER", "antoine"),
    "password": os.getenv("CCF_DB_PASS", ""),
}

THEMATIC_FRAMES = [
    ("political_frame",      "Political"),
    ("economic_frame",       "Economic"),
    ("scientific_frame",     "Scientific"),
    ("environmental_frame",  "Environmental"),
    ("justice_frame",        "Justice"),
    ("cultural_frame",       "Cultural"),
    ("health_frame",         "Health"),
    ("security_frame",       "Security"),
]


def _connect():
    return psycopg2.connect(**DB_PARAMS)


def _style():
    plt.style.use("default")
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size":          10,
        "axes.labelsize":     11,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    9,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#333333",
        "figure.dpi":         100,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
    })


FRAME_COLOURS = {
    "political_frame":     "#1a2742",  # midnight blue
    "economic_frame":      "#b8860b",  # dark goldenrod
    "scientific_frame":    "#2e7d32",  # green 800
    "environmental_frame": "#558b2f",  # light green 800
    "justice_frame":       "#6a1b9a",  # purple 800
    "cultural_frame":      "#c2185b",  # pink 700
    "health_frame":        "#c62828",  # red 800
    "security_frame":      "#5d4037",  # brown 700
}


def build_frames_lines(start_year: int = 1980, end_year: int = 2024) -> Path:
    """Line plot of mean per-article frame share by year."""
    cols = ", ".join(f"AVG(a.prop_{code})::numeric AS {code}"
                     for code, _ in THEMATIC_FRAMES)
    sql = (
        f'SELECT EXTRACT(YEAR FROM f.date)::INT AS yr, {cols} '
        f'FROM "CCF_article_aggregates" a '
        f'JOIN "CCF_full_data" f USING (doc_id) '
        f'WHERE EXTRACT(YEAR FROM f.date) BETWEEN {start_year} AND {end_year} '
        f'GROUP BY yr ORDER BY yr;'
    )
    with _connect() as conn:
        df = pd.read_sql(sql, conn)

    df = df.set_index("yr").astype(float) * 100.0
    years = df.index.values

    _style()
    fig, ax = plt.subplots(figsize=(11, 4.6))

    # Plot from most prevalent to least so the legend stacks naturally.
    order = sorted(
        ((code, label) for code, label in THEMATIC_FRAMES),
        key=lambda cl: -float(df[cl[0]].mean()),
    )
    for code, label in order:
        ax.plot(years, df[code].values,
                color=FRAME_COLOURS[code], linewidth=1.8,
                marker="o", markersize=3.2, label=label)

    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean per-article share of sentences (%)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # x-ticks every 5 years to avoid clutter.
    xticks = [y for y in years if y % 5 == 0]
    if years[-1] not in xticks:
        xticks.append(int(years[-1]))
    ax.set_xticks(xticks)

    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.0,
    )
    for line in leg.get_lines():
        line.set_linewidth(2.2)

    out = FIG_DIR / "data_overview_frames_lines.png"
    fig.savefig(out, facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return out


def _escape_latex(text: str) -> str:
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for a, b in replacements:
        text = text.replace(a, b)
    return text


def build_entities_table(top_k: int = 10) -> Path:
    """Top-K named entities by type from CCF_article_entities."""
    sql = (
        'SELECT entity, COUNT(*) AS n '
        'FROM (SELECT jsonb_array_elements_text({col}) AS entity '
        '      FROM "CCF_article_entities") s '
        'WHERE entity IS NOT NULL AND LENGTH(entity) >= 2 '
        "AND entity !~ '^[[:punct:][:space:]]+$' "
        'GROUP BY entity '
        'ORDER BY n DESC LIMIT {k};'
    )

    out_rows = {}
    with _connect() as conn:
        for col, key in (("entities_per", "PER"),
                         ("entities_org", "ORG"),
                         ("entities_loc", "LOC")):
            df = pd.read_sql(sql.format(col=col, k=top_k), conn)
            out_rows[key] = df

    lines = [
        r"% Auto-generated by Results/Scripts/3_data_overview.py — do not edit by hand.",
        r"\begin{tabular}{r l r  l r  l r}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Persons (PER)}} & "
        r"\multicolumn{2}{c}{\textbf{Organisations (ORG)}} & "
        r"\multicolumn{2}{c}{\textbf{Locations (LOC)}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\textbf{\#} & \textbf{Entity} & \textbf{Articles} & "
        r"\textbf{Entity} & \textbf{Articles} & "
        r"\textbf{Entity} & \textbf{Articles} \\",
        r"\midrule",
    ]
    n = min(top_k, *(len(out_rows[k]) for k in out_rows))
    for i in range(n):
        per = out_rows["PER"].iloc[i]
        org = out_rows["ORG"].iloc[i]
        loc = out_rows["LOC"].iloc[i]
        lines.append(
            f"{i + 1} & {_escape_latex(per['entity'])} & {int(per['n']):,} & "
            f"{_escape_latex(org['entity'])} & {int(org['n']):,} & "
            f"{_escape_latex(loc['entity'])} & {int(loc['n']):,} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = TAB_DIR / "table_data_overview_entities.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_descriptives_table() -> Path:
    """Article-level descriptive statistics, compact two-block layout.

    Left block: main frames (mean per-article share + top-frame %).
    Right block: primary categories and tone (mean per-article share).
    A short header line above carries the corpus-size figures.
    """
    with _connect() as conn:
        meta = pd.read_sql(
            'SELECT COUNT(*)::BIGINT          AS n_articles, '
            '       ROUND(AVG(words_count))   AS mean_words, '
            '       ROUND(STDDEV(words_count)) AS sd_words, '
            '       percentile_cont(0.5) WITHIN GROUP (ORDER BY words_count) AS median_words '
            'FROM "CCF_full_data";', conn).iloc[0]

        n_sentences = pd.read_sql(
            'SELECT COUNT(*) AS n FROM "CCF_processed_data";', conn
        ).iloc[0]["n"]

        agg_cols = (
            [f"prop_{code}" for code, _ in THEMATIC_FRAMES]
            + ["prop_messenger", "prop_event", "prop_solution",
               "prop_canada", "prop_urgency",
               "prop_tone_positive", "prop_tone_negative", "prop_tone_neutral"]
        )
        sel = ", ".join(f"AVG({c})::numeric AS {c}" for c in agg_cols)
        agg_means = pd.read_sql(
            f'SELECT {sel} FROM "CCF_article_aggregates";', conn
        ).iloc[0]

        dom = pd.read_sql(
            'SELECT top_frame, COUNT(*)::BIGINT AS n '
            'FROM "CCF_article_aggregates" '
            'WHERE top_frame IS NOT NULL '
            'GROUP BY top_frame;', conn)

    total = int(meta["n_articles"])

    dom_pct = {row["top_frame"]: 100.0 * int(row["n"]) / total
               for _, row in dom.iterrows()}

    frame_rows = []
    for code, label in THEMATIC_FRAMES:
        share = float(agg_means[f"prop_{code}"]) * 100.0
        d = dom_pct.get(code, 0.0)
        frame_rows.append((f"{label} frame", share, d))

    primary_rows = [
        ("Messenger present",  float(agg_means["prop_messenger"])      * 100),
        ("Event present",      float(agg_means["prop_event"])          * 100),
        ("Solution present",   float(agg_means["prop_solution"])       * 100),
        ("Canadian context",   float(agg_means["prop_canada"])         * 100),
        ("Urgency to act",     float(agg_means["prop_urgency"])        * 100),
        ("Positive tone",      float(agg_means["prop_tone_positive"])  * 100),
        ("Negative tone",      float(agg_means["prop_tone_negative"])  * 100),
        ("Neutral tone",       float(agg_means["prop_tone_neutral"])   * 100),
    ]

    # Pad shorter column with empty cells so the table is rectangular.
    n_rows = max(len(frame_rows), len(primary_rows))
    while len(frame_rows) < n_rows:
        frame_rows.append(("", None, None))
    while len(primary_rows) < n_rows:
        primary_rows.append(("", None))

    corpus_line_1 = (
        rf"\textit{{Corpus:}} {total:,} articles, "
        rf"{int(n_sentences):,} two-sentence units;"
    )
    corpus_line_2 = (
        rf"{int(meta['mean_words']):,} words/article on average "
        rf"(SD $=$ {int(meta['sd_words']):,}; "
        rf"median $=$ {int(meta['median_words']):,})."
    )

    lines = [
        r"% Auto-generated by Results/Scripts/3_data_overview.py — do not edit by hand.",
        r"\begin{tabular}{l r r @{\hspace{1.3em}} l r}",
        r"\toprule",
        rf"\multicolumn{{5}}{{l}}{{{corpus_line_1}}} \\",
        rf"\multicolumn{{5}}{{l}}{{{corpus_line_2}}} \\",
        r"\addlinespace[2pt]",
        r"\multicolumn{3}{c}{\textbf{Main frames}} & "
        r"\multicolumn{2}{c}{\textbf{Primary categories \& tone}} \\",
        r"\cmidrule(lr){1-3}\cmidrule(lr){4-5}",
        r" & \textbf{Mean} & \textbf{Top} & & \textbf{Mean} \\",
        r" & \textbf{share (\%)} & \textbf{frame (\%)} & & \textbf{share (\%)} \\",
        r"\midrule",
    ]

    for (flabel, fshare, fdom), (plabel, pshare) in zip(frame_rows, primary_rows):
        left = (f"{flabel} & {fshare:.1f} & {fdom:.1f}"
                if flabel else r" & & ")
        right = (f"{plabel} & {pshare:.1f}"
                 if plabel else r" & ")
        lines.append(f"{left} & {right} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    out = TAB_DIR / "table_data_overview_descriptives.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    print(f"[3/3] Data overview artefacts → {FIG_DIR} and {TAB_DIR}")
    fig = build_frames_lines()
    print(f"  • figure written: {fig.name}")
    t1 = build_descriptives_table()
    print(f"  • table written: {t1.name}")
    t2 = build_entities_table()
    print(f"  • table written: {t2.name}")
    print("Done.")


if __name__ == "__main__":
    main()
