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
2_temporal_f1_validation.py

MAIN OBJECTIVE:
---------------
Generate the temporal F1 evolution plot (Figure 9) for CCF model validation
by ACTUALLY computing pooled macro F1 scores per decade from the 1,000-sentence
expert gold standard against the deployed model predictions stored in
PostgreSQL. Nothing is hard-coded: the figure is fully reproducible end-to-end.

METHOD (exactly reproduces the canonical Table S7 protocol of
Scripts/Annotation/11_Annotation_metrics.py):
- Gold standard: Database/Training_data/manual_annotations_JSONL/all.jsonl
  (the 1,000 expert-annotated sentences with full metadata; this is the file
  Table S7 / final_annotation_metrics.csv was computed against).
- Each protocol label (e.g. "Just_Detection", "Messenger_9_SUB") is mapped to
  its canonical S3 category code (e.g. justice_frame, msg_official) through
  the exact same chain used by the normalized metrics pipeline:
  LABEL_MAPPING (11_Annotation_metrics.py) -> normalize_category
  (15_normalize_categories.py).
- Predictions: the 65 annotation columns of "CCF_processed_data"
  (database CCF_Database_texts), matched on (doc_id, sentence_id).
- NULL predictions are SKIPPED, not coerced to 0 (hierarchical annotation:
  sub-category models only run when the parent detection fired). This is the
  canonical S7 behaviour; it reproduces the published supports exactly
  (n=30,330 comparisons; 6,401 gold positives) and the published scores
  (ALL 0.866 / EN 0.869 / FR 0.864).
  [For reference: coercing NULL->0 gives 0.872; scoring the stricter
  consensus_gold_standard.jsonl label sets gives 0.793 — neither matches
  the published Table S7 protocol.]
- Pooled macro F1 = mean(F1 of class 1, F1 of class 0) computed on a single
  binary confusion matrix pooling all categories x sentences.
- Decades are taken from the article date stored in the database.

SELF-VALIDATION:
----------------
Before producing the figure, the script recomputes the GLOBAL pooled macro F1
and requires it to match the canonical Table S7 OVERALL value (0.866 +/- 0.005).
If validation fails, no figure is produced.

Dependencies:
-------------
- matplotlib
- numpy
- psycopg2

Author:
-------
Antoine Lemor
"""

import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import psycopg2

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]          # repo root
ANNOTATION_DIR = PROJECT_ROOT / "Scripts" / "Annotation"
GOLD_JSONL = (PROJECT_ROOT / "Database" / "Training_data" /
              "manual_annotations_JSONL" / "all.jsonl")
FIG_OUT_RESULTS = SCRIPT_DIR.parent / "Outputs" / "Figures" / "temporal_f1_evolution.png"
FIG_OUT_LATEX = SCRIPT_DIR.parents[1] / "Latex" / "Figures" / "temporal_f1_evolution.png"

# Canonical Table S7 OVERALL pooled macro F1 (final_annotation_metrics.csv, ALL/ALL)
CANONICAL_OVERALL_F1 = 0.866
VALIDATION_TOLERANCE = 0.005

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Connect through the local Unix socket as the current OS user by default
# (peer authentication); env variables override when set.
DB_PARAMS = {
    "dbname": os.getenv("CCF_DB_NAME", "CCF_Database_texts"),
    "user": os.getenv("CCF_DB_USER", os.getenv("USER", "postgres")),
}
if os.getenv("CCF_DB_HOST"):
    DB_PARAMS["host"] = os.getenv("CCF_DB_HOST")
    DB_PARAMS["port"] = int(os.getenv("CCF_DB_PORT", 5432))
if os.getenv("CCF_DB_PASS"):
    DB_PARAMS["password"] = os.getenv("CCF_DB_PASS")
TABLE_NAME = "CCF_processed_data"

# =============================================================================
# LABEL MAPPING (imported from the canonical scripts — single source of truth)
# =============================================================================
sys.path.insert(0, str(ANNOTATION_DIR))
_norm = importlib.import_module("15_normalize_categories")     # ALL_CATEGORIES, normalize_category
_metrics = importlib.import_module("11_Annotation_metrics")    # LABEL_MAPPING

ALL_CODES = [code for _name, code in _norm.ALL_CATEGORIES.values()]  # 65 canonical codes


def gold_label_to_code(protocol_label: str):
    """Map a gold protocol label (e.g. 'Just_Detection') to its canonical
    S3 category code (e.g. 'justice_frame'), exactly as the normalized
    metrics pipeline does."""
    readable = _metrics.LABEL_MAPPING.get(protocol_label, protocol_label)
    num, _official, code = _norm.normalize_category(readable)
    if num is None or code not in ALL_CODES:
        return None
    return code


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gold(path: Path):
    """Load the gold JSONL; return a list of dicts with matched keys and the
    set of canonical category codes annotated as positive."""
    entries = []
    with path.open("r", encoding="utf-8") as fo:
        for line in fo:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            codes = set()
            for lab in rec.get("label", []):
                code = gold_label_to_code(lab)
                if code is not None:
                    codes.add(code)
            entries.append({
                "doc_id": meta.get("doc_id"),
                "sentence_id": meta.get("sentence_id"),
                "language": meta.get("language"),
                "gold": codes,
            })
    return entries


def fetch_predictions(entries):
    """Fetch date + the 65 annotation columns for every gold sentence.
    Returns {(doc_id, sentence_id): (language, date, {code: value_or_None})}."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cols = ", ".join(f'"{c}"' for c in ALL_CODES)
    keys = tuple((e["doc_id"], e["sentence_id"]) for e in entries)
    cur.execute(
        f'SELECT doc_id, sentence_id, language, date, {cols} '
        f'FROM "{TABLE_NAME}" WHERE (doc_id, sentence_id) IN %s',
        (keys,),
    )
    rows = {}
    for r in cur.fetchall():
        rows[(r[0], r[1])] = (r[2], r[3], dict(zip(ALL_CODES, r[4:])))
    cur.close()
    conn.close()
    return rows


# =============================================================================
# POOLED MACRO F1
# =============================================================================

def _prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def pooled_macro_f1(pairs):
    """Pooled macro F1 over (gold, pred) binary pairs: single confusion
    matrix across all categories x sentences; macro = mean(F1_class1, F1_class0)."""
    tp = fp = fn = tn = 0
    for g, p in pairs:
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    f1_pos = _prf1(tp, fp, fn)
    # Class 0 viewed as the positive class: TP=tn, FP=fn, FN=fp
    f1_neg = _prf1(tn, fn, fp)
    n = tp + fp + fn + tn
    return 0.5 * (f1_pos + f1_neg), n, tp + fn


def collect_pairs(entries, rows):
    """Yield (decade, language, gold, pred) for every scored comparison.
    NULL predictions are skipped (canonical S7 behaviour)."""
    out = []
    for e in entries:
        key = (e["doc_id"], e["sentence_id"])
        if key not in rows:
            continue
        lang, date, preds = rows[key]
        decade = f"{(date.year // 10) * 10}s" if date is not None else None
        for code in ALL_CODES:
            v = preds[code]
            if v is None:
                continue
            g = 1 if code in e["gold"] else 0
            out.append((decade, lang, g, int(v)))
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"[INFO] Loading gold standard: {GOLD_JSONL}")
    entries = load_gold(GOLD_JSONL)
    print(f"[INFO] {len(entries):,} gold sentences loaded.")

    print(f"[INFO] Fetching predictions from {DB_PARAMS['dbname']}.\"{TABLE_NAME}\"…")
    rows = fetch_predictions(entries)
    print(f"[INFO] {len(rows):,} sentences matched in the database.")

    pairs = collect_pairs(entries, rows)

    # -------------------------------------------------------------------------
    # SELF-VALIDATION against the canonical Table S7 OVERALL value
    # -------------------------------------------------------------------------
    overall_f1, n_all, sup1_all = pooled_macro_f1([(g, p) for _, _, g, p in pairs])
    print(f"[INFO] Global pooled macro F1 = {overall_f1:.4f} "
          f"(n={n_all:,} comparisons, gold positives={sup1_all:,})")
    if abs(overall_f1 - CANONICAL_OVERALL_F1) > VALIDATION_TOLERANCE:
        raise SystemExit(
            f"[FATAL] Validation failed: global pooled macro F1 {overall_f1:.4f} "
            f"does not reproduce the canonical Table S7 OVERALL value "
            f"{CANONICAL_OVERALL_F1} (±{VALIDATION_TOLERANCE}). Figure NOT produced."
        )
    print(f"[INFO] Validation OK: reproduces Table S7 OVERALL "
          f"({CANONICAL_OVERALL_F1} ±{VALIDATION_TOLERANCE}).")

    # -------------------------------------------------------------------------
    # PER-DECADE POOLED MACRO F1 (ALL / EN / FR)
    # -------------------------------------------------------------------------
    time_periods = ["1980s", "1990s", "2000s", "2010s", "2020s"]

    def series(lang_filter):
        vals = []
        for dec in time_periods:
            sel = [(g, p) for d, l, g, p in pairs
                   if d == dec and (lang_filter is None or l == lang_filter)]
            if not sel:
                vals.append(np.nan)
                continue
            f1, _, _ = pooled_macro_f1(sel)
            vals.append(f1)
        return vals

    f1_all = series(None)
    f1_en = series("EN")
    f1_fr = series("FR")

    print("\n[INFO] Pooled macro F1 by decade:")
    print(f"  {'decade':<8}{'ALL':>8}{'EN':>8}{'FR':>8}")
    for i, dec in enumerate(time_periods):
        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "  —  "
        print(f"  {dec:<8}{fmt(f1_all[i]):>8}{fmt(f1_en[i]):>8}{fmt(f1_fr[i]):>8}")

    valid_all = [v for v in f1_all if not np.isnan(v)]
    delta_max = max(valid_all) - min(valid_all)
    print(f"\n[INFO] Max ΔF1 across decades (ALL): {delta_max:.3f} ({delta_max*100:.1f} pts)")

    # -------------------------------------------------------------------------
    # FIGURE (same style/outputs as before)
    # -------------------------------------------------------------------------
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

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_periods, f1_all,
            color='#1f77b4',
            linewidth=2.5,
            marker='o',
            markersize=8,
            markerfacecolor='#1f77b4',
            markeredgecolor='white',
            markeredgewidth=2,
            label='Overall (EN+FR)')

    ax.plot(time_periods, f1_en,
            color='#ff7f0e',
            linewidth=2,
            marker='s',
            markersize=7,
            markerfacecolor='#ff7f0e',
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='English',
            alpha=0.8)

    ax.plot(time_periods, f1_fr,
            color='#2ca02c',
            linewidth=2,
            marker='^',
            markersize=7,
            markerfacecolor='#2ca02c',
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='French',
            alpha=0.8)

    # Horizontal reference line at the recomputed global validation F1
    ax.axhline(y=overall_f1, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
               label=f'Final Validation (F1={overall_f1:.3f})')

    ax.set_xlabel('Decade', fontweight='semibold', fontsize=12)
    ax.set_ylabel('F1 Macro Score', fontweight='semibold', fontsize=12)

    # y-limits framed around the observed values
    all_vals = [v for s in (f1_all, f1_en, f1_fr) for v in s if not np.isnan(v)]
    ymin = min(all_vals + [overall_f1]) - 0.010
    ymax = max(all_vals + [overall_f1]) + 0.010
    ax.set_ylim(ymin, ymax)

    # Value labels on points
    for i in range(len(time_periods)):
        if not np.isnan(f1_all[i]):
            ax.text(i, f1_all[i] + 0.002, f'{f1_all[i]:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        if not np.isnan(f1_en[i]):
            # Avoid overprinting the Overall label when the two values coincide:
            # stack the English label above it (French labels sit below points)
            if not np.isnan(f1_all[i]) and abs(f1_en[i] - f1_all[i]) < 0.002:
                y_en = max(f1_en[i], f1_all[i]) + 0.0045
            else:
                y_en = f1_en[i] + 0.002
            ax.text(i, y_en, f'{f1_en[i]:.3f}', ha='center', va='bottom',
                    fontsize=8, color='#ff7f0e')
        if not np.isnan(f1_fr[i]):
            ax.text(i, f1_fr[i] - 0.002, f'{f1_fr[i]:.3f}', ha='center', va='top',
                    fontsize=8, color='#2ca02c')

    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Statistics box (computed, not asserted). Report BOTH spreads
    # explicitly: the drift over the well-supported decades
    # (1990s-2020s) and the larger spread once the sparsely sampled
    # 1980s decade is included, as discussed in the paper caption.
    n_1980s = sum(1 for e in entries
                  if (e["doc_id"], e["sentence_id"]) in rows
                  and rows[(e["doc_id"], e["sentence_id"])][1] is not None
                  and rows[(e["doc_id"], e["sentence_id"])][1].year // 10 * 10 == 1980)
    valid_ex80 = [v for dec, v in zip(time_periods, f1_all)
                  if dec != "1980s" and not np.isnan(v)]
    delta_ex80 = max(valid_ex80) - min(valid_ex80)
    stats_text = (f'Temporal Drift Analysis\n'
                  f'ΔF1 = {delta_ex80*100:.1f} pts (1990s–2020s);\n'
                  f'{delta_max*100:.1f} pts incl. 1980s (n={n_1980s} sentences, EN only)')
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                      edgecolor='gray', linewidth=1.5))

    plt.tight_layout()

    for outpath in (FIG_OUT_RESULTS, FIG_OUT_LATEX):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close(fig)


if __name__ == "__main__":
    main()
