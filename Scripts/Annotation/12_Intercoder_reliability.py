#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
12_Intercoder_reliability.py

MAIN OBJECTIVE:
---------------
This script computes comprehensive inter-coder reliability metrics by comparing
annotations from two independent coders: the primary coder (all.jsonl with
technical English labels) and a second coder (second_coder.jsonl with French
labels). The script produces detailed reliability statistics including Cohen's
Kappa, Krippendorff's Alpha, Gwet's AC, and other standard agreement measures
for all annotation categories. Additionally, it tracks the second coder's
learning progression by computing metrics in cumulative 100-label increments,
analyzes reliability separately for sentences before and after #600 (where
after 600 the annotation was completely blind without discussions), and
creates a consensus-based gold standard by reconciling both coders' judgments.

Dependencies:
-------------
- json
- csv
- os
- pathlib.Path
- collections.Counter
- typing (Any, Dict, List, Tuple, Set)
- numpy ≥ 1.20
- scikit-learn ≥ 1.0
- tqdm ≥ 4.65
- krippendorff (install via: pip install krippendorff)

MAIN FEATURES:
--------------
1) Bilingual label mapping – Maps French labels from second_coder.jsonl to
   technical English labels using label_mapping_second_coder.csv
2) Comprehensive reliability metrics – Computes Cohen's Kappa, Krippendorff's
   Alpha, Gwet's AC1, Scott's Pi, percent agreement, and F1-based agreement
   for binary classification per label
3) Global and per-label analysis – Calculates both overall agreement across
   all labels and category-specific reliability statistics
4) Learning curve detection – Tracks second coder's progression by computing
   cumulative agreement metrics in 100-label increments
5) Before/After 600 analysis – Separately analyzes reliability for sentences
   before and after #600 (with meetings vs. completely blind annotation)
6) Consensus gold standard creation – Generates enhanced gold standard using
   multiple reconciliation strategies (consensus, union, intersection)
7) Publication-ready CSV export – Outputs comprehensive, well-formatted tables
   with all metrics, organized by analysis type and category

Author:
-------
[Anonymous]
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm.auto import tqdm

MAX_WORKERS = min(32, (os.cpu_count() or 1))

# Import Krippendorff's Alpha (requires: pip install krippendorff)
try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    print("[WARNING] krippendorff package not found. Install with: pip install krippendorff")

# Adjust this BASE_DIR according to your directory structure
BASE_DIR = Path(__file__).resolve().parent

# Input paths
MANUAL_DIR = (BASE_DIR / ".." / ".." / "Database" / "Training_data" /
              "manual_annotations_JSONL").resolve()
PRIMARY_CODER_JSONL = MANUAL_DIR / "all.jsonl"
SECOND_CODER_JSONL = MANUAL_DIR / "second_coder.jsonl"
LABEL_MAPPING_CSV = MANUAL_DIR / "label_mapping_second_coder.csv"

# Output paths
OUTPUT_CSV = MANUAL_DIR / "intercoder_reliability.csv"
CONSENSUS_GOLD_JSONL = MANUAL_DIR / "consensus_gold_standard.jsonl"

# Columns we never treat as "annotation"
NON_ANNOT_COLS: Set[str] = {
    "language", "sentences", "id_article", "Unnamed: 0", "doc_id",
    "sentence_id", "words_count_updated", "words_count",
    "news_type", "title", "author", "media", "date", "page_number", "ner_entities",
}

##############################################################################
#                          HELPER UTILITIES                                  #
##############################################################################

def load_label_mapping(path: Path) -> Dict[str, str]:
    """
    Loads the label mapping CSV and returns a dict:
    {french_label: technical_english_label}
    """
    mapping = {}
    with path.open("r", encoding="utf-8") as fo:
        reader = csv.DictReader(fo)
        for row in reader:
            mapping[row["second_coder_label_fr"]] = row["technical_label_en"]
    return mapping


def load_jsonl_annotations(path: Path, label_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Loads a JSONL file with manual annotations.

    Parameters
    ----------
    path : Path
        Path to the JSONL file
    label_mapping : Dict[str, str], optional
        If provided, maps labels using this dictionary (for second coder)

    Returns
    -------
    List of dicts with keys: doc_id, sentence_id, language, gold_labels (set)
    """
    entries = []
    with path.open("r", encoding="utf-8") as fo:
        for ln in fo:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            meta = rec.get("meta", {})
            doc_id = meta.get("doc_id")
            sentence_id = meta.get("sentence_id")
            language = meta.get("language")

            # Get labels
            raw_labels = rec.get("label", [])

            # Map labels if mapping provided (for second coder)
            if label_mapping:
                mapped_labels = set()
                for lbl in raw_labels:
                    if lbl in label_mapping:
                        mapped_labels.add(label_mapping[lbl])
                    else:
                        print(f"[WARNING] Label '{lbl}' not in mapping, skipping")
                gold_labels = mapped_labels
            else:
                gold_labels = set(raw_labels)

            entries.append({
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "language": language,
                "gold_labels": gold_labels
            })
    return entries


def percent_agreement(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Calculate simple percent agreement between two arrays."""
    if len(labels1) != len(labels2):
        raise ValueError("Arrays must have same length")
    if len(labels1) == 0:
        return 0.0
    return np.mean(labels1 == labels2)


def scotts_pi(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Calculate Scott's Pi, which accounts for chance agreement.
    Similar to Cohen's Kappa but assumes both coders have same distribution.
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0

    # Observed agreement
    p_o = np.mean(labels1 == labels2)

    # Expected agreement (marginal probabilities averaged)
    n = len(labels1)
    all_labels = np.concatenate([labels1, labels2])
    unique_labels = np.unique(all_labels)

    p_e = 0.0
    for label in unique_labels:
        p_joint = (np.sum(labels1 == label) + np.sum(labels2 == label)) / (2 * n)
        p_e += p_joint ** 2

    # Scott's Pi
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def gwet_ac1(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Calculate Gwet's AC1 coefficient.
    More stable than Kappa, especially with high agreement or skewed distributions.
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0

    # Observed agreement
    p_o = np.mean(labels1 == labels2)

    # Expected agreement by chance (uniform distribution assumption)
    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    q = len(unique_labels)

    if q <= 1:
        return 1.0

    # Gwet's chance agreement formula
    p_e = 1.0 / q

    # AC1 coefficient
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def calculate_krippendorff_alpha(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Calculate Krippendorff's Alpha for two coders on nominal data.

    Uses the krippendorff package when available, otherwise falls back to an
    internal implementation.
    """
    if len(labels1) != len(labels2):
        raise ValueError("Arrays must have the same length")
    if len(labels1) == 0:
        return np.nan

    labels1 = np.asarray(labels1, dtype=float)
    labels2 = np.asarray(labels2, dtype=float)

    # Edge case: both coders constant and identical → perfect agreement
    if len(np.unique(labels1)) == 1 and len(np.unique(labels2)) == 1:
        return 1.0 if labels1[0] == labels2[0] else -1.0

    # Edge case: one coder constant → fall back to simple agreement
    if len(np.unique(labels1)) == 1 or len(np.unique(labels2)) == 1:
        return float(np.mean(labels1 == labels2))

    reliability_data = np.vstack([labels1, labels2])

    if HAS_KRIPPENDORFF:
        try:
            alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
            if alpha is not None and not np.isnan(alpha):
                return float(alpha)
        except Exception as exc:
            print(f"[WARNING] Krippendorff's Alpha (library) failed: {exc}")

    # Fallback computation
    fallback_alpha = _krippendorff_alpha_nominal(reliability_data)
    if np.isnan(fallback_alpha):
        print("[WARNING] Krippendorff's Alpha fallback returned NaN (insufficient variance)")
    return fallback_alpha


def calculate_global_krippendorff_alpha(
    coder1_matrix: np.ndarray,
    coder2_matrix: np.ndarray,
) -> float:
    """
    Calculate global Krippendorff's Alpha using pre-built coder matrices.

    Parameters
    ----------
    coder1_matrix, coder2_matrix : np.ndarray
        Binary matrices of shape (n_labels, n_sentences).
    """
    if coder1_matrix.size == 0 or coder2_matrix.size == 0:
        return np.nan

    reliability_data = np.vstack([
        coder1_matrix.reshape(-1, order="F"),
        coder2_matrix.reshape(-1, order="F"),
    ])

    if HAS_KRIPPENDORFF:
        try:
            alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
            if alpha is not None and not np.isnan(alpha):
                return float(alpha)
        except Exception as exc:
            print(f"[WARNING] Global Krippendorff's Alpha (library) failed: {exc}")

    return _krippendorff_alpha_nominal(reliability_data)


def calculate_f1_agreement(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """
    Calculate F1 score treating one coder as 'gold' and other as 'prediction'.
    Symmetrized by averaging both directions.
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0

    try:
        f1_12 = f1_score(labels1, labels2, average='binary', zero_division=0)
        f1_21 = f1_score(labels2, labels1, average='binary', zero_division=0)
        return (f1_12 + f1_21) / 2.0
    except Exception:
        return 0.0


def _build_binary_matrices(
    lookup1: Dict[Tuple, Dict[str, Any]],
    lookup2: Dict[Tuple, Dict[str, Any]],
    ordered_keys: List[Tuple],
    sorted_labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute binary label matrices for both coders."""
    n_labels = len(sorted_labels)
    n_sentences = len(ordered_keys)

    coder1_matrix = np.zeros((n_labels, n_sentences), dtype=np.uint8)
    coder2_matrix = np.zeros((n_labels, n_sentences), dtype=np.uint8)

    label_index = {label: idx for idx, label in enumerate(sorted_labels)}

    for col, key in enumerate(ordered_keys):
        entry1 = lookup1.get(key)
        entry2 = lookup2.get(key)
        if entry1 is None or entry2 is None:
            continue

        for lbl in entry1["gold_labels"]:
            idx = label_index.get(lbl)
            if idx is not None:
                coder1_matrix[idx, col] = 1

        for lbl in entry2["gold_labels"]:
            idx = label_index.get(lbl)
            if idx is not None:
                coder2_matrix[idx, col] = 1

    return coder1_matrix, coder2_matrix


def _compute_metrics_from_matrices(
    coder1_matrix: np.ndarray,
    coder2_matrix: np.ndarray,
    sorted_labels: List[str],
    show_progress: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute per-label and overall metrics from binary label matrices."""
    results: Dict[str, Dict[str, float]] = {}
    n_labels, n_sentences = coder1_matrix.shape

    if n_labels == 0 or n_sentences == 0:
        results["OVERALL"] = {"sample_size": 0}
        return results

    def compute_single(idx: int) -> Tuple[int, Dict[str, float]]:
        arr1 = coder1_matrix[idx]
        arr2 = coder2_matrix[idx]

        metrics: Dict[str, float] = {}
        metrics["sample_size"] = n_sentences
        metrics["coder1_prevalence"] = float(arr1.mean())
        metrics["coder2_prevalence"] = float(arr2.mean())

        try:
            metrics["percent_agreement"] = percent_agreement(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] Percent agreement failed for {sorted_labels[idx]}: {exc}")
            metrics["percent_agreement"] = np.nan

        try:
            metrics["cohens_kappa"] = cohen_kappa_score(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] Cohen's Kappa failed for {sorted_labels[idx]}: {exc}")
            metrics["cohens_kappa"] = np.nan

        try:
            metrics["scotts_pi"] = scotts_pi(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] Scott's Pi failed for {sorted_labels[idx]}: {exc}")
            metrics["scotts_pi"] = np.nan

        try:
            metrics["gwet_ac1"] = gwet_ac1(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] Gwet's AC1 failed for {sorted_labels[idx]}: {exc}")
            metrics["gwet_ac1"] = np.nan

        try:
            metrics["krippendorff_alpha"] = calculate_krippendorff_alpha(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] Krippendorff's Alpha failed for {sorted_labels[idx]}: {exc}")
            metrics["krippendorff_alpha"] = np.nan

        try:
            metrics["f1_agreement"] = calculate_f1_agreement(arr1, arr2)
        except Exception as exc:
            print(f"[WARNING] F1 agreement failed for {sorted_labels[idx]}: {exc}")
            metrics["f1_agreement"] = np.nan

        return idx, metrics

    workers = min(MAX_WORKERS, n_labels) or 1
    label_indices = list(range(n_labels))
    if workers <= 1:
        iter_results = map(compute_single, label_indices)
        if show_progress:
            iterator = tqdm(iter_results, total=n_labels, desc="Computing reliability metrics")
        else:
            iterator = iter_results
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iter_results = executor.map(compute_single, label_indices)
            if show_progress:
                iterator = tqdm(iter_results, total=n_labels, desc="Computing reliability metrics")
            else:
                iterator = iter_results

    for idx, metrics in iterator:
        results[sorted_labels[idx]] = metrics

    overall: Dict[str, float] = {}
    metric_names = ["percent_agreement", "cohens_kappa", "scotts_pi", "gwet_ac1", "f1_agreement"]

    for metric_name in metric_names:
        values = [
            results[label][metric_name]
            for label in sorted_labels
            if metric_name in results[label] and not np.isnan(results[label][metric_name])
        ]
        overall[metric_name] = float(np.mean(values)) if values else np.nan

    overall["krippendorff_alpha"] = calculate_global_krippendorff_alpha(coder1_matrix, coder2_matrix)
    overall["sample_size"] = n_sentences
    results["OVERALL"] = overall

    return results


def _krippendorff_alpha_nominal(reliability_data: np.ndarray) -> float:
    """
    Internal fallback computation of Krippendorff's alpha for nominal data.

    Parameters
    ----------
    reliability_data : np.ndarray
        Array with shape (n_coders, n_units).

    Returns
    -------
    float
        Krippendorff's alpha. Returns np.nan if it cannot be computed.
    """
    data = np.asarray(reliability_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("reliability_data must be 2-dimensional")

    n_coders, n_units = data.shape
    if n_coders < 2 or n_units == 0:
        return np.nan

    disagreements = 0.0
    valid_units = 0

    for unit_idx in range(n_units):
        values = data[:, unit_idx]
        # Drop missing values (NaN)
        values = values[~np.isnan(values)]
        if len(values) < 2:
            continue

        unit_pairs = 0
        unit_disagreements = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                unit_pairs += 1
                if values[i] != values[j]:
                    unit_disagreements += 1

        if unit_pairs == 0:
            continue

        disagreements += unit_disagreements / unit_pairs
        valid_units += 1

    if valid_units == 0:
        return np.nan

    do = disagreements / valid_units

    flat_values = data.flatten()
    flat_values = flat_values[~np.isnan(flat_values)]
    if flat_values.size == 0:
        return np.nan

    counts = Counter(flat_values)
    total = float(sum(counts.values()))
    proportions = [count / total for count in counts.values()]
    de = 1.0 - sum(p ** 2 for p in proportions)

    if de == 0.0:
        return 1.0 if do == 0.0 else np.nan

    return 1.0 - (do / de)


##############################################################################
#                    INTER-CODER RELIABILITY COMPUTATION                     #
##############################################################################

def compute_reliability_metrics(
    coder1_entries: List[Dict[str, Any]],
    coder2_entries: List[Dict[str, Any]],
    all_labels: Set[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive reliability metrics for all labels.

    Returns
    -------
    Dict with structure:
    {
        "OVERALL": {metric_name: value},
        label: {metric_name: value}
    }
    """
    lookup1 = {(e["doc_id"], e["sentence_id"]): e for e in coder1_entries}
    lookup2 = {(e["doc_id"], e["sentence_id"]): e for e in coder2_entries}

    common_key_set = set(lookup1.keys()) & set(lookup2.keys())
    if not common_key_set:
        print("[WARNING] No overlapping sentences between coders")
        return {}

    ordered_keys = sorted(common_key_set)
    sorted_labels = sorted(all_labels)

    coder1_matrix, coder2_matrix = _build_binary_matrices(
        lookup1, lookup2, ordered_keys, sorted_labels
    )

    print(f"[INFO] Found {len(ordered_keys)} sentences annotated by both coders")
    print("[INFO] Computing global Krippendorff's Alpha across all labels...")

    results = _compute_metrics_from_matrices(coder1_matrix, coder2_matrix, sorted_labels)
    if "OVERALL" in results:
        results["OVERALL"]["sample_size"] = len(ordered_keys)
        results["OVERALL"]["num_labels"] = len(sorted_labels)

    return results


def compute_reliability_metrics_split_600(
    coder1_entries: List[Dict[str, Any]],
    coder2_entries: List[Dict[str, Any]],
    all_labels: Set[str],
    threshold: int = 600
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Compute reliability metrics split at sentence threshold (default 600).

    Before 600: sentences annotated with meetings between coders
    After 600: sentences annotated completely blind without discussion

    Returns
    -------
    Tuple of (before_600_metrics, after_600_metrics)
    """
    # Create lookups
    lookup1 = {(e["doc_id"], e["sentence_id"]): e for e in coder1_entries}
    lookup2 = {(e["doc_id"], e["sentence_id"]): e for e in coder2_entries}

    # Get ordered common keys (by order in second_coder file for chronological order)
    coder2_keys = [(e["doc_id"], e["sentence_id"]) for e in coder2_entries]
    common_keys = [k for k in coder2_keys if k in lookup1]

    if not common_keys:
        print("[WARNING] No overlapping sentences between coders")
        return {}, {}

    # Split keys before and after threshold
    keys_before = common_keys[:threshold]
    keys_after = common_keys[threshold:]

    sorted_labels = sorted(all_labels)

    # Compute metrics for before 600
    print(f"\n[INFO] Computing metrics for first {threshold} sentences (with meetings)...")
    if keys_before:
        coder1_matrix_before, coder2_matrix_before = _build_binary_matrices(
            lookup1, lookup2, keys_before, sorted_labels
        )
        print(f"      Found {len(keys_before)} sentences before threshold")
        results_before = _compute_metrics_from_matrices(
            coder1_matrix_before, coder2_matrix_before, sorted_labels, show_progress=True
        )
        if "OVERALL" in results_before:
            results_before["OVERALL"]["sample_size"] = len(keys_before)
            results_before["OVERALL"]["num_labels"] = len(sorted_labels)
    else:
        results_before = {}

    # Compute metrics for after 600
    print(f"\n[INFO] Computing metrics for sentences after {threshold} (completely blind)...")
    if keys_after:
        coder1_matrix_after, coder2_matrix_after = _build_binary_matrices(
            lookup1, lookup2, keys_after, sorted_labels
        )
        print(f"      Found {len(keys_after)} sentences after threshold")
        results_after = _compute_metrics_from_matrices(
            coder1_matrix_after, coder2_matrix_after, sorted_labels, show_progress=True
        )
        if "OVERALL" in results_after:
            results_after["OVERALL"]["sample_size"] = len(keys_after)
            results_after["OVERALL"]["num_labels"] = len(sorted_labels)
    else:
        results_after = {}

    return results_before, results_after


##############################################################################
#                      LEARNING PROGRESSION ANALYSIS                         #
##############################################################################

def compute_learning_progression(
    coder1_entries: List[Dict[str, Any]],
    coder2_entries: List[Dict[str, Any]],
    all_labels: Set[str],
    increment: int = 100
) -> List[Dict[str, Any]]:
    """
    Compute reliability metrics in cumulative increments to track learning.

    Parameters
    ----------
    increment : int
        Number of labels to add per increment (default: 100)

    Returns
    -------
    List of dicts with progression metrics
    """
    # Create lookup
    lookup1 = {(e["doc_id"], e["sentence_id"]): e for e in coder1_entries}
    lookup2 = {(e["doc_id"], e["sentence_id"]): e for e in coder2_entries}

    # Get ordered common keys (by order in second_coder file)
    coder2_keys = [(e["doc_id"], e["sentence_id"]) for e in coder2_entries]
    common_keys = [k for k in coder2_keys if k in lookup1]

    if len(common_keys) == 0:
        return []

    progression = []
    sorted_labels = sorted(all_labels)
    coder1_matrix, coder2_matrix = _build_binary_matrices(lookup1, lookup2, common_keys, sorted_labels)

    for n in range(increment, len(common_keys) + 1, increment):
        # Take first n sentences
        coder1_slice = coder1_matrix[:, :n]
        coder2_slice = coder2_matrix[:, :n]
        metrics = _compute_metrics_from_matrices(coder1_slice, coder2_slice, sorted_labels, show_progress=False)

        if "OVERALL" in metrics:
            progress_row = {
                "n_sentences": n,
                "percent_agreement": metrics["OVERALL"].get("percent_agreement", np.nan),
                "cohens_kappa": metrics["OVERALL"].get("cohens_kappa", np.nan),
                "scotts_pi": metrics["OVERALL"].get("scotts_pi", np.nan),
                "gwet_ac1": metrics["OVERALL"].get("gwet_ac1", np.nan),
                "krippendorff_alpha": metrics["OVERALL"].get("krippendorff_alpha", np.nan),
                "f1_agreement": metrics["OVERALL"].get("f1_agreement", np.nan),
                "sample_size": metrics["OVERALL"].get("sample_size", n),
                "num_labels": len(sorted_labels),
            }
            progression.append(progress_row)

    return progression


##############################################################################
#                    CONSENSUS GOLD STANDARD CREATION                        #
##############################################################################

def create_consensus_gold_standard(
    coder1_entries: List[Dict[str, Any]],
    coder2_entries: List[Dict[str, Any]],
    strategy: str = "consensus"
) -> List[Dict[str, Any]]:
    """
    Create a consensus gold standard by reconciling both coders' annotations.

    Parameters
    ----------
    strategy : str
        - "consensus": Include label only if both coders agree (INTERSECTION)
        - "union": Include label if at least one coder marked it (UNION)
        - "majority": Same as consensus for 2 coders

    Returns
    -------
    List of consensus entries in same format as input
    """
    lookup1 = {(e["doc_id"], e["sentence_id"]): e for e in coder1_entries}
    lookup2 = {(e["doc_id"], e["sentence_id"]): e for e in coder2_entries}

    common_keys = set(lookup1.keys()) & set(lookup2.keys())

    consensus = []

    for key in sorted(common_keys):
        e1 = lookup1[key]
        e2 = lookup2[key]

        labels1 = e1["gold_labels"]
        labels2 = e2["gold_labels"]

        if strategy == "consensus" or strategy == "majority":
            # Intersection: both must agree
            consensus_labels = labels1 & labels2
        elif strategy == "union":
            # Union: at least one must mark it
            consensus_labels = labels1 | labels2
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        consensus.append({
            "doc_id": e1["doc_id"],
            "sentence_id": e1["sentence_id"],
            "language": e1["language"],
            "gold_labels": consensus_labels
        })

    return consensus




##############################################################################
#                          CSV EXPORT FUNCTIONS                              #
##############################################################################

def export_to_csv(
    reliability_results: Dict[str, Dict[str, float]],
    reliability_before_600: Dict[str, Dict[str, float]],
    reliability_after_600: Dict[str, Dict[str, float]],
    progression_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Export all results to multiple well-formatted CSV files.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = output_path.parent / output_path.stem

    # ========================================================================
    # TABLE 1: OVERALL RELIABILITY SUMMARY
    # ========================================================================
    overall_rows = []

    # Add section headers and data for each period
    sections = [
        ("OVERALL (All 1000 sentences)", reliability_results.get("OVERALL", {})),
        ("BEFORE 600 (With meetings)", reliability_before_600.get("OVERALL", {})),
        ("AFTER 600 (Completely blind)", reliability_after_600.get("OVERALL", {}))
    ]

    for section_name, section_data in sections:
        if not section_data:
            continue

        # Add section header
        overall_rows.append({
            "metric": f"=== {section_name} ===",
            "value": "",
            "interpretation": "",
            "quality_threshold": ""
        })

        # Add metrics for this section
        overall_rows.extend([
            {
                "metric": "Sample Size",
                "value": f"{section_data.get('sample_size', 0):.0f}",
                "interpretation": "Number of sentences annotated by both coders",
                "quality_threshold": "N/A"
            },
            {
                "metric": "Percent Agreement",
                "value": f"{section_data.get('percent_agreement', 0):.3f}",
                "interpretation": "Proportion of exact label agreement (not chance-corrected)",
                "quality_threshold": ">0.80 = Good"
            },
            {
                "metric": "Cohen's Kappa",
                "value": f"{section_data.get('cohens_kappa', 0):.3f}",
                "interpretation": "Chance-corrected agreement (sensitive to prevalence)",
                "quality_threshold": ">0.60 = Good, >0.80 = Excellent"
            },
            {
                "metric": "Scott's Pi",
                "value": f"{section_data.get('scotts_pi', 0):.3f}",
                "interpretation": "Alternative chance-correction assuming same distribution",
                "quality_threshold": ">0.60 = Good, >0.80 = Excellent"
            },
            {
                "metric": "Gwet's AC1",
                "value": f"{section_data.get('gwet_ac1', 0):.3f}",
                "interpretation": "Most robust to prevalence and high agreement",
                "quality_threshold": ">0.60 = Good, >0.80 = Excellent"
            },
            {
                "metric": "Krippendorff's Alpha",
                "value": f"{section_data.get('krippendorff_alpha', 0):.3f}",
                "interpretation": "Handles missing data and multiple coders",
                "quality_threshold": ">0.667 = Acceptable, >0.80 = Good"
            },
            {
                "metric": "F1 Agreement",
                "value": f"{section_data.get('f1_agreement', 0):.3f}",
                "interpretation": "Symmetric F1 score between coders",
                "quality_threshold": ">0.70 = Good"
            },
        ])

        # Add empty row for spacing
        overall_rows.append({
            "metric": "",
            "value": "",
            "interpretation": "",
            "quality_threshold": ""
        })

    with (base_path.parent / f"{base_path.name}_1_overall_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fo:
        writer = csv.DictWriter(
            fo,
            fieldnames=["metric", "value", "interpretation", "quality_threshold"]
        )
        writer.writeheader()
        writer.writerows(overall_rows)

    print(f"[INFO] Table 1 saved → {base_path.name}_1_overall_summary.csv")

    # ========================================================================
    # TABLE 2: PER-LABEL RELIABILITY METRICS (WIDE FORMAT)
    # ========================================================================
    per_label_rows = []

    for label, metrics in sorted(reliability_results.items()):
        if label == "OVERALL":
            continue

        sample_size = metrics.get("sample_size", 0)
        coder1_count = int(round(metrics.get("coder1_prevalence", 0) * sample_size))
        coder2_count = int(round(metrics.get("coder2_prevalence", 0) * sample_size))

        per_label_rows.append({
            "label": label,
            "n": f"{metrics.get('sample_size', 0):.0f}",
            "count_coder1": coder1_count,
            "count_coder2": coder2_count,
            "prev_coder1": f"{metrics.get('coder1_prevalence', 0):.3f}",
            "prev_coder2": f"{metrics.get('coder2_prevalence', 0):.3f}",
            "percent_agree": f"{metrics.get('percent_agreement', 0):.3f}",
            "cohens_kappa": f"{metrics.get('cohens_kappa', 0):.3f}",
            "scotts_pi": f"{metrics.get('scotts_pi', 0):.3f}",
            "gwet_ac1": f"{metrics.get('gwet_ac1', 0):.3f}",
            "krippendorff_alpha": f"{metrics.get('krippendorff_alpha', 0):.3f}",
            "f1_agreement": f"{metrics.get('f1_agreement', 0):.3f}",
        })

    with (base_path.parent / f"{base_path.name}_2_per_label_reliability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fo:
        writer = csv.DictWriter(
            fo,
            fieldnames=[
                "label", "n", "count_coder1", "count_coder2",
                "prev_coder1", "prev_coder2", "percent_agree",
                "cohens_kappa", "scotts_pi", "gwet_ac1",
                "krippendorff_alpha", "f1_agreement"
            ]
        )
        writer.writeheader()
        writer.writerows(per_label_rows)

    print(f"[INFO] Table 2 saved → {base_path.name}_2_per_label_reliability.csv")

    # ========================================================================
    # TABLE 3: LEARNING PROGRESSION
    # ========================================================================
    with (base_path.parent / f"{base_path.name}_3_learning_progression.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fo:
        if progression_results:
            fieldnames = list(progression_results[0].keys())
            writer = csv.DictWriter(fo, fieldnames=fieldnames)
            writer.writeheader()

            for prog in progression_results:
                formatted_prog = {
                    k: f"{v:.3f}" if isinstance(v, float) else v
                    for k, v in prog.items()
                }
                writer.writerow(formatted_prog)

    print(f"[INFO] Table 3 saved → {base_path.name}_3_learning_progression.csv")

    # ========================================================================
    # TABLE 4: RELIABILITY METRICS BEFORE/AFTER 600 SENTENCES
    # ========================================================================
    comparison_rows = []

    # Get all labels from both periods
    all_labels = sorted(set(
        list(reliability_before_600.keys()) +
        list(reliability_after_600.keys())
    ) - {"OVERALL"})

    # Add overall row first
    if "OVERALL" in reliability_before_600 or "OVERALL" in reliability_after_600:
        overall_row = {"label": "OVERALL"}

        if "OVERALL" in reliability_before_600:
            before = reliability_before_600["OVERALL"]
            overall_row.update({
                "before_n": f"{before.get('sample_size', 0):.0f}",
                "before_kappa": f"{before.get('cohens_kappa', 0):.3f}",
                "before_gwet": f"{before.get('gwet_ac1', 0):.3f}",
                "before_kripp": f"{before.get('krippendorff_alpha', 0):.3f}",
                "before_agree": f"{before.get('percent_agreement', 0):.3f}",
            })

        if "OVERALL" in reliability_after_600:
            after = reliability_after_600["OVERALL"]
            overall_row.update({
                "after_n": f"{after.get('sample_size', 0):.0f}",
                "after_kappa": f"{after.get('cohens_kappa', 0):.3f}",
                "after_gwet": f"{after.get('gwet_ac1', 0):.3f}",
                "after_kripp": f"{after.get('krippendorff_alpha', 0):.3f}",
                "after_agree": f"{after.get('percent_agreement', 0):.3f}",
            })

        comparison_rows.append(overall_row)

    # Add per-label rows
    for label in all_labels:
        row = {"label": label}

        if label in reliability_before_600:
            before = reliability_before_600[label]
            row.update({
                "before_n": f"{before.get('sample_size', 0):.0f}",
                "before_kappa": f"{before.get('cohens_kappa', 0):.3f}",
                "before_gwet": f"{before.get('gwet_ac1', 0):.3f}",
                "before_kripp": f"{before.get('krippendorff_alpha', 0):.3f}",
                "before_agree": f"{before.get('percent_agreement', 0):.3f}",
            })

        if label in reliability_after_600:
            after = reliability_after_600[label]
            row.update({
                "after_n": f"{after.get('sample_size', 0):.0f}",
                "after_kappa": f"{after.get('cohens_kappa', 0):.3f}",
                "after_gwet": f"{after.get('gwet_ac1', 0):.3f}",
                "after_kripp": f"{after.get('krippendorff_alpha', 0):.3f}",
                "after_agree": f"{after.get('percent_agreement', 0):.3f}",
            })

        comparison_rows.append(row)

    with (base_path.parent / f"{base_path.name}_4_before_after_600.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fo:
        fieldnames = [
            "label",
            "before_n", "before_kappa", "before_gwet", "before_kripp", "before_agree",
            "after_n", "after_kappa", "after_gwet", "after_kripp", "after_agree"
        ]
        writer = csv.DictWriter(fo, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"[INFO] Table 4 saved → {base_path.name}_4_before_after_600.csv")

    print(f"\n[INFO] All 4 tables successfully exported to: {base_path.parent}")


def export_consensus_gold_jsonl(consensus_entries: List[Dict[str, Any]], output_path: Path) -> None:
    """Export consensus gold standard to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fo:
        for entry in consensus_entries:
            obj = {
                "meta": {
                    "doc_id": entry["doc_id"],
                    "sentence_id": entry["sentence_id"],
                    "language": entry["language"]
                },
                "label": list(entry["gold_labels"])
            }
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[INFO] Consensus gold standard saved → {output_path}")


##############################################################################
#                          MAIN EVALUATION LOGIC                             #
##############################################################################

def main() -> None:
    print("="*80)
    print("INTER-CODER RELIABILITY ANALYSIS")
    print("="*80)

    # Step 1: Load label mapping
    print("\n[1/7] Loading label mapping...")
    label_mapping = load_label_mapping(LABEL_MAPPING_CSV)
    print(f"      Loaded {len(label_mapping)} label mappings")

    # Step 2: Load annotations from both coders
    print("\n[2/7] Loading primary coder annotations...")
    coder1_entries = load_jsonl_annotations(PRIMARY_CODER_JSONL, label_mapping=None)
    print(f"      Loaded {len(coder1_entries)} sentences from primary coder")

    print("\n[3/7] Loading second coder annotations...")
    coder2_entries = load_jsonl_annotations(SECOND_CODER_JSONL, label_mapping=label_mapping)
    print(f"      Loaded {len(coder2_entries)} sentences from second coder")

    # Get all unique labels
    all_labels = set()
    for e in coder1_entries + coder2_entries:
        all_labels.update(e["gold_labels"])
    print(f"      Found {len(all_labels)} unique labels across both coders")

    # Step 3: Compute overall reliability metrics
    print("\n[4/7] Computing overall inter-coder reliability metrics...")
    reliability_results = compute_reliability_metrics(coder1_entries, coder2_entries, all_labels)

    if "OVERALL" in reliability_results:
        overall = reliability_results["OVERALL"]
        print(f"      Overall Cohen's Kappa: {overall.get('cohens_kappa', 0):.4f}")
        print(f"      Overall Gwet's AC1: {overall.get('gwet_ac1', 0):.4f}")
        print(f"      Overall Percent Agreement: {overall.get('percent_agreement', 0):.4f}")

    # Step 4: Compute split reliability metrics (before/after 600)
    print("\n[5/7] Computing split reliability metrics (before/after sentence 600)...")
    reliability_before_600, reliability_after_600 = compute_reliability_metrics_split_600(
        coder1_entries, coder2_entries, all_labels, threshold=600
    )

    if "OVERALL" in reliability_before_600:
        before = reliability_before_600["OVERALL"]
        print(f"\n      Before 600 (with meetings):")
        print(f"        Cohen's Kappa: {before.get('cohens_kappa', 0):.4f}")
        print(f"        Gwet's AC1: {before.get('gwet_ac1', 0):.4f}")
        print(f"        Percent Agreement: {before.get('percent_agreement', 0):.4f}")

    if "OVERALL" in reliability_after_600:
        after = reliability_after_600["OVERALL"]
        print(f"\n      After 600 (completely blind):")
        print(f"        Cohen's Kappa: {after.get('cohens_kappa', 0):.4f}")
        print(f"        Gwet's AC1: {after.get('gwet_ac1', 0):.4f}")
        print(f"        Percent Agreement: {after.get('percent_agreement', 0):.4f}")

    # Step 5: Compute learning progression
    print("\n[6/7] Analyzing second coder learning progression...")
    progression_results = compute_learning_progression(coder1_entries, coder2_entries, all_labels)
    print(f"      Computed {len(progression_results)} progression checkpoints")

    # Step 6: Create consensus gold standard
    print("\n[7/7] Creating consensus gold standard...")
    consensus_entries = create_consensus_gold_standard(
        coder1_entries, coder2_entries, strategy="consensus"
    )
    print(f"      Created consensus for {len(consensus_entries)} sentences")

    # Export consensus to JSONL
    export_consensus_gold_jsonl(consensus_entries, CONSENSUS_GOLD_JSONL)

    # Step 7: Export comprehensive CSV
    print("\n[8/8] Exporting results to CSV...")
    export_to_csv(
        reliability_results,
        reliability_before_600,
        reliability_after_600,
        progression_results,
        OUTPUT_CSV
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutput files generated:")
    print(f"  1. Overall reliability summary")
    print(f"  2. Per-label reliability metrics")
    print(f"  3. Learning progression analysis")
    print(f"  4. Before/After 600 comparison")
    print(f"\n  Consensus gold standard: {CONSENSUS_GOLD_JSONL.name}")
    print(f"\nAll files saved to: {OUTPUT_CSV.parent}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — exiting.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
