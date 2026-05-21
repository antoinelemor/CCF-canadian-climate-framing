#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
16_build_normalized_csvs.py

MAIN OBJECTIVE:
---------------
Single entry point for building the four normalised CSVs that describe
the per-category reliability, tier assignment and training provenance of
the 65 annotation categories of the CCF Methodology paper. The script
writes:

  (a) ``per_category_reliability_normalized.csv``
      One row per Table B1 category with full-sample, training-phase
      and blind-phase inter-coder reliability metrics (Cohen's kappa,
      Scott's pi, Gwet's AC1, Krippendorff's alpha, percent agreement
      and F1 agreement). Built by ``build_per_category_reliability()``.

  (b) ``reliability_tiers.csv``
      One row per Table B1 category with tier assignments (A, B, C)
      for the combined corpus and for each language, the
      prevalence-induced reliability deflation flag (column
      ``paradox_kappa``, kept under that name for backwards
      compatibility with the deposited database), the exclusion reason
      for the five untrained categories, and the underlying F1 and
      kappa metrics. Built by ``build_reliability_tiers()``. Depends on
      (a).

  (c) ``training_hyperparameters_normalized.csv``
      One row per (Table B1 number, language) with the full training
      provenance: best epoch, training phase, validation losses,
      per-class precision / recall / F1 / support, executed epochs in
      each phase, model artefact mtime, and the technical label used
      by the training pipeline. Built by
      ``build_training_hyperparameters()``.

  (d) ``training_static_configuration.csv``
      One row per static hyperparameter, mirroring the constants of
      ``06_Training_best_models.py`` plus the campaign-wide timings
      derived from (c). Written together with (c) by
      ``build_training_hyperparameters()``.

All three builders reuse the canonical normalisation API exposed by
``15_normalize_categories.py`` (``ALL_CATEGORIES``, ``normalize_category``,
``OLD_CSV_TO_NEW_NUMBER``) through a dynamic loader, so the four CSVs
share the canonical 65-category Table B1 reference system.

Input files:
- Database/Training_data/manual_annotations_JSONL/
      intercoder_reliability_2_per_label_reliability.csv
      intercoder_reliability_4_before_after_600.csv
      label_mapping_second_coder.csv
- Database/Training_data/final_annotation_metrics.csv
- Database/Training_data/non_trained_models.csv
- Database/Training_data/Training_logs/<Category>_<LANG>/
      training_metrics.csv
      reinforced_training_metrics.csv (only when reinforcement ran)
      best_models.csv
- models/<code>_<LANG>.jsonl.model/pytorch_model.bin
      Used solely for its mtime to estimate per-model elapsed time.
- Scripts/Annotation/15_normalize_categories.py (imported)

Output files:
- Database/Training_data/per_category_reliability_normalized.csv
- Database/Training_data/reliability_tiers.csv
- Database/Training_data/training_hyperparameters_normalized.csv
- Database/Training_data/training_static_configuration.csv

Dependencies:
-------------
- pandas
- pathlib
- importlib
- datetime

MAIN FEATURES:
--------------
1) Single ordered pipeline - ``main()`` runs the three builders in the
   required dependency order: per-category reliability first (consumed
   by the tier assignment), then reliability tiers, then training
   hyperparameters.
2) Canonical normalisation - Every builder loads
   ``15_normalize_categories.py`` once and reuses
   ``normalize_category()`` so the resulting CSVs share the same Table
   B1 numbers, official names and codes.
3) Strict validation - Each builder aborts with an explicit error if
   any technical label cannot be mapped to Table B1, so the output is
   guaranteed to be exhaustive and self-consistent.

Author:
-------
Antoine Lemor
"""

from __future__ import annotations

##############################################################################
#                          IMPORTS & CONFIGURATION                           #
##############################################################################

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

##############################################################################
#                                  PATHS                                     #
##############################################################################

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "Database" / "Training_data"
OUTPUT_DIR = DATA_DIR
JSONL_DIR = DATA_DIR / "manual_annotations_JSONL"
LOGS_DIR = DATA_DIR / "Training_logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Per-category reliability inputs / output
INPUT_PER_LABEL = JSONL_DIR / "intercoder_reliability_2_per_label_reliability.csv"
INPUT_BEFORE_AFTER = JSONL_DIR / "intercoder_reliability_4_before_after_600.csv"
INPUT_LABEL_MAPPING = JSONL_DIR / "label_mapping_second_coder.csv"
OUTPUT_RELIABILITY = OUTPUT_DIR / "per_category_reliability_normalized.csv"

# Reliability tier inputs / output
INPUT_VALIDATION = DATA_DIR / "final_annotation_metrics.csv"
INPUT_RELIABILITY_TIERS = OUTPUT_RELIABILITY  # produced by builder (a)
INPUT_NON_TRAINED = DATA_DIR / "non_trained_models.csv"
OUTPUT_TIERS = OUTPUT_DIR / "reliability_tiers.csv"

# Training hyperparameters output
OUTPUT_HYPERPARAMS = OUTPUT_DIR / "training_hyperparameters_normalized.csv"
OUTPUT_STATIC = OUTPUT_DIR / "training_static_configuration.csv"

##############################################################################
#                            TIER THRESHOLDS                                 #
##############################################################################

F1_TIER_A = 0.80
KAPPA_TIER_A = 0.60
F1_TIER_C = 0.60
KAPPA_TIER_C = 0.40

# Paradox flag: F1 acceptable but kappa collapses through low prevalence.
PARADOX_F1_MIN = 0.70
PARADOX_KAPPA_MAX = 0.40

##############################################################################
#                STATIC HYPERPARAMETERS (from 06_Training_best_models.py)    #
# All values below are constants of the pipeline. They are stored in a       #
# separate CSV so the per-model file remains compact.                        #
##############################################################################

STATIC_CONFIG = {
    "english_model": "bert-base-uncased",
    "french_model": "camembert-base",
    "model_size_parameters": "110M (per language)",
    "optimizer": "AdamW",
    "optimizer_eps": 1e-8,
    "learning_rate_normal": 5e-5,
    "learning_rate_reinforced": 1e-5,
    "scheduler": "linear with 0 warm-up steps",
    "batch_size_normal": 32,
    "batch_size_reinforced": 64,
    "max_seq_length_tokens": 512,
    "n_epochs_normal_max": 20,
    "n_epochs_reinforced_max": 20,
    "reinforcement_trigger": "positive-class F1 < 0.60 at best normal epoch",
    "best_epoch_criterion": "0.7 * F1_class_1 + 0.3 * macro_F1 (combined)",
    "class_weighting_normal": "pos_weight from class frequencies (weighted cross-entropy)",
    "class_weighting_reinforced": "WeightedRandomSampler (oversample minority)",
    "loss_function": "BCEWithLogitsLoss (binary classification per category)",
    "random_seed_python": 42,
    "random_seed_numpy": 42,
    "random_seed_torch": 42,
    "random_seed_torch_cuda": 42,
    "random_seed_weighted_random_sampler": (
        "not fixed (PyTorch default generator); known limitation, to be "
        "addressed in a future release"
    ),
    "training_hardware": (
        "Apple Mac Studio (M2 Ultra, 24-core CPU, 60-core GPU, 128 GB unified "
        "memory, 2023). Acceleration via Metal Performance Shaders (MPS) when "
        "available, otherwise CPU."
    ),
    "training_framework": (
        "PyTorch (transformers library) through the AugmentedSocialScientist "
        "fork (Lemor 2025, https://github.com/antoinelemor/AugmentedSocialScientistFork)"
    ),
}


##############################################################################
#               DYNAMIC IMPORT OF 15_normalize_categories.py                 #
# The module name begins with a digit, so the standard import statement is   #
# invalid. importlib.util.spec_from_file_location is used instead.           #
##############################################################################


def _load_normalization_module():
    """Load 15_normalize_categories.py and return its module object."""
    spec_path = SCRIPT_DIR / "15_normalize_categories.py"
    spec = importlib.util.spec_from_file_location("ccf_normalization", spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load normalisation module at {spec_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ccf_normalization"] = module
    spec.loader.exec_module(module)
    return module


##############################################################################
#               (A) PER-CATEGORY INTER-CODER RELIABILITY                     #
##############################################################################


def _build_technical_to_readable(mapping_csv: Path) -> Dict[str, str]:
    """Return {technical_label_en: readable_label_en} from the bridge CSV."""
    df = pd.read_csv(mapping_csv)
    expected_cols = {"technical_label_en", "readable_label_en"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"label_mapping_second_coder.csv is missing columns: {missing}"
        )
    return dict(zip(df["technical_label_en"], df["readable_label_en"]))


def _normalise_category(name: str, normaliser) -> Optional[Tuple[int, str, str]]:
    """Return (B1 number, official name, code) or None if not mapped."""
    num, official, code = normaliser(name)
    if num is None or num == 99:
        return None
    return num, official, code


def _build_per_category_reliability_table() -> pd.DataFrame:
    """Produce the consolidated per-category reliability table.

    Returns
    -------
    pd.DataFrame
        One row per Table B1 category with full-sample, training-phase
        and blind-phase reliability metrics.
    """
    normalisation = _load_normalization_module()
    tech_to_readable = _build_technical_to_readable(INPUT_LABEL_MAPPING)

    df_full = pd.read_csv(INPUT_PER_LABEL)
    df_phase = pd.read_csv(INPUT_BEFORE_AFTER)

    # Drop the OVERALL row from the per-phase CSV.
    df_phase = df_phase[df_phase["label"] != "OVERALL"].copy()

    # Build a single working frame keyed on the technical label.
    df_full = df_full.rename(
        columns={
            "n": "n_full",
            "count_coder1": "count_coder1_full",
            "count_coder2": "count_coder2_full",
            "prev_coder1": "prev_coder1_full",
            "prev_coder2": "prev_coder2_full",
            "percent_agree": "percent_agree_full",
            "cohens_kappa": "cohens_kappa_full",
            "scotts_pi": "scotts_pi_full",
            "gwet_ac1": "gwet_ac1_full",
            "krippendorff_alpha": "krippendorff_alpha_full",
            "f1_agreement": "f1_agreement_full",
        }
    )
    df_phase = df_phase.rename(
        columns={
            "before_n": "n_training_phase",
            "before_kappa": "cohens_kappa_training_phase",
            "before_gwet": "gwet_ac1_training_phase",
            "before_kripp": "krippendorff_alpha_training_phase",
            "before_agree": "percent_agree_training_phase",
            "after_n": "n_blind_phase",
            "after_kappa": "cohens_kappa_blind_phase",
            "after_gwet": "gwet_ac1_blind_phase",
            "after_kripp": "krippendorff_alpha_blind_phase",
            "after_agree": "percent_agree_blind_phase",
        }
    )

    merged = df_full.merge(df_phase, on="label", how="outer")
    unmapped: list[str] = []
    rows = []

    for _, row in merged.iterrows():
        technical = row["label"]
        readable = tech_to_readable.get(technical)
        if readable is None:
            unmapped.append(f"{technical} (not in label_mapping_second_coder.csv)")
            continue
        normalised = _normalise_category(readable, normalisation.normalize_category)
        if normalised is None:
            unmapped.append(
                f"{technical} -> {readable} (excluded from Table B1)"
            )
            continue
        num, official, code = normalised
        rows.append(
            {
                "number": num,
                "category": official,
                "code": code,
                "technical_label": technical,
                # Full sample (1,000 sentences)
                "n_full": row.get("n_full"),
                "count_coder1_full": row.get("count_coder1_full"),
                "count_coder2_full": row.get("count_coder2_full"),
                "prev_coder1_full": row.get("prev_coder1_full"),
                "prev_coder2_full": row.get("prev_coder2_full"),
                "percent_agree_full": row.get("percent_agree_full"),
                "cohens_kappa_full": row.get("cohens_kappa_full"),
                "scotts_pi_full": row.get("scotts_pi_full"),
                "gwet_ac1_full": row.get("gwet_ac1_full"),
                "krippendorff_alpha_full": row.get("krippendorff_alpha_full"),
                "f1_agreement_full": row.get("f1_agreement_full"),
                # Training phase (sentences 1-600, with coder meetings)
                "n_training_phase": row.get("n_training_phase"),
                "percent_agree_training_phase": row.get("percent_agree_training_phase"),
                "cohens_kappa_training_phase": row.get("cohens_kappa_training_phase"),
                "gwet_ac1_training_phase": row.get("gwet_ac1_training_phase"),
                "krippendorff_alpha_training_phase": row.get(
                    "krippendorff_alpha_training_phase"
                ),
                # Blind phase (sentences 601-1000, no communication between coders)
                "n_blind_phase": row.get("n_blind_phase"),
                "percent_agree_blind_phase": row.get("percent_agree_blind_phase"),
                "cohens_kappa_blind_phase": row.get("cohens_kappa_blind_phase"),
                "gwet_ac1_blind_phase": row.get("gwet_ac1_blind_phase"),
                "krippendorff_alpha_blind_phase": row.get(
                    "krippendorff_alpha_blind_phase"
                ),
            }
        )

    if unmapped:
        raise RuntimeError(
            "Encountered categories that could not be mapped to Table B1:\n  - "
            + "\n  - ".join(unmapped)
        )

    table = pd.DataFrame(rows).sort_values("number").reset_index(drop=True)
    return table


def build_per_category_reliability() -> pd.DataFrame:
    """Build, persist and return the per-category reliability CSV."""
    print("=" * 72)
    print("Per-category inter-coder reliability — canonical normalisation")
    print("=" * 72)
    for path in (INPUT_PER_LABEL, INPUT_BEFORE_AFTER, INPUT_LABEL_MAPPING):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        print(f"  input : {path.relative_to(PROJECT_ROOT)}")

    table = _build_per_category_reliability_table()
    print(f"\n  rows produced : {len(table)}")
    print(f"  unique B1 numbers : {table['number'].nunique()}")
    print(
        "  blind-phase kappa available : "
        f"{table['cohens_kappa_blind_phase'].notna().sum()} / {len(table)}"
    )

    OUTPUT_RELIABILITY.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_RELIABILITY, index=False)
    print(f"\n  output: {OUTPUT_RELIABILITY.relative_to(PROJECT_ROOT)}")
    return table


##############################################################################
#                          (B) RELIABILITY TIERS                             #
##############################################################################


def _assign_tier(f1: Optional[float], kappa: Optional[float]) -> str:
    """Return ``A``, ``B``, ``C``, or ``no_data`` for a single (F1, kappa).

    The kappa value is allowed to be missing (returns the most informative
    tier given F1 alone). The F1 value is required.
    """
    if pd.isna(f1):
        return "no_data"
    if f1 < F1_TIER_C:
        return "C"
    if pd.notna(kappa) and kappa < KAPPA_TIER_C:
        return "C"
    if f1 >= F1_TIER_A and (pd.isna(kappa) or kappa >= KAPPA_TIER_A):
        return "A"
    return "B"


def _paradox_flag(f1: Optional[float], kappa: Optional[float]) -> bool:
    """Return True iff the category exhibits prevalence-induced
    reliability deflation.

    A high F1 (>= PARADOX_F1_MIN) combined with a depressed kappa
    (< PARADOX_KAPPA_MAX) is the hallmark of the phenomenon: the
    chance-corrected agreement is dragged down by extreme class
    imbalance rather than by systematic disagreement between coders.
    """
    if pd.isna(f1) or pd.isna(kappa):
        return False
    return (f1 >= PARADOX_F1_MIN) and (kappa < PARADOX_KAPPA_MAX)


def _build_validation_table(normalisation) -> pd.DataFrame:
    """Pivot final_annotation_metrics.csv into one row per category.

    The CSV stores three rows per category (EN / FR / ALL). The frame is
    reshaped into f1_en, f1_fr, f1_all (plus the matching positive-class
    supports) after dropping the aggregate ``ALL,ALL`` row.
    """
    df = pd.read_csv(INPUT_VALIDATION)
    df = df[df["label"] != "ALL"].copy()

    # Normalise each label to its Table B1 number/code.
    norm_records = []
    for label in df["label"].unique():
        num, official, code = normalisation.normalize_category(label)
        if num is None:
            # Excluded from Table B1 (e.g. Positive Health Impacts).
            continue
        norm_records.append(
            {"label": label, "number": num, "category": official, "code": code}
        )
    if not norm_records:
        raise RuntimeError("No labels from final_annotation_metrics.csv mapped to Table B1.")
    norm_df = pd.DataFrame(norm_records)

    df = df.merge(norm_df, on="label", how="inner")
    pivot = df.pivot_table(
        index=["number", "category", "code"],
        columns="language",
        values=["f1_macro", "support_1"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{lang.lower()}" for metric, lang in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot.rename(
        columns={
            "f1_macro_en": "f1_en",
            "f1_macro_fr": "f1_fr",
            "f1_macro_all": "f1_all",
            "support_1_en": "support_en",
            "support_1_fr": "support_fr",
            "support_1_all": "support_all",
        }
    )
    return pivot


def _build_reliability_subset() -> pd.DataFrame:
    """Read per_category_reliability_normalized.csv and keep only what is needed."""
    df = pd.read_csv(INPUT_RELIABILITY_TIERS)
    columns_kept = [
        "number",
        "cohens_kappa_blind_phase",
        "gwet_ac1_blind_phase",
        "krippendorff_alpha_blind_phase",
        "n_blind_phase",
        "prev_coder1_full",
        "prev_coder2_full",
    ]
    return df[columns_kept]


def _build_excluded_set() -> Dict[int, str]:
    """Return {Table B1 number: reason} for categories with no trained model.

    Reads non_trained_models.csv produced by the training pipeline and
    maps the raw model basenames to canonical Table B1 numbers.
    """
    normalisation = _load_normalization_module()
    df = pd.read_csv(INPUT_NON_TRAINED)
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        base = row["model_name"].replace(".jsonl", "").rsplit("_", 1)[0]
        num, _official, _code = normalisation.normalize_category(base)
        if num is None or num == 99:
            continue
        out[num] = "insufficient_training_data"
    return out


def _build_reliability_tiers_table() -> pd.DataFrame:
    """Produce the reliability_tiers table.

    Returns
    -------
    pd.DataFrame
        Sorted by Table B1 number, with one row per category.
    """
    normalisation = _load_normalization_module()
    val = _build_validation_table(normalisation)
    rel = _build_reliability_subset()
    excluded = _build_excluded_set()

    merged = val.merge(rel, on="number", how="left")

    # Apply tier assignment.
    merged["tier_en"] = merged.apply(
        lambda r: _assign_tier(r["f1_en"], r["cohens_kappa_blind_phase"]),
        axis=1,
    )
    merged["tier_fr"] = merged.apply(
        lambda r: _assign_tier(r["f1_fr"], r["cohens_kappa_blind_phase"]),
        axis=1,
    )
    merged["tier_overall"] = merged.apply(
        lambda r: _assign_tier(r["f1_all"], r["cohens_kappa_blind_phase"]),
        axis=1,
    )
    merged["paradox_kappa"] = merged.apply(
        lambda r: _paradox_flag(r["f1_all"], r["cohens_kappa_blind_phase"]),
        axis=1,
    )

    # Overlay excluded categories. They are forced to Tier C in every column
    # and assigned a dedicated reason field for transparency.
    merged["exclusion_reason"] = merged["number"].map(excluded).fillna("")
    for col in ("tier_en", "tier_fr", "tier_overall"):
        merged.loc[merged["exclusion_reason"].ne(""), col] = "C"

    # Final column order, sorted by Table B1 number.
    out = merged.sort_values("number").reset_index(drop=True)
    column_order = [
        "number",
        "category",
        "code",
        "tier_overall",
        "tier_en",
        "tier_fr",
        "paradox_kappa",
        "exclusion_reason",
        "f1_all",
        "f1_en",
        "f1_fr",
        "support_all",
        "support_en",
        "support_fr",
        "cohens_kappa_blind_phase",
        "gwet_ac1_blind_phase",
        "krippendorff_alpha_blind_phase",
        "n_blind_phase",
        "prev_coder1_full",
        "prev_coder2_full",
    ]
    out = out[column_order]
    return out


def _print_tier_summary(table: pd.DataFrame) -> None:
    """Pretty-print the tier distribution for human verification."""
    print("\nTier distribution (overall):")
    print(table["tier_overall"].value_counts().sort_index().to_string())
    print("\nTier distribution (English):")
    print(table["tier_en"].value_counts().sort_index().to_string())
    print("\nTier distribution (French):")
    print(table["tier_fr"].value_counts().sort_index().to_string())
    n_paradox = int(table["paradox_kappa"].sum())
    print(f"\nCategories flagged with prevalence-induced reliability deflation : {n_paradox}")
    if n_paradox:
        print(
            table[table["paradox_kappa"]][
                [
                    "number",
                    "code",
                    "f1_all",
                    "cohens_kappa_blind_phase",
                    "gwet_ac1_blind_phase",
                ]
            ].to_string(index=False)
        )
    n_excluded = (table["exclusion_reason"] != "").sum()
    print(f"\nCategories forced to Tier C (excluded from training) : {n_excluded}")
    if n_excluded:
        print(
            table[table["exclusion_reason"] != ""][
                ["number", "code", "exclusion_reason"]
            ].to_string(index=False)
        )


def build_reliability_tiers() -> pd.DataFrame:
    """Build, persist and return the reliability tier CSV."""
    print("\n" + "=" * 72)
    print("Reliability tier assignment — F1 (validation) x kappa (blind)")
    print("=" * 72)
    for path in (INPUT_VALIDATION, INPUT_RELIABILITY_TIERS, INPUT_NON_TRAINED):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
        print(f"  input : {path.relative_to(PROJECT_ROOT)}")
    print(
        f"\n  thresholds: A (F1 >= {F1_TIER_A:.2f} and kappa >= {KAPPA_TIER_A:.2f}); "
        f"C (F1 < {F1_TIER_C:.2f} or kappa < {KAPPA_TIER_C:.2f}); "
        f"paradox (F1 >= {PARADOX_F1_MIN:.2f} and kappa < {PARADOX_KAPPA_MAX:.2f})."
    )

    table = _build_reliability_tiers_table()
    assert len(table) == 65, f"Expected 65 rows, got {len(table)}"
    assert table["number"].is_unique
    assert set(table["number"]) == set(range(1, 66))

    OUTPUT_TIERS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_TIERS, index=False)
    print(f"\n  output: {OUTPUT_TIERS.relative_to(PROJECT_ROOT)}")
    _print_tier_summary(table)
    return table


##############################################################################
#                  (C) TRAINING HYPERPARAMETERS PROVENANCE                   #
##############################################################################


def _build_technical_to_readable_for_training() -> Dict[str, str]:
    """Return the {technical_label: readable_label} bridge for training logs.

    The training pipeline names model directories with technical labels
    (e.g. ``Sci_1_SUB``). The normalisation table in
    ``15_normalize_categories.py`` is keyed on readable labels (e.g.
    ``Scientific Controversy``). The bridge CSV produced for the
    second-coder analysis closes the gap.

    Two extra mappings are added because the training pipeline uses
    ``Emotion:_Negative`` / ``Emotion:_Positive`` / ``Emotion:_Neutral``
    in directory names whereas the bridge CSV records them as the
    canonical Table B1 readable forms (``Negative Emotion`` etc.).
    """
    df = pd.read_csv(INPUT_LABEL_MAPPING)
    mapping = dict(zip(df["technical_label_en"], df["readable_label_en"]))
    # Emotion directories use a non-standard naming with a colon; explicitly
    # bridge them to the canonical readable labels in Table B1.
    mapping.setdefault("Emotion:_Negative", "Negative Emotion")
    mapping.setdefault("Emotion:_Positive", "Positive Emotion")
    mapping.setdefault("Emotion:_Neutral", "Neutral Emotion")
    return mapping


def _split_dir_name(name: str) -> Optional[tuple[str, str]]:
    """Split a Training_logs directory name into (technical, language).

    The naming convention used by the training pipeline is
    ``<technical_label>_<EN|FR>``. This helper returns None if the
    directory does not match.
    """
    for suffix in ("_EN", "_FR"):
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return None


def _best_epoch_row(best_csv: Path) -> Optional[pd.Series]:
    """Return the row corresponding to the final best epoch.

    best_models.csv accumulates every epoch that improved the combined
    criterion. The final row is therefore the best model that was
    ultimately retained.
    """
    if not best_csv.exists():
        return None
    df = pd.read_csv(best_csv)
    if df.empty:
        return None
    return df.iloc[-1]


def _max_epoch_in_phase(csv_path: Path) -> Optional[int]:
    """Return the maximum epoch value in a metrics CSV, or None if absent."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return int(df["epoch"].max())


def _model_dir(code: str, lang: str) -> Path:
    """Return the path to the trained model directory for (code, language)."""
    return MODELS_DIR / f"{code}_{lang}.jsonl.model"


def _model_mtime(code: str, lang: str) -> Optional[datetime]:
    """Return the pytorch_model.bin modification time for a model, if any."""
    bin_path = _model_dir(code, lang) / "pytorch_model.bin"
    if not bin_path.exists():
        return None
    return datetime.fromtimestamp(bin_path.stat().st_mtime, tz=timezone.utc)


def _build_hyperparameter_table() -> pd.DataFrame:
    """Walk Training_logs/ and produce one row per (category, language)."""
    normalisation = _load_normalization_module()
    tech_to_readable = _build_technical_to_readable_for_training()

    rows: List[Dict] = []
    unmapped: List[str] = []

    for log_dir in sorted(LOGS_DIR.iterdir()):
        if not log_dir.is_dir():
            continue
        parts = _split_dir_name(log_dir.name)
        if parts is None:
            unmapped.append(log_dir.name)
            continue
        technical, language = parts

        readable = tech_to_readable.get(technical)
        if readable is None:
            unmapped.append(
                f"{log_dir.name} -> '{technical}' "
                "(no readable equivalent in label_mapping_second_coder.csv)"
            )
            continue
        num, official, code = normalisation.normalize_category(readable)
        if num is None or num == 99:
            unmapped.append(f"{log_dir.name} -> '{readable}' (not in Table B1)")
            continue

        best_row = _best_epoch_row(log_dir / "best_models.csv")
        max_normal = _max_epoch_in_phase(log_dir / "training_metrics.csv")
        max_reinforced = _max_epoch_in_phase(log_dir / "reinforced_training_metrics.csv")
        reinforcement_ran = max_reinforced is not None
        model_mtime = _model_mtime(code, language)

        row = {
            "number": num,
            "category": official,
            "code": code,
            "language": language,
            "technical_label": technical,
            "reinforcement_triggered": reinforcement_ran,
            "n_epochs_normal_executed": max_normal,
            "n_epochs_reinforced_executed": max_reinforced,
            "best_epoch": int(best_row["epoch"]) if best_row is not None else None,
            "best_training_phase": (
                best_row["training_phase"] if best_row is not None else None
            ),
            "best_train_loss": (
                float(best_row["train_loss"]) if best_row is not None else None
            ),
            "best_val_loss": (
                float(best_row["val_loss"]) if best_row is not None else None
            ),
            "best_precision_0": (
                float(best_row["precision_0"]) if best_row is not None else None
            ),
            "best_recall_0": (
                float(best_row["recall_0"]) if best_row is not None else None
            ),
            "best_f1_0": (
                float(best_row["f1_0"]) if best_row is not None else None
            ),
            "support_0": (
                int(best_row["support_0"]) if best_row is not None else None
            ),
            "best_precision_1": (
                float(best_row["precision_1"]) if best_row is not None else None
            ),
            "best_recall_1": (
                float(best_row["recall_1"]) if best_row is not None else None
            ),
            "best_f1_1": (
                float(best_row["f1_1"]) if best_row is not None else None
            ),
            "support_1": (
                int(best_row["support_1"]) if best_row is not None else None
            ),
            "best_macro_f1": (
                float(best_row["macro_f1"]) if best_row is not None else None
            ),
            "saved_model_path": (
                best_row["saved_model_path"] if best_row is not None else None
            ),
            "model_artifact_mtime_utc": (
                model_mtime.isoformat() if model_mtime is not None else None
            ),
        }
        rows.append(row)

    if unmapped:
        print("WARNING: the following log directories were skipped:")
        for item in unmapped:
            print(f"  - {item}")

    table = pd.DataFrame(rows).sort_values(["number", "language"]).reset_index(drop=True)
    return table


def _derive_campaign_timings(table: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return start, end, and elapsed durations for the training campaign.

    Two durations are reported:
      - ``campaign_span_hours`` is the calendar span between the first and
        last trained model (typically inflated by inactive periods);
      - ``campaign_bulk_span_hours`` is the span restricted to the densest
        contiguous training window, defined as the longest stretch of
        consecutive models whose adjacent mtimes are less than 6 hours
        apart. This estimate is closer to the actual wall-clock time
        spent training.
    """
    mtimes = pd.to_datetime(
        table["model_artifact_mtime_utc"].dropna(), utc=True, format="ISO8601"
    )
    mtimes = mtimes.dropna().sort_values().reset_index(drop=True)
    if mtimes.empty:
        return {
            "campaign_start_utc": None,
            "campaign_end_utc": None,
            "campaign_span_hours": None,
            "campaign_bulk_start_utc": None,
            "campaign_bulk_end_utc": None,
            "campaign_bulk_span_hours": None,
            "n_models_with_mtime": 0,
        }

    # Identify the densest contiguous block: largest set of consecutive
    # mtimes where each gap is below 6 hours (a quiet threshold for an
    # actively running training campaign on a workstation).
    gap_threshold_seconds = 6 * 3600
    gaps = mtimes.diff().dt.total_seconds().fillna(0)
    # block_id increments each time a gap exceeds the threshold.
    block_id = (gaps > gap_threshold_seconds).cumsum()
    block_counts = block_id.value_counts()
    densest_block = block_counts.idxmax()
    bulk_mtimes = mtimes[block_id == densest_block]

    return {
        "campaign_start_utc": mtimes.iloc[0].isoformat(),
        "campaign_end_utc": mtimes.iloc[-1].isoformat(),
        "campaign_span_hours": round(
            (mtimes.iloc[-1] - mtimes.iloc[0]).total_seconds() / 3600.0, 2
        ),
        "campaign_bulk_start_utc": bulk_mtimes.iloc[0].isoformat(),
        "campaign_bulk_end_utc": bulk_mtimes.iloc[-1].isoformat(),
        "campaign_bulk_span_hours": round(
            (bulk_mtimes.iloc[-1] - bulk_mtimes.iloc[0]).total_seconds() / 3600.0, 2
        ),
        "n_models_with_mtime": int(len(mtimes)),
        "n_models_in_bulk_block": int(len(bulk_mtimes)),
    }


def build_training_hyperparameters() -> pd.DataFrame:
    """Build, persist and return the per-model hyperparameter CSV.

    The static configuration CSV is written as a by-product so the two
    artefacts remain consistent (campaign timings derived from the
    per-model mtimes are appended to the static configuration table).
    """
    print("\n" + "=" * 72)
    print("Training hyperparameters — canonical extraction")
    print("=" * 72)
    if not LOGS_DIR.exists():
        raise FileNotFoundError(f"Missing logs directory: {LOGS_DIR}")
    print(f"  logs root : {LOGS_DIR.relative_to(PROJECT_ROOT)}")

    table = _build_hyperparameter_table()
    print(f"\n  rows produced : {len(table)}")
    n_reinforced = int(table["reinforcement_triggered"].sum())
    n_best_in_reinforced = int(
        table["best_training_phase"].fillna("").str.startswith("reinforced").sum()
    )
    print(
        f"  reinforcement triggered : {n_reinforced} / {len(table)} models "
        f"(best epoch retained in reinforced phase: {n_best_in_reinforced})"
    )
    timings = _derive_campaign_timings(table)
    print(f"  campaign start (UTC)        : {timings['campaign_start_utc']}")
    print(f"  campaign end   (UTC)        : {timings['campaign_end_utc']}")
    print(f"  campaign calendar span (h)  : {timings['campaign_span_hours']}")
    print(f"  bulk window start (UTC)     : {timings['campaign_bulk_start_utc']}")
    print(f"  bulk window end   (UTC)     : {timings['campaign_bulk_end_utc']}")
    print(f"  bulk window span (h)        : {timings['campaign_bulk_span_hours']}")
    print(
        f"  models in bulk block        : "
        f"{timings.get('n_models_in_bulk_block')} / {timings['n_models_with_mtime']}"
    )

    OUTPUT_HYPERPARAMS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_HYPERPARAMS, index=False)
    print(f"\n  output (per-model): {OUTPUT_HYPERPARAMS.relative_to(PROJECT_ROOT)}")

    # Save the static configuration with campaign timings appended.
    static = dict(STATIC_CONFIG)
    static.update({
        "campaign_start_utc": timings["campaign_start_utc"],
        "campaign_end_utc": timings["campaign_end_utc"],
        "campaign_span_hours": timings["campaign_span_hours"],
        "campaign_bulk_start_utc": timings["campaign_bulk_start_utc"],
        "campaign_bulk_end_utc": timings["campaign_bulk_end_utc"],
        "campaign_bulk_span_hours": timings["campaign_bulk_span_hours"],
        "n_models_with_artifact_mtime": timings["n_models_with_mtime"],
        "n_models_in_bulk_block": timings.get("n_models_in_bulk_block"),
        "n_models_total": len(table),
        "n_models_reinforced": n_reinforced,
        "n_models_best_epoch_in_reinforced_phase": n_best_in_reinforced,
    })
    static_df = pd.DataFrame(
        [(k, v) for k, v in static.items()], columns=["parameter", "value"]
    )
    static_df.to_csv(OUTPUT_STATIC, index=False)
    print(f"  output (static)   : {OUTPUT_STATIC.relative_to(PROJECT_ROOT)}")
    return table


##############################################################################
#                                  MAIN                                      #
##############################################################################


def main() -> None:
    """Run the three builders in the required dependency order."""
    build_per_category_reliability()
    build_reliability_tiers()
    build_training_hyperparameters()
    print("\nDone.")


if __name__ == "__main__":
    main()
