"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7_Annotation.py

MAIN OBJECTIVE:
---------------
This script loads the CCF.media_processed_texts.csv database and annotates
each sentence using trained English and French models. It also can save/resume
progress to handle interruptions.

Dependencies:
-------------
- os
- glob
- re
- pandas
- numpy
- torch
- transformers
- tqdm

MAIN FEATURES:
--------------
1) Loads the CCF.media_processed_texts.csv file.
2) Removes approximate string matching (strict suffix detection only).
3) Supports intermediate saving to resume annotation if interrupted.
4) Prints distribution stats after each category and at the end.

Author:
-------
Antoine Lemor 
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


##############################################################################
#                 A. STRICT PARSING OF MODEL FILENAMES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictly extracts:
      - 'base_category' (e.g., "Event_Detection" or "Cult_1_SUB")
      - 'EN' or 'FR' language (must strictly end with "_EN" or "_FR")
      - 'Detection', 'SUB', or 'Other' type (must strictly end with "_Detection" or "_SUB")

    If it doesn't match the strict rules, return None for lang / type accordingly.
    """
    # Remove the extension
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    # 1) Detect language strictly
    lang = None
    if name.endswith('_EN'):
        lang = 'EN'
        name = name[:-3]  # remove "_EN"
    elif name.endswith('_FR'):
        lang = 'FR'
        name = name[:-3]  # remove "_FR"

    # 2) Determine model type strictly
    if name.endswith('_Detection'):
        model_type = 'Detection'
        # base_category = name (on garde tel quel, ex: "Eco_Detection")
        base_category = name
    elif name.endswith('_SUB'):
        model_type = 'SUB'
        base_category = name
    else:
        # ni _Detection ni _SUB => 'Other'
        model_type = 'Other'
        base_category = name

    return base_category, lang, model_type


##############################################################################
#                B. BUILDING A MODEL DICTIONARY (STRICT)
##############################################################################
def load_all_models_strict(models_dir):
    """
    Scans all *.model files and for each one detects:
      - base_category (e.g. "Event_Detection")
      - lang (EN/FR) strictly
      - type (Detection/SUB/Other)
    Builds a dictionary of the form:
    {
      "Event_Detection": {"EN": "/path/Event_Detection_EN.model", "FR": ...},
      "Event_1_SUB": {"EN": ..., "FR": ...},
      ...
    }
    Models that do not follow the strict naming convention are skipped.
    """
    model_files = glob.glob(os.path.join(models_dir, "*.model"))
    model_dict = {}

    for filepath in model_files:
        filename = os.path.basename(filepath)
        base_cat, lang, model_type = parse_model_filename_strict(filename)

        # Vérif stricte de la langue => si None, on skip
        if lang is None:
            print(f"[WARNING] File '{filename}' ignored (strict language suffix not found: must end with _EN or _FR).")
            continue

        # Note d'info
        print(f"[INFO] Mapping model file '{filename}' -> base_cat='{base_cat}', lang='{lang}', type='{model_type}'")

        if base_cat not in model_dict:
            model_dict[base_cat] = {}

        # On stocke juste le chemin.
        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#                      C. DEVICE DETECTION (GPU / CPU)
##############################################################################
def get_device():
    """
    Detects if a GPU (CUDA) or MPS (Mac Silicon) is available, otherwise CPU.
    """
    if torch.cuda.is_available():
        print("Using CUDA GPU for computations.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS GPU for computations.")
        return torch.device("mps")
    else:
        print("Using CPU for computations.")
        return torch.device("cpu")


##############################################################################
#        D. LOADING MODEL + TOKENIZER FROM PATH
##############################################################################
def load_model_and_tokenizer(model_path, lang, device):
    """
    Loads the model + tokenizer from model_path based on language.
    """
    if lang == 'FR':
        base_model_name = 'camembert-base'
    else:
        base_model_name = 'bert-base-uncased'

    # Load the base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512

    # Load the fine-tuned model from the local path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f"[DEBUG] Model '{model_path}' loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


##############################################################################
#     E. GENERIC ANNOTATION FUNCTION: PREDICT LABELS (0/1) PER BATCH
##############################################################################
def predict_labels(df, indices, text_column, model, tokenizer, device, output_col):
    """
    Annotates DataFrame df[output_col] on rows 'indices' with the given model/tokenizer.
    Performs batch-inference (batch_size=16).
    """
    if len(indices) == 0:
        return

    batch_size = 16
    texts = df.loc[indices, text_column].tolist()

    predictions = []
    with tqdm(total=len(texts), desc=f"Annotating '{output_col}'", unit="txt") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            pbar.update(len(batch_texts))

    df.loc[indices, output_col] = predictions


##############################################################################
#            F. MAIN ANNOTATION LOGIC (STRICT ORDER)
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path):
    """
    1) Retrieves the list of categories (Detection, SUB, Other) by the suffix of the name.
    2) Annotates each category, language by language.
    3) After each language within each category, the distribution is printed (0/1/NaN).
    4) An intermediate save is performed after each category-language pair.
    5) At the end, prints the final distribution for all columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be annotated.
    model_dict : dict
        Dictionary of model paths keyed by category and language.
    device : torch.device
        The computation device (CPU/CUDA/MPS).
    output_path : str
        CSV path to save the intermediate/final annotated DataFrame.

    Returns:
    --------
    df : pandas.DataFrame
        The updated DataFrame, annotated with new columns.
    """

    text_col = "sentences"
    lang_col = "language"

    # Classify categories strictly:
    #   - those ending with "_Detection" => detection
    #   - those ending with "_SUB"       => sub
    #   - otherwise => other
    categories_detection = []
    categories_sub = []
    categories_other = []

    # Sort categories alphabetically for deterministic order
    sorted_categories = sorted(model_dict.keys())

    for base_cat in sorted_categories:
        if base_cat.endswith('_Detection'):
            categories_detection.append(base_cat)
        elif base_cat.endswith('_SUB'):
            categories_sub.append(base_cat)
        else:
            categories_other.append(base_cat)

    # --- 1) MAIN CATEGORIES (DETECTION) ---------------------------------------
    print("\n[ANNOTATION] Step 1: Main Categories (Detection)")
    for cat_det in categories_detection:
        # Create the column if it doesn't exist
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        # For each language (EN/FR) present in model_dict[cat_det], annotate separately
        for lang, model_path in model_dict[cat_det].items():
            print(f"\n -> Now annotating category='{cat_det}' (type=Detection) for lang='{lang}' with model='{model_path}'")

            # Filter rows to annotate:
            #   - language == lang
            #   - text_col not missing
            #   - cat_det is NaN (not yet annotated)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_det}' / lang={lang}. (Already done or no data)")
                continue

            # Load model + tokenizer
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. "
                      f"Setting {cat_det} to NaN for these lines.\n   Reason: {e}")
                df.loc[idx, cat_det] = np.nan
                continue

            # Prediction
            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_det)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. "
                      f"Setting {cat_det} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_det] = np.nan
                continue

            # Intermediate save
            df.to_csv(output_path, index=False)

            # Print the distribution specifically for this language
            dist_lang = df.loc[df[lang_col] == lang, cat_det].value_counts(dropna=False)
            print(f"Distribution for '{cat_det}' (lang={lang}) after annotation:\n{dist_lang}")

        # After annotating all languages for this category, print overall distribution
        dist_after = df[cat_det].value_counts(dropna=False)
        print(f"\nOverall distribution for '{cat_det}' after all languages:\n{dist_after}")

    # --- 2) SUB-CATEGORIES (SUB) ----------------------------------------------
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # We derive the parent category: remove "_X_SUB" or "_SUB" from the end and add "_Detection"
        # For example: "Solutions_2_SUB" => "Solutions_Detection"
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # remove '_1_SUB', '_2_SUB', etc.
        main_category += '_Detection'

        # If the parent column doesn't exist, skip
        if main_category not in df.columns:
            print(f" - Warning: main category '{main_category}' does not exist for '{cat_sub}'. Skipping sub-annotation.")
            continue

        # For each language (EN/FR)
        for lang, model_path in model_dict[cat_sub].items():
            print(f"\n -> Now annotating sub-category='{cat_sub}' (type=SUB) for lang='{lang}' with model='{model_path}'")

            # We only target rows where main_category == 1
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &      # parent must be 1
                (df[cat_sub].isna())           # sub-cat not yet annotated
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_sub}' / lang={lang}. "
                      "(either no positives or already annotated).")
                continue

            # Load model + tokenizer
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. Setting {cat_sub} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_sub] = np.nan
                continue

            # Prediction
            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_sub)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. Setting {cat_sub} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_sub] = np.nan
                continue

            # Intermediate save
            df.to_csv(output_path, index=False)

            # Print the distribution specifically for this language
            dist_lang = df.loc[df[lang_col] == lang, cat_sub].value_counts(dropna=False)
            print(f"Distribution for '{cat_sub}' (lang={lang}) after annotation:\n{dist_lang}")

        # After annotating all languages for this sub-category, print overall distribution
        dist_after = df[cat_sub].value_counts(dropna=False)
        print(f"\nOverall distribution for '{cat_sub}' after all languages:\n{dist_after}")

    # --- 3) OTHER MODELS -------------------------------------------------------
    print("\n[ANNOTATION] Step 3: Other models (neither Detection nor SUB)")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        # For each language
        for lang, model_path in model_dict[cat_other].items():
            print(f"\n -> Now annotating other-category='{cat_other}' (type=Other) for lang='{lang}' "
                  f"with model='{model_path}'")

            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_other}' / lang={lang} "
                      f"(already done or no data).")
                continue

            # Load model + tokenizer
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. Setting {cat_other} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_other] = np.nan
                continue

            # Prediction
            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_other)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. Setting {cat_other} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_other] = np.nan
                continue

            # Save intermediate
            df.to_csv(output_path, index=False)

            # Print the distribution specifically for this language
            dist_lang = df.loc[df[lang_col] == lang, cat_other].value_counts(dropna=False)
            print(f"Distribution for '{cat_other}' (lang={lang}) after annotation:\n{dist_lang}")

        # After annotating all languages for this "other" category, print overall distribution
        dist_after = df[cat_other].value_counts(dropna=False)
        print(f"\nOverall distribution for '{cat_other}' after all languages:\n{dist_after}")

    # --- FINAL DISTRIBUTION FOR ALL ANNOTATED COLUMNS --------------------------
    print("\n[ANNOTATION] Final distribution for all annotated columns:")
    annotated_cols = categories_detection + categories_sub + categories_other
    for col in annotated_cols:
        if col in df.columns:
            dist_col = df[col].value_counts(dropna=False)
            print(f" - {col}: \n{dist_col}\n")

    return df


##############################################################################
#                   G. MAIN FUNCTION
##############################################################################
def main():
    """
    Main pipeline:
    1) Define paths (data, models, output).
    2) Read or resume an already-annotated CSV (if it exists).
    3) Scan the model directory (strict suffix detection).
    4) Annotation (3 steps: Detection, SUB, Other).
    5) Save and display final distributions.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1) Paths
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    models_dir = os.path.join(base_path, "..", "..", "models")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # 2) Load or resume
    if os.path.exists(output_path):
        print(f"[main] Existing annotation file detected: '{output_path}'. Resuming annotation...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] No existing annotation file. Loading initial CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into the DataFrame.")

    # 3) Load models (strict)
    print("[main] Loading model files (strict matching)...")
    model_dict = load_all_models_strict(models_dir)
    print(f"[main] Number of detected base_categories: {len(model_dict)}")

    # 4) Detect GPU/CPU
    device = get_device()

    # 5) Annotate
    print("[main] Starting annotation...")
    df_annotated = annotate_dataframe(df, model_dict, device, output_path)

    # 6) Final save
    print("[main] Saving the fully annotated DataFrame...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation complete. Output file: {output_path}")


if __name__ == "__main__":
    main()
