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
2) Uses approximate string matching for suffix recognition.
3) Supports intermediate saving to resume annotation if interrupted.

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

from difflib import SequenceMatcher  # For string similarity


##############################################################################
#                          A. SIMILARITY UTILITIES
##############################################################################
def approximate_match(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Returns True if the similarity between a and b (case-insensitive)
    is >= threshold, based on SequenceMatcher ratio.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def approximate_endswith(name: str, suffix: str, threshold: float = 0.95) -> bool:
    """
    Checks if the end of `name` is "close" to `suffix` based on a similarity threshold.
    Example: approximate_endswith('Solution_detection', '_Detection') => True
    if the ratio is >= threshold.
    """
    # Only consider the part of name with length similar to suffix
    if len(name) < len(suffix):
        return False
    end_part = name[-len(suffix):]
    return approximate_match(end_part, suffix, threshold=threshold)


##############################################################################
#                      B. MODEL FILENAME PARSING
##############################################################################
def parse_model_filename(filename):
    """
    Extracts:
      - 'base_category' (e.g., "Event_Detection" or "Cult_1_SUB")
      - 'EN' or 'FR' language
      - type ('Detection', 'SUB', or 'Other') for info.

    Uses similarity to tolerate minor errors: "Solution_detectio" ≈ "_Detection"
    or "Solutons_2_SUB" ≈ "_SUB", etc.
    """

    # Remove the extension
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    # 1) Detect the language (approx): test if the end is close to "_EN" or "_FR"
    lang = None
    if approximate_endswith(name, '_EN', 0.95):
        lang = 'EN'
        # Remove the ending part corresponding to _EN
        name = name[:-3]  # remove exactly 3 characters
    elif approximate_endswith(name, '_FR', 0.95):
        lang = 'FR'
        name = name[:-3]

    # 2) Determine if it's a Detection, SUB, or Other, by looking at the end
    #    of the string (approx)
    if approximate_endswith(name, '_Detection', 0.95):
        model_type = 'Detection'
        # remove the suffix (length len('_Detection')=10)
        base_category = name  # the base category will include the end "_Detection" (or approx)
    elif approximate_endswith(name, '_SUB', 0.95):
        model_type = 'SUB'
        base_category = name
    else:
        model_type = 'Other'
        base_category = name

    return base_category, lang, model_type


##############################################################################
#         C. CREATING MODEL DICTIONARY STRUCTURE
##############################################################################
def load_all_models(models_dir):
    """
    Iterates through all *.model files in the directory, parses their names, and builds
    a dict in the form:
    {
      "Event_Detection": {
          "EN": "/path/Event_Detection_EN.model",
          "FR": "/path/Event_Detection_FR.model"
      },
      "Event_1_SUB": {
          "EN": "/path/Event_1_SUB_EN.model",
          "FR": "/path/Event_1_SUB_FR.model"
      },
      "Emotion:_Positive": {
          "EN": "...",
          "FR": "..."
      },
      ...
    }

    This dictionary unifies EN/FR under the same key (base_category).
    """
    model_files = glob.glob(os.path.join(models_dir, "*.model"))
    model_dict = {}

    for filepath in model_files:
        filename = os.path.basename(filepath)
        base_cat, lang, _ = parse_model_filename(filename)

        # If the language is not detected (None), ignore it.
        if lang is None:
            print(f"[WARNING] File {filename} ignored (language not found).")
            continue

        if base_cat not in model_dict:
            model_dict[base_cat] = {}

        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#                 D. DEVICE DETECTION (GPU / CPU)
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
#        E. LOADING MODEL + TOKENIZER FROM PATH
##############################################################################
def load_model_and_tokenizer(model_path, lang, device):
    """
    Loads the model + tokenizer from model_path.
    """
    if lang == 'FR':
        base_model_name = 'camembert-base'
    else:
        base_model_name = 'bert-base-uncased'

    # Load the base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512

    # Load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Add logs to verify the device of the model
    print(f"[DEBUG] Model loaded on device: {next(model.parameters()).device}")

    return model, tokenizer


##############################################################################
#           F. GENERIC ANNOTATION FUNCTION WITH MODEL
##############################################################################
def predict_labels(df, indices, text_column, model, tokenizer, device, output_col):
    """
    Annotates (with tqdm) DataFrame df[output_col] on rows 'indices'
    using 'model'/'tokenizer'. Performs batch-inference (16).
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

    # Update the df
    df.loc[indices, output_col] = predictions


##############################################################################
#                    G. MAIN LOGIC FOR ANNOTATION
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path):
    """
    1) For each key in model_dict (e.g., "Event_Detection", "Cult_1_SUB", ...),
       determine if it's a Detection, SUB, or Other, and annotate accordingly.
    2) Create a single column (base_category) to merge EN/FR.
    3) After each category (or sub-category), save the CSV to resume in case of interruption.
    """

    text_col = "sentences"
    lang_col = "language"

    # -- List the order (Detection, SUB, Other) to annotate main categories first,
    #    then sub-categories, then the rest.
    categories_detection = []
    categories_sub = []
    categories_other = []

    # Sort the keys to avoid random order
    sorted_categories = sorted(model_dict.keys())

    for base_cat in sorted_categories:
        if approximate_endswith(base_cat, '_Detection', 0.95):
            categories_detection.append(base_cat)
        elif approximate_endswith(base_cat, '_SUB', 0.95):
            categories_sub.append(base_cat)
        else:
            categories_other.append(base_cat)

    # --- 1) Main categories: Annotate all rows
    print("\n[ANNOTATION] Step 1: Main Categories (Detection)...")
    for cat_det in categories_detection:
        if cat_det not in df.columns:
            df[cat_det] = np.nan  # create column if it doesn't exist

        # Annotate language by language
        for lang, model_path in model_dict[cat_det].items():
            # Select rows not yet annotated
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())  # Do not re-annotate already filled rows
            ].index

            if len(idx) == 0:
                print(f" - No rows to annotate for '{cat_det}' / lang={lang}. (Already done or no data)")
                continue

            print(f" - Annotating category {cat_det} (lang={lang}) on {len(idx)} rows.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERROR] Unable to load model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_det)
            except Exception as e:
                print(f" [ERROR] Annotation failed for model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            # Intermediate save to resume in case of abrupt stop
            df.to_csv(output_path, index=False)

    # --- 2) Sub-categories (SUB): Annotate only positive rows (==1)
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)...")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # Reconstruct main category: if named "Xxx_1_SUB",
        # main category is "Xxx_Detection" (remove "_1_SUB" or "_SUB"
        # and replace with "_Detection").
        # Handle cases like "Solutions_2_SUB" => "Solutions_Detection"
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # remove "_1_SUB" or "_2_SUB", etc.
        main_category = main_category + '_Detection'

        if main_category not in df.columns:
            print(f" - Warning: main category '{main_category}' does not exist for '{cat_sub}'. Skipping.")
            continue

        for lang, model_path in model_dict[cat_sub].items():
            # Do not re-annotate if already filled
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &    # main category must be 1
                (df[cat_sub].isna())         # not already annotated
            ].index

            if len(idx) == 0:
                print(f" - No rows to annotate for '{cat_sub}' / lang={lang} (either no positives or already annotated).")
                continue

            print(f" - Annotating sub-category {cat_sub} (lang={lang}) on {len(idx)} rows.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERROR] Unable to load model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_sub)
            except Exception as e:
                print(f" [ERROR] Annotation failed for model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            # Intermediate save
            df.to_csv(output_path, index=False)

    # --- 3) Other models: Annotate all rows
    print("\n[ANNOTATION] Step 3: Other models (neither Detection nor SUB)...")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        for lang, model_path in model_dict[cat_other].items():
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f" - No rows to annotate for '{cat_other}' / lang={lang} (already done or no data).")
                continue

            print(f" - Annotating category {cat_other} (lang={lang}) on {len(idx)} rows.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERROR] Unable to load model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_other)
            except Exception as e:
                print(f" [ERROR] Annotation failed for model '{model_path}'. Skipping this model.\n   Reason: {e}")
                continue

            # Intermediate save
            df.to_csv(output_path, index=False)

    return df


##############################################################################
#                   H. MAIN FUNCTION
##############################################################################
def main():
    """
    The main entry point for executing the entire annotation pipeline:
    1) Defines and checks all necessary file paths.
    2) Loads or resumes a partially annotated DataFrame.
    3) Scans a directory of model files and organizes them by category.
    4) Detects GPU/CPU availability for optimized performance.
    5) Annotates sentences across multiple language-specific models.
    6) Saves intermediate results to facilitate resuming on unexpected interruptions.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1) Define paths
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    models_dir = os.path.join(base_path, "..", "..", "models")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # 2) Check if an annotation file already exists
    if os.path.exists(output_path):
        print(f"[main] Existing annotation file detected: '{output_path}'. Resuming annotation...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] No existing annotation file. Loading initial CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into the DataFrame.")

    # 3) Load the list of models
    print("[main] Loading model files...")
    model_dict = load_all_models(models_dir)
    print(f"[main] Number of detected categories: {len(model_dict)}")

    # 4) Detect device
    device = get_device()

    # 5) Annotate
    print("[main] Starting annotation...")
    df_annotated = annotate_dataframe(df, model_dict, device, output_path)

    # 6) Save final result
    print("[main] Saving the annotated DataFrame...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation complete. Output file: {output_path}")


if __name__ == "__main__":
    main()
