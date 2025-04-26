"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7bis_Personnalised_annotation.py

MAIN OBJECTIVE:
---------------
This script provides a way to selectively annotate the CCF.media_processed_texts.csv 
database using one or more retrained models. The user can choose which model(s) to use 
for re-annotation. Once chosen, any existing annotations (labels) for those model(s) 
are cleared and replaced by fresh predictions. This is especially useful if a model 
has been re-trained or updated and we want to update the dataset accordingly.

NEW FEATURES COMPARED TO 7_Annotation.py:
-----------------------------------------
1) Interactive selection of one or more specific model(s) for annotation.
2) Prior to re-annotation, existing labels for the selected model(s) are erased in 
   the CSV to ensure a clean re-run (fresh predictions overwrite old labels).
3) The same error reporting for sentences exceeding the 512-token limit and 
   distribution metrics logging are preserved.

Dependencies:
-------------
- os
- glob
- re
- sys
- pandas
- numpy
- torch
- tqdm
- warnings
- AugmentedSocialScientist 

MAIN STEPS (SUMMARY):
---------------------
1) Load or resume the base CSV (if already annotated).
2) Discover available models in the models directory (strict naming).
3) Prompt the user to select one or more models for re-annotation.
4) For each chosen model, clear the existing annotation column in the CSV, then re-annotate:
   a) "Detection" models are annotated for all sentences in the specified language.
   b) "SUB" models (e.g. "Cult_1_SUB") are only annotated for rows where the parent 
      detection column is == 1.
   c) "Other" models are annotated for all rows in the specified language.
5) Update CSV with predictions, log errors for oversize texts in 
   `sentences_annotation_error.csv`, and append distribution metrics to 
   `annotated_label_metrics.csv`.
6) Final CSV saved to `CCF.media_processed_texts_annotated.csv`.

Author:
-------
Antoine Lemor 
"""

import os
import glob
import re
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm.auto import tqdm

# --- Importing AugmentedSocialScientist models ---
from AugmentedSocialScientist.models import Camembert, Bert

##############################################################################
#                A. STRICT PARSING OF MODEL FILE NAMES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictly parse the model filename to extract:
      - base_category (e.g., "Event_Detection" or "Cult_1_SUB")
      - language 'EN' or 'FR' (must end with "_EN" or "_FR")
      - model_type in {'Detection', 'SUB', 'Other'}

    Returns
    -------
    base_category : str
    lang : str or None
        'EN' or 'FR' if found, else None
    model_type : str
        'Detection', 'SUB', or 'Other'
    """
    # Remove possible extensions
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    # Language
    lang = None
    if name.endswith('_EN'):
        lang = 'EN'
        name = name[:-3]  # remove "_EN"
    elif name.endswith('_FR'):
        lang = 'FR'
        name = name[:-3]  # remove "_FR"

    # Model type
    if name.endswith('_Detection'):
        model_type = 'Detection'
        base_category = name  # e.g., "Event_Detection"
    elif name.endswith('_SUB'):
        model_type = 'SUB'
        base_category = name  # e.g., "Cult_1_SUB"
    else:
        model_type = 'Other'
        base_category = name

    return base_category, lang, model_type


##############################################################################
#          B. LOADING ALL MODELS INTO A DICT (STRICT NAME)
##############################################################################
def load_all_models_strict(models_dir):
    """
    Scans all *.model files in the folder and applies parse_model_filename_strict.
    
    Returns
    -------
    model_dict : dict
        A nested dictionary of structure:
          {
            "Event_Detection": {
                "EN": "/path/Event_Detection_EN.model", 
                "FR": ...
            },
            "Cult_1_SUB": {
                "EN": ..., 
                "FR": ...
            },
            ...
          }
        Each key is the base_category (e.g. "Cult_1_SUB"), and the 
        sub-dictionary is keyed by language with the path to the model.
    """
    model_files = glob.glob(os.path.join(models_dir, "*.model"))
    model_dict = {}

    for filepath in model_files:
        filename = os.path.basename(filepath)
        base_cat, lang, model_type = parse_model_filename_strict(filename)

        if lang is None:
            print(f"[WARNING] File '{filename}' ignored (no _EN or _FR suffix).")
            continue

        print(f"[INFO] Model detected: '{filename}' -> base='{base_cat}', lang='{lang}', type='{model_type}'")

        if base_cat not in model_dict:
            model_dict[base_cat] = {}

        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#          C. DEVICE DETECTION (GPU / MPS / CPU)
##############################################################################
def get_device():
    """
    Returns a torch.device:
      - GPU CUDA if torch.cuda.is_available()
      - GPU MPS if torch.backends.mps.is_available()
      - Otherwise CPU
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
#       D. HANDLING TOO LONG SENTENCES -> ERROR CSV
##############################################################################
def check_text_exceeds_length_limit(text, tokenizer, max_length=512):
    """
    Checks if the tokenized sequence exceeds the imposed limit (512 tokens).

    Returns
    -------
    bool
        True if the text exceeds 512 tokens (likely triggers a warning message).
    """
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return (len(encoded) > max_length)


def init_error_csv(error_csv_path):
    """
    Creates a CSV file with a header if it does not exist. Used to log
    sentences that exceed the 512-token limit.
    """
    if not os.path.exists(os.path.dirname(error_csv_path)):
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)

    if not os.path.exists(error_csv_path):
        df = pd.DataFrame(columns=["row_id", "lang", "category", "text"])
        df.to_csv(error_csv_path, index=False)


def append_to_error_csv(error_csv_path, rows):
    """
    Appends a list of dicts (keys: row_id, lang, category, text) to the 
    existing error CSV in append mode (no header).
    """
    if not rows:
        return
    df_err = pd.DataFrame(rows)
    df_err.to_csv(error_csv_path, mode='a', header=False, index=False)


##############################################################################
#       E. METRICS CSV (FOR DISTRIBUTION OF LABELS)
##############################################################################
def init_metrics_csv(metrics_csv_path):
    """
    Creates a CSV file with a header if it does not exist. Used to log 
    distribution metrics for annotated labels (0,1,NaN).
    """
    if not os.path.exists(os.path.dirname(metrics_csv_path)):
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

    if not os.path.exists(metrics_csv_path):
        cols = ["category", "lang", "label_value", "count"]
        pd.DataFrame(columns=cols).to_csv(metrics_csv_path, index=False)


def append_to_metrics_csv(metrics_csv_path, category, lang, value_counts):
    """
    Appends distribution metrics to 'annotated_label_metrics.csv' in the form:
      category | lang | label_value | count
    """
    if value_counts.empty:
        return

    rows = []
    for label_val, nb in value_counts.items():
        rows.append({
            "category": category,
            "lang": lang,
            "label_value": label_val,
            "count": nb
        })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_csv_path, mode='a', header=False, index=False)


##############################################################################
#    F. PREDICTION HELPER (BATCH ANNOTATION, TOKEN LENGTH ERRORS)
##############################################################################
def predict_labels(
    df,
    indices,
    text_column,
    model_path,
    lang,
    device,
    output_col,
    error_csv_path,
    batch_size=200
):
    """
    Given a list of indices in 'df', predict labels for df[output_col] using the 
    model at 'model_path' (Camembert or Bert) on the specified device. 

    Steps:
    ------
    1. Select the relevant text samples by 'indices' from df[text_column].
    2. Instantiate the correct model (Camembert if lang=FR, Bert if lang=EN).
    3. Batch-encode and predict probabilities => argmax => label.
    4. If any sentence exceeds 512 tokens, log it to the error CSV.
    5. Assign predictions to df.loc[indices, output_col].
    """
    if len(indices) == 0:
        return

    texts = df.loc[indices, text_column].tolist()

    # Instantiate model
    if lang == 'FR':
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    # We'll keep track of predictions in a list
    predictions = []
    index_list = list(indices)  # to track row IDs for errors, etc.

    with tqdm(total=len(texts), desc=f"Annot '{output_col}'", unit="txt") as pbar:
        for start_i in range(0, len(texts), batch_size):
            batch_texts = texts[start_i:start_i + batch_size]
            batch_idx = index_list[start_i:start_i + batch_size]

            # Check length for each text to see if it surpasses 512 tokens
            error_rows = []
            for local_i, t in enumerate(batch_texts):
                if check_text_exceeds_length_limit(t, model.tokenizer):
                    row_id = batch_idx[local_i]
                    error_rows.append({
                        "row_id": row_id,
                        "lang": lang,
                        "category": output_col,
                        "text": t
                    })
            if error_rows:
                append_to_error_csv(error_csv_path, error_rows)

            # Encode for prediction
            batch_loader = model.encode(
                batch_texts,
                labels=None,
                batch_size=len(batch_texts),
                progress_bar=False
            )

            # Predict
            try:
                probs = model.predict_with_model(
                    batch_loader,
                    model_path=model_path,
                    proba=True,
                    progress_bar=False
                )
                batch_preds = np.argmax(probs, axis=1).tolist()

            except Exception as e:
                print(f"   [ERROR] Prediction failed on model='{model_path}'. Reason: {e}")
                # If an error occurs, fill with NaN
                batch_preds = [np.nan] * len(batch_texts)

            predictions.extend(batch_preds)
            pbar.update(len(batch_texts))

    df.loc[indices, output_col] = predictions


##############################################################################
#            G. CORE ANNOTATION PER MODEL (DETECTION, SUB, OR OTHER)
##############################################################################
def annotate_one_category(df, category, lang, model_path, device, error_csv_path, metrics_csv_path, output_path):
    """
    Annotates a single category (e.g., "Cult_1_SUB" or "Event_Detection") 
    for a given language in the DataFrame 'df'.

    Steps:
    ------
    1. Determine model type (Detection, SUB, or Other).
    2. If needed for SUB, filter to rows where the parent detection == 1.
    3. Predict labels => store in df[category].
    4. Update CSV with distribution stats in 'annotated_label_metrics.csv'.
    5. Save intermediate DF to disk.

    Returns
    -------
    df : pd.DataFrame
        The updated dataframe.
    """
    text_col = "sentences"
    lang_col = "language"

    # Identify the model type based on suffix
    if category.endswith("_Detection"):
        model_type = "Detection"
    elif category.endswith("_SUB"):
        model_type = "SUB"
    else:
        model_type = "Other"

    # Ensure the column exists in df
    if category not in df.columns:
        df[category] = np.nan

    print(f"\n[ANNOTATION] Annotating '{category}' (type='{model_type}') for lang='{lang}' using '{model_path}'")

    # For a SUB model, we need the parent detection column
    if model_type == "SUB":
        # Example: "Cult_1_SUB" => parent = "Cult_Detection"
        # If "Health_2_SUB" => parent = "Health_Detection"
        # Basically, remove the trailing digits + "_SUB", then add "_Detection"
        parent_cat = re.sub(r'_?\d*_SUB$', '', category)
        parent_cat += '_Detection'

        if parent_cat not in df.columns:
            print(f"   [WARN] Parent detection column '{parent_cat}' not in DF. Skipping annotation.")
            return df

        # We only annotate where parent_cat == 1
        idx = df[
            (df[lang_col] == lang) &
            (df[text_col].notna()) &
            (df[parent_cat] == 1)
        ].index

    elif model_type == "Detection":
        idx = df[
            (df[lang_col] == lang) &
            (df[text_col].notna())
        ].index

    else:  # model_type == "Other"
        idx = df[
            (df[lang_col] == lang) &
            (df[text_col].notna())
        ].index

    # Clear existing annotation for the selected category & language
    # so that we're re-annotating from scratch for these rows
    df.loc[idx, category] = np.nan

    if len(idx) == 0:
        print(f"   => No rows to annotate for category='{category}' / lang='{lang}'. Skipping.")
        return df

    # Predict
    predict_labels(
        df=df,
        indices=idx,
        text_column=text_col,
        model_path=model_path,
        lang=lang,
        device=device,
        output_col=category,
        error_csv_path=error_csv_path,
        batch_size=200
    )

    # Intermediate save
    df.to_csv(output_path, index=False)

    # Distribution
    dist_after = df.loc[df[lang_col] == lang, category].value_counts(dropna=False)
    print(f"[INFO] Distribution for '{category}' (lang='{lang}'):\n{dist_after}")

    # Add metrics
    append_to_metrics_csv(metrics_csv_path, category, lang, dist_after)

    return df


##############################################################################
#   H. PROMPT USER FOR WHICH MODEL(S) TO ANNOTATE
##############################################################################
def prompt_model_selection(model_dict):
    """
    Displays all discovered models and prompts user to select one or more 
    (by index). Returns a list of (category, lang) pairs that the user wants 
    to re-annotate.

    The 'model_dict' has structure:
      { base_cat: { 'EN': path, 'FR': path, ... }, ... }
    """

    # Build a flat list of (cat, lang, model_path)
    all_models = []
    for cat, subd in model_dict.items():
        for lang, path in subd.items():
            all_models.append((cat, lang, path))

    if not all_models:
        print("[ERROR] No valid models found in directory. Exiting.")
        sys.exit(1)

    # Display them
    print("\n[MODEL SELECTION] Available models:")
    for i, (cat, lang, path) in enumerate(all_models):
        print(f"[{i}] category='{cat}', lang='{lang}', path='{os.path.basename(path)}'")

    # Prompt user
    print("\nEnter the indices of the model(s) you wish to re-annotate, separated by commas.\n"
          "For example: 0,2 or 1,3,5. Press Enter when done.")
    valid_indices = list(range(len(all_models)))
    chosen_indices = []

    while True:
        user_input = input("Indices: ").strip()
        if not user_input:
            print("No model indices entered. Exiting.")
            sys.exit(0)
        try:
            # Parse user input
            raw_list = [x.strip() for x in user_input.split(',')]
            chosen_indices = [int(x) for x in raw_list]
            # Check if all in valid range
            if all(x in valid_indices for x in chosen_indices):
                break
            else:
                print(f"Invalid selection. Valid indices are 0 to {len(all_models)-1}.")
        except ValueError:
            print("Invalid input. Please enter comma-separated integers only.")

    # Deduplicate and sort user indices
    chosen_indices = sorted(set(chosen_indices))

    # Build the final selection as a list of (category, lang, path)
    selected_models = [all_models[i] for i in chosen_indices]
    return selected_models


##############################################################################
#                           I. MAIN FUNCTION
##############################################################################
def main():
    """
    1) Define paths (data, models, output).
    2) Load or resume the CSV if already annotated.
    3) Discover available models (strict suffix parsing).
    4) Prompt the user which model(s) they wish to re-annotate.
    5) For each chosen model, clear old annotations in the relevant column 
       and re-run the predictions. Log distribution & errors.
    6) Final save.
    """

    print("Script started.")  # Debug message in English

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Input / output files
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # Model directory
    models_dir = os.path.join(base_path, "..", "..", "models")

    # Error CSV (token length) and metrics CSV
    error_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "sentences_annotation_error.csv")
    metrics_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotated_label_metrics.csv")

    # Load or resume DF
    if os.path.exists(output_path):
        print(f"[main] Annotated file already found at '{output_path}'. Loading...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        if not os.path.exists(data_path):
            print(f"[ERROR] No base file found at '{data_path}'. Exiting.")
            sys.exit(1)
        print(f"[main] No existing annotation file found. Loading base CSV from '{data_path}'...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into DataFrame.")

    # Discover available models
    print("[main] Scanning models directory for valid .model files...")
    model_dict = load_all_models_strict(models_dir)

    # If no models found, exit
    if not model_dict:
        print("[ERROR] No strict-suffix models detected. Exiting.")
        sys.exit(1)

    # Prompt the user to select one or more models
    selected_models = prompt_model_selection(model_dict)

    # Initialize error & metrics CSV
    init_error_csv(error_csv_path)
    init_metrics_csv(metrics_csv_path)

    # Device for prediction
    device = get_device()

    # Annotate each chosen model
    for (category, lang, model_path) in selected_models:
        df = annotate_one_category(
            df=df,
            category=category,
            lang=lang,
            model_path=model_path,
            device=device,
            error_csv_path=error_csv_path,
            metrics_csv_path=metrics_csv_path,
            output_path=output_path
        )

    # Final save
    print("[main] Saving final annotated DataFrame...")
    df.to_csv(output_path, index=False)
    print(f"[main] Annotation completed and saved at '{output_path}'")

    print("Script ended.")  # Debug message in English


if __name__ == "__main__":
    main()
