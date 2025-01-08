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
each sentence using pre-trained English or French models
(using the AugmentedSocialScientist library).
It can save/resume progress to handle potential interruptions.

NEW FEATURES:
--------------------------
1) The _SUB categories are only annotated for rows where the
   parent category _Detection = 1 (others are not processed).
2) A CSV `sentences_annotation_error.csv` is created in Database/Training_data
   if some sentences exceed the 512 token limit and trigger
   the length warning ("Token indices sequence length is longer ...").
   It records index, sentence, category, etc.
3) A CSV `annotated_label_metrics.csv` is maintained in Database/Training_data.
   For each annotated category/language, it records the distribution
   (value_counts) of labels (0,1,NaN).

Dependencies:
-------------
- os
- glob
- re
- pandas
- numpy
- torch
- tqdm
- warnings
- AugmentedSocialScientist (pip install AugmentedSocialScientist)

MAIN FEATURES (inherited):
-------------------------
1) Loads the CCF.media_processed_texts.csv database.
2) Searches for EN/FR models in a directory.
3) Manages a strict suffix system to separate (Detection, SUB, Other).
4) Saves at each step to allow resuming if interrupted.
5) Writes the final annotated DataFrame to CCF.media_processed_texts_annotated.csv

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
import warnings
from tqdm.auto import tqdm

# --- Importing AugmentedSocialScientist models ---
# For FR: Camembert
# For EN: Bert
from AugmentedSocialScientist.models import Camembert, Bert

##############################################################################
#                A. STRICT PARSING OF MODEL FILE NAMES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictly, we look for:
      - "base_category" (e.g., "Event_Detection" or "Cult_1_SUB")
      - the language 'EN' or 'FR' (must end with "_EN" or "_FR")
      - the type 'Detection', 'SUB' or 'Other' (must end with "_Detection" or "_SUB")

    If the name is not compliant, we return lang=None/type=None.
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
        base_category = name
    elif name.endswith('_SUB'):
        model_type = 'SUB'
        base_category = name
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
    Returns a dictionary:
    {
      "Event_Detection": {"EN": "/path/Event_Detection_EN.model", "FR": ...},
      "Cult_1_SUB":      {"EN": ..., "FR": ...},
      ...
    }
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
    Checks if the tokenized sequence exceeds the imposed limit (512).
    Returns True if the text exceeds 512 tokens (thus potentially
    triggering a warning message).
    """
    # Disable truncation to measure the actual size
    # add_special_tokens=True => includes [CLS], [SEP]
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return (len(encoded) > max_length)


##############################################################################
# E. PREPARING CSV FOR ERRORS AND METRICS
##############################################################################
def init_error_csv(error_csv_path):
    """
    Initializes (or opens in append mode) the CSV that will contain
    sentences with length issues.
    """
    if not os.path.exists(os.path.dirname(error_csv_path)):
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)

    if not os.path.exists(error_csv_path):
        # Create the file with a header
        pd.DataFrame(columns=["row_id", "lang", "category", "text"]).to_csv(error_csv_path, index=False)


def append_to_error_csv(error_csv_path, rows):
    """
    Appends to the error CSV (in append mode) the list of dictionaries `rows`.
    Each element of `rows` must be a dict with
    { "row_id":..., "lang":..., "category":..., "text":... }
    """
    if not rows:
        return
    df_err = pd.DataFrame(rows)
    df_err.to_csv(error_csv_path, mode='a', header=False, index=False)


def init_metrics_csv(metrics_csv_path):
    """
    Initializes the CSV that will contain distribution metrics.
    """
    if not os.path.exists(os.path.dirname(metrics_csv_path)):
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

    if not os.path.exists(metrics_csv_path):
        cols = ["category", "lang", "label_value", "count"]
        pd.DataFrame(columns=cols).to_csv(metrics_csv_path, index=False)


def append_to_metrics_csv(metrics_csv_path, category, lang, value_counts):
    """
    value_counts is a Series => index=label_value (0.0,1.0,NaN), values=count
    Feeds a CSV "annotated_label_metrics.csv" in append mode,
    with columns = category, lang, label_value, count
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
#   F. GENERIC ANNOTATION (PREDICTION) FUNCTION BY BATCHES + ERROR HANDLING
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
    batch_size=250
):
    """
    This function annotates df[output_col] on the 'indices' rows using
    a pre-trained model (whose folder is model_path) via AugmentedSocialScientist.

    1) Retrieve texts in df.loc[indices, text_column].
    2) Instantiate a Camembert() or Bert() model depending on lang, on device.
    3) By batch, encode => check length to detect "errors".
       - If a text exceeds 512 tokens, log it in the error CSV.
         Continue prediction (as huggingface will truncate).
    4) Use 'predict_with_model' to get probabilities, then argmax => label (0/1).
    5) Update df[output_col] for the relevant rows.
    """

    if len(indices) == 0:
        return

    # Select texts
    texts = df.loc[indices, text_column].tolist()

    # Instantiate the model
    if lang == 'FR':
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    # Prepare an iterator on indices => to log the error correctly
    indices_list = list(indices)  # to iterate along with texts

    # Batch loop
    predictions = []
    with tqdm(total=len(texts), desc=f"Annot '{output_col}'", unit="txt") as pbar:
        for start_i in range(0, len(texts), batch_size):
            batch_texts = texts[start_i:start_i + batch_size]
            batch_idx = indices_list[start_i:start_i + batch_size]

            # Check the length of each sentence before encoding
            # to detect those that will exceed 512 tokens
            error_rows = []
            for local_i, t in enumerate(batch_texts):
                if check_text_exceeds_length_limit(t, model.tokenizer, max_length=512):
                    row_id = batch_idx[local_i]
                    error_rows.append({
                        "row_id": row_id,
                        "lang": lang,
                        "category": output_col,
                        "text": t
                    })

            # Append these potential errors to the CSV
            if error_rows:
                append_to_error_csv(error_csv_path, error_rows)

            # Actual encoding
            # (AugmentedSocialScientist handles automatic truncation,
            # so even if >512, it won't crash, it will truncate. We just logged it.)
            batch_loader = model.encode(
                batch_texts,
                labels=None,         # we don't have labels
                batch_size=len(batch_texts),
                progress_bar=False
            )

            try:
                # Prediction => shape (N,2) if binary
                probs = model.predict_with_model(
                    batch_loader,
                    model_path=model_path,
                    proba=True,         # we want probabilities
                    progress_bar=False
                )
                batch_preds = np.argmax(probs, axis=1).tolist()

            except Exception as e:
                # In case of exception, log and set NaN
                print(f"   [ERROR] Failed predict_with_model on '{model_path}'. Reason: {e}")
                batch_preds = [np.nan] * len(batch_texts)

            predictions.extend(batch_preds)

            pbar.update(len(batch_texts))

    # Update the DF
    df.loc[indices, output_col] = predictions


##############################################################################
#           G. MAIN ANNOTATION LOGIC (IN 3 STEPS)
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path, error_csv_path, metrics_csv_path):
    """
    - First, identify "Detection", "SUB", "Other" categories
      via their strict suffix (_Detection, _SUB, otherwise Other).
    - Annotate in order: Detection => SUB => Other.
    - After each category/language, save the CSV + update
      the metrics CSV (annotated_label_metrics.csv).
    """

    text_col = "sentences"
    lang_col = "language"

    # 1) Classification of categories by suffix
    categories_detection = []
    categories_sub = []
    categories_other = []

    sorted_categories = sorted(model_dict.keys())
    for base_cat in sorted_categories:
        if base_cat.endswith('_Detection'):
            categories_detection.append(base_cat)
        elif base_cat.endswith('_SUB'):
            categories_sub.append(base_cat)
        else:
            categories_other.append(base_cat)

    # ----------------------------------------------------------------
    # STEP 1: DETECTION
    # ----------------------------------------------------------------
    print("\n[ANNOTATION] Step 1: Main Categories (Detection)")
    for cat_det in categories_detection:
        # If the column does not already exist in the DF, create it
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        # Annotate for each language where a model exists
        for lang, model_path in model_dict[cat_det].items():
            print(f"\n -> Now annotating '{cat_det}' (Detection) for lang='{lang}' with model='{model_path}'")

            # Select rows that do not yet have a label (isna)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_det}' / lang={lang}.")
                continue

            # Prediction
            predict_labels(
                df, idx, text_col, 
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_det,
                error_csv_path=error_csv_path,
                batch_size=250
            )
            # Intermediate save
            df.to_csv(output_path, index=False)

            # Partial distribution for the language
            dist_lang = df.loc[df[lang_col] == lang, cat_det].value_counts(dropna=False)
            print(f"Distribution for '{cat_det}' (lang={lang}):\n{dist_lang}")

            # Feed the metrics CSV
            append_to_metrics_csv(metrics_csv_path, cat_det, lang, dist_lang)

        # Overall distribution
        dist_after = df[cat_det].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_det}':\n{dist_after}")

    # ----------------------------------------------------------------
    # STEP 2: SUB
    # ----------------------------------------------------------------
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # Determine the parent category
        # (e.g., "Cult_1_SUB" => parent = "Cult_Detection")
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # remove '_1_SUB', '_2_SUB', ...
        main_category += '_Detection'

        if main_category not in df.columns:
            print(f"   [WARN] Parent category '{main_category}' missing for '{cat_sub}'. Skipping.")
            continue

        # For each language
        for lang, model_path in model_dict[cat_sub].items():
            print(f"\n -> Now annotating '{cat_sub}' (SUB) for lang='{lang}' with model='{model_path}'")

            # Unlike "Detection", annotate only where:
            # - language = lang
            # - sentences notna
            # - parent detection = 1
            # - cat_sub is NaN (not yet annotated)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &
                (df[cat_sub].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No positive rows to annotate for '{cat_sub}' / lang={lang}.")
                continue

            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_sub,
                error_csv_path=error_csv_path,
                batch_size=250
            )
            df.to_csv(output_path, index=False)

            dist_lang = df.loc[df[lang_col] == lang, cat_sub].value_counts(dropna=False)
            print(f"Distribution for '{cat_sub}' (lang={lang}):\n{dist_lang}")

            # Add to metrics CSV
            append_to_metrics_csv(metrics_csv_path, cat_sub, lang, dist_lang)

        dist_after = df[cat_sub].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_sub}':\n{dist_after}")

    # ----------------------------------------------------------------
    # STEP 3: OTHER
    # ----------------------------------------------------------------
    print("\n[ANNOTATION] Step 3: Other models (neither Detection nor SUB)")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        for lang, model_path in model_dict[cat_other].items():
            print(f"\n -> Now annotating '{cat_other}' (Other) for lang='{lang}' with model='{model_path}'")

            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_other}' / lang={lang}.")
                continue

            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_other,
                error_csv_path=error_csv_path,
                batch_size=250
            )
            df.to_csv(output_path, index=False)

            dist_lang = df.loc[df[lang_col] == lang, cat_other].value_counts(dropna=False)
            print(f"Distribution for '{cat_other}' (lang={lang}):\n{dist_lang}")

            # Add to metrics CSV
            append_to_metrics_csv(metrics_csv_path, cat_other, lang, dist_lang)

        dist_after = df[cat_other].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_other}':\n{dist_after}")

    # --- Final summary
    print("\n[ANNOTATION] Final distribution summary:")
    all_cols = categories_detection + categories_sub + categories_other
    for col in all_cols:
        if col in df.columns:
            dist_col = df[col].value_counts(dropna=False)
            print(f" - {col} : \n{dist_col}\n")

    return df


##############################################################################
#                           H. MAIN FUNCTION
##############################################################################
def main():
    """
    1) Define paths (data, models, output).
    2) Load or resume the CSV if already annotated.
    3) Scan the model directory (strict suffixes).
    4) Annotate!
    5) Final save.
    6) Sentences >512 tokens are logged in sentences_annotation_error.csv
    7) Label distributions are logged incrementally in annotated_label_metrics.csv
    """

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Input / output files
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # Model directory
    models_dir = os.path.join(base_path, "..", "..", "models")

    # Error CSV (token length)
    error_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "sentences_annotation_error.csv")

    # Metrics CSV
    metrics_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotated_label_metrics.csv")

    # Loading / resuming
    if os.path.exists(output_path):
        print(f"[main] Annotation file already exists: '{output_path}'. Resuming...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] No existing annotation file. Initial loading of CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into DataFrame.")

    # Loading available models
    print("[main] Loading model files (strict suffixes)...")
    model_dict = load_all_models_strict(models_dir)
    print(f"[main] Number of detected categories: {len(model_dict)}")

    # Device
    device = get_device()

    # Initialize error & metrics CSV
    init_error_csv(error_csv_path)
    init_metrics_csv(metrics_csv_path)

    # Main annotation
    print("[main] Starting annotation...")
    df_annotated = annotate_dataframe(
        df=df,
        model_dict=model_dict,
        device=device,
        output_path=output_path,
        error_csv_path=error_csv_path,
        metrics_csv_path=metrics_csv_path
    )

    # Final save
    print("[main] Saving final annotated DataFrame...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation completed. Final file: {output_path}")


if __name__ == "__main__":
    main()
