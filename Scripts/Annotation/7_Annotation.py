#!/usr/bin/env python3
"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7_Annotation_parallel.py

MAIN OBJECTIVE:
---------------
This script loads the CCF.media_processed_texts.csv database and annotates
each sentence using pre-trained English or French models (using the
AugmentedSocialScientist library). It can save/resume progress to handle
potential interruptions.

NEW FEATURES:
-------------
1) The _SUB categories are only annotated for rows where the
   parent category _Detection = 1 (others are not processed).
2) A CSV `sentences_annotation_error.csv` is created in Database/Training_data
   if some sentences exceed the 512 token limit and trigger
   the length warning ("Token indices sequence length is longer ...").
   It records index, sentence, category, etc.
3) A CSV `annotated_label_metrics.csv` is maintained in Database/Training_data.
   For each annotated category/language, it records the distribution
   (value_counts) of labels (0,1,NaN).
4) A sentence-level tqdm progress bar is displayed in each process to track
   the progress within each chunk.
5) After a model finishes annotating its target category for a given language,
   an intermediate save is made to the output CSV (i.e. save per category).

ADDED PARALLELIZATION + LANGUAGE CHOICE:
----------------------------------------
1) At the start, the user is prompted to choose which language(s) to annotate:
   - 'EN' only
   - 'FR' only
   - 'both'
   This choice decides which rows are processed and the final output file name
   will end with either '_EN.csv', '_FR.csv', or '_full.csv'.

2) Multiple processes are spawned to annotate chunks of data in parallel.
   A shared pool of tasks is built and each worker – preaffecté à un device via
   une file partagée – récupère les chunks à traiter dès qu’il devient disponible.
   TQDM bars track both chunk-level and sentence-level progress and a global tqdm
   bar shows the number of chunks successfully processed.

ADDITIONAL FEATURE:
-------------------
The inference batch size is now separated:
   - GPU_BATCH_SIZE: batch size when processing on GPU (e.g., "cuda" or "mps")
   - CPU_BATCH_SIZE: batch size when processing on CPU
The code now asks the user to enter the desired CPU and GPU batch sizes.
This way, you only need to modify the values at a single place.

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
- multiprocessing / concurrent.futures
- AugmentedSocialScientist (pip install AugmentedSocialScientist)

Author:
-------
Antoine Lemor
"""

import os
import glob
import re
import warnings
import pandas as pd
import numpy as np
import torch

from tqdm.auto import tqdm
from multiprocessing import Pool, Manager, cpu_count

# --- Importing AugmentedSocialScientist models ---
# For FR: Camembert
# For EN: Bert
from AugmentedSocialScientist.models import Camembert, Bert


##############################################################################
#                A. STRICT PARSING OF MODEL FILE NAMES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictly parse the model filename to extract:
      - base_category (e.g., "Event_Detection" or "Cult_1_SUB")
      - language ('EN' or 'FR')
      - type: 'Detection', 'SUB', or 'Other'

    If the name is not compliant (missing _EN or _FR),
    we return (base_category, None, model_type).

    :param filename: Model filename (string)
    :return: (base_category, lang, model_type)
    """
    name = filename.replace('.jsonl.model', '').replace('.model', '')
    
    # Identify language
    lang = None
    if name.endswith('_EN'):
        lang = 'EN'
        name = name[:-3]
    elif name.endswith('_FR'):
        lang = 'FR'
        name = name[:-3]
    
    # Identify model type by suffix
    if name.endswith('_Detection'):
        model_type = 'Detection'
        base_category = name
    elif name.endswith('_SUB'):
        model_type = 'SUB'
        base_category = name
    else:
        # If it has no recognized suffix, we treat it as "Other"
        model_type = 'Other'
        base_category = name

    return base_category, lang, model_type


##############################################################################
#          B. LOADING ALL MODELS INTO A DICT (STRICT NAME)
##############################################################################
def load_all_models_strict(models_dir):
    """
    Scans all *.model files in the given folder and
    applies parse_model_filename_strict() to each.

    Returns a dictionary of the form:
    {
      "Event_Detection": {"EN": "/path/Event_Detection_EN.model", "FR": ...},
      "Cult_1_SUB":      {"EN": ..., "FR": ...},
      ...
    }

    :param models_dir: Directory containing the .model files
    :return: Dictionary { base_category: {lang: filepath}, ... }
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

        # We store by language
        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#          C. DEVICE DETECTION (GPU / MPS / CPU)
##############################################################################
def get_main_device():
    """
    Detects a primary device to use:
      - CUDA GPU if torch.cuda.is_available()
      - MPS GPU if torch.backends.mps.is_available()
      - Otherwise CPU

    :return: torch.device object
    """
    if torch.cuda.is_available():
        print("Using CUDA GPU for computations.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS GPU for computations (Apple Silicon).")
        return torch.device("mps")
    else:
        print("Using CPU for computations.")
        return torch.device("cpu")


##############################################################################
#       D. HANDLING TOO LONG SENTENCES -> ERROR CSV
##############################################################################
def check_text_exceeds_length_limit(text, tokenizer, max_length=512):
    """
    Checks if the tokenized sequence exceeds the max_length (512).
    If True, we log it as a potential warning.

    :param text: The text to tokenize
    :param tokenizer: HuggingFace-like tokenizer
    :param max_length: Token limit (int)
    :return: bool (True if exceeds)
    """
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return (len(encoded) > max_length)


##############################################################################
# E. PREPARING CSV FOR ERRORS AND METRICS
##############################################################################
def init_error_csv(error_csv_path):
    """
    Ensures the error CSV (token length issues) is initialized with headers.

    :param error_csv_path: Path to the CSV
    """
    if not os.path.exists(os.path.dirname(error_csv_path)):
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)

    if not os.path.exists(error_csv_path):
        pd.DataFrame(columns=["row_id", "lang", "category", "text"]).to_csv(error_csv_path, index=False)


def append_to_error_csv(error_csv_path, rows):
    """
    Appends to the error CSV a list of rows, each a dict with
    { "row_id":..., "lang":..., "category":..., "text":... }.

    :param error_csv_path: Path to the CSV
    :param rows: List of dictionaries with error info
    """
    if not rows:
        return
    df_err = pd.DataFrame(rows)
    df_err.to_csv(error_csv_path, mode='a', header=False, index=False)


def init_metrics_csv(metrics_csv_path):
    """
    Ensures the metrics CSV is initialized with headers.

    :param metrics_csv_path: Path to the CSV
    """
    if not os.path.exists(os.path.dirname(metrics_csv_path)):
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

    if not os.path.exists(metrics_csv_path):
        cols = ["category", "lang", "label_value", "count"]
        pd.DataFrame(columns=cols).to_csv(metrics_csv_path, index=False)


def append_to_metrics_csv(metrics_csv_path, category, lang, value_counts):
    """
    Appends the distribution of label values to the metrics CSV:
    columns = category, lang, label_value, count

    :param metrics_csv_path: Path to the CSV
    :param category: The category name (string)
    :param lang: 'EN' or 'FR'
    :param value_counts: Pandas Series with label counts (index=label, value=count)
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
# Global variables to be set in each worker during initialization
##############################################################################
WORKER_DEVICE = None
WORKER_BATCH_SIZE = None

##############################################################################
# Worker initializer for dynamic device assignment
##############################################################################
def _worker_init(shared_device_queue, cpu_bs, gpu_bs):
    """
    Assigns each worker a device from the shared_device_queue.
    The first processes get the GPU (if available) and the rest use CPU.
    
    :param shared_device_queue: A Manager list containing device strings.
    :param cpu_bs: Batch size for CPU.
    :param gpu_bs: Batch size for GPU.
    """
    global WORKER_DEVICE, WORKER_BATCH_SIZE
    # Pop a device from the shared list if available; otherwise default to CPU.
    if len(shared_device_queue) > 0:
        WORKER_DEVICE = shared_device_queue.pop(0)
    else:
        WORKER_DEVICE = "cpu"
    WORKER_BATCH_SIZE = gpu_bs if WORKER_DEVICE in ["cuda", "mps"] else cpu_bs
    print(f"[Worker Init] PID {os.getpid()} assigned device {WORKER_DEVICE} with batch size {WORKER_BATCH_SIZE}")


##############################################################################
# F. WORKER FUNCTION FOR PARALLEL PREDICTION
##############################################################################
def _worker_predict_chunk(args):
    """
    Worker function used by parallel_predict_labels() to handle
    one chunk of data. It loads the model on the worker’s pre-assigned device
    (set by _worker_init) and performs annotation on the chunk.

    :param args: A tuple containing:
        (
            chunk_indices,           # the row indices to process
            df_chunk,                # a DF containing text_column for those rows
            model_path,              # path to the .model
            lang,                    # 'EN' or 'FR'
            category,                # column to annotate in DF
            error_csv_path,          # path to error CSV
            pos_dict                 # Manager dict to assign unique tqdm positions per process
        )
    :return: a dict with:
        {
          "predictions": {row_id: pred_label, ...},
          "errors": [ {row_id, lang, category, text}, ... ]
        }
    """
    # Unpack task arguments (no device or batch size passed)
    (chunk_indices, df_chunk, model_path, lang, category, error_csv_path, pos_dict) = args

    # Get device and batch size from global variables assigned in _worker_init.
    global WORKER_DEVICE, WORKER_BATCH_SIZE
    device = torch.device(WORKER_DEVICE)

    # Load model based on language
    if lang == 'FR':
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    results = {
        "predictions": {},
        "errors": []
    }

    texts = df_chunk["sentences"].tolist()
    row_ids = df_chunk.index.tolist()

    # Assign a unique position for this worker's progress bar using pos_dict
    pid = os.getpid()
    if pid not in pos_dict:
        pos_dict[pid] = len(pos_dict)
    # Position for the worker's inner tqdm (start from 1 because main bar is position 0)
    position = pos_dict[pid] + 1

    # Determine device label for tqdm display
    device_label = "GPU" if WORKER_DEVICE in ["cuda", "mps"] else "CPU"

    # Create a tqdm progress bar to track sentence-level processing in this chunk
    pbar = tqdm(total=len(texts),
                desc=f"PID {pid} {device_label} sentences",
                unit="sentence",
                position=position,
                leave=False)

    # Process sentences in sub-batches within this chunk using WORKER_BATCH_SIZE
    for start_i in range(0, len(texts), WORKER_BATCH_SIZE):
        batch_texts = texts[start_i:start_i + WORKER_BATCH_SIZE]
        batch_idx = row_ids[start_i:start_i + WORKER_BATCH_SIZE]

        # Check length for each text in the batch
        error_rows = []
        for local_i, t in enumerate(batch_texts):
            if check_text_exceeds_length_limit(t, model.tokenizer, max_length=512):
                row_id = batch_idx[local_i]
                error_rows.append({
                    "row_id": row_id,
                    "lang": lang,
                    "category": category,
                    "text": t
                })
        results["errors"].extend(error_rows)

        # Encode and predict for the batch
        batch_loader = model.encode(
            batch_texts,
            labels=None,
            batch_size=len(batch_texts),
            progress_bar=False
        )
        try:
            probs = model.predict_with_model(
                batch_loader,
                model_path=model_path,
                proba=True,
                progress_bar=False
            )
            batch_preds = np.argmax(probs, axis=1).tolist()
        except Exception as e:
            print(f"[ERROR] Failed predict_with_model on '{model_path}'. Reason: {e}")
            batch_preds = [np.nan] * len(batch_texts)

        # Store predictions
        for idx_, pred_ in zip(batch_idx, batch_preds):
            results["predictions"][idx_] = pred_

        # Update the progress bar with the number of processed sentences
        pbar.update(len(batch_texts))

    pbar.close()
    return results


##############################################################################
#   G. PARALLEL PREDICTION DISPATCHER
##############################################################################
def parallel_predict_labels(
    df,
    indices,
    model_path,
    lang,
    category,
    error_csv_path,
    device_list,
    gpu_batch_size,
    cpu_batch_size,
    chunk_size=5000
):
    """
    Splits the given 'indices' into chunks, then spawns multiple processes
    to annotate in parallel. Each worker is assigned a device dynamically during
    its initialization (via _worker_init) using a shared device queue.
    Tasks are then progressively fed to the workers.

    :param df: The entire DataFrame containing the text and placeholders
    :param indices: The row indices (df.index) that need annotation
    :param model_path: Path to the .model
    :param lang: 'EN' or 'FR'
    :param category: Column name where predictions will be stored
    :param error_csv_path: Path to error logging CSV
    :param device_list: A list of device strings, e.g. ['cuda', 'cpu', 'cpu', ...]
    :param gpu_batch_size: Batch size for inference on GPU.
    :param cpu_batch_size: Batch size for inference on CPU.
    :param chunk_size: Number of rows per chunk.
    :return: A dict of {row_id: label_prediction} for each index
    """
    if len(indices) == 0:
        return {}

    indices_list = list(indices)
    indices_list.sort()

    # Build chunks from indices_list
    chunks = []
    for start_i in range(0, len(indices_list), chunk_size):
        chunk_idx = indices_list[start_i:start_i + chunk_size]
        chunks.append(chunk_idx)

    # Create a Manager dict for tqdm positioning (shared among workers)
    manager = Manager()
    pos_dict = manager.dict()

    # Prepare tasks: note that device information and batch sizes will be set
    # in each worker via _worker_init.
    tasks = []
    for chunk_indices in chunks:
        df_chunk = df.loc[chunk_indices, ["sentences"]].copy()
        task_args = (
            chunk_indices,
            df_chunk,
            model_path,
            lang,
            category,
            error_csv_path,
            pos_dict  # Shared dict for tqdm positioning
        )
        tasks.append(task_args)

    print(f"  [parallel_predict_labels] Launching {len(chunks)} chunk(s) across {len(device_list)} process(es).")
    print(f"  [parallel_predict_labels] category='{category}', lang='{lang}', model='{model_path}'")

    all_predictions = {}
    all_errors = []

    # Build a shared device queue from the provided device_list.
    manager_for_devices = Manager()
    shared_device_queue = manager_for_devices.list(device_list)

    # Create a Pool with number of processes equal to len(device_list) and use the custom initializer.
    pool_processes = len(device_list)
    with Pool(processes=pool_processes,
              initializer=_worker_init,
              initargs=(shared_device_queue, cpu_batch_size, gpu_batch_size)
              ) as pool:
        results_iter = pool.imap_unordered(_worker_predict_chunk, tasks)
        chunk_pbar = tqdm(total=len(tasks),
                          desc=f"Annot '{category}'",
                          unit="chunk",
                          position=0,
                          leave=True)
        for res in results_iter:
            chunk_pbar.update(1)
            for rid, pred in res["predictions"].items():
                all_predictions[rid] = pred
            all_errors.extend(res["errors"])
        chunk_pbar.close()

    # Write all encountered errors to CSV
    append_to_error_csv(error_csv_path, all_errors)

    return all_predictions


##############################################################################
#           H. MAIN ANNOTATION LOGIC (3 STEPS: Detection, SUB, Other)
##############################################################################
def annotate_dataframe(
    df,
    model_dict,
    device_list,
    output_path,
    error_csv_path,
    metrics_csv_path,
    langs_to_annotate,
    gpu_batch_size,
    cpu_batch_size
):
    """
    Annotates the DataFrame in 3 steps:
      1) Detection
      2) SUB
      3) Other

    Each category is annotated only for the relevant language(s) found
    in model_dict[category]. If 'langs_to_annotate' is 'EN' or 'FR', we skip
    the other language. If 'both', we do all available languages.

    After each category is fully annotated for a given language, an intermediate
    save is made to update the output CSV.

    :param df: The main DataFrame
    :param model_dict: The dictionary of categories -> {lang: model_path}
    :param device_list: A list of device strings to parallelize
    :param output_path: Path to the main CSV to save partial annotation
    :param error_csv_path: Path to the error CSV
    :param metrics_csv_path: Path to the metrics CSV
    :param langs_to_annotate: 'EN', 'FR', or 'both'
    :param gpu_batch_size: Batch size for GPU processes
    :param cpu_batch_size: Batch size for CPU processes
    :return: Annotated DataFrame
    """
    text_col = "sentences"
    lang_col = "language"

    # Classify categories by suffix
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

    # ---------------------------
    # STEP 1: DETECTION
    # ---------------------------
    print("\n[ANNOTATION] Step 1: Main Categories (Detection)")
    for cat_det in categories_detection:
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        for lang, model_path in model_dict[cat_det].items():
            if langs_to_annotate != 'both' and lang != langs_to_annotate:
                continue

            print(f"\n -> Annotating '{cat_det}' (Detection) for lang='{lang}'")
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_det}' / lang={lang}.")
                continue

            all_preds = parallel_predict_labels(
                df=df,
                indices=idx,
                model_path=model_path,
                lang=lang,
                category=cat_det,
                error_csv_path=error_csv_path,
                device_list=device_list,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=cpu_batch_size,
                chunk_size=5000
            )

            for rid, pred_val in all_preds.items():
                df.at[rid, cat_det] = pred_val

            df.to_csv(output_path, index=False)
            print(f"  [SAVE] Partial results saved for category '{cat_det}' (lang={lang}).")

            dist_lang = df.loc[df[lang_col] == lang, cat_det].value_counts(dropna=False)
            print(f"Distribution for '{cat_det}' (lang={lang}):\n{dist_lang}")
            append_to_metrics_csv(metrics_csv_path, cat_det, lang, dist_lang)

        dist_after = df[cat_det].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_det}':\n{dist_after}")

    # ---------------------------
    # STEP 2: SUB
    # ---------------------------
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub) + '_Detection'
        if main_category not in df.columns:
            print(f"[WARN] Missing parent category '{main_category}' for '{cat_sub}'. Skipping.")
            continue

        for lang, model_path in model_dict[cat_sub].items():
            if langs_to_annotate != 'both' and lang != langs_to_annotate:
                continue

            print(f"\n -> Annotating '{cat_sub}' (SUB) for lang='{lang}'")
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &
                (df[cat_sub].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No positive rows to annotate for '{cat_sub}' / lang={lang}.")
                continue

            all_preds = parallel_predict_labels(
                df=df,
                indices=idx,
                model_path=model_path,
                lang=lang,
                category=cat_sub,
                error_csv_path=error_csv_path,
                device_list=device_list,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=cpu_batch_size,
                chunk_size=5000
            )

            for rid, pred_val in all_preds.items():
                df.at[rid, cat_sub] = pred_val

            df.to_csv(output_path, index=False)
            print(f"  [SAVE] Partial results saved for category '{cat_sub}' (lang={lang}).")

            dist_lang = df.loc[df[lang_col] == lang, cat_sub].value_counts(dropna=False)
            print(f"Distribution for '{cat_sub}' (lang={lang}):\n{dist_lang}")
            append_to_metrics_csv(metrics_csv_path, cat_sub, lang, dist_lang)

        dist_after = df[cat_sub].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_sub}':\n{dist_after}")

    # ---------------------------
    # STEP 3: OTHER
    # ---------------------------
    print("\n[ANNOTATION] Step 3: Other models (neither Detection nor SUB)")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        for lang, model_path in model_dict[cat_other].items():
            if langs_to_annotate != 'both' and lang != langs_to_annotate:
                continue

            print(f"\n -> Annotating '{cat_other}' (Other) for lang='{lang}'")
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_other}' / lang={lang}.")
                continue

            all_preds = parallel_predict_labels(
                df=df,
                indices=idx,
                model_path=model_path,
                lang=lang,
                category=cat_other,
                error_csv_path=error_csv_path,
                device_list=device_list,
                gpu_batch_size=gpu_batch_size,
                cpu_batch_size=cpu_batch_size,
                chunk_size=5000
            )

            for rid, pred_val in all_preds.items():
                df.at[rid, cat_other] = pred_val

            df.to_csv(output_path, index=False)
            print(f"  [SAVE] Partial results saved for category '{cat_other}' (lang={lang}).")

            dist_lang = df.loc[df[lang_col] == lang, cat_other].value_counts(dropna=False)
            print(f"Distribution for '{cat_other}' (lang={lang}):\n{dist_lang}")
            append_to_metrics_csv(metrics_csv_path, cat_other, lang, dist_lang)

        dist_after = df[cat_other].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_other}':\n{dist_after}")

    # Final summary distribution
    print("\n[ANNOTATION] Final distribution summary:")
    all_cols = categories_detection + categories_sub + categories_other
    for col in all_cols:
        if col in df.columns:
            dist_col = df[col].value_counts(dropna=False)
            print(f" - {col} : \n{dist_col}\n")

    return df


##############################################################################
#                           I. MAIN FUNCTION
##############################################################################
def main():
    """
    1) Prompt which language(s) to annotate: 'EN', 'FR', or 'both'.
    2) Define input/output paths.
    3) Load or resume DataFrame from existing annotation CSV if present.
    4) Load available models from the models directory.
    5) Initialize error/metrics CSV logs.
    6) Build a device list for parallel usage (GPU + CPU if available).
    7) Ask the user for the GPU and CPU batch sizes.
    8) Annotate in 3 steps: Detection, SUB, Other.
    9) Save the final annotated DataFrame.
    """

    # ------------------ 1) Prompt the language choice -------------------
    def get_language_choice():
        """
        Ask the user which language(s) to annotate, returns 'EN', 'FR', or 'both'.
        """
        print("Please choose which sentences to annotate:")
        print("   1) EN only")
        print("   2) FR only")
        print("   3) BOTH (EN and FR)")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == '1':
            return 'EN'
        elif choice == '2':
            return 'FR'
        else:
            return 'both'

    langs_to_annotate = get_language_choice()
    if langs_to_annotate == 'EN':
        suffix = "_EN.csv"
    elif langs_to_annotate == 'FR':
        suffix = "_FR.csv"
    else:
        suffix = "_full.csv"

    # ------------------ 2) Define paths -------------------
    base_path = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    output_path = os.path.join(
        base_path, "..", "..", "Database", "Database", f"CCF.media_processed_texts_annotated{suffix}"
    )

    models_dir = os.path.join(base_path, "..", "..", "models")

    error_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "sentences_annotation_error.csv")
    metrics_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotated_label_metrics.csv")

    # ------------------ 3) Load or resume DataFrame -------------------
    if os.path.exists(output_path):
        print(f"[main] Annotation file already exists: '{output_path}'. Resuming...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] No existing annotation file. Loading base CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into DataFrame.")

    # ------------------ 4) Load models -------------------
    print("[main] Loading model files (strict suffixes)...")
    model_dict = load_all_models_strict(models_dir)
    print(f"[main] Number of detected categories: {len(model_dict)}")

    # ------------------ 5) Initialize CSV logs -------------------
    init_error_csv(error_csv_path)
    init_metrics_csv(metrics_csv_path)

    # ------------------ 6) Build device list for parallel usage -------------------
    main_dev = get_main_device()

    # For parallel usage, decide on the number of GPU and CPU processes.
    # For MPS/CUDA, we restrict to 1 GPU process.
    NUM_GPU_PROCS = 1 if (main_dev.type in ["cuda", "mps"]) else 0
    NUM_CPU_PROCS = max(1, cpu_count() - 2)

    # device_list now contains one or more GPU (if available) and the remaining CPU processes.
    device_list = []
    for _ in range(NUM_GPU_PROCS):
        device_list.append(main_dev.type)
    for _ in range(NUM_CPU_PROCS):
        device_list.append("cpu")

    print(f"[main] Creating device list for parallel annotation: {device_list}")
    if not device_list:
        device_list = ["cpu"]

    # ------------------ 7) Ask for batch sizes -------------------
    try:
        cpu_bs_input = input("Please enter the desired CPU batch size: ").strip()
        gpu_bs_input = input("Please enter the desired GPU batch size: ").strip()
        CPU_BATCH_SIZE = int(cpu_bs_input)
        GPU_BATCH_SIZE = int(gpu_bs_input)
    except Exception as e:
        print("Error reading batch size inputs. Using default values: CPU=25, GPU=50.")
        CPU_BATCH_SIZE = 25
        GPU_BATCH_SIZE = 50

    print(f"[main] Using CPU batch size: {CPU_BATCH_SIZE} and GPU batch size: {GPU_BATCH_SIZE}.")

    # ------------------ 8) Annotate -------------------
    print("[main] Starting annotation steps...")
    df_annotated = annotate_dataframe(
        df=df,
        model_dict=model_dict,
        device_list=device_list,
        output_path=output_path,
        error_csv_path=error_csv_path,
        metrics_csv_path=metrics_csv_path,
        langs_to_annotate=langs_to_annotate,
        gpu_batch_size=GPU_BATCH_SIZE,
        cpu_batch_size=CPU_BATCH_SIZE
    )

    # ------------------ 9) Final save -------------------
    print("[main] Saving final annotated DataFrame...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation completed. Final file: {output_path}")


if __name__ == "__main__":
    # Multiprocessing guard for Windows or other OS.
    main()