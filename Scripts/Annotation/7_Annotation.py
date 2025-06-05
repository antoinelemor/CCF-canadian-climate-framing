"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7_Annotation.py

MAIN OBJECTIVE:
-------------------
Automate the annotation of every sentence stored in the
`CCF_processed_data` table of the PostgreSQL database CCF by
running pretrained CamemBERT (FR) or BERT-base (EN) classifiers.  
The script handles large-scale inference in parallel, writes the
predictions back to the database in bulk, and logs both processing
metrics and any problematic rows (e.g., texts exceeding 512 tokens).

Dependencies:
-------------
- os, sys, time, queue, random, warnings
- csv, glob, re, json (indirect via models)
- multiprocessing (Pool, Manager, cpu_count, RLock)
- collections.defaultdict (internal helpers)
- psycopg2 / psycopg2.extras 
- numpy, pandas             
- torch                     
- tqdm.auto                  

MAIN FEATURES:
----------------------------
1) Robust DB connection layer with interactive retry, credential
   override, and “offline predictions” CSV export on repeated failure.
2) Dynamic device allocation: user-selectable CPU / GPU / hybrid
   mode with automatic batch-size adjustment.
3) Strict model discovery: scans `../../models/*.model`, parses
   filenames to match base category, language, and model type
   (Detection / SUB / Other), ignoring malformed files.
4) Parallel inference pool: distributes work across all logical
   CPUs (or a user-defined subset), each worker receiving its own
   device and tqdm position.
5) 512-token safeguard: detects texts that overflow the transformer
   context window and logs them to `sentences_annotation_error.csv`
   for manual inspection.
6) Bulk, transaction-safe updates: writes predictions using a
   temporary UNLOGGED table and a single `UPDATE … FROM` join,
   minimising row-level locks and WAL traffic.
7) Offline buffer & flush: if any DB update fails, predictions are
   kept in memory, then re-attempted; as a last resort they are dumped
   to `offline_predictions.csv`.
8) Per-category metrics: after each batch, the script appends the
   distribution of predicted labels (0/1/None) to
   `annotated_label_metrics.csv`, enabling downstream quality checks.
9) Language & device prompts: interactive CLI lets the user choose
   to annotate EN only, FR only, or both, and select the hardware mode
   before processing starts.
10) Extensible fallback models: when the real package
    *AugmentedSocialScientist* is absent, lightweight placeholders with
    random outputs let the rest of the pipeline run for testing.

Author :
--------
Antoine Lemor
"""

import os
import sys
import csv
import time
import queue
import random
import warnings
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
import torch

from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from multiprocessing import Pool, Manager, cpu_count, RLock, current_process

# Attempt to import the actual models
try:
    from AugmentedSocialScientist.models import Camembert, Bert
except ImportError:
    warnings.warn(
        "Could not import AugmentedSocialScientist. "
        "Falling back to placeholders with random predictions."
    )
    class Camembert:
        def __init__(self, device: torch.device): 
            """
            Placeholder Camembert class for random predictions.
            """
            pass
        def encode(self, texts, labels, batch_size, progress_bar=False):
            """
            Dummy text encoding returning a simple range object.
            """
            return range(len(texts))
        def predict_with_model(self, loader, model_path, proba=True, progress_bar=False):
            """
            Returns random probabilities of size (n, 2).
            """
            n = len(loader)
            return np.random.rand(n, 2)

    class Bert:
        def __init__(self, device: torch.device): 
            """
            Placeholder Bert class for random predictions.
            """
            pass
        def encode(self, texts, labels, batch_size, progress_bar=False):
            """
            Dummy text encoding returning a simple range object.
            """
            return range(len(texts))
        def predict_with_model(self, loader, model_path, proba=True, progress_bar=False):
            """
            Returns random probabilities of size (n, 2).
            """
            n = len(loader)
            return np.random.rand(n, 2)


##############################################################################
#                         1. DATABASE PARAMETERS
##############################################################################

DB_PARAMS: Dict[str, Any] = {
    "host": "localhost",
    "port": 5432,
    "dbname": "CCF",
    "user": "antoine",
    "password": "Absitreverentiavero19!",
}

TABLE_NAME = "CCF_processed_data"

# Name of columns in the DB table:
DOC_ID_COL       = "doc_id"       # text or int
SENTENCE_ID_COL  = "sentence_id"  # int
TEXT_COL         = "sentences"    # text
LANG_COL         = "language"     # either 'EN' or 'FR'

# If you want to limit the number of rows read for testing, set a positive int.
# Otherwise, set to None to process all rows.
LIMIT_ROWS = None

# The chunk size for parallel tasks
CHUNK_SIZE = 5000

# CSV logs for error & metrics
ERROR_CSV_PATH   = "sentences_annotation_error.csv"
METRICS_CSV_PATH = "annotated_label_metrics.csv"


##############################################################################
#                 2. OFFLINE BUFFER FOR PREDICTIONS
##############################################################################
OFFLINE_PREDICTIONS = []  # type: List[Dict[str, Any]]

def store_offline_predictions(
    column: str,
    updates: List[Tuple[Optional[int], str, int]]
) -> None:
    """
    Store predictions in a global buffer so they are not lost
    if we fail to update the DB due to a connection error.

    Parameters
    ----------
    column : str
        The target DB column being updated.
    updates : List[Tuple[Optional[int], str, int]]
        Each tuple is (pred_label, doc_id, sentence_id).
    """
    global OFFLINE_PREDICTIONS
    OFFLINE_PREDICTIONS.append({
        "column": column,
        "updates": updates
    })


def export_offline_predictions_csv(filename: str) -> None:
    """
    Export all offline predictions to CSV with columns:
      doc_id, sentence_id, column, pred

    Parameters
    ----------
    filename : str
        Path to the CSV file to create.
    """
    global OFFLINE_PREDICTIONS
    rows = []
    for entry in OFFLINE_PREDICTIONS:
        col  = entry["column"]
        for (pred, doc_id, sent_id) in entry["updates"]:
            rows.append((doc_id, sent_id, col, pred))

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "sentence_id", "column", "pred"])
        writer.writerows(rows)


def flush_offline_predictions(cur: psycopg2.extensions.cursor) -> None:
    """
    Attempt to flush all offline predictions from OFFLINE_PREDICTIONS
    into the DB using the current cursor. If successful, clears the buffer.

    Raises
    ------
    psycopg2.Error
        If the update fails, the exception is thrown for the caller to handle.
    """
    global OFFLINE_PREDICTIONS
    while OFFLINE_PREDICTIONS:
        entry = OFFLINE_PREDICTIONS.pop(0)
        col = entry["column"]
        updates = entry["updates"]
        bulk_update_column(cur, col, updates)


##############################################################################
#            3. DB CONNECTION + HELPERS (SAFE RECONNECT LOGIC)
##############################################################################
def connect_to_db(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str
) -> psycopg2.extensions.connection:
    """
    Attempt to connect to PostgreSQL with given parameters.

    Raises
    ------
    psycopg2.Error
        If the connection fails.
    """
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )


def safe_get_connection() -> psycopg2.extensions.connection:
    """
    Continuously try to get a valid DB connection (autocommit=True),
    prompting the user for retries, new credentials, or exporting
    offline predictions to CSV if the connection fails repeatedly.

    Returns
    -------
    psycopg2.extensions.connection
        A valid, autocommit-enabled connection.
    """
    global DB_PARAMS

    while True:
        try:
            conn = connect_to_db(**DB_PARAMS)
            conn.autocommit = True
            return conn
        except psycopg2.Error as e:
            print(f"\n[ERROR] Unable to connect to the DB: {e}")
            choice = input(
                "\nOptions:\n"
                "   1) Retry with the same credentials\n"
                "   2) Enter new credentials\n"
                "   3) Export offline predictions to CSV and exit\n"
                "Select [1/2/3]: "
            ).strip()
            if choice == "1":
                continue
            elif choice == "2":
                # Prompt new credentials
                new_host = input("host (default=localhost): ").strip() or "localhost"
                new_port_str = input("port (default=5432): ").strip() or "5432"
                new_dbname = input("dbname (default=CCF): ").strip() or "CCF"
                new_user = input("user (default=antoine): ").strip() or "antoine"
                new_password = input("password (default=Absitreverentiavero19!): ").strip() or "Absitreverentiavero19!"

                DB_PARAMS.update({
                    "host": new_host,
                    "port": int(new_port_str),
                    "dbname": new_dbname,
                    "user": new_user,
                    "password": new_password,
                })
                continue
            else:
                # Export to CSV and exit
                filename = "offline_predictions.csv"
                print(f"[INFO] Exporting offline predictions to «{filename}».")
                export_offline_predictions_csv(filename)
                sys.exit("[EXIT] Predictions have been saved to CSV. Exiting script.")


def ensure_column(cur: psycopg2.extensions.cursor, column: str) -> None:
    """
    Ensure an INTEGER column named `column` exists in TABLE_NAME.
    If it doesn't exist, create it.

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
        Active DB cursor.
    column : str
        Column name to create if missing.
    """
    cur.execute(f"""
        ALTER TABLE public."{TABLE_NAME}"
        ADD COLUMN IF NOT EXISTS "{column}" INTEGER;
    """)


def fetch_to_annotate_detection(
    cur: psycopg2.extensions.cursor,
    category: str
) -> List[Tuple[str, int, str]]:
    """
    Fetch rows needing annotation for a Detection category (is NULL),
    ignoring rows where `sentences` is null, and reading all languages
    from the 'language' column. We only fetch those with (category IS NULL)
    so we do not re-predict existing ones.

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
    category : str
        The detection column name (e.g. "Event_Detection").

    Returns
    -------
    rows : List[Tuple[doc_id, sentence_id, text]]
    """
    limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
    cur.execute(f"""
        SELECT "{DOC_ID_COL}",
               "{SENTENCE_ID_COL}",
               "{TEXT_COL}"
        FROM public."{TABLE_NAME}"
        WHERE "{category}" IS NULL
          AND "{TEXT_COL}" IS NOT NULL
        {limit_sql};
    """)
    return cur.fetchall()


def fetch_to_annotate_sub(
    cur: psycopg2.extensions.cursor,
    sub_category: str,
    parent_detection_col: str
) -> List[Tuple[str, int, str]]:
    """
    Fetch rows needing annotation for a SUB category (is NULL),
    only where the parent Detection == 1, ignoring rows with null text.

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
    sub_category : str
        The sub-category column name (e.g. "Cult_1_SUB")
    parent_detection_col : str
        The parent detection column name (e.g. "Cult_Detection")

    Returns
    -------
    rows : List[Tuple[str, int, str]]
    """
    limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
    cur.execute(f"""
        SELECT "{DOC_ID_COL}",
               "{SENTENCE_ID_COL}",
               "{TEXT_COL}"
        FROM public."{TABLE_NAME}"
        WHERE "{sub_category}" IS NULL
          AND "{parent_detection_col}" = 1
          AND "{TEXT_COL}" IS NOT NULL
        {limit_sql};
    """)
    return cur.fetchall()


def fetch_to_annotate_other(
    cur: psycopg2.extensions.cursor,
    category: str
) -> List[Tuple[str, int, str]]:
    """
    Fetch rows needing annotation for a category that is neither
    detection nor sub. (Category is NULL, text is not null.)

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
    category : str
        The column name.

    Returns
    -------
    rows : List[Tuple[doc_id, sentence_id, text]]
    """
    limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
    cur.execute(f"""
        SELECT "{DOC_ID_COL}",
               "{SENTENCE_ID_COL}",
               "{TEXT_COL}"
        FROM public."{TABLE_NAME}"
        WHERE "{category}" IS NULL
          AND "{TEXT_COL}" IS NOT NULL
        {limit_sql};
    """)
    return cur.fetchall()


def bulk_update_column(
    cur: psycopg2.extensions.cursor,
    column: str,
    updates: List[Tuple[Optional[int], str, int]]
) -> None:
    """
    Perform a bulk update of (column) in TABLE_NAME, for rows identified
    by (doc_id, sentence_id). We use an unlogged temporary table approach
    for speed.

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
        Active DB cursor.
    column : str
        The target column to update (must already exist in the table).
    updates : List[Tuple[Optional[int], str, int]]
        Each tuple is (prediction_label, doc_id, sentence_id).
    """
    if not updates:
        return

    # We create an unlogged temp table
    tmp_name = f"_tmp_{abs(hash(column)) % 10_000_000}"
    cur.execute(f"DROP TABLE IF EXISTS {tmp_name};")
    cur.execute(f"""
        CREATE UNLOGGED TABLE {tmp_name} (
            pred INTEGER,
            doc_id BIGINT,
            sentence_id INTEGER
        );
    """)

    # Insert via execute_values
    psycopg2.extras.execute_values(
        cur,
        f"INSERT INTO {tmp_name} (pred, doc_id, sentence_id) VALUES %s",
        updates,
        page_size=1000
    )

    # Update from the temp table
    cur.execute(f"""
        UPDATE public."{TABLE_NAME}" AS t
           SET "{column}" = s.pred
          FROM {tmp_name} AS s
         WHERE t.doc_id      = s.doc_id
           AND t.sentence_id = s.sentence_id;
    """)

    # Drop temp table
    cur.execute(f"DROP TABLE {tmp_name};")


##############################################################################
#                4. MODEL DISCOVERY (STRICT NAME PARSING)
##############################################################################
import glob
import re

def parse_model_filename_strict(filename: str) -> Tuple[str, Optional[str], str]:
    """
    Parse a model filename and return:
        (base_category, lang, model_type)

    - Accepts extensions: .model, .json.model, .jsonl.model
    - Removes trailing whitespaces
    - Detects language suffixes _EN / _FR
    - Determines model_type: Detection / SUB / Other
    """
    name = filename.strip()

    # 1) Remove any recognized double-extension, longest first
    for ext in ('.jsonl.model', '.json.model', '.model'):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break

    # 2) Language suffix
    lang = None
    if name.endswith('_EN'):
        lang, name = 'EN', name[:-3]
    elif name.endswith('_FR'):
        lang, name = 'FR', name[:-3]

    # 3) Model type
    if   name.endswith('_Detection'):
        model_type = 'Detection'
    elif name.endswith('_SUB'):
        model_type = 'SUB'
    else:
        model_type = 'Other'

    return name, lang, model_type


def load_all_models_strict(models_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Scan all *.model files in `models_dir` and parse them to build a dict:
      {
        base_cat_1: {"EN": path, "FR": path, ...},
        base_cat_2: {"EN": path, "FR": ...},
        ...
      }

    We do not load the model into memory here; just record the file paths.

    Parameters
    ----------
    models_dir : str
        Directory containing .model files.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Mapping base_category -> (lang -> model_path)
    """
    model_files = glob.glob(os.path.join(models_dir, "*.model"))
    model_dict: Dict[str, Dict[str, str]] = {}

    for filepath in model_files:
        filename = os.path.basename(filepath)
        base_cat, lang, model_type = parse_model_filename_strict(filename)

        if lang is None:
            print(f"[WARNING] Model file '{filename}' ignored (no _EN or _FR suffix).")
            continue

        print(f"[INFO] Found model '{filename}' -> base='{base_cat}', lang='{lang}', type='{model_type}'")
        if base_cat not in model_dict:
            model_dict[base_cat] = {}
        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#       5. CHECK SENTENCE LENGTH (FOR ERROR CSV) & CSV LOGGING
##############################################################################
def check_text_exceeds_length_limit(text: str, tokenizer, max_length: int = 512) -> bool:
    """
    Return True if the tokenized sequence length exceeds max_length=512.
    """
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return (len(encoded) > max_length)


def init_error_csv(path: str) -> None:
    """
    Ensure the error CSV is initialized with the proper headers.

    Parameters
    ----------
    path : str
        Path to the error CSV file.
    """
    if not os.path.exists(path):
        df = pd.DataFrame(columns=["doc_id", "sentence_id", "lang", "category", "text"])
        df.to_csv(path, index=False)


def append_to_error_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Append a list of records to the error CSV. Each record should
    have the keys: doc_id, sentence_id, lang, category, text.

    Parameters
    ----------
    path : str
        Path to the error CSV file.
    rows : List[Dict[str, Any]]
        Each item must have keys corresponding to the error CSV columns.
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=False, index=False)


def init_metrics_csv(path: str) -> None:
    """
    Ensure the metrics CSV is initialized with the proper headers.

    Parameters
    ----------
    path : str
        Path to the metrics CSV file.
    """
    if not os.path.exists(path):
        cols = ["category", "lang", "label_value", "count"]
        pd.DataFrame(columns=cols).to_csv(path, index=False)


def append_to_metrics_csv(
    path: str,
    category: str,
    lang: str,
    value_counts: pd.Series
) -> None:
    """
    Append the distribution of label values to the metrics CSV:
    with columns: category, lang, label_value, count

    Parameters
    ----------
    path : str
        Path to the metrics CSV file.
    category : str
        The category name (e.g. "Event_Detection").
    lang : str
        The language ("EN" or "FR").
    value_counts : pd.Series
        Distribution of predicted values for this category & language.
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
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=False, index=False)


##############################################################################
#         6. DEVICE CHOOSING & PARALLEL POOL INITIALIZATION
##############################################################################
def pick_device() -> torch.device:
    """
    Return the best available device: GPU (CUDA), MPS, or CPU.

    Returns
    -------
    torch.device
        The best available device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_devices_mode(user_choice: str, max_workers: int) -> List[str]:
    """
    Create a list of device names for each worker based on the user's choice:
      - 'cpu': all workers on CPU
      - 'gpu': only one worker on GPU (if available), else CPU fallback
      - 'both': one GPU worker (if available) + the rest CPU

    The length of this list = number of worker processes to spawn.

    Parameters
    ----------
    user_choice : str
        Either "cpu", "gpu", or "both".
    max_workers : int
        Number of workers to create.

    Returns
    -------
    List[str]
        List of device string identifiers, e.g. ["cpu", "cpu", "cuda", ...].
    """
    dev = pick_device().type
    if user_choice == "gpu":
        if dev in ("cuda", "mps"):
            return [dev] * min(1, max_workers)
        else:
            print("[WARN] No GPU/MPS available. Falling back to CPU-only.")
            return ["cpu"] * max_workers
    elif user_choice == "cpu":
        return ["cpu"] * max_workers
    elif user_choice == "both":
        if dev in ("cuda", "mps"):
            if max_workers == 1:
                return [dev]  # only 1 worker -> GPU
            else:
                return [dev] + ["cpu"] * (max_workers - 1)
        else:
            print("[WARN] No GPU/MPS available. Using CPU-only for all workers.")
            return ["cpu"] * max_workers
    else:
        print("[WARN] Unrecognized device mode. Defaulting to CPU-only.")
        return ["cpu"] * max_workers


##############################################################################
#            7. WORKER FUNCTIONS FOR PARALLEL ANNOTATION
##############################################################################
WORKER_DEVICE = "cpu"
WORKER_BS_CPU = 25
WORKER_BS_GPU = 50
WORKER_POSITION = 0

def _worker_init(
    devices_q: "queue.Queue[str]",
    bs_cpu: int,
    bs_gpu: int,
    pos_q: "queue.Queue[int]"
) -> None:
    """
    Assign each worker a device from devices_q and set the appropriate batch size.
    Also retrieve a TQDM position for local progress bars.

    Parameters
    ----------
    devices_q : queue.Queue
        Queue of device names to assign ("cpu", "cuda", "mps", ...).
    bs_cpu : int
        Batch size for CPU inference.
    bs_gpu : int
        Batch size for GPU inference.
    pos_q : queue.Queue
        Queue of integer positions for TQDM progress bar placement.
    """
    global WORKER_DEVICE, WORKER_BS_CPU, WORKER_BS_GPU, WORKER_POSITION

    try:
        WORKER_DEVICE = devices_q.get_nowait()
    except queue.Empty:
        WORKER_DEVICE = "cpu"

    WORKER_BS_CPU = bs_cpu
    WORKER_BS_GPU = bs_gpu

    try:
        WORKER_POSITION = pos_q.get_nowait()
    except queue.Empty:
        WORKER_POSITION = 0

    # Optionally reduce transformers logging
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


def predict_batch_safely(
    model: Bert or Camembert,
    texts: List[str],
    model_path: str
) -> List[Optional[int]]:
    """
    Predict a list of texts, returning numeric class indices [0 or 1]
    or None on error. If the entire batch fails, tries a fallback of
    predicting text-by-text individually.

    Parameters
    ----------
    model : Bert or Camembert
        The loaded model instance.
    texts : List[str]
        The texts to be classified.
    model_path : str
        Path to the model file.

    Returns
    -------
    List[Optional[int]]
        List of predicted class labels (0 or 1), or None for a failure.
    """
    try:
        loader = model.encode(texts, labels=None, batch_size=len(texts), progress_bar=False)
        probs = model.predict_with_model(loader, model_path, proba=True, progress_bar=False)
        return np.argmax(probs, axis=1).tolist()
    except Exception:
        # Fallback: predict individually
        results = []
        for t in texts:
            try:
                loader_1 = model.encode([t], labels=None, batch_size=1, progress_bar=False)
                probs_1 = model.predict_with_model(loader_1, model_path, proba=True, progress_bar=False)
                pred = int(np.argmax(probs_1, axis=1)[0])
                results.append(pred)
            except Exception:
                results.append(None)
        return results


def _worker_annotate_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    The worker function that handles a chunk of rows for a single category & model.

    Parameters
    ----------
    task : Dict[str, Any]
        Includes the keys:
          - "category" : str
            The DB column to update.
          - "model_path" : str
            Path to the .model file.
          - "lang" : str
            "EN" or "FR".
          - "rows" : List[Tuple[str, int, str]]
            Each tuple: (doc_id, sentence_id, text).

    Returns
    -------
    Dict[str, Any]
        Contains:
         - "category" : str
         - "lang" : str
         - "updates" : List[Tuple[Optional[int], str, int]]
           The list of predictions for updating the DB.
         - "errors" : List[Dict[str, Any]]
           Rows that had length issues, to be logged in CSV.
    """
    global WORKER_DEVICE, WORKER_BS_CPU, WORKER_BS_GPU, WORKER_POSITION

    category   = task["category"]
    model_path = task["model_path"]
    lang       = task["lang"]
    rows       = task["rows"]

    # Decide which model class to instantiate based on language
    device_label = WORKER_DEVICE
    device = torch.device(WORKER_DEVICE)
    if lang == "FR":
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    # TQDM progress bar for these rows
    pid = current_process().name
    pbar = tqdm(
        total=len(rows),
        desc=f"[{pid} - {device_label}] {category}",
        position=WORKER_POSITION,
        leave=False,
        dynamic_ncols=True,
        unit="sent",
    )

    # Decide batch size based on device
    if WORKER_DEVICE in ("cuda", "mps"):
        batch_size = WORKER_BS_GPU
    else:
        batch_size = WORKER_BS_CPU

    # We'll gather predictions in the form (pred_label, doc_id, sentence_id)
    updates = []
    # We'll also gather any too-long text errors
    error_logs = []

    # Process the rows in sub-batches
    i = 0
    while i < len(rows):
        sub = rows[i : i + batch_size]
        i += batch_size

        sub_texts = [r[2] for r in sub]  # the "sentences"
        local_errors = []
        for idx_sub, text in enumerate(sub_texts):
            # Check if the text exceeds 512 tokens
            if check_text_exceeds_length_limit(text, model.tokenizer, max_length=512):
                docid = sub[idx_sub][0]
                sentid = sub[idx_sub][1]
                local_errors.append({
                    "doc_id": docid,
                    "sentence_id": sentid,
                    "lang": lang,
                    "category": category,
                    "text": text
                })
        error_logs.extend(local_errors)

        # Inference on the sub-batch
        preds = predict_batch_safely(model, sub_texts, model_path)
        for (doc_id, sent_id, _), pred_label in zip(sub, preds):
            updates.append((pred_label, doc_id, sent_id))

        pbar.update(len(sub))

    pbar.close()

    return {
        "category": category,
        "lang": lang,
        "updates": updates,       # List[ (pred, doc_id, sentence_id) ]
        "errors": error_logs      # List[ {doc_id, sentence_id, lang, category, text} ]
    }


##############################################################################
#               8. MAIN ANNOTATION LOGIC: DETECTION, SUB, OTHER
##############################################################################
def classify_categories(model_dict: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[str], List[str]]:
    """
    From a dictionary of {base_category -> {lang -> model_path}},
    classify them into 3 lists: detection, sub, other, sorted by name.

    Parameters
    ----------
    model_dict : Dict[str, Dict[str, str]]
        The parsed model dictionary.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (detection_list, sub_list, other_list)
        Each is a list of base category names that end with:
         - "_Detection"
         - "_SUB"
         - neither
    """
    detection, subcat, other = [], [], []
    for base_cat in sorted(model_dict.keys()):
        if base_cat.endswith("_Detection"):
            detection.append(base_cat)
        elif base_cat.endswith("_SUB"):
            subcat.append(base_cat)
        else:
            other.append(base_cat)
    return (detection, subcat, other)


def annotate_category(pool: Pool,
                      category: str,
                      lang: str,
                      model_path: str,
                      fetch_func,
                      error_csv: str) -> List[Tuple[Optional[int], str, int]]:
    """
    Annotate a single category for one language, dispatch chunks to the pool,
    then return the list of (pred_label, doc_id, sentence_id) updates.

    Parameters
    ----------
    pool        : multiprocessing.Pool already configured
    category    : DB column to update (e.g. "Secu_Detection")
    lang        : "EN" or "FR"
    model_path  : Path to the .model file used for inference
    fetch_func  : Callable that returns rows to annotate
    error_csv   : Path to CSV where length-overflow errors are logged
    """
    conn = safe_get_connection()
    cur = conn.cursor()

    # Ensure target column exists
    ensure_column(cur, category)

    # ─── FETCH ROWS TO ANNOTATE ───────────────────────────────────────────
    base_main = re.sub(r'_?\d*_SUB$', '', category) + "_Detection"
    try:
        rows = fetch_func(cur, category)          # (cur, category)
    except TypeError:
        rows = fetch_func(cur, category, base_main)  # (cur, sub_cat, parent_det)

    if not rows:
        print(f" -> No rows need annotation for category='{category}', lang='{lang}'.")
        cur.close()
        conn.close()
        return []

    n_rows = len(rows)
    print(f" -> Annotating category='{category}' for lang='{lang}' on {n_rows} rows.")

    # ─── PREPARE CHUNKS & PARALLEL DISPATCH ──────────────────────────────
    tasks, all_updates, all_errors = [], [], []
    for i in range(0, n_rows, CHUNK_SIZE):
        tasks.append({
            "category": category,
            "lang": lang,
            "model_path": model_path,
            "rows": rows[i:i + CHUNK_SIZE],
        })

    chunk_bar = tqdm(total=len(tasks), desc=f"{category}({lang})",
                     position=0, leave=True, unit="chunk")

    for result in pool.imap_unordered(_worker_annotate_chunk, tasks):
        all_updates.extend(result["updates"])
        all_errors.extend(result["errors"])
        chunk_bar.update(1)

    chunk_bar.close()
    cur.close()
    conn.close()

    # Log sentences >512 tokens
    append_to_error_csv(error_csv, all_errors)
    return all_updates


def compute_and_log_metrics(
    category: str,
    lang: str,
    updates: List[Tuple[Optional[int], str, int]],
    metrics_csv: str
) -> None:
    """
    Compute label distribution from newly annotated rows and log to CSV.

    Parameters
    ----------
    category : str
        The category name.
    lang : str
        "EN" or "FR".
    updates : List[Tuple[Optional[int], str, int]]
        The newly made predictions: (pred_label, doc_id, sentence_id).
    metrics_csv : str
        Path to the metrics CSV.
    """
    if not updates:
        return
    # Filter out None predictions and do a quick value_counts
    arr = [u[0] for u in updates if u[0] is not None]
    if not arr:
        return

    s = pd.Series(arr).value_counts(dropna=False)
    print(f"Distribution for '{category}' (lang={lang}):\n{s}")
    append_to_metrics_csv(metrics_csv, category, lang, s)


##############################################################################
#                               9. MAIN FUNCTION
##############################################################################
def main():
    """
    Main entry point:

    1) Prompt user for language annotation choice: 
       a) English only
       b) French only
       c) Both languages.
    2) Prompt user for device usage mode (CPU, GPU, or both).
    3) Prompt user for CPU/GPU batch sizes.
    4) Load all .model files from the "models" directory into model_dict.
    5) Sort categories into detection, sub, other.
    6) Create a Pool with dynamic device assignment.
    7) For each category in detection → sub → other:
         - For each language in the category (but only if the language is allowed
           by the user's choice), annotate those rows in parallel and update the DB.
         - Log metrics distribution.
    8) Flush any offline predictions if DB connection fails.
    9) Close the pool.
    """
    # Prompt user for language choice
    print(
        "Which lines do you want to annotate?\n"
        "   1) Only English lines (EN)\n"
        "   2) Only French lines (FR)\n"
        "   3) Both English and French lines\n"
    )
    lang_choice = input("Select your choice [1/2/3]: ").strip()
    if lang_choice == "1":
        user_lang_mode = ["EN"]
    elif lang_choice == "2":
        user_lang_mode = ["FR"]
    else:
        # Default to both if unrecognized or "3"
        user_lang_mode = ["EN", "FR"]

    # Prompt device usage
    print(
        "\nChoose device mode for inference:\n"
        "   1) CPU only\n"
        "   2) GPU only\n"
        "   3) Both CPU and GPU\n"
    )
    device_choice = input("Select device mode [1/2/3]: ").strip()
    if device_choice == "1":
        mode = "cpu"
    elif device_choice == "2":
        mode = "gpu"
    elif device_choice == "3":
        mode = "both"
    else:
        print("[WARN] Unrecognized input. Defaulting to CPU-only.")
        mode = "cpu"

    # CPU/GPU batch sizes
    def_cpu_bs = 25
    def_gpu_bs = 50
    try:
        cpu_bs_in = input(f"CPU batch size (default {def_cpu_bs}): ").strip()
        cpu_bs = int(cpu_bs_in) if cpu_bs_in else def_cpu_bs
    except ValueError:
        cpu_bs = def_cpu_bs

    dev = pick_device().type
    if mode in ["gpu", "both"] and dev in ("cuda", "mps"):
        try:
            gpu_bs_in = input(f"GPU batch size (default {def_gpu_bs}): ").strip()
            gpu_bs = int(gpu_bs_in) if gpu_bs_in else def_gpu_bs
        except ValueError:
            gpu_bs = def_gpu_bs
    else:
        gpu_bs = cpu_bs  # no GPU effectively

    print(f"[INFO] Using CPU batch size={cpu_bs}, GPU batch size={gpu_bs}.")

    # Initialize CSV logs
    init_error_csv(ERROR_CSV_PATH)
    init_metrics_csv(METRICS_CSV_PATH)

    # Determine the directory containing the .model files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "..", "..", "models")

    # Load model files
    model_dict = load_all_models_strict(models_dir)
    print(f"[INFO] Found {len(model_dict)} base categories in {models_dir}.")

    # Classify them: detection, sub, other
    cat_det, cat_sub, cat_other = classify_categories(model_dict)
    print(f"[INFO] Categories (detection): {cat_det}")
    print(f"[INFO] Categories (sub): {cat_sub}")
    print(f"[INFO] Categories (other): {cat_other}")

    # Decide number of processes
    max_procs = cpu_count()
    num_workers = max_procs  # you may set a different policy if desired

    # Build the device list
    devices_list = configure_devices_mode(mode, num_workers)
    print(f"[INFO] Devices list for pool: {devices_list}")

    # Prepare manager queues
    manager = Manager()
    devices_q = manager.Queue()
    for d in devices_list:
        devices_q.put(d)

    pos_q = manager.Queue()
    # We reserve position=0 for the chunk-level bar,
    # so the workers get positions from 1 up to num_workers
    for wpos in range(1, 1 + num_workers):
        pos_q.put(wpos)

    tqdm.set_lock(RLock())  # for safe multi-process TQDM

    pool = Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(devices_q, cpu_bs, gpu_bs, pos_q),
    )

    try:
        # Loop over categories in detection → sub → other
        for cat_list in [cat_det, cat_sub, cat_other]:
            for category in cat_list:
                # For each category, see which languages are available
                # in model_dict[category] (e.g. {"EN": path, "FR": path}).
                available_langs = model_dict[category].keys()
                # We only process those languages the user selected
                # and for which we have a .model file.
                for lang in available_langs:
                    if lang not in user_lang_mode:
                        continue

                    model_path = model_dict[category][lang]

                    # To replicate your logic of restricting DB queries by language,
                    # we can define local fetch functions that only pick rows of `lang`.
                    def fetch_detection_for_lang(cur, cat_col):
                        limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
                        cur.execute(f"""
                            SELECT "{DOC_ID_COL}",
                                   "{SENTENCE_ID_COL}",
                                   "{TEXT_COL}"
                            FROM public."{TABLE_NAME}"
                            WHERE "{cat_col}" IS NULL
                              AND "{TEXT_COL}" IS NOT NULL
                              AND "{LANG_COL}" = %s
                            {limit_sql};
                        """, (lang,))
                        return cur.fetchall()

                    def fetch_sub_for_lang(cur, sub_cat, parent_det):
                        limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
                        cur.execute(f"""
                            SELECT "{DOC_ID_COL}",
                                   "{SENTENCE_ID_COL}",
                                   "{TEXT_COL}"
                            FROM public."{TABLE_NAME}"
                            WHERE "{sub_cat}" IS NULL
                              AND "{parent_det}" = 1
                              AND "{TEXT_COL}" IS NOT NULL
                              AND "{LANG_COL}" = %s
                            {limit_sql};
                        """, (lang,))
                        return cur.fetchall()

                    def fetch_other_for_lang(cur, cat_col):
                        limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
                        cur.execute(f"""
                            SELECT "{DOC_ID_COL}",
                                   "{SENTENCE_ID_COL}",
                                   "{TEXT_COL}"
                            FROM public."{TABLE_NAME}"
                            WHERE "{cat_col}" IS NULL
                              AND "{TEXT_COL}" IS NOT NULL
                              AND "{LANG_COL}" = %s
                            {limit_sql};
                        """, (lang,))
                        return cur.fetchall()

                    # Decide fetch function according to suffix
                    if category.endswith("_Detection"):
                        fetch_func = lambda c, cat=category: fetch_detection_for_lang(c, cat)
                    elif category.endswith("_SUB"):
                        # We'll rely on the general sub approach but incorporate language filter
                        def sub_fetch_wrapper(cur, sub_col, parent_col):
                            return fetch_sub_for_lang(cur, sub_col, parent_col)
                        fetch_func = sub_fetch_wrapper
                    else:
                        fetch_func = lambda c, cat=category: fetch_other_for_lang(c, cat)

                    # Annotate
                    updates = annotate_category(
                        pool=pool,
                        category=category,
                        lang=lang,
                        model_path=model_path,
                        fetch_func=fetch_func,
                        error_csv=ERROR_CSV_PATH
                    )

                    # Bulk update in DB
                    if updates:
                        start_time = time.perf_counter()
                        conn = safe_get_connection()
                        cur = conn.cursor()
                        try:
                            bulk_update_column(cur, category, updates)
                        except psycopg2.Error as e:
                            print(f"[ERROR] DB update failed for {category} (lang={lang}): {e}")
                            store_offline_predictions(category, updates)
                            # Attempt flush offline
                            while True:
                                try:
                                    conn = safe_get_connection()
                                    cur = conn.cursor()
                                    flush_offline_predictions(cur)
                                    break
                                except psycopg2.Error as e2:
                                    print(f"[ERROR] Still cannot flush offline predictions: {e2}")
                                    retry_opt = input(
                                        "Options:\n"
                                        "  1) Try again\n"
                                        "  2) Export offline to CSV and exit\n"
                                        "Choice [1/2]: "
                                    ).strip()
                                    if retry_opt == "2":
                                        filename = "offline_predictions.csv"
                                        export_offline_predictions_csv(filename)
                                        sys.exit(f"[EXIT] Predictions saved to «{filename}». Exiting.")
                        else:
                            elapsed = time.perf_counter() - start_time
                            print(f"[DB] Updated {len(updates)} rows for '{category}' (lang={lang}) in {elapsed:.2f}s.")

                        cur.close()
                        conn.close()

                        # Compute metrics
                        compute_and_log_metrics(category, lang, updates, METRICS_CSV_PATH)

    finally:
        # Attempt final flush of offline predictions
        if OFFLINE_PREDICTIONS:
            print("\n[INFO] Attempting final flush of offline predictions...")
            while True:
                try:
                    conn = safe_get_connection()
                    cur = conn.cursor()
                    flush_offline_predictions(cur)
                    print("[INFO] All offline predictions have been successfully flushed.")
                    break
                except psycopg2.Error as e:
                    print(f"[ERROR] Flushing offline predictions failed: {e}")
                    choice2 = input(
                        "Options:\n"
                        "  1) Try again\n"
                        "  2) Export offline to CSV and exit\n"
                        "Choice [1/2]: "
                    ).strip()
                    if choice2 == "2":
                        filename = "offline_predictions.csv"
                        export_offline_predictions_csv(filename)
                        sys.exit(f"[EXIT] Predictions saved to «{filename}». Exiting.")

        pool.close()
        pool.join()

    print("\n[DONE] All categories have been processed. Annotation complete.")


if __name__ == "__main__":
    main()