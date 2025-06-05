"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
8_NER.py

MAIN OBJECTIVE:
---------------
This script performs large-scale Named Entity Recognition (PER, ORG, LOC) on the
sentence-level data stored in the PostgreSQL table CCF_processed_data.
It parallelises inference across CPU and/or GPU workers, writes the recognised
entities back to the database in bulk, and guarantees no data loss through an
offline-buffer mechanism that can be exported to CSV if the DB becomes
unavailable.

Dependencies:
-------------
- Standard library : os, sys, csv, json, time, queue, random, re, argparse,
  unicodedata, datetime, contextlib, typing, multiprocessing
- Third-party     : psycopg2, pandas, torch, tqdm, joblib,
  spaCy (fr_core_news_lg), transformers (Hugging Face)

MAIN FEATURES:
--------------
1) Resilient DB workflow
   - Safe connection loop with interactive credential override.  
   - Automatic creation of the ner_entities column.  
   - Bulk updates via a temporary table to avoid row-by-row commits.

2) Language-aware NER pipelines
   - French : spaCy for PER + CamemBERT for ORG/LOC.  
   - English : BERT-base-NER for PER/ORG/LOC.  
   - Unicode normalisation and span-merging to reduce fragment noise.

3) Adaptive parallelisation
   - Workers inherit a device (CPU / CUDA / MPS) from a shared queue.  
   - User-defined batch sizes per device type.  
   - TQDM progress bars with distinct positions for clean multi-process logs.

4) Offline-buffer fail-safe
   - All row-level updates are cached in memory if a DB write fails.  
   - Users can retry, supply new credentials, or export the buffer to
     offline_predictions.csv and exit gracefully.

5) Chunked processing at scale
   - Rows lacking NER are fetched once, chunked (default 5 000) and dispatched
     to the worker pool.  
   - Results are merged and flushed back in a single transaction.

6) Device negotiation helper
   - `pick_device()` auto-detects the best available accelerator.  
   - `configure_devices_mode()` builds a per-worker device map for
     cpu, gpu, or both modes.

Author :
--------
Antoine Lemor
"""


from __future__ import annotations
import os
import sys
import csv
import time
import queue
import random
import unicodedata
import re
import json
import argparse
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional

import psycopg2
import psycopg2.extras
import pandas as pd
import torch

from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count, RLock, current_process

import joblib
from joblib import Parallel, delayed

# NER-specific imports
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# ###########################################################################
# GLOBAL CONFIG - DB & TABLE
# ###########################################################################
DB_PARAMS: Dict[str, Any] = {
    "host": "localhost",
    "port": 5432,
    "dbname": "CCF",  
    "user": "antoine",
    "password": "Absitreverentiavero19!",
}

TABLE_NAME = "CCF_processed_data"

DOC_ID_COL       = "doc_id"
SENTENCE_ID_COL  = "sentence_id"
TEXT_COL         = "sentences"
LANG_COL         = "language"

# Name of the target column for the recognized entities
NER_COLUMN       = "ner_entities"

# If you want to limit the number of rows read for debugging,
# set to a positive integer. Otherwise, set to None to process all.
LIMIT_ROWS       = None

# Chunk size for parallel tasks
CHUNK_SIZE = 5000

# Buffer for storing offline predictions if DB update fails
OFFLINE_PREDICTIONS = []  # type: List[Dict[str, Any]]


# ###########################################################################
# 1) OFFLINE BUFFER FUNCTIONS
# ###########################################################################
def store_offline_predictions(
    updates: List[Tuple[str, int, str]]
) -> None:
    """
    Store row-level NER updates in a global buffer so they are not lost
    if we fail to update the DB due to a connection error.

    Each update tuple has the form (doc_id, sentence_id, ner_json).

    Parameters
    ----------
    updates : List[Tuple[str, int, str]]
        Each tuple is (doc_id, sentence_id, ner_json).
    """
    global OFFLINE_PREDICTIONS
    OFFLINE_PREDICTIONS.append({
        "updates": updates
    })


def export_offline_predictions_csv(filename: str) -> None:
    """
    Export all offline predictions to CSV with columns:
      doc_id, sentence_id, ner_entities

    Parameters
    ----------
    filename : str
        Path to the CSV file to create.
    """
    global OFFLINE_PREDICTIONS
    rows = []
    for entry in OFFLINE_PREDICTIONS:
        for (doc_id, sent_id, ner_json) in entry["updates"]:
            rows.append((doc_id, sent_id, ner_json))

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "sentence_id", "ner_entities"])
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
        updates = entry["updates"]
        bulk_update_ner_column(cur, updates)


# ###########################################################################
# 2) DB CONNECTION & HELPERS
# ###########################################################################
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


def ensure_ner_column(cur: psycopg2.extensions.cursor) -> None:
    """
    Ensure an ner_entities column (TEXT) exists in TABLE_NAME.
    If it doesn't exist, create it.

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
        Active DB cursor.
    """
    cur.execute(f"""
        ALTER TABLE public."{TABLE_NAME}"
        ADD COLUMN IF NOT EXISTS "{NER_COLUMN}" TEXT;
    """)


def fetch_rows_for_ner(cur: psycopg2.extensions.cursor) -> List[Tuple[str, int, str, str]]:
    """
    Fetch rows needing NER annotation (where ner_entities IS NULL),
    ignoring rows with null 'sentences'. Optionally limit via LIMIT_ROWS.

    Returns
    -------
    rows : List[Tuple[doc_id, sentence_id, text, language]]
    """
    limit_sql = f"LIMIT {LIMIT_ROWS}" if LIMIT_ROWS else ""
    cur.execute(f"""
        SELECT "{DOC_ID_COL}",
               "{SENTENCE_ID_COL}",
               "{TEXT_COL}",
               "{LANG_COL}"
        FROM public."{TABLE_NAME}"
        WHERE "{NER_COLUMN}" IS NULL
          AND "{TEXT_COL}" IS NOT NULL
        {limit_sql};
    """)
    return cur.fetchall()


def bulk_update_ner_column(
    cur: psycopg2.extensions.cursor,
    updates: List[Tuple[str, int, str]]
) -> None:
    """
    Bulk-update the *ner_entities* column in ``TABLE_NAME`` using a
    temporary table and a single ``UPDATE ... FROM`` statement.

    The previous implementation stored ``doc_id`` as *TEXT* in the
    temporary table, while the production table defines it as *BIGINT*,
    provoking the PostgreSQL error::

        operator does not exist: bigint = text

    To guarantee type alignment we:

    1. Define ``doc_id`` as **BIGINT** in the temporary table.
    2. Cast every incoming ``doc_id`` to ``int`` (Python’s
       built-in maps to PostgreSQL *bigint* via psycopg2).

    Parameters
    ----------
    cur : psycopg2.extensions.cursor
        An open, autocommit-enabled cursor.
    updates : List[Tuple[str, int, str]]
        Triplets of *(doc_id, sentence_id, ner_json)* produced by the
        worker processes.
    """
    if not updates:
        return

    # ###########################################################################
    # 1) Create an unlogged temporary table with matching column types
    # ###########################################################################
    tmp_name = f"_tmp_ner_{abs(hash(NER_COLUMN)) % 10_000_000}"
    cur.execute(f"DROP TABLE IF EXISTS {tmp_name};")
    cur.execute(f"""
        CREATE UNLOGGED TABLE {tmp_name} (
            doc_id      BIGINT,          -- matches public."{TABLE_NAME}".doc_id
            sentence_id INTEGER,
            ner_text    TEXT
        );
    """)

    # ###########################################################################
    # 2) Insert the payload, ensuring proper casting of *doc_id*
    # ###########################################################################
    # Many callers pass doc_id as str; convert to int for type safety
    cleaned_updates = [
        (int(doc_id), sentence_id, ner_json)
        for doc_id, sentence_id, ner_json in updates
    ]

    psycopg2.extras.execute_values(
        cur,
        f"INSERT INTO {tmp_name} (doc_id, sentence_id, ner_text) VALUES %s",
        cleaned_updates,
        page_size=1000
    )

    # ###########################################################################
    # 3) Single-statement bulk update from the temporary table
    # ###########################################################################
    cur.execute(f"""
        UPDATE public."{TABLE_NAME}" AS t
           SET "{NER_COLUMN}" = s.ner_text
          FROM {tmp_name} AS s
         WHERE t.doc_id      = s.doc_id    -- types now align (BIGINT)
           AND t.sentence_id = s.sentence_id;
    """)

    # ###########################################################################
    # 4) House-keeping
    # ###########################################################################
    cur.execute(f"DROP TABLE {tmp_name};")


# ###########################################################################
# 3) NER LOGIC (same as original script, just integrated for DB usage)
# ###########################################################################
MODEL_FR = "Jean-Baptiste/camembert-ner"
MODEL_EN = "dslim/bert-base-NER"
_EMPTY_JSON = json.dumps({"PER": [], "ORG": [], "LOC": []}, ensure_ascii=False)

# spaCy FR model for PERSON
_nlp_spacy = spacy.load("fr_core_news_lg")

# Cache for CamemBERT pipelines per device
_PIPELINE_FR_CAMEMBERT: Dict[int, pipeline] = {}

def _get_fr_camembert(device: str) -> pipeline:
    """Load (or retrieve) CamemBERT-NER pipeline for ORG/LOC (FR)."""
    if device.startswith("cuda"):
        dev_id = int(device.split(":")[-1])
    elif device.startswith("mps"):
        # huggingface pipeline might treat MPS as GPU with dev_id=0 or -1, 
        # but let's unify as -1 for safe usage. Adjust if needed.
        dev_id = -1  
    else:
        dev_id = -1

    if dev_id not in _PIPELINE_FR_CAMEMBERT:
        tok = AutoTokenizer.from_pretrained(MODEL_FR, use_fast=True, add_prefix_space=True)
        mdl = AutoModelForTokenClassification.from_pretrained(MODEL_FR)
        _PIPELINE_FR_CAMEMBERT[dev_id] = pipeline(
            task="ner",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
            device=dev_id,
        )
    return _PIPELINE_FR_CAMEMBERT[dev_id]


def _normalize_fr(text: str) -> str:
    """Unicode NFC + standard apostrophes."""
    return unicodedata.normalize("NFC", text.replace("’", "'"))


def _merge_spans(offsets: List[Tuple[int,int]], text: str) -> List[str]:
    """Merge adjacent spans separated by ≤5 non-word chars."""
    if not offsets:
        return []
    offsets = sorted(offsets, key=lambda x: x[0])
    merged, (s0, e0) = [], offsets[0]
    for s, e in offsets[1:]:
        # If there's a short non-word gap between [e0 : s], merge them
        if re.fullmatch(r"\W{1,5}", text[e0:s]):
            e0 = e
        else:
            merged.append(text[s0:e0])
            s0, e0 = s, e
    merged.append(text[s0:e0])
    return merged


def _extract_persons_spacy(text: str) -> List[str]:
    """Extract PER via spaCy FR."""
    doc = _nlp_spacy(text)
    seen, persons = set(), []
    for ent in doc.ents:
        if ent.label_ == "PER" and ent.text not in seen:
            seen.add(ent.text)
            persons.append(ent.text)
    return persons


def _extract_org_loc_camembert(text: str, device: str) -> Dict[str, List[str]]:
    """Extract ORG/LOC via CamemBERT-NER FR."""
    preds = _get_fr_camembert(device)(text)
    offsets = {"ORG": [], "LOC": []}
    for ent in preds:
        grp = ent.get("entity_group")
        if grp in offsets:
            offsets[grp].append((ent["start"], ent["end"]))
    return {lbl: _merge_spans(offsets[lbl], text) for lbl in offsets}


# EN
_PIPELINE_EN_BERT: Dict[int, pipeline] = {}

def _get_en_bert(device: str) -> pipeline:
    """Load (or retrieve) BERT-NER pipeline for PER/ORG/LOC (EN)."""
    if device.startswith("cuda"):
        dev_id = int(device.split(":")[-1])
    elif device.startswith("mps"):
        dev_id = -1
    else:
        dev_id = -1

    if dev_id not in _PIPELINE_EN_BERT:
        tok = AutoTokenizer.from_pretrained(MODEL_EN, use_fast=True)
        _PIPELINE_EN_BERT[dev_id] = pipeline(
            task="token-classification",
            model=MODEL_EN,
            tokenizer=tok,
            aggregation_strategy="first",
            device=dev_id,
        )
    return _PIPELINE_EN_BERT[dev_id]


def _normalize_en(text: str) -> str:
    """Replace curly quotes + NFC normalization."""
    txt = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return unicodedata.normalize("NFC", txt)


def _extract_en_entities(text: str, device: str) -> Dict[str, List[str]]:
    """Extract PER/ORG/LOC via HF BERT-NER EN."""
    norm = _normalize_en(text)
    preds = _get_en_bert(device)(norm)
    offsets: Dict[str, List[Tuple[int,int]]] = {"PER": [], "ORG": [], "LOC": []}
    for ent in preds:
        grp = ent.get("entity_group")
        if grp in offsets:
            offsets[grp].append((ent["start"], ent["end"]))
    # Merge offsets into strings
    return {lbl: _merge_spans(offsets[lbl], norm) for lbl in offsets}


def ner_sentence(text: str | float, lang: str, device: str) -> str:
    """
    Route to language-specific NER, return JSON string or empty JSON.
    FR → (spaCy PER + CamemBERT ORG/LOC)
    EN → BERT NER (PER/ORG/LOC)
    """
    if not isinstance(text, str) or not text.strip():
        return _EMPTY_JSON
    try:
        lang = lang.upper()
        if lang == "FR":
            txt = _normalize_fr(text)
            persons = _extract_persons_spacy(txt)
            orgloc = _extract_org_loc_camembert(txt, device)
            entities = {"PER": persons, "ORG": orgloc["ORG"], "LOC": orgloc["LOC"]}
        elif lang == "EN":
            entities = _extract_en_entities(text, device)
        else:
            return _EMPTY_JSON
        return json.dumps(entities, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] NER failed ({lang}): {e}")
        return _EMPTY_JSON


# ###########################################################################
# 4) DEVICE HANDLING & PARALLEL POOL
# ###########################################################################
def pick_device() -> str:
    """
    Return the best available device: 'cuda' (GPU), 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_devices_mode(user_choice: str, max_workers: int) -> List[str]:
    """
    Create a list of device names for each worker based on the user's choice:
      - 'cpu': all workers on CPU
      - 'gpu': only one worker on GPU (if available), else CPU fallback
      - 'both': one GPU worker (if available) + the rest CPU
    The length of this list = number of worker processes.
    """
    dev = pick_device()
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
                return [dev]  # only 1 worker
            else:
                return [dev] + ["cpu"] * (max_workers - 1)
        else:
            print("[WARN] No GPU/MPS available. Using CPU-only for all workers.")
            return ["cpu"] * max_workers
    else:
        print("[WARN] Unrecognized device mode. Defaulting to CPU-only.")
        return ["cpu"] * max_workers


WORKER_DEVICE = "cpu"
WORKER_BS_CPU = 2000
WORKER_BS_GPU = 2000
WORKER_POSITION = 0


def _worker_init(
    devices_q: "queue.Queue[str]",
    bs_cpu: int,
    bs_gpu: int,
    pos_q: "queue.Queue[int]"
) -> None:
    """
    Assign each worker a device from devices_q and set the appropriate batch size.
    Retrieve a TQDM position for local progress bars.
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


# ###########################################################################
# 5) WORKER FUNCTION
# ###########################################################################
def _worker_process_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function that handles a chunk of rows for NER annotation.

    task dict keys:
      - "chunk_rows" (List[Tuple[doc_id, sentence_id, text, lang]])
    """
    global WORKER_DEVICE, WORKER_BS_CPU, WORKER_BS_GPU, WORKER_POSITION

    chunk_rows = task["chunk_rows"]
    device_label = WORKER_DEVICE

    pid = current_process().name
    pbar = tqdm(
        total=len(chunk_rows),
        desc=f"[{pid} - {device_label}] NER",
        position=WORKER_POSITION,
        leave=False,
        dynamic_ncols=True,
        unit="sent",
    )

    # Decide batch size
    if WORKER_DEVICE in ("cuda", "mps"):
        batch_size = WORKER_BS_GPU
    else:
        batch_size = WORKER_BS_CPU

    all_updates = []  # (doc_id, sentence_id, ner_json)
    i = 0
    while i < len(chunk_rows):
        sub = chunk_rows[i : i + batch_size]
        i += batch_size

        sub_updates = []
        for (doc_id, sent_id, text, lang) in sub:
            ner_json = ner_sentence(text, lang, WORKER_DEVICE)
            sub_updates.append((doc_id, sent_id, ner_json))

        all_updates.extend(sub_updates)
        pbar.update(len(sub))

    pbar.close()
    return {"updates": all_updates}


# ###########################################################################
# 6) MAIN FUNCTION
# ###########################################################################
def main():
    """
    1) Prompt user for device usage (CPU, GPU, or both).
    2) Prompt user for CPU/GPU batch sizes.
    3) Read rows from 'CCF_processed_data' where ner_entities is null, in chunks.
    4) Distribute chunks across multiple worker processes, each loading the
       specified device (CPU/GPU) and performing NER on the assigned chunk.
    5) Collect results, do a bulk update in the DB.
    6) In case of DB failure, store predictions offline and re-attempt or exit.
    """
    print(
        "Choose device mode for NER inference:\n"
        "   1) CPU only\n"
        "   2) GPU only\n"
        "   3) Both CPU and GPU\n"
    )
    choice = input("Select device mode [1/2/3]: ").strip()
    if choice == "1":
        mode = "cpu"
    elif choice == "2":
        mode = "gpu"
    elif choice == "3":
        mode = "both"
    else:
        print("[WARN] Unrecognized input. Defaulting to CPU-only.")
        mode = "cpu"

    # Prompt for batch sizes
    def_cpu_bs = 2000
    def_gpu_bs = 2000
    try:
        cpu_bs_in = input(f"CPU batch size (default {def_cpu_bs}): ").strip()
        cpu_bs = int(cpu_bs_in) if cpu_bs_in else def_cpu_bs
    except ValueError:
        cpu_bs = def_cpu_bs

    # If user chooses GPU or BOTH, ask for GPU BS if device is actually available
    dev = pick_device()
    if mode in ["gpu", "both"] and dev in ("cuda", "mps"):
        try:
            gpu_bs_in = input(f"GPU batch size (default {def_gpu_bs}): ").strip()
            gpu_bs = int(gpu_bs_in) if gpu_bs_in else def_gpu_bs
        except ValueError:
            gpu_bs = def_gpu_bs
    else:
        gpu_bs = cpu_bs

    print(f"[INFO] Using CPU batch size={cpu_bs}, GPU batch size={gpu_bs}.")

    # Prepare DB & table
    conn = safe_get_connection()
    cur = conn.cursor()
    ensure_ner_column(cur)
    cur.close()
    conn.close()

    # Gather rows that need NER
    conn = safe_get_connection()
    cur = conn.cursor()
    rows = fetch_rows_for_ner(cur)
    cur.close()
    conn.close()

    n_rows = len(rows)
    print(f"[INFO] Total rows needing NER: {n_rows}")
    if not n_rows:
        print("[DONE] No rows to annotate. Exiting.")
        return

    # Decide number of processes
    num_workers = cpu_count()
    devices_list = configure_devices_mode(mode, num_workers)
    print(f"[INFO] Devices list for pool: {devices_list}")

    manager = Manager()
    devices_q = manager.Queue()
    for d in devices_list:
        devices_q.put(d)

    pos_q = manager.Queue()
    # We'll reserve 1 for the chunk-level bar, so worker bars start at position=1
    for wpos in range(1, 1 + num_workers):
        pos_q.put(wpos)

    tqdm.set_lock(RLock())  # safe multi-process TQDM

    pool = Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(devices_q, cpu_bs, gpu_bs, pos_q),
    )

    # Split rows into chunks
    chunks = [rows[i : i + CHUNK_SIZE] for i in range(0, n_rows, CHUNK_SIZE)]

    all_updates: List[Tuple[str, int, str]] = []
    pbar_chunks = tqdm(total=len(chunks), desc="All chunks", position=0, unit="chunk")

    try:
        # Process chunks in parallel
        tasks = [{"chunk_rows": ch} for ch in chunks]

        for result in pool.imap_unordered(_worker_process_chunk, tasks):
            chunk_updates = result["updates"]  # List[ (doc_id, sentence_id, ner_json) ]
            all_updates.extend(chunk_updates)
            pbar_chunks.update(1)

        pbar_chunks.close()

        # Bulk update
        if all_updates:
            start_time = time.perf_counter()
            conn = safe_get_connection()
            cur = conn.cursor()
            try:
                bulk_update_ner_column(cur, all_updates)
            except psycopg2.Error as e:
                print(f"[ERROR] DB update failed for ner_entities: {e}")
                store_offline_predictions(all_updates)
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
                print(f"[DB] Updated {len(all_updates)} rows with ner_entities in {elapsed:.2f}s.")

            cur.close()
            conn.close()

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

    print("[DONE] NER annotation completed successfully.")


if __name__ == "__main__":
    main()