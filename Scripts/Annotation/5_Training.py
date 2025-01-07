"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
5_Training.py

MAIN OBJECTIVE:
---------------
This script manages the training process for Camembert (FR) and Bert (EN) 
models by loading data, detecting necessary files, and orchestrating 
training routines.

Dependencies:
-------------
- os
- json
- sys
- glob
- shutil
- pandas
- torch
- AugmentedSocialScientist (Camembert, Bert)

MAIN FEATURES:
--------------
1) Detects which models have been fully trained, partially trained, or not started 
   and logs these states.  
2) Loads JSONL data into DataFrames and checks label distributions.  
3) Instantiates and trains Camembert (FR) or Bert (EN) depending on the language.  
4) Handles skipping of training if only zero labels are found or data is insufficient.  
5) Produces a CSV file listing all non-trained models.  

Author:
-------
Antoine Lemor
"""

import json
import sys
import os
import glob
import shutil  # <-- pour pouvoir supprimer des dossiers
import pandas as pd
import torch

from AugmentedSocialScientist.models import Camembert, Bert

# --------------------------------------------------------------------
# FUNCTION TO GET THE DEVICE (CUDA, MPS, OR CPU)
# --------------------------------------------------------------------
def get_device():
    """
    Detects an available GPU (CUDA or MPS) or defaults to CPU.

    Returns:
    --------
    torch.device
        The device to use for model computations (GPU or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA) for computations.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (MPS) for computations.")
    else:
        device = torch.device("cpu")
        print("Using CPU for computations.")
    return device

# Debugging print to check script execution
print("Script started.")

# Get the base path of the script
base_path = os.path.dirname(os.path.abspath(__file__))

# Paths (relative)
annotation_base_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotation_bases")
model_output_dir = os.path.join(base_path, "..", "..", "models")  # Directly points to 'models'
log_output_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "Annotation_logs")
training_data_dir = os.path.join(base_path, "..", "..", "Database", "Training_data")  # To save the final CSV

print(f"Model output directory: {model_output_dir}")  # Debug

# Check if the log directory exists
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)

# Check if the model output directory exists
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

# --------------------------------------------------------------------
# LOGGER CLASS FOR LOGGING
# --------------------------------------------------------------------
class Logger(object):
    """
    A simple logger to redirect stdout to a file and the terminal.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

# --------------------------------------------------------------------
# FUNCTION TO LOAD JSONL -> PANDAS
# --------------------------------------------------------------------
def load_jsonl_to_dataframe(filepath):
    """
    Loads a JSONL file into a pandas DataFrame.

    Parameters:
    -----------
    filepath : str
        The path to the JSONL file.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the loaded JSON lines.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# FUNCTION TO DETERMINE REQUIRED FILES BASED ON LANGUAGE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str):
    """
    Returns the list of required files for a given language model.

    Parameters:
    -----------
    language : str
        Either "FR" for Camembert or another code for Bert.

    Returns:
    --------
    list
        A list of required file names.
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    else:
        return ["config.json", "pytorch_model.bin", "vocab.txt"]

# --------------------------------------------------------------------
# FUNCTION TO CHECK HOW MANY MODEL FILES ARE PRESENT FOR A GIVEN LANGUAGE
# --------------------------------------------------------------------
def get_model_file_count(model_dir, language):
    """
    Counts how many required model files exist in 'model_dir'.

    Parameters:
    -----------
    model_dir : str
        The directory where model files should be stored.
    language : str
        Language code used to determine the required files.

    Returns:
    --------
    int
        The number of required files present in 'model_dir'.
    """
    required_files = get_required_files_for_language(language)
    present_count = 0

    if not os.path.exists(model_dir):
        # The directory does not exist
        return 0

    for file_name in required_files:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            present_count += 1

    return present_count

# --------------------------------------------------------------------
# CUSTOM EXCEPTION TO SKIP TRAINING
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """
    Custom exception raised to indicate training should be skipped.
    """
    pass

# --------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------------------------
def train_models(base_dir, model_output_dir, log_output_dir):
    """
    Main function that coordinates model training. It checks the training 
    status of each model, loads data, initiates training, and handles logs 
    and final summaries.

    Parameters:
    -----------
    base_dir : str
        Path to annotation bases for training.
    model_output_dir : str
        Path where trained models will be stored.
    log_output_dir : str
        Path where training logs will be stored.
    """
    # Get the device (GPU or CPU)
    device = get_device()

    # For global reporting
    fully_trained_count = 0
    partial_count = 0
    not_started_count = 0
    skipped_count = 0  # Counter for skipped models

    # List to store non-trained models (due to insufficient annotations or other reasons)
    # and their distribution
    non_trained_models = []

    # ----------------------------------------------------------------
    # 1) GET THE LIST OF ALL MODELS AND DETERMINE THEIR STATUS
    # ----------------------------------------------------------------
    models_info = []

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path):
            continue

        # Process for each language
        for language in ['EN', 'FR']:
            # Search for training and validation files
            train_filepath_pattern = os.path.join(label_path, 'train', language, f"*train*_{language}.jsonl")
            test_filepath_pattern = os.path.join(label_path, 'validation', language, f"*validation*_{language}.jsonl")

            train_files = glob.glob(train_filepath_pattern)
            test_files = glob.glob(test_filepath_pattern)

            if not train_files:
                continue
            if not test_files:
                continue

            # Associate training files with corresponding validation files
            for train_file in train_files:
                base_name = os.path.basename(train_file).replace('_train_', '_validation_')
                matching_test_file = None
                for test_file in test_files:
                    if base_name in test_file:
                        matching_test_file = test_file
                        break

                if not matching_test_file:
                    continue

                # Determine the model name (for logs and directory)
                model_name = os.path.basename(train_file).replace('_train_', '_')

                # The final model directory: e.g., "Cult_4_SUB_FR.jsonl.model"
                model_dir = os.path.join(model_output_dir, f"{model_name}.model")

                # Calculate the status
                file_count = get_model_file_count(model_dir, language)
                required_count = len(get_required_files_for_language(language))

                if file_count == required_count:
                    status = "fully_trained"
                    fully_trained_count += 1
                elif file_count == 0:
                    status = "not_started"
                    not_started_count += 1
                else:
                    status = "partial"
                    partial_count += 1

                # Store for the training step
                models_info.append({
                    "label": label,
                    "language": language,
                    "train_file": train_file,
                    "test_file": matching_test_file,
                    "model_name": model_name,
                    "model_dir": model_dir,
                    "status": status
                })

    # ----------------------------------------------------------------
    # 2) DISPLAY A SUMMARY IN THE TERMINAL
    # ----------------------------------------------------------------
    print("===== OVERVIEW OF MODEL STATES =====")
    print(f"Fully trained models: {fully_trained_count}")
    print(f"Not started models: {not_started_count}")
    print(f"Partially trained models: {partial_count}")
    print("===============================================")

    # ----------------------------------------------------------------
    # 3) TRAINING LOOP FOR INCOMPLETE MODELS
    # ----------------------------------------------------------------
    for info in models_info:
        if info["status"] == "fully_trained":
            # Ignore already complete models
            print(f"[INFO] Model already fully trained: {info['model_name']}")
            continue

        label = info["label"]
        language = info["language"]
        train_file = info["train_file"]
        test_file = info["test_file"]
        model_name = info["model_name"]
        model_dir = info["model_dir"]

        print(f"[TRAIN] Starting training for label: {label}, language: {language} -> {model_name}")

        # Instantiate the model based on the language
        if language == 'FR':
            print("Instantiating Camembert model for French.")
            model = Camembert(device=device)
        else:
            print("Instantiating Bert model for English.")
            model = Bert(device=device)

        # Attempt to send the model to the device (if possible)
        try:
            model.to(device)
        except AttributeError:
            print("Warning: Your model class may not support .to(device).")
            print("Please check the API of your Camembert / Bert classes.")

        # Configure the Logger
        log_filepath = os.path.join(log_output_dir, f"{model_name}_training_log.txt")
        print(f"Setting up logging to: {log_filepath}")
        logger = Logger(log_filepath)
        sys.stdout = logger
        print(f"[LOG] Logging started for {label} in {language}")

        scores = None  # Initialize scores
        train_label_counts = None
        test_label_counts = None

        try:
            # Load training and validation data
            train_data = load_jsonl_to_dataframe(train_file)
            test_data = load_jsonl_to_dataframe(test_file)
            print(f"Data loaded successfully for label: {label}, language: {language}")

            if train_data.empty or test_data.empty:
                print(f"Training or test data is empty for {label} in {language}")
                raise ValueError("Training or test data is empty")

            # Label distribution
            train_label_counts = train_data['label'].value_counts()
            test_label_counts = test_data['label'].value_counts()
            print(f"Training label distribution for {label} in {language}:")
            print(train_label_counts)
            print(f"Validation label distribution for {label} in {language}:")
            print(test_label_counts)

            # ----------------------------------------------------------------
            # Checking that there is at least 1 positive (label > 0) in TRAIN and TEST
            # ----------------------------------------------------------------
            train_has_positive = (train_data['label'] > 0).any()
            test_has_positive = (test_data['label'] > 0).any()

            if not train_has_positive or not test_has_positive:
                print("[SKIP] Training skipped because there is no positive label in either train or test set.")
                skipped_count += 1
                raise SkipTrainingException("Data containing no positive labels in train or test.")

            # Check if there are enough annotations overall
            min_annotations = 1
            if len(train_data) < min_annotations or len(test_data) < min_annotations:
                print(f"Not enough annotations for {label} in {language}. Need at least {min_annotations} samples.")
                raise ValueError("Not enough annotations")

            # Prepare DataLoader (or equivalent) via the model's encode() method
            train_loader = model.encode(
                train_data.text.values,
                train_data.label.values.astype(int)  # Explicit conversion to int
            )
            test_loader = model.encode(
                test_data.text.values,
                test_data.label.values.astype(int)  # Explicit conversion to int
            )
            print(f"Data encoding completed for label: {label}, language: {language}")

            # Relative path (relative to model_output_dir) for saving
            relative_model_output_path = f"{model_name}.model"
            print(f"Saving model to (relative path): {relative_model_output_path}")

            # Train and save the model
            scores = model.run_training(
                train_loader,
                test_loader,
                lr=5e-5,
                n_epochs=20,
                random_state=42,
                save_model_as=relative_model_output_path
            )
            print(f"Training completed successfully for {model_name}")

        except SkipTrainingException as ste:
            print(f"[INFO] {ste}")
            # Consider this model as non-trained
            non_trained_models.append({
                "model_name": model_name,
                "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                "test_distribution": test_label_counts.to_dict() if test_label_counts is not None else {}
            })
            # Delete the model directory if it exists
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"Directory deleted for non-trained model: {model_dir}")

        except Exception as e:
            print(f"Error during training for {model_name}: {e}")
            # Consider this model as non-trained as well (for "any other reason")
            non_trained_models.append({
                "model_name": model_name,
                "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                "test_distribution": test_label_counts.to_dict() if test_label_counts is not None else {}
            })
            # Delete the model directory if it exists
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"Directory deleted for non-trained model: {model_dir}")

        finally:
            # Close the logger and restore stdout
            sys.stdout = sys.__stdout__
            logger.close()
            print(f"[LOG] Logging closed for {label} in {language}")

        if scores is not None:
            print(f"[TRAIN] Training completed for {model_name}, scores: {scores}")
        else:
            # If no scores, training was not performed or abandoned
            if train_label_counts is not None and test_label_counts is not None:
                print(f"[TRAIN] Training skipped or failed for {model_name} (see logs).")
            else:
                print(f"[TRAIN] Training not started for {model_name} (no data or exception before loading).")

    # ----------------------------------------------------------------
    # 4) CREATE CSV FILE FOR NON-TRAINED MODELS
    # ----------------------------------------------------------------
    if non_trained_models:
        non_trained_csv_path = os.path.join(training_data_dir, "non_trained_models.csv")
        df_non_trained = pd.DataFrame(non_trained_models)
        df_non_trained.to_csv(non_trained_csv_path, index=False, encoding='utf-8')
        print(f"[INFO] non_trained_models.csv created at: {non_trained_csv_path}")

    # ----------------------------------------------------------------
    # 5) DISPLAY A FINAL SUMMARY
    # ----------------------------------------------------------------
    print("===== FINAL OVERVIEW =====")
    print(f"Models fully trained: {fully_trained_count}")
    print(f"Models not started: {not_started_count}")
    print(f"Models partially trained: {partial_count}")
    print(f"Models skipped (no positive labels in train or test): {skipped_count}")
    print("================================")

# --------------------------------------------------------------------
# START THE TRAINING PROCESS
# --------------------------------------------------------------------
train_models(annotation_base_dir, model_output_dir, log_output_dir)

# Debugging print to check the end of execution
print("Script ended.")
