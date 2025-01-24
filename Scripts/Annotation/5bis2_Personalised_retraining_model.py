"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
5bis2_Personalised_retraining_model.py

MAIN OBJECTIVE:
---------------
This script allows a user to selectively retrain a single Camembert/Bert model using 
the labeled data. It lists all currently available models (whether fully trained, partially 
trained, or not trained) in the terminal. The user can then:
1) Choose which model they wish to retrain.
2) Specify the number of epochs for retraining.

After the model is selected, this script:
- Deletes the old model folder (if it exists).
- Retrains a fresh model using the same data splits (train/validation).
- Logs progress and saves final model artifacts.

Dependencies:
-------------
- os
- sys
- glob
- shutil
- json
- pandas
- torch
- AugmentedSocialScientist

MAIN FEATURES:
--------------
1) Detect GPU (CUDA/MPS) or default to CPU.
2) Enumerate all potential models from train/validation JSONL files in subdirectories.
3) Prompt user to select one model and epoch count for retraining.
4) Remove old model files/directories.
5) Train the model on the specified number of epochs.
6) Produce a training log and display final messages.

Author:
-------
Antoine Lemor 
"""

import json
import sys
import os
import glob
import shutil
import pandas as pd
import torch

from AugmentedSocialScientist.models import Camembert, Bert

# --------------------------------------------------------------------
# FUNCTION TO GET DEVICE (CUDA, MPS, OR CPU)
# --------------------------------------------------------------------
def get_device():
    """
    Detects if GPU (CUDA or MPS) is available; otherwise falls back to CPU.
    
    Returns
    -------
    torch.device
        Device for computations.
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

# --------------------------------------------------------------------
# LOGGER CLASS FOR LOGGING
# --------------------------------------------------------------------
class Logger(object):
    """
    Handles logging to both console and file.
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
    
    Parameters
    ----------
    filepath : str
        Path to the JSONL file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the loaded data.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# FUNCTION TO DETERMINE REQUIRED FILES PER LANGUAGE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str):
    """
    Returns the list of files needed for the given language.
    - For FR (Camembert), it's: ['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model']
    - For EN (Bert), it's:      ['config.json', 'pytorch_model.bin', 'vocab.txt']
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    else:
        return ["config.json", "pytorch_model.bin", "vocab.txt"]

# --------------------------------------------------------------------
# FUNCTION TO CHECK MODEL FILE COUNT
# --------------------------------------------------------------------
def get_model_file_count(model_dir, language):
    """
    Counts how many required model files are present for a given language.
    
    Parameters
    ----------
    model_dir : str
        Path to the model directory.
    language : str
        The language identifier ('EN' or 'FR').

    Returns
    -------
    int
        Number of required files that are present in the model_dir.
    """
    required_files = get_required_files_for_language(language)
    present_count = 0

    if not os.path.exists(model_dir):
        # The folder doesn't even exist
        return 0

    for file_name in required_files:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            present_count += 1

    return present_count

# --------------------------------------------------------------------
# CUSTOM EXCEPTION FOR SKIPPING TRAINING
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """
    Exception raised when training must be skipped.
    """
    pass

# --------------------------------------------------------------------
# FUNCTION TO LIST ALL POTENTIAL MODELS AND THEIR TRAIN/VAL FILES
# --------------------------------------------------------------------
def list_all_models(annotation_base_dir, model_output_dir):
    """
    Lists all potential models from the annotation base directories 
    (train/validation JSONL files), along with their status (fully_trained, 
    partial, or not_started). 

    Parameters
    ----------
    annotation_base_dir : str
        The base directory containing labeled data subfolders.
    model_output_dir : str
        The directory where models are saved.

    Returns
    -------
    list of dict
        Each dict contains:
        {
            "label": str,
            "language": str ('EN' or 'FR'),
            "train_file": str,
            "test_file": str,
            "model_name": str,
            "model_dir": str,
            "status": str ("fully_trained", "partial", or "not_started")
        }
    """
    models_info = []
    fully_trained_count = 0
    partial_count = 0
    not_started_count = 0

    for label in os.listdir(annotation_base_dir):
        label_path = os.path.join(annotation_base_dir, label)
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

            # Match training files with corresponding validation files
            for train_file in train_files:
                base_name = os.path.basename(train_file).replace('_train_', '_validation_')
                matching_test_file = None
                for test_file in test_files:
                    if base_name in test_file:
                        matching_test_file = test_file
                        break

                if not matching_test_file:
                    continue

                # Determine the model name
                model_name = os.path.basename(train_file).replace('_train_', '_')
                model_dir = os.path.join(model_output_dir, f"{model_name}.model")

                # Check how many required model files exist
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

                models_info.append({
                    "label": label,
                    "language": language,
                    "train_file": train_file,
                    "test_file": matching_test_file,
                    "model_name": model_name,
                    "model_dir": model_dir,
                    "status": status
                })

    # Display summary in the terminal
    print("===== MODEL STATE SUMMARY =====")
    print(f"Fully trained models : {fully_trained_count}")
    print(f"Not started models : {not_started_count}")
    print(f"Partially trained models : {partial_count}")
    print("===============================================")

    return models_info

# --------------------------------------------------------------------
# FUNCTION TO PROMPT USER FOR MODEL SELECTION AND EPOCHS
# --------------------------------------------------------------------
def prompt_user_model_choice(models_info):
    """
    Prompts the user to select a single model to retrain, from a provided list, 
    and also requests the desired number of epochs.

    Parameters
    ----------
    models_info : list of dict
        The list of available models with their training info.

    Returns
    -------
    (dict, int)
        - The dictionary containing the selected model information.
        - The integer specifying the number of epochs to train.
    """
    if not models_info:
        print("[ERROR] No models found. Exiting.")
        sys.exit(1)

    print("\nPlease select a model to retrain by typing its index:")
    for idx, info in enumerate(models_info):
        print(f"[{idx}] {info['model_name']} (Status: {info['status']}, Language: {info['language']})")

    # Validate user index choice
    valid_indices = list(range(len(models_info)))
    while True:
        try:
            user_idx = int(input("\nEnter the index of the model to retrain: "))
            if user_idx in valid_indices:
                break
            else:
                print(f"Invalid selection. Please choose an index between 0 and {len(models_info)-1}.")
        except ValueError:
            print("Invalid input. Please enter a valid index (integer).")

    # Prompt user for epochs
    while True:
        try:
            user_epochs = int(input("Enter the number of epochs for retraining (e.g., 5, 10, 20): "))
            if user_epochs <= 0:
                print("Number of epochs must be positive. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer for epochs.")

    selected_model = models_info[user_idx]
    return selected_model, user_epochs

# --------------------------------------------------------------------
# MAIN FUNCTION TO RETRAIN A SELECTED MODEL
# --------------------------------------------------------------------
def main():
    """
    Main entry point for selective retraining:
    1) Gathers a list of potential models from annotation subfolders.
    2) Prompts the user to choose which model to retrain and how many epochs.
    3) Deletes the old model directory (if exists).
    4) Loads data (train/validation).
    5) Retrains and saves the new model.
    6) Produces a log file and final summary.
    """
    print("Script started.")  # Debugging print in English

    # ----------------------------------------------------------------
    # 0) PATH SETUP
    # ----------------------------------------------------------------
    base_path = os.path.dirname(os.path.abspath(__file__))

    annotation_base_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotation_bases")
    model_output_dir = os.path.join(base_path, "..", "..", "models")  # Points directly to 'models'
    log_output_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "Annotation_logs")

    # Ensure directories exist
    if not os.path.exists(log_output_dir):
        os.makedirs(log_output_dir)

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # ----------------------------------------------------------------
    # 1) LIST ALL MODELS
    # ----------------------------------------------------------------
    models_info = list_all_models(annotation_base_dir, model_output_dir)

    # ----------------------------------------------------------------
    # 2) PROMPT USER TO SELECT A MODEL AND EPOCHS
    # ----------------------------------------------------------------
    selected_model, user_epochs = prompt_user_model_choice(models_info)
    label = selected_model["label"]
    language = selected_model["language"]
    train_file = selected_model["train_file"]
    test_file = selected_model["test_file"]
    model_name = selected_model["model_name"]
    model_dir = selected_model["model_dir"]
    status = selected_model["status"]

    print(f"\n[INFO] You have chosen to retrain the model: {model_name}")
    print(f"Label: {label}, Language: {language}, Current Status: {status}")
    print(f"Number of epochs chosen: {user_epochs}\n")

    # ----------------------------------------------------------------
    # 3) REMOVE OLD MODEL FOLDER IF IT EXISTS
    # ----------------------------------------------------------------
    if os.path.exists(model_dir):
        print(f"[INFO] Removing old model directory: {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)

    # ----------------------------------------------------------------
    # 4) DEVICE SETUP
    # ----------------------------------------------------------------
    device = get_device()

    # ----------------------------------------------------------------
    # 5) INSTANTIATE THE MODEL
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 6) SETUP LOG
    # ----------------------------------------------------------------
    log_filepath = os.path.join(log_output_dir, f"{model_name}_personalised_retraining_log.txt")
    print(f"Setting up logging to: {log_filepath}")
    logger = Logger(log_filepath)
    sys.stdout = logger
    print(f"[LOG] Personalised Retraining started for {label} in {language}")

    train_label_counts = None
    test_label_counts = None
    scores = None

    # ----------------------------------------------------------------
    # 7) LOAD AND CHECK DATA
    # ----------------------------------------------------------------
    try:
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

        # Check if there are positive values in the labels
        train_has_positive = (train_data['label'] > 0).any()
        test_has_positive = (test_data['label'] > 0).any()

        if not train_has_positive or not test_has_positive:
            print(f"[SKIP] Training or validation data for {label} in {language} contain only 0s. Cannot retrain.")
            raise SkipTrainingException("Data contain only 0s.")

        # Check if there are enough annotations
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

        # ----------------------------------------------------------------
        # 8) TRAINING PROCESS
        # ----------------------------------------------------------------
        relative_model_output_path = f"{model_name}.model"
        print(f"[INFO] Retraining {model_name} for {user_epochs} epoch(s).")
        print(f"Will save model to (relative path): {relative_model_output_path}")

        scores = model.run_training(
            train_loader,
            test_loader,
            lr=5e-5,
            n_epochs=user_epochs,
            random_state=42,
            save_model_as=relative_model_output_path
        )
        print(f"Retraining completed successfully for {model_name} with user-defined epochs: {user_epochs}")

        # Move the newly saved model artifacts into the final model directory
        final_path = os.path.join(os.path.dirname(model_dir), os.path.basename(relative_model_output_path))
        if os.path.exists(relative_model_output_path):
            print(f"[INFO] Moving {relative_model_output_path} -> {final_path}")
            shutil.move(relative_model_output_path, final_path)

    except SkipTrainingException as ste:
        print(f"[INFO] {ste}")
        print("[INFO] Personalised retraining aborted due to insufficient label distribution.")
    except Exception as e:
        print(f"Error during retraining for {model_name}: {e}")
        # If training fails, we can remove partial files
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
            print(f"Directory removed for the partially trained model: {model_dir}")

    finally:
        # Close the logger and restore stdout
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"[LOG] Logging closed for {label} in {language}")

    if scores is not None:
        print(f"[TRAIN] Retraining completed for {model_name}, scores: {scores}")
    else:
        print(f"[TRAIN] Retraining was skipped or failed for {model_name} (see logs).")

    print("Script ended.")  # Debugging print in English

# --------------------------------------------------------------------
# LAUNCH
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
