"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
6_Training_best_models.py

MAIN OBJECTIVE:
---------------
This script automates the end-to-end training pipeline for a large collection
of English (BERT) and French (CamemBERT) text-classification models.  
It (i) scans existing model folders or a CSV file, or builds fresh models from
raw JSONL data; (ii) trains each model with a rigorous early-stopping / best-epoch
selection strategy; (iii) triggers a reinforced-training phase when the F1-score
on the positive class is unsatisfactory; and (iv) logs exhaustive performance
metrics while safely archiving the final best checkpoints.

DEPENDENCIES:
-------------
- csv, json, os, sys, glob, shutil, datetime, time, random
- collections.defaultdict
- numpy, pandas, torch, tqdm
- scikit-learn (classification_report, precision_recall_fscore_support)
- transformers (Bert, Camembert, AdamW, get_linear_schedule_with_warmup)
- scipy (softmax)

MAIN FEATURES:
--------------
1) Flexible model selection  
   - Option 1 : retrain all models found in `./models/*.model`  
   - Option 2 : retrain models listed in an external CSV summary  
   - Option 3 : start from scratch by discovering every train/validation pair
     in `annotation_bases/*/train|validation/*`

2) Language-aware architecture  
   Automatically routes French files to CamemBERT and English files to BERT,
   preserving identical training logic across languages.

3) Robust training loop (`run_training`)  
   - Smart GPU/MPS/CPU detection  
   - Weighted cross-entropy support for class imbalance  
   - Best-epoch pick based on a combined metric
     (macro-F1, weighted class-1 F1)  
   - Seamless checkpointing and CSV-based metric tracking

4) Reinforced training phase  
   If the positive-class F1 < 0.70:  
   - Oversamples minority class via `WeightedRandomSampler`  
   - Lowers learning rate, increases batch size and class weights  
   - Includes a *rescue* override: any improvement from F1 = 0 triggers a save  
   - Safety net forces a checkpoint at the 5ᵉ epoch if F1 remains 0

5) Comprehensive logging  
   Writes per-epoch metrics, reinforced metrics, and consolidated best-model
   summaries to `Database/Training_data/Retraining_logs/…` and appends the final
   results to `Database/Training_data/all_best_models.csv`.

6) Utility helpers  
   - Tokenisation & padding capped at 512 tokens  
   - Probability or logit prediction helpers  
   - Time formatter for console outputs

Author:
-------
[Anonymous]
"""

import os
import sys
import csv
import json
import shutil
import datetime
import random
import time
import glob

import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Any, Dict, Optional
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    CamembertForSequenceClassification,
    CamembertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    CONFIG_NAME
)
from scipy.special import softmax


###############################################################################
# BERTBase Class
###############################################################################
class BertBase:
    """
    General class to train and evaluate a BERT (or similar) model. It handles:
      - Data encoding and tokenization
      - Training loop (run_training) with best-model selection
      - Reinforced training if class-1 F1 < 0.6
      - Rescue logic to allow model saving if class-1 F1 was 0 but slightly improves
      - Prediction methods and model loading
    """

    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        tokenizer: Any = BertTokenizer,
        model_sequence_classifier: Any = BertForSequenceClassification,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the BertBase object.

        Parameters
        ----------
        model_name : str, optional
            A HuggingFace model name (e.g. 'bert-base-cased', 'bert-base-uncased'),
            by default 'bert-base-cased'.
        tokenizer : Any, optional
            Tokenizer class to use, by default BertTokenizer.
        model_sequence_classifier : Any, optional
            The sequence classification model class (from transformers),
            by default BertForSequenceClassification.
        device : torch.device or None, optional
            The torch device to use. If None, auto-detect CUDA/MPS/CPU.
        """
        self.model_name = model_name
        self.tokenizer = tokenizer.from_pretrained(self.model_name)
        self.model_sequence_classifier = model_sequence_classifier
        self.dict_labels = None

        # Automatic device detection if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print('→ GPU CUDA is available:',
                      torch.cuda.get_device_name(torch.cuda.current_device()))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print('→ Apple Silicon (MPS) GPU is available!')
            else:
                self.device = torch.device("cpu")
                print('→ No GPU available. Using CPU.')
        else:
            self.device = device

    def encode(
        self,
        sequences: List[str],
        labels: Optional[List[str]] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        add_special_tokens: bool = True
    ) -> DataLoader:
        """
        Tokenize and prepare a PyTorch DataLoader.

        Parameters
        ----------
        sequences : list of str
            The textual samples to tokenize.
        labels : list of str or None, optional
            Labels (0/1 or text-based) for training. If None, it returns a
            DataLoader for inference/prediction only.
        batch_size : int, optional
            The batch size, by default 32.
        progress_bar : bool, optional
            Whether to display a progress bar during tokenization, by default True.
        add_special_tokens : bool, optional
            Whether to add special tokens ([CLS], [SEP]), by default True.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader ready for training or prediction.
        """
        input_ids = []
        if progress_bar:
            sent_loader = tqdm(sequences, desc="Tokenizing")
        else:
            sent_loader = sequences

        # Tokenization loop
        for sent in sent_loader:
            encoded_sent = self.tokenizer.encode(
                sent,
                add_special_tokens=add_special_tokens
            )
            input_ids.append(encoded_sent)

        # Determine maximum sequence length, capped at 512
        max_len = min(max(len(sen) for sen in input_ids), 512)

        # Pad/truncate to max_len
        pad = np.full((len(input_ids), max_len), 0, dtype='long')
        for idx, s in enumerate(input_ids):
            trunc = s[:max_len]
            pad[idx, :len(trunc)] = trunc
        input_ids = pad

        # Build attention masks
        attention_masks = []
        if progress_bar:
            input_loader = tqdm(input_ids, desc="Creating attention masks")
        else:
            input_loader = input_ids

        for sent in input_loader:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # If no labels provided, create a DataLoader for inference
        if labels is None:
            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)
            data = TensorDataset(inputs_tensors, masks_tensors)
            sampler = SequentialSampler(data)
            return DataLoader(data, sampler=sampler, batch_size=batch_size)
        else:
            # Build dict_labels if needed
            label_names = np.unique(labels)
            self.dict_labels = dict(zip(label_names, range(len(label_names))))
            if progress_bar:
                print(f"label ids: {self.dict_labels}")

            inputs_tensors = torch.tensor(input_ids)
            masks_tensors = torch.tensor(attention_masks)
            labels_tensors = torch.tensor([self.dict_labels[x] for x in labels])

            data = TensorDataset(inputs_tensors, masks_tensors, labels_tensors)
            sampler = SequentialSampler(data)
            return DataLoader(data, sampler=sampler, batch_size=batch_size)

    def run_training(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        n_epochs: int = 3,
        lr: float = 5e-5,
        random_state: int = 42,
        save_model_as: Optional[str] = None,
        pos_weight: Optional[torch.Tensor] = None,
        metrics_output_dir: str = "./training_logs",
        best_model_criteria: str = "combined",
        f1_class_1_weight: float = 0.7,
        reinforced_learning: bool = True,
        n_epochs_reinforced: int = 2,
        rescue_low_class1_f1: bool = False,
        f1_1_rescue_threshold: float = 0.0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]]:
        """
        Train and evaluate the model, automatically saving the best model,
        and optionally perform reinforced training if F1 for class 1 is < 0.6.

        Additionally, a "rescue" mechanism can be enabled to allow saving a
        reinforced model if class-1 F1 was extremely low (e.g., 0) and any
        small improvement is achieved.

        Parameters
        ----------
        train_dataloader : DataLoader
            Training DataLoader.
        test_dataloader : DataLoader
            Validation DataLoader.
        n_epochs : int, optional
            Number of epochs for the standard training, by default 3.
        lr : float, optional
            Learning rate, by default 5e-5.
        random_state : int, optional
            Seed for reproducibility, by default 42.
        save_model_as : str or None, optional
            Base name under which the best model is saved in ./models/<save_model_as>.
            If None, the model is not saved to disk.
        pos_weight : torch.Tensor or None, optional
            Class weighting to handle class imbalance, by default None.
        metrics_output_dir : str, optional
            Path where CSV logs (training_metrics.csv, best_models.csv, etc.) are saved,
            by default "./training_logs".
        best_model_criteria : str, optional
            Criterion for choosing the best model. "combined" uses a weighted average
            of class-1 F1 and macro-F1, by default "combined".
        f1_class_1_weight : float, optional
            Weight of class-1 F1 in the combined metric, by default 0.7.
        reinforced_learning : bool, optional
            If True, triggers a second training round if class-1 F1 < 0.6, by default True.
        n_epochs_reinforced : int, optional
            Number of epochs for the reinforced training phase, by default 2.
        rescue_low_class1_f1 : bool, optional
            If True, then if the best normal-training F1 for class 1 is 0, any
            improvement above f1_1_rescue_threshold in reinforced learning
            will be considered a new best model, by default False.
        f1_1_rescue_threshold : float, optional
            Threshold above which a class-1 F1 improvement from 0 is considered
            sufficient to override the standard combined metric, by default 0.0.

        Returns
        -------
        (precision, recall, f1, support) : tuple of np.ndarray
            The scores from precision_recall_fscore_support for the final best model.
        final_info : dict
            A dictionary containing final best model metrics to store globally, including:
            {
              "epoch", "train_loss", "val_loss",
              "precision_0", "recall_0", "f1_0", "support_0",
              "precision_1", "recall_1", "f1_1", "support_1",
              "macro_f1", "saved_model_path", "training_phase"
            }
            for the final chosen model (normal or reinforced).
        """
        os.makedirs(metrics_output_dir, exist_ok=True)
        training_metrics_csv = os.path.join(metrics_output_dir, "training_metrics.csv")
        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Overwrite or create the training_metrics.csv file
        with open(training_metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1"
            ])

        # Overwrite or create the best_models.csv file
        with open(best_models_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1",
                "saved_model_path",
                "training_phase"
            ])

        # Collect all the labels from the test set (for classification metrics)
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()
        num_labels = np.unique(test_labels).size

        if self.dict_labels is None:
            label_names = None
        else:
            # Sort label names by their index
            label_names = [str(x[0]) for x in sorted(self.dict_labels.items(), key=lambda x: x[1])]

        # Set seeds
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Initialize model
        model = self.model_sequence_classifier.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * n_epochs
        )

        best_metric_val = -1.0
        best_model_path = None
        best_scores = None

        # For final metrics logging
        final_info = {
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "precision_0": None,
            "recall_0": None,
            "f1_0": None,
            "support_0": None,
            "precision_1": None,
            "recall_1": None,
            "f1_1": None,
            "support_1": None,
            "macro_f1": None,
            "saved_model_path": None,
            "training_phase": "normal"
        }

        ################################################################
        # NORMAL TRAINING
        ################################################################
        for epoch in range(n_epochs):
            print(f"\n======== Epoch {epoch + 1}/{n_epochs} ========")
            print("Training...")

            t0 = time.time()
            total_train_loss = 0.0
            model.train()

            for step, train_batch in enumerate(train_dataloader):
                if step % 40 == 0 and step != 0:
                    elapsed = self.format_time(time.time() - t0)
                    print(f"  Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.")

                b_inputs = train_batch[0].to(self.device)
                b_masks = train_batch[1].to(self.device)
                b_labels = train_batch[2].to(self.device)

                model.zero_grad()
                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                logits = outputs[0]

                if pos_weight is not None:
                    # Weighted cross-entropy for class imbalance
                    weight_tensor = torch.tensor([1.0, pos_weight.item()], device=self.device)
                    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                else:
                    criterion = torch.nn.CrossEntropyLoss()

                loss = criterion(logits, b_labels)
                total_train_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"  Average training loss: {avg_train_loss:.2f} (Epoch {epoch + 1})")
            print("  Training epoch took: ", self.format_time(time.time() - t0))

            # ======================= Validation =======================
            print("\nRunning Validation...")
            t0 = time.time()
            model.eval()

            total_val_loss = 0.0
            logits_complete = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(
                        b_inputs,
                        token_type_ids=None,
                        attention_mask=b_masks,
                        labels=b_labels
                    )

                total_val_loss += outputs.loss.item()
                logits_complete.append(outputs.logits.detach().cpu().numpy())

            logits_complete = np.concatenate(logits_complete, axis=0)
            avg_val_loss = total_val_loss / len(test_dataloader)
            print(f"  Validation loss: {avg_val_loss:.2f}")
            print("  Validation took: ", self.format_time(time.time() - t0))

            preds = np.argmax(logits_complete, axis=1).flatten()
            report = classification_report(test_labels, preds, target_names=label_names, output_dict=True)

            class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            macro_avg = report.get("macro avg", {"f1-score": 0})

            precision_0 = class_0_metrics["precision"]
            recall_0 = class_0_metrics["recall"]
            f1_0 = class_0_metrics["f1-score"]
            support_0 = class_0_metrics["support"]

            precision_1 = class_1_metrics["precision"]
            recall_1 = class_1_metrics["recall"]
            f1_1 = class_1_metrics["f1-score"]
            support_1 = class_1_metrics["support"]

            macro_f1 = macro_avg["f1-score"]
            print(classification_report(test_labels, preds, target_names=label_names))

            # Save metrics for this epoch
            with open(training_metrics_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ])

            # ───────────── Metric used to pick the "best" epoch ───────────── #
            if best_model_criteria == "class1":
                combined_metric = f1_1
            elif best_model_criteria == "combined":
                combined_metric = (
                    f1_class_1_weight * f1_1
                    + (1.0 - f1_class_1_weight) * macro_f1
                )
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            # Check if this is a new best
            if combined_metric > best_metric_val + 1e-6:
                print(f"→ New best model at epoch {epoch + 1} with metric={combined_metric:.4f}")
                # Remove old best model directory if exists
                if best_model_path is not None and os.path.isdir(best_model_path):
                    shutil.rmtree(best_model_path, ignore_errors=True)

                best_metric_val = combined_metric
                if save_model_as is not None:
                    best_model_path = os.path.join("models", f"{save_model_as}_epoch_{epoch+1}")
                    os.makedirs(best_model_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model

                    output_model_file = os.path.join(best_model_path, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path)
                else:
                    best_model_path = None

                best_scores = precision_recall_fscore_support(test_labels, preds)

                # Log best model into best_models.csv
                with open(best_models_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        precision_0,
                        recall_0,
                        f1_0,
                        support_0,
                        precision_1,
                        recall_1,
                        f1_1,
                        support_1,
                        macro_f1,
                        best_model_path if best_model_path else "Not saved to disk",
                        "normal"
                    ])

                # Store final info (candidate for best final)
                final_info = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "precision_0": precision_0,
                    "recall_0": recall_0,
                    "f1_0": f1_0,
                    "support_0": support_0,
                    "precision_1": precision_1,
                    "recall_1": recall_1,
                    "f1_1": f1_1,
                    "support_1": support_1,
                    "macro_f1": macro_f1,
                    "saved_model_path": best_model_path if best_model_path else "Not saved to disk",
                    "training_phase": "normal"
                }

        print("\nNormal training phase completed!")
        final_path = None
        if save_model_as and best_model_path is not None:
            final_path = os.path.join("models", save_model_as)
            # Clean up any existing directory
            if os.path.exists(final_path):
                shutil.rmtree(final_path, ignore_errors=True)
            os.rename(best_model_path, final_path)
            best_model_path = final_path
            print(f"→ Final best model (normal) saved to: {best_model_path}")
            final_info["saved_model_path"] = best_model_path

        ################################################################
        # REINFORCED TRAINING (if needed)
        ################################################################
        if best_scores is not None:
            best_f1_1 = best_scores[2][1]  # indices: precision=0, recall=1, f1=2, support=3
            if best_f1_1 < 0.7 and reinforced_learning:
                print(f"\nClass-1 F1 ({best_f1_1:.3f}) < 0.70 → starting reinforced training.")

                # ----------------------------------------------------------
                # CHECK IF BEST NORMAL EPOCH IS 1 WITH F1_1=0
                # If yes, we pass a flag to forcibly save epoch 5 in case
                # F1_1 never improves above 0 in reinforced training.
                # ----------------------------------------------------------
                best_normal_epoch = final_info["epoch"]
                best_normal_f1_1 = final_info["f1_1"]
                force_save_5th_if_no_progress = (
                    best_normal_epoch == 1 and
                    abs(best_normal_f1_1) < 1e-9  # i.e., best_normal_f1_1 == 0
                )

                new_best_val, new_best_path, new_best_scores, new_info = self.reinforced_training(
                    train_dataloader,
                    test_dataloader,
                    base_model_path=best_model_path,
                    random_state=random_state,
                    metrics_output_dir=metrics_output_dir,
                    save_model_as=save_model_as,
                    best_model_criteria=best_model_criteria,
                    f1_class_1_weight=f1_class_1_weight,
                    previous_best_metric=best_metric_val,
                    n_epochs_reinforced=n_epochs_reinforced,
                    rescue_low_class1_f1=rescue_low_class1_f1,
                    f1_1_rescue_threshold=f1_1_rescue_threshold,
                    force_save_5th_reinforced_if_no_progress=force_save_5th_if_no_progress,
                    best_normal_f1_1=best_normal_f1_1
                )

                # If the reinforced training yields a better metric, update final info
                if new_best_val > best_metric_val:
                    best_metric_val = new_best_val
                    best_model_path = new_best_path
                    best_scores = new_best_scores
                    final_info = new_info
                else:
                    print("Reinforced training did not improve metrics.")

                # If forced-save occurred or a new path was used
                if (new_info is not None) and (new_info.get("training_phase", "").startswith("reinforced")):
                    is_forced = (new_info.get("training_phase") == "reinforced_5_forced")
                    path_changed = (new_info.get("saved_model_path") != final_info.get("saved_model_path"))

                    if is_forced or path_changed:
                        print("Adopting forced-save model from reinforced training.")
                        final_info = new_info
                        best_model_path = new_best_path
                        best_scores = new_best_scores if new_best_scores is not None else best_scores
                        best_metric_val = new_best_val
            else:
                print("No reinforced training (class-1 F1 >= 0.7).")
        else:
            print("No best_scores found after normal training. Skipping reinforced phase.")

        return best_scores, final_info

    def reinforced_training(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        base_model_path: Optional[str],
        random_state: int = 42,
        metrics_output_dir: str = "./training_logs",
        save_model_as: Optional[str] = None,
        best_model_criteria: str = "combined",
        f1_class_1_weight: float = 0.7,
        previous_best_metric: float = -1.0,
        n_epochs_reinforced: int = 2,
        rescue_low_class1_f1: bool = False,
        f1_1_rescue_threshold: float = 0.0,
        force_save_5th_reinforced_if_no_progress: bool = False,
        best_normal_f1_1: float = 0.0
    ) -> Tuple[float, Optional[str],
               Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
               Dict[str, Any]]:
        """
        Reinforced training phase: e.g., oversampling for the minority class,
        reduced LR, heavier class weighting, etc.

        Includes a "rescue" mechanism: if the best normal-training F1 for class 1
        was 0, and any epoch's class-1 F1 surpasses f1_1_rescue_threshold,
        we consider that a new best even if the combined metric isn't higher.

        NEW LOGIC:
        ----------
        1. If `force_save_5th_reinforced_if_no_progress` is True and normal best epoch was 1
           with F1_1=0, we will forcibly save the 5th epoch model if we never see F1_1 > 0.
        2. This ensures we don't lose all checkpoints if class 1 is never predicted.

        Parameters
        ----------
        train_dataloader : DataLoader
            The original training DataLoader.
        test_dataloader : DataLoader
            The validation DataLoader.
        base_model_path : str or None
            Path to the best model from the normal phase. If None, uses a fresh model.
        random_state : int, optional
            Random seed, by default 42.
        metrics_output_dir : str, optional
            Path where CSV logs are saved, by default "./training_logs".
        save_model_as : str or None, optional
            Base name for saving the best reinforced model, by default None.
        best_model_criteria : str, optional
            Criterion for best-model selection, by default "combined".
        f1_class_1_weight : float, optional
            Weight for the class-1 F1 in the combined metric, by default 0.7.
        previous_best_metric : float, optional
            The best metric value so far (from normal training), by default -1.0.
        n_epochs_reinforced : int, optional
            Number of epochs for the reinforced training, by default 2.
        rescue_low_class1_f1 : bool, optional
            If True, then if normal training had class-1 F1=0, any RL epoch with
            class-1 F1 > f1_1_rescue_threshold is considered an improvement.
        f1_1_rescue_threshold : float, optional
            If normal training F1_1=0, any new F1_1 above this threshold triggers
            a "rescue" override.
        force_save_5th_reinforced_if_no_progress : bool, optional
            If True, forcibly save the 5th epoch of reinforced if F1_1 never improved.
        best_normal_f1_1 : float, optional
            Best class-1 F1 from normal training phase, by default 0.0.

        Returns
        -------
        best_metric_val : float
            The new best metric value after reinforced training.
        best_model_path_local : str or None
            Path to the final best reinforced model.
        best_scores : tuple or None
            (precision, recall, f1, support) for the final best model after reinforcement.
        final_info : dict
            Dictionary of final best model metrics from the reinforced phase if improved,
            or from the prior phase if no improvement occurred.
        """
        print("=== Reinforced Training Phase ===")
        reinforced_metrics_csv = os.path.join(metrics_output_dir, "reinforced_training_metrics.csv")
        with open(reinforced_metrics_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1"
            ])

        best_models_csv = os.path.join(metrics_output_dir, "best_models.csv")

        # Retrieve the dataset and labels
        dataset = train_dataloader.dataset
        labels = dataset.tensors[2].numpy()
        class_sample_count = np.bincount(labels)
        weight_per_class = 1.0 / class_sample_count
        sample_weights = [weight_per_class[t] for t in labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        # Adjust hyperparameters for reinforced training
        new_batch_size = 64
        new_train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=new_batch_size)
        new_lr = 5e-6
        pos_weight_val = 3.5  # heavier weighting for class 1
        weight_tensor = torch.tensor([1.0, pos_weight_val], dtype=torch.float)

        # Seeds
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        # Load the base model
        if base_model_path:
            model = self.model_sequence_classifier.from_pretrained(base_model_path)
            print(f"Base model loaded from: {base_model_path}")
        else:
            model = self.model_sequence_classifier.from_pretrained(
                self.model_name,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
            print("No base_model_path provided. Using a fresh model.")
        model.to(self.device)

        optimizer = AdamW(model.parameters(), lr=new_lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(new_train_dataloader) * n_epochs_reinforced
        )

        best_metric_val = previous_best_metric
        best_model_path_local = base_model_path
        best_scores = None

        # Prepare final info structure
        final_info = {
            "epoch": None,
            "train_loss": None,
            "val_loss": None,
            "precision_0": None,
            "recall_0": None,
            "f1_0": None,
            "support_0": None,
            "precision_1": None,
            "recall_1": None,
            "f1_1": None,
            "support_1": None,
            "macro_f1": None,
            "saved_model_path": base_model_path if base_model_path else "Not saved to disk",
            "training_phase": "reinforced"
        }

        # Extract test labels
        test_labels = []
        for batch in test_dataloader:
            test_labels += batch[2].numpy().tolist()

        # Track whether we ever see an F1_1 > 0
        f1_1_ever_above_zero = False
        forced_stats = None
        ephemeral_epoch_5_path = None

        # Start reinforced training loop
        for epoch in range(n_epochs_reinforced):
            print(f"\n=== Reinforced: Epoch {epoch + 1}/{n_epochs_reinforced} ===")
            t0 = time.time()
            model.train()
            running_loss = 0.0
            criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(self.device))

            for step, train_batch in enumerate(new_train_dataloader):
                b_inputs = train_batch[0].to(self.device)
                b_masks = train_batch[1].to(self.device)
                b_labels = train_batch[2].to(self.device)

                model.zero_grad()
                outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks)
                logits = outputs[0]

                loss = criterion(logits, b_labels)
                running_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = running_loss / len(new_train_dataloader)
            print(f"  [Reinforced] Training loss: {avg_train_loss:.4f}, Elapsed: {self.format_time(time.time() - t0)}")

            # Validation
            model.eval()
            total_val_loss = 0.0
            logits_complete = []

            for test_batch in test_dataloader:
                b_inputs = test_batch[0].to(self.device)
                b_masks = test_batch[1].to(self.device)
                b_labels = test_batch[2].to(self.device)

                with torch.no_grad():
                    outputs = model(b_inputs, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                total_val_loss += outputs.loss.item()
                logits_complete.append(outputs.logits.detach().cpu().numpy())

            avg_val_loss = total_val_loss / len(test_dataloader)
            logits_complete = np.concatenate(logits_complete, axis=0)
            val_preds = np.argmax(logits_complete, axis=1).flatten()

            report = classification_report(test_labels, val_preds, output_dict=True)
            class_0_metrics = report.get("0", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            class_1_metrics = report.get("1", {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            macro_avg = report.get("macro avg", {"f1-score": 0})

            precision_0 = class_0_metrics["precision"]
            recall_0 = class_0_metrics["recall"]
            f1_0 = class_0_metrics["f1-score"]
            support_0 = class_0_metrics["support"]

            precision_1 = class_1_metrics["precision"]
            recall_1 = class_1_metrics["recall"]
            f1_1 = class_1_metrics["f1-score"]
            support_1 = class_1_metrics["support"]

            macro_f1 = macro_avg["f1-score"]
            print(classification_report(test_labels, val_preds))

            # Update flag if F1_1 is above 0
            if f1_1 > 0:
                f1_1_ever_above_zero = True

            # Log reinforced metrics
            with open(reinforced_metrics_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    avg_train_loss,
                    avg_val_loss,
                    precision_0,
                    recall_0,
                    f1_0,
                    support_0,
                    precision_1,
                    recall_1,
                    f1_1,
                    support_1,
                    macro_f1
                ])

            # ───────────── Metric used to pick the "best" epoch ───────────── #
            if best_model_criteria == "class1":
                combined_metric = f1_1
            elif best_model_criteria == "combined":
                combined_metric = (
                    f1_class_1_weight * f1_1
                    + (1.0 - f1_class_1_weight) * macro_f1
                )
            else:
                combined_metric = (f1_1 + macro_f1) / 2.0

            # "Rescue" logic: if normal F1_1 was 0, any new F1_1 > threshold is an improvement
            rescue_override = False
            if rescue_low_class1_f1 and best_normal_f1_1 == 0.0:
                if f1_1 > f1_1_rescue_threshold:
                    rescue_override = True

            if rescue_override:
                new_metric_val = combined_metric + 9999.0
                print(f"[Rescue Logic Triggered] Class-1 F1 moved from 0.0 to {f1_1:.4f}, "
                      f"exceeding threshold {f1_1_rescue_threshold:.4f}")
            else:
                new_metric_val = combined_metric

            # -------------------------------------------------------
            # Ephemeral save for 5th epoch if still no progress
            # -------------------------------------------------------
            if (
                force_save_5th_reinforced_if_no_progress
                and (epoch + 1 == 5)
                and (not f1_1_ever_above_zero)
            ):
                forced_stats = {
                    "train_loss":  avg_train_loss,
                    "val_loss":    avg_val_loss,
                    "precision_0": precision_0,
                    "recall_0":    recall_0,
                    "f1_0":        f1_0,
                    "support_0":   support_0,
                    "precision_1": precision_1,
                    "recall_1":    recall_1,
                    "f1_1":        f1_1,
                    "support_1":   support_1,
                    "macro_f1":    macro_f1
                }

                ephemeral_epoch_5_path = os.path.join(
                    "models", f"{save_model_as}_reinforced_epoch_5_no_progress_safety"
                )
                print(
                    f"[INFO] Creating ephemeral checkpoint at epoch 5 (f1_1 still 0): "
                    f"{ephemeral_epoch_5_path}"
                )
                if os.path.isdir(ephemeral_epoch_5_path):
                    shutil.rmtree(ephemeral_epoch_5_path, ignore_errors=True)
                os.makedirs(ephemeral_epoch_5_path, exist_ok=True)

                model_to_save = model.module if hasattr(model, "module") else model
                output_model_file  = os.path.join(ephemeral_epoch_5_path, WEIGHTS_NAME)
                output_config_file = os.path.join(ephemeral_epoch_5_path, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                self.tokenizer.save_vocabulary(ephemeral_epoch_5_path)

            # Check if we have a new best model
            if new_metric_val > best_metric_val:
                print(f"→ New best (reinforced) at epoch {epoch + 1}, metric={combined_metric:.4f}")
                if best_model_path_local and os.path.isdir(best_model_path_local):
                    shutil.rmtree(best_model_path_local, ignore_errors=True)

                best_metric_val = new_metric_val
                if save_model_as is not None:
                    best_model_path_local = os.path.join("models", f"{save_model_as}_reinforced_epoch_{epoch+1}")
                    os.makedirs(best_model_path_local, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model

                    output_model_file = os.path.join(best_model_path_local, WEIGHTS_NAME)
                    output_config_file = os.path.join(best_model_path_local, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    self.tokenizer.save_vocabulary(best_model_path_local)
                else:
                    best_model_path_local = None

                with open(best_models_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        avg_train_loss,
                        avg_val_loss,
                        precision_0,
                        recall_0,
                        f1_0,
                        support_0,
                        precision_1,
                        recall_1,
                        f1_1,
                        support_1,
                        macro_f1,
                        best_model_path_local if best_model_path_local else "Not saved to disk",
                        "reinforced"
                    ])

                best_scores = precision_recall_fscore_support(test_labels, val_preds)
                final_info = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "precision_0": precision_0,
                    "recall_0": recall_0,
                    "f1_0": f1_0,
                    "support_0": support_0,
                    "precision_1": precision_1,
                    "recall_1": recall_1,
                    "f1_1": f1_1,
                    "support_1": support_1,
                    "macro_f1": macro_f1,
                    "saved_model_path": best_model_path_local if best_model_path_local else "Not saved to disk",
                    "training_phase": "reinforced"
                }

        # End of reinforced training
        # --------------------------------------------------------------
        # FORCED SAVE OF 5TH EPOCH IF F1_1 NEVER IMPROVED ABOVE 0
        # --------------------------------------------------------------
        if force_save_5th_reinforced_if_no_progress and not f1_1_ever_above_zero:
            # We forcibly save the 5th epoch ONLY IF we actually created that ephemeral checkpoint
            # and no other improvement has been seen.
            if ephemeral_epoch_5_path and os.path.isdir(ephemeral_epoch_5_path):
                print("[FORCED SAVE] F1_1 never rose above 0 throughout reinforced training.")
                print("→ Forcibly saving the model from the 5th epoch of reinforced.")

                final_path = None
                if save_model_as is not None:
                    final_path = os.path.join("models", save_model_as)
                    if os.path.exists(final_path):
                        shutil.rmtree(final_path, ignore_errors=True)
                    if os.path.abspath(ephemeral_epoch_5_path) != os.path.abspath(final_path):
                        os.rename(ephemeral_epoch_5_path, final_path)
                    best_model_path_local = final_path
                else:
                    best_model_path_local = ephemeral_epoch_5_path

                stats = forced_stats or {}
                final_info = {
                    "epoch":       5,
                    "train_loss":  stats.get("train_loss"),
                    "val_loss":    stats.get("val_loss"),
                    "precision_0": stats.get("precision_0"),
                    "recall_0":    stats.get("recall_0"),
                    "f1_0":        stats.get("f1_0"),
                    "support_0":   stats.get("support_0"),
                    "precision_1": stats.get("precision_1", 0.0),
                    "recall_1":    stats.get("recall_1"),
                    "f1_1":        stats.get("f1_1", 0.0),
                    "support_1":   stats.get("support_1"),
                    "macro_f1":    stats.get("macro_f1"),
                    "saved_model_path": best_model_path_local if best_model_path_local else "Not saved to disk",
                    "training_phase": "reinforced_5_forced"
                }
                # We do not modify the best_metric_val here if F1_1=0.

            else:
                print("[INFO] Did not reach epoch 5 or no checkpoint found – no forced save.")

        # If a genuinely better model was found
        if best_model_path_local and (best_model_path_local != base_model_path):
            # If we have a new best reinforced path
            if save_model_as is not None and os.path.isdir(best_model_path_local):
                final_path = os.path.join("models", save_model_as)
                if os.path.exists(final_path):
                    shutil.rmtree(final_path, ignore_errors=True)
                if os.path.abspath(best_model_path_local) != os.path.abspath(final_path):
                    os.rename(best_model_path_local, final_path)
                best_model_path_local = final_path
                print(f"→ Final best (reinforced) model saved to: {best_model_path_local}")
                final_info["saved_model_path"] = best_model_path_local

        print("Reinforced training completed.\n")
        return best_metric_val, best_model_path_local, best_scores, final_info

    def predict(
        self,
        dataloader: DataLoader,
        model: Any,
        proba: bool = True,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Generate predictions (logits or probability distributions) using a trained model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader with the input data to predict on.
        model : nn.Module
            A loaded PyTorch model to use for inference.
        proba : bool, optional
            If True, returns probabilities (via softmax). If False, returns raw logits.
            By default True.
        progress_bar : bool, optional
            Whether to display a progress bar, by default True.

        Returns
        -------
        np.ndarray
            Numpy array of shape (n_samples, n_classes), containing probabilities or logits.
        """
        logits_complete = []
        if progress_bar:
            loader = tqdm(dataloader, desc="Predicting")
        else:
            loader = dataloader

        model.eval()
        for batch in loader:
            batch = tuple(t.to(self.device) for t in batch)
            if len(batch) == 3:
                b_input_ids, b_input_mask, _ = batch
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0].detach().cpu().numpy()
            logits_complete.append(logits)

            del outputs
            torch.cuda.empty_cache()

        pred = np.concatenate(logits_complete, axis=0)
        return softmax(pred, axis=1) if proba else pred

    def load_model(self, model_path: str) -> Any:
        """
        Load a saved model from a given directory.

        Parameters
        ----------
        model_path : str
            Path to the directory containing the saved model.

        Returns
        -------
        nn.Module
            A loaded PyTorch transformer model.
        """
        return self.model_sequence_classifier.from_pretrained(model_path)

    def predict_with_model(
        self,
        dataloader: DataLoader,
        model_path: str,
        proba: bool = True,
        progress_bar: bool = True
    ) -> np.ndarray:
        """
        Load a model from disk and predict on a given DataLoader.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader to run inference on.
        model_path : str
            Path to the saved model directory.
        proba : bool, optional
            If True, returns probabilities (via softmax). If False, returns raw logits.
            By default True.
        progress_bar : bool, optional
            Whether to show a progress bar, by default True.

        Returns
        -------
        np.ndarray
            Predictions (probabilities or logits).
        """
        model = self.load_model(model_path)
        model.to(self.device)
        return self.predict(dataloader, model, proba, progress_bar)

    def format_time(self, elapsed: float) -> str:
        """
        Convert a time in seconds to a string in the format hh:mm:ss.

        Parameters
        ----------
        elapsed : float
            Elapsed time in seconds.

        Returns
        -------
        str
            Time in hh:mm:ss.
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))


###############################################################################
# CamembertBase Class
###############################################################################
class CamembertBase(BertBase):
    """
    A variant of BertBase adapted for CamemBERT, using CamembertTokenizer and
    CamembertForSequenceClassification.
    """

    def __init__(
        self,
        model_name: str = 'camembert-base',
        tokenizer: Any = CamembertTokenizer,
        model_sequence_classifier: Any = CamembertForSequenceClassification,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the CamembertBase object.

        Parameters
        ----------
        model_name : str, optional
            A HuggingFace model name (e.g. 'camembert-base'),
            by default 'camembert-base'.
        tokenizer : Any, optional
            The tokenizer class for CamemBERT, by default CamembertTokenizer.
        model_sequence_classifier : Any, optional
            CamembertForSequenceClassification or derivative,
            by default CamembertForSequenceClassification.
        device : torch.device or None, optional
            The torch device to use. If None, auto-detect.
        """
        super().__init__(
            model_name=model_name,
            tokenizer=tokenizer,
            model_sequence_classifier=model_sequence_classifier,
            device=device
        )


####################################################2###########################
# Main Script
###############################################################################
def main():
    """
    Main function to:
     1) Ask the user for training mode:
        - (1) Retrain ALL models in 'models/' directory
        - (2) Use the CSV file listing model names
        - (3) Train from scratch: clear 'models/' and rebuild every model found in 'annotation_bases'
     2) If retrain ALL, automatically scan 'models/' for folders ending with '.model'.
     3) If CSV, read the file `Database/Training_data/models_metrics_summary_advanced_retrained.csv`.
     4) If train from scratch, remove all existing models in 'models/', then parse
        every train/validation file pair in 'annotation_bases' to build a list of models to train.
     5) For each selected model, detect language (EN/FR), build the train/validation paths,
        and train for 20 epochs. If class-1 F1 < 0.7, trigger a reinforced training
        for 20 more epochs (with rescue logic).
     6) Save logs to CSV in `Database/Training_data/Retraining_logs` and each final
        best model in the `./models` directory.
     7) Append final model metrics to `Database/Training_data/all_best_models.csv`.

    Note: "Rescue" logic is in place. If the best normal training epoch
    yields class-1 F1=0, any improvement in the reinforced phase triggers a save.
    """

    # --- Paths and Setup ---
    csv_path = os.path.join("Database", "Training_data", "models_metrics_summary_advanced_retrained.csv")
    annotation_base = os.path.join("Database", "Training_data", "annotation_bases")
    LOGS_DIR = os.path.join("Database", "Training_data", "Retraining_logs")
    ALL_BEST_MODELS_CSV = os.path.join("Database", "Training_data", "all_best_models.csv")

    # Prepare the global best-model CSV (create header if it doesn't exist)
    if not os.path.exists(ALL_BEST_MODELS_CSV):
        with open(ALL_BEST_MODELS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "precision_0",
                "recall_0",
                "f1_0",
                "support_0",
                "precision_1",
                "recall_1",
                "f1_1",
                "support_1",
                "macro_f1",
                "saved_model_path",
                "training_phase"
            ])

    os.makedirs("models", exist_ok=True)

    # Prompt user for choice
    choice = input(
        "Choose an option:\n"
        "1) Retrain ALL models from 'models/'\n"
        "2) Use CSV file with model names\n"
        "3) Train from scratch (clear 'models/' and rebuild all models from 'annotation_bases')\n"
        "Enter your choice [1/2/3]: "
    ).strip().lower()

    def detect_language(model_name: str) -> str:
        """
        Detect language (EN or FR) from model file name by checking the suffix.

        Parameters
        ----------
        model_name : str
            Model filename (e.g. "Just_1_SUB_EN.jsonl" or "Eco_2_SUB_FR.jsonl").

        Returns
        -------
        str
            "EN" or "FR".
        """
        lower_name = model_name.lower()
        if lower_name.endswith("_fr.jsonl") or "_fr" in lower_name:
            return "FR"
        return "EN"

    def find_main_folder(model_name: str, lang: str) -> str:
        """
        Infer the main detection folder from the model name.
        Example:
          - "Just_1_SUB_EN.jsonl" -> "Just_Detection"
          - "Eco_2_SUB_FR.jsonl"  -> "Eco_Detection"
          - "Emotion:_Negative_FR.jsonl" -> "Emotion:_Negative"
          - "Solutions_Detection_FR.jsonl" -> "Solutions_Detection"

        If the name already contains "Detection", return it directly. Otherwise,
        use a dictionary of known prefixes or return the base name if uncertain.

        Parameters
        ----------
        model_name : str
            The filename of the model, e.g. "Just_1_SUB_EN.jsonl".
        lang : str
            "EN" or "FR".

        Returns
        -------
        str
            The name of the subfolder (e.g. "Just_Detection").
        """
        suffix = f"_{lang}.jsonl"
        base = model_name.replace(suffix, "")

        # Map of known prefixes
        candidates = {
            "Cult_": "Cult_Detection",
            "Eco_": "Eco_Detection",
            "Envt_": "Envt_Detection",
            "Event_": "Event_Detection",
            "Just_": "Just_Detection",
            "Messenger_": "Messenger_Detection",
            "Pbh_": "Pbh_Detection",
            "Pol_": "Pol_Detection",
            "RED_": "RED_Detection",
            "Sci_": "Sci_Detection",
            "Secu_": "Secu_Detection",
            "Solutions_": "Solutions_Detection",
            # Emotions
            "Emotion:_Negative": "Emotion:_Negative",
            "Emotion:_Positive": "Emotion:_Positive",
            "Emotion:_Neutral": "Emotion:_Neutral",
            # Other
            "Location_": "Location_Detection"
        }
        if "Detection" in base:
            return base
        for k, v in candidates.items():
            if base.startswith(k):
                return v
        return base

    def gather_scratch_models() -> pd.DataFrame:
        """
        Gather all JSONL train files from the annotation_base directory to train from scratch.
        For each train file 'X_train_EN.jsonl' (or '_FR.jsonl'), we find the matching
        validation file 'X_validation_EN.jsonl' (or '_FR.jsonl').

        Returns
        -------
        pd.DataFrame
            A DataFrame with a single column 'model_name' (e.g. "Cult_1_SUB_EN.jsonl"),
            representing all models to train.
        """
        model_entries = []
        # Walk through each folder in annotation_bases
        for detection_folder in os.listdir(annotation_base):
            folder_path = os.path.join(annotation_base, detection_folder)
            if not os.path.isdir(folder_path):
                continue

            # Potential subfolders: "train" and "validation"
            train_subfolder = os.path.join(folder_path, "train")
            if not os.path.isdir(train_subfolder):
                continue

            # For each language sub-subfolder
            for lang_folder in os.listdir(train_subfolder):
                lang_path = os.path.join(train_subfolder, lang_folder)
                if not os.path.isdir(lang_path):
                    continue

                # Gather all train files
                train_files = [
                    f for f in os.listdir(lang_path)
                    if f.endswith(".jsonl") and "_train_" in f
                ]
                for tf in train_files:
                    # Example tf: "Cult_1_SUB_train_EN.jsonl"
                    # Build the final model name => "Cult_1_SUB_EN.jsonl"
                    # by removing "_train_" part
                    if lang_folder not in ["EN", "FR"]:
                        continue  # skip unknown subfolder

                    if "_train_" not in tf:
                        continue

                    # Derive the final name used in the existing logic
                    model_name_csv = tf.replace("_train_", "_")
                    # We'll also check if the matching validation file exists
                    validation_tf = tf.replace("_train_", "_validation_")
                    valid_file_candidate = os.path.join(
                        annotation_base, detection_folder, "validation",
                        lang_folder, validation_tf
                    )
                    if not os.path.exists(valid_file_candidate):
                        print(f"[WARNING] Could not find a matching validation file for {tf}. Skipping.")
                        continue

                    model_entries.append({"model_name": model_name_csv})

        # Convert to a DataFrame
        if not model_entries:
            print("[INFO] No train/validation pairs found in annotation_bases for scratch training.")
            return pd.DataFrame()

        df_scratch = pd.DataFrame(model_entries)
        # Remove duplicates if any
        df_scratch = df_scratch.drop_duplicates(subset=["model_name"])
        df_scratch = df_scratch.sort_values("model_name").reset_index(drop=True)
        return df_scratch

    # ---------------------------
    # Choice: 1) Retrain all
    # ---------------------------
    if choice in ["1", "yes", "all"]:
        # Gather all .model folders from 'models/'
        all_model_dirs = sorted(
            d for d in os.listdir("models")
            if d.endswith(".model") and os.path.isdir(os.path.join("models", d))
        )
        # Build a structure similar to reading from CSV
        model_list = []
        for folder_name in all_model_dirs:
            # Example: "Cult_1_SUB_EN.jsonl.model"
            # Remove the trailing ".model" so we treat it like "Cult_1_SUB_EN.jsonl"
            raw_name = folder_name.replace(".model", "")
            if not raw_name.lower().endswith(".jsonl"):
                raw_name += ".jsonl"
            model_list.append({"model_name": raw_name})
        df = pd.DataFrame(model_list)
        df = df.drop_duplicates(subset=["model_name"])
        df["base_name"] = df["model_name"].apply(
        lambda x: find_main_folder(x, detect_language(x))
        )
        df = df.sort_values(["base_name", "model_name"]).reset_index(drop=True)

    # ---------------------------
    # Choice: 2) Use CSV
    # ---------------------------
    elif choice in ["2", "no", "csv"]:
        # Read the CSV with the list of models to retrain
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"[ERROR] Could not find the CSV file at {csv_path}. Please check the path.")
            sys.exit(1)

        if df.empty:
            print("[ERROR] The CSV for retraining is empty. No models to retrain.")
            sys.exit(1)

    # ---------------------------
    # Choice: 3) Train from scratch
    # ---------------------------
    elif choice in ["3", "scratch", "train from scratch"]:
        print("→ Option selected: Train from scratch. Clearing 'models/' and rebuilding all models.")
        # 1) Clear 'models/' entirely
        for item in os.listdir("models"):
            item_path = os.path.join("models", item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)

        # 2) Gather all model names from annotation_bases
        df = gather_scratch_models()
        df["base_name"] = df["model_name"].apply(
        lambda x: find_main_folder(x, detect_language(x))
        )
        df = df.sort_values(["base_name", "model_name"]).reset_index(drop=True)
        if df.empty:
            print("[ERROR] No data found to train from scratch. Exiting.")
            sys.exit(1)
    else:
        print("[ERROR] Unrecognized choice. Please run again and enter 1, 2, or 3.")
        sys.exit(1)

    if df.empty:
        print("[ERROR] No models to retrain. Exiting.")
        sys.exit(1)

    # Common function logic after we have a DataFrame of model names
    for idx, row in df.iterrows():
        model_name_csv = str(row.get("model_name", "")).strip()
        if not model_name_csv:
            continue

        lang = detect_language(model_name_csv)
        folder = find_main_folder(model_name_csv, lang)

        # Build the train/validation file paths
        base_no_lang = model_name_csv.replace(f"_{lang}.jsonl", "")
        train_file = os.path.join(
            annotation_base, folder, "train", lang, f"{base_no_lang}_train_{lang}.jsonl")
        valid_file = os.path.join(
            annotation_base, folder, "validation", lang, f"{base_no_lang}_validation_{lang}.jsonl")

        if not os.path.exists(train_file):
            print(f"[WARNING] Training file not found: {train_file}. Skipping...")
            continue
        if not os.path.exists(valid_file):
            print(f"[WARNING] Validation file not found: {valid_file}. Skipping...")
            continue

        # Load the JSONL data for train/validation
        train_records = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                train_records.append(json.loads(line))
        train_df = pd.DataFrame(train_records)

        valid_records = []
        with open(valid_file, "r", encoding="utf-8") as f:
            for line in f:
                valid_records.append(json.loads(line))
        valid_df = pd.DataFrame(valid_records)

        if train_df.empty or valid_df.empty:
            print(f"[ERROR] Empty training/validation data for {model_name_csv}. Skipping.")
            continue
        if "label" not in train_df.columns or "label" not in valid_df.columns:
            print(f"[ERROR] No 'label' column for {model_name_csv}. Skipping.")
            continue
        if "text" not in train_df.columns or "text" not in valid_df.columns:
            print(f"[ERROR] No 'text' column for {model_name_csv}. Skipping.")
            continue

        print("============================================================")
        print(f"→ Processing model: {model_name_csv}")
        print(f"   Main folder      : {folder}")
        print(f"   Detected language: {lang}")
        print(f"   Train file       : {train_file}")
        print(f"   Validation file  : {valid_file}")
        print(f"   #train samples   : {len(train_df)}, #val samples: {len(valid_df)}")

        # Choose BERT or Camembert
        if lang == "FR":
            model_class = CamembertBase
            chosen_model_name = "camembert-base"
        else:
            model_class = BertBase
            chosen_model_name = "bert-base-cased"

        # Instantiate the model
        model = model_class(model_name=chosen_model_name)

        # Encode data
        train_loader = model.encode(train_df["text"].tolist(), train_df["label"].tolist(), batch_size=32)
        valid_loader = model.encode(valid_df["text"].tolist(), valid_df["label"].tolist(), batch_size=32)

        # Create logs directory for this model
        logs_dir = os.path.join(LOGS_DIR, base_no_lang + f"_{lang}")
        os.makedirs(logs_dir, exist_ok=True)

        # Define final model name
        save_as = base_no_lang + f"_{lang}.jsonl.model"
        print(f"[INFO] Training for 20 epochs + possible reinforced (20 epochs) → {save_as}")

        # Run training
        best_scores, final_info = model.run_training(
            train_loader,
            valid_loader,
            n_epochs=20,
            lr=5e-5,
            random_state=42,
            save_model_as=save_as,
            pos_weight=None,
            metrics_output_dir=logs_dir,
            best_model_criteria="combined",
            f1_class_1_weight=0.98,
            reinforced_learning=True,
            n_epochs_reinforced=20,
            rescue_low_class1_f1=True,  # rescue logic
            f1_1_rescue_threshold=0.0
        )

        print(f"→ Finished training model {model_name_csv}, final scores: {best_scores}")

        # Append final info to the global CSV of best models
        with open(ALL_BEST_MODELS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                final_info["epoch"],
                final_info["train_loss"],
                final_info["val_loss"],
                final_info["precision_0"],
                final_info["recall_0"],
                final_info["f1_0"],
                final_info["support_0"],
                final_info["precision_1"],
                final_info["recall_1"],
                final_info["f1_1"],
                final_info["support_1"],
                final_info["macro_f1"],
                final_info["saved_model_path"],
                final_info["training_phase"]
            ])

        print("Final best-model metrics appended to global 'all_best_models.csv'.")
        print("============================================================\n")


if __name__ == "__main__":
    main()
