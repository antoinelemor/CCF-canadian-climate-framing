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
Antoine Lemor (modifications by ChatGPT)
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
    Parcours tous les fichiers *.model et, pour chacun, détecte:
      - base_category (ex: "Event_Detection")
      - lang (EN/FR) strict
      - type (Detection/SUB/Other)
    Construit ensuite un dict sous la forme:
    {
      "Event_Detection": {"EN": "/path/Event_Detection_EN.model", "FR": ...},
      "Event_1_SUB": {"EN": ..., "FR": ...},
      ...
    }
    Les modèles qui ne respectent pas la nomenclature sont ignorés (lang=None).
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
    1) Récupère la liste des catégories (Detection, SUB, Other) en se basant sur la fin du nom.
    2) Annoter chaque catégorie, langue par langue.
    3) Après chaque catégorie, on affiche la distribution (0/1/NaN).
    4) Sauvegarde intermédiaire après chaque catégorie.
    5) À la fin, on affiche la distribution finale pour toutes les colonnes annotées.
    """

    text_col = "sentences"
    lang_col = "language"

    # On va classer strictement: 
    #   - tout ce qui finit en "_Detection" => detection
    #   - tout ce qui finit en "_SUB"       => sub
    #   - sinon => other
    categories_detection = []
    categories_sub = []
    categories_other = []

    # On trie simplement par ordre alphabétique pour avoir un ordre déterministe
    sorted_categories = sorted(model_dict.keys())

    for base_cat in sorted_categories:
        if base_cat.endswith('_Detection'):
            categories_detection.append(base_cat)
        elif base_cat.endswith('_SUB'):
            categories_sub.append(base_cat)
        else:
            categories_other.append(base_cat)

    # --- 1) Main categories: Annotate all rows
    print("\n[ANNOTATION] Step 1: Main Categories (Detection)")
    for cat_det in categories_detection:
        # Créer la colonne si elle n'existe pas
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        # Pour chaque langue (EN / FR) qu'on a dans model_dict[cat_det], on annote
        for lang, model_path in model_dict[cat_det].items():
            print(f"\n -> Now annotating category='{cat_det}' (type=Detection) for lang='{lang}' with model='{model_path}'")

            # Sélection des lignes à annoter : 
            #   - language == lang
            #   - text_col non vide
            #   - cat_det est NaN (non encore annoté)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_det}' / lang={lang}. (Already done or no data)")
                continue

            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. Setting {cat_det} to NaN for these lines.\n   Reason: {e}")
                df.loc[idx, cat_det] = np.nan  # on met tout à NaN pour relancer plus tard
                continue

            # Prévision
            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_det)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. Setting {cat_det} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_det] = np.nan
                continue

            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

        # Après avoir annoté (EN + FR) pour cette catégorie, affichons sa distribution
        dist_after = df[cat_det].value_counts(dropna=False)
        print(f"Distribution for '{cat_det}' after annotation:\n{dist_after}")

    # --- 2) Sub-categories (SUB): Annotate only rows where main_category == 1
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # On déduit la catégorie mère: on retire "_X_SUB" ou "_SUB" et on ajoute "_Detection"
        # par ex: "Solutions_2_SUB" => "Solutions_Detection"
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # enlève '_1_SUB', '_2_SUB', etc.
        main_category += '_Detection'

        # Si la colonne parente n'existe pas, skip
        if main_category not in df.columns:
            print(f" - Warning: main category '{main_category}' does not exist for '{cat_sub}'. Skipping sub-annotation.")
            continue

        # Pour chaque langue dispo (EN / FR)
        for lang, model_path in model_dict[cat_sub].items():
            print(f"\n -> Now annotating sub-category='{cat_sub}' (type=SUB) for lang='{lang}' with model='{model_path}'")

            # On cible uniquement les lignes où main_category == 1
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &      # parent must be 1
                (df[cat_sub].isna())           # sub-cat non encore annoté
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_sub}' / lang={lang}. (either no positives or already annotated).")
                continue

            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. Setting {cat_sub} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_sub] = np.nan
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_sub)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. Setting {cat_sub} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_sub] = np.nan
                continue

            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

        dist_after = df[cat_sub].value_counts(dropna=False)
        print(f"Distribution for '{cat_sub}' after annotation:\n{dist_after}")

    # --- 3) Other models: Annotate all rows
    print("\n[ANNOTATION] Step 3: Other models (neither Detection nor SUB)")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        # Pour chaque langue
        for lang, model_path in model_dict[cat_other].items():
            print(f"\n -> Now annotating other-category='{cat_other}' (type=Other) for lang='{lang}' with model='{model_path}'")

            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f"   => No rows to annotate for '{cat_other}' / lang={lang} (already done or no data).")
                continue

            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f"   [ERROR] Unable to load model '{model_path}'. Setting {cat_other} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_other] = np.nan
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_other)
            except Exception as e:
                print(f"   [ERROR] Annotation failed for model '{model_path}'. Setting {cat_other} to NaN.\n   Reason: {e}")
                df.loc[idx, cat_other] = np.nan
                continue

            df.to_csv(output_path, index=False)

        dist_after = df[cat_other].value_counts(dropna=False)
        print(f"Distribution for '{cat_other}' after annotation:\n{dist_after}")

    # --- Distribution finale pour toutes les colonnes qu'on a annotées
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
    1) Définition des chemins (données, modèles, sortie).
    2) Lecture ou reprise du CSV déjà annoté (s'il existe).
    3) Scan du répertoire de modèles (suffixes stricts).
    4) Annotation (en 3 étapes: Detection, SUB, Other).
    5) Sauvegarde et affichage final des distributions.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1) Paths
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    models_dir = os.path.join(base_path, "..", "..", "models")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # 2) Chargement ou reprise
    if os.path.exists(output_path):
        print(f"[main] Existing annotation file detected: '{output_path}'. Resuming annotation...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] No existing annotation file. Loading initial CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} rows loaded into the DataFrame.")

    # 3) Charger les modèles (strict)
    print("[main] Loading model files (strict matching)...")
    model_dict = load_all_models_strict(models_dir)
    print(f"[main] Number of detected base_categories: {len(model_dict)}")

    # 4) Détecter GPU/CPU
    device = get_device()

    # 5) Annoter
    print("[main] Starting annotation...")
    df_annotated = annotate_dataframe(df, model_dict, device, output_path)

    # 6) Sauvegarde finale
    print("[main] Saving the fully annotated DataFrame...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation complete. Output file: {output_path}")


if __name__ == "__main__":
    main()
