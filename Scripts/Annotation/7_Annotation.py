"""
PROJECT:
-------
CCF-Canadian-Climate-Framing

TITLE:
------
7_Annotation.py

MAIN OBJECTIVE:
---------------
Ce script charge la base de données CCF.media_processed_texts.csv et annote
chaque phrase à l’aide de modèles anglais ou français déjà entraînés 
(en utilisant la librairie AugmentedSocialScientist).
Il peut sauvegarder/reprendre la progression pour gérer d’éventuelles interruptions.

Dependencies:
-------------
- os
- glob
- re
- pandas
- numpy
- torch
- tqdm
- AugmentedSocialScientist (pip install AugmentedSocialScientist)

MAIN FEATURES:
--------------
1) Charge la base CCF.media_processed_texts.csv.
2) Recherche les modèles EN/FR dans un répertoire.
3) Gère un système de suffixes stricts pour séparer (Detection, SUB, Other).
4) Sauvegarde à chaque étape pour pouvoir reprendre si interrompu.

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
from tqdm.auto import tqdm

# --- Import des modèles AugmentedSocialScientist ---
# Pour le FR: Camembert
# Pour l'EN: Bert
from AugmentedSocialScientist.models import Camembert, Bert


##############################################################################
#           A. PARSING STRICT DES NOMS DE FICHIERS DE MODÈLES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictement, on cherche :
      - "base_category" (ex: "Event_Detection" ou "Cult_1_SUB")
      - la langue 'EN' ou 'FR' (doit se terminer par "_EN" ou "_FR")
      - le type 'Detection', 'SUB' ou 'Other' (doit se terminer par "_Detection" ou "_SUB")

    Si le nom n'est pas conforme, on renvoie lang=None/type=None.
    """
    # Retirer les extensions
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    lang = None
    if name.endswith('_EN'):
        lang = 'EN'
        name = name[:-3]  # retire "_EN"
    elif name.endswith('_FR'):
        lang = 'FR'
        name = name[:-3]  # retire "_FR"

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
#          B. CONSTRUCTION D'UN DICTIONNAIRE DE MODÈLES (STRICT)
##############################################################################
def load_all_models_strict(models_dir):
    """
    Parcourt tous les fichiers *.model du dossier et applique parse_model_filename_strict.
    Renvoie un dictionnaire:
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
            print(f"[WARNING] Fichier '{filename}' ignoré (pas de suffixe _EN ou _FR).")
            continue

        print(f"[INFO] Modèle détecté : '{filename}' -> base='{base_cat}', lang='{lang}', type='{model_type}'")
        if base_cat not in model_dict:
            model_dict[base_cat] = {}

        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#     C. DÉTECTION DU DEVICE (GPU / MPS / CPU)
##############################################################################
def get_device():
    """
    Retourne un torch.device:
      - GPU CUDA si torch.cuda.is_available()
      - GPU MPS si torch.backends.mps.is_available()
      - Sinon CPU
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
# D. FONCTION GÉNÉRIQUE POUR ANNOTER (PRÉDIRE) LES LABELS (0/1) PAR LOTS
##############################################################################
def predict_labels(
    df,
    indices,
    text_column,
    model_path,
    lang,
    device,
    output_col,
    batch_size=400
):
    """
    Cette fonction annote df[output_col] sur les lignes 'indices' en utilisant
    un modèle déjà entraîné (dont le dossier est model_path) via AugmentedSocialScientist.

    1) On récupère les textes dans df.loc[indices, text_column].
    2) On instancie un model Camembert() ou Bert() selon lang, sur device.
    3) On encode par batches, on fait 'predict_with_model' pour obtenir les probabilités,
       puis on prend argmax(logits) comme label (0 ou 1).
    """

    if len(indices) == 0:
        return

    # Sélection des textes
    texts = df.loc[indices, text_column].tolist()

    # Instanciation du modèle correspond à la langue
    if lang == 'FR':
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    predictions = []
    # On va traiter les textes par batch pour éviter d'encoder tout d'un coup si c'est trop gros
    # (même si AugmentedSocialScientist sait déjà batcher, on le contrôle ici).
    with tqdm(total=len(texts), desc=f"Annot '{output_col}'", unit="txt") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 1) Encode
            batch_loader = model.encode(
                batch_texts,
                labels=None,         # on n'a pas de labels
                batch_size=len(batch_texts),
                progress_bar=False
            )
            # 2) Predict
            try:
                probs = model.predict_with_model(
                    batch_loader,
                    model_path=model_path,
                    proba=True,         # on veut des proba
                    progress_bar=False
                )
                # 3) On prend argmax pour un problème binaire => 0 ou 1
                #    Les colonnes du array "probs" = shape (N, nb_labels).
                #    Si binaire, shape=(N,2). On prend le col max:
                batch_preds = np.argmax(probs, axis=1).tolist()
                predictions.extend(batch_preds)

            except Exception as e:
                print(f"   [ERROR] Échec de predict_with_model sur {model_path}. Raison: {e}")
                # Si gros plantage, on annule tout en mettant NaN
                # (on aurait pu faire un fallback ; ici on part du principe qu'on skip)
                predictions.extend([np.nan]*len(batch_texts))

            pbar.update(len(batch_texts))

    # Mise à jour du DF
    df.loc[indices, output_col] = predictions


##############################################################################
#           E. LOGIQUE PRINCIPALE D'ANNOTATION (EN 3 ÉTAPES)
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path):
    """
    - On repère d'abord les catégories "Detection", "SUB", "Other"
      via leur suffixe strict (_Detection, _SUB, sinon Other).
    - On annote dans l'ordre : Detection => SUB => Other.
    - Après chaque catégorie/langue, on sauvegarde le CSV.
    """

    text_col = "sentences"
    lang_col = "language"

    # 1) Catégories "Detection", "SUB", "Other" selon suffixe
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
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        # Annoter chaque langue
        for lang, model_path in model_dict[cat_det].items():
            print(f"\n -> Now annotating '{cat_det}' (Detection) for lang='{lang}' with model='{model_path}'")

            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => Pas de lignes à annoter pour '{cat_det}' / lang={lang}.")
                continue

            # On prédit
            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_det
            )

            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

            # Distribution partielle
            dist_lang = df.loc[df[lang_col] == lang, cat_det].value_counts(dropna=False)
            print(f"Distribution pour '{cat_det}' (lang={lang}):\n{dist_lang}")

        # Distribution globale
        dist_after = df[cat_det].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_det}':\n{dist_after}")

    # ----------------------------------------------------------------
    # STEP 2: SUB
    # ----------------------------------------------------------------
    print("\n[ANNOTATION] Step 2: Sub-categories (SUB)")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # Retrouver la catégorie "parent"
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # enlève '_1_SUB', '_2_SUB', etc.
        main_category += '_Detection'

        if main_category not in df.columns:
            print(f"   [WARN] Catégorie parent '{main_category}' manquante pour '{cat_sub}'. On skip.")
            continue

        for lang, model_path in model_dict[cat_sub].items():
            print(f"\n -> Now annotating '{cat_sub}' (SUB) for lang='{lang}' with model='{model_path}'")

            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &  # on n'annote que si le parent = 1
                (df[cat_sub].isna())
            ].index

            if len(idx) == 0:
                print(f"   => Pas de lignes positives à annoter pour '{cat_sub}' / lang={lang}.")
                continue

            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_sub
            )
            df.to_csv(output_path, index=False)
            dist_lang = df.loc[df[lang_col] == lang, cat_sub].value_counts(dropna=False)
            print(f"Distribution pour '{cat_sub}' (lang={lang}):\n{dist_lang}")

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
                print(f"   => Pas de lignes à annoter pour '{cat_other}' / lang={lang}.")
                continue

            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_other
            )
            df.to_csv(output_path, index=False)

            dist_lang = df.loc[df[lang_col] == lang, cat_other].value_counts(dropna=False)
            print(f"Distribution pour '{cat_other}' (lang={lang}):\n{dist_lang}")

        dist_after = df[cat_other].value_counts(dropna=False)
        print(f"Overall distribution for '{cat_other}':\n{dist_after}")

    # --- Résumé final
    print("\n[ANNOTATION] Final distribution summary:")
    all_cols = categories_detection + categories_sub + categories_other
    for col in all_cols:
        if col in df.columns:
            dist_col = df[col].value_counts(dropna=False)
            print(f" - {col} : \n{dist_col}\n")

    return df


##############################################################################
#                       F. FONCTION PRINCIPALE
##############################################################################
def main():
    """
    1) Définit les chemins (data, models, output).
    2) Charge ou reprend le CSV si déjà annoté.
    3) Scanne le répertoire de modèles (strict naming).
    4) Annotate !
    5) Sauvegarde finale.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    models_dir = os.path.join(base_path, "..", "..", "models")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # Chargement / reprise
    if os.path.exists(output_path):
        print(f"[main] Fichier d'annotation déjà existant : '{output_path}'. Reprise...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] Aucun fichier d'annotation existant. Chargement initial du CSV...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} lignes chargées dans le DataFrame.")

    # Chargement des modèles disponibles
    print("[main] Chargement des fichiers de modèles (suffixes stricts)...")
    model_dict = load_all_models_strict(models_dir)
    print(f"[main] Nombre de catégories détectées : {len(model_dict)}")

    # Device
    device = get_device()

    # Annotation
    print("[main] Démarrage de l'annotation...")
    df_annotated = annotate_dataframe(df, model_dict, device, output_path)

    # Sauvegarde finale
    print("[main] Sauvegarde du DataFrame annoté final...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation terminée. Fichier final : {output_path}")


if __name__ == "__main__":
    main()
