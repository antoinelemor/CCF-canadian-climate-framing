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

NOUVELLES FONCTIONNALITÉS:
--------------------------
1) Les catégories _SUB ne sont annotées que pour les lignes où la
   catégorie parente _Detection = 1 (on ne traite pas les autres).
2) Un CSV `sentences_annotation_error.csv` est créé dans Database/Training_data
   si certaines phrases dépassent la limite 512 tokens et déclenchent
   l'avertissement de longueur ("Token indices sequence length is longer ...").
   On y enregistre index, phrase, catégorie, etc.
3) Un CSV `annotated_label_metrics.csv` est maintenu dans Database/Training_data.
   À chaque catégorie/chaque langue annotée, on y inscrit la distribution
   (value_counts) des labels (0,1,NaN).

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

MAIN FEATURES (héritées):
-------------------------
1) Charge la base CCF.media_processed_texts.csv.
2) Recherche les modèles EN/FR dans un répertoire.
3) Gère un système de suffixes stricts pour séparer (Detection, SUB, Other).
4) Sauvegarde à chaque étape pour pouvoir reprendre si interrompu.
5) Écrit le DataFrame final annoté dans CCF.media_processed_texts_annotated.csv

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

# --- Import des modèles AugmentedSocialScientist ---
# Pour le FR: Camembert
# Pour l'EN: Bert
from AugmentedSocialScientist.models import Camembert, Bert

##############################################################################
#                A. PARSING STRICT DES NOMS DE FICHIERS DE MODÈLES
##############################################################################
def parse_model_filename_strict(filename):
    """
    Strictement, on cherche :
      - "base_category" (ex: "Event_Detection" ou "Cult_1_SUB")
      - la langue 'EN' ou 'FR' (doit se terminer par "_EN" ou "_FR")
      - le type 'Detection', 'SUB' ou 'Other' (doit se terminer par "_Detection" ou "_SUB")

    Si le nom n'est pas conforme, on renvoie lang=None/type=None.
    """
    # Retirer les extensions possibles
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    # Langue
    lang = None
    if name.endswith('_EN'):
        lang = 'EN'
        name = name[:-3]  # retire "_EN"
    elif name.endswith('_FR'):
        lang = 'FR'
        name = name[:-3]  # retire "_FR"

    # Type de modèle
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
#          B. CHARGEMENT DE TOUS LES MODÈLES DANS UN DICT (NOM STRICT)
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
#          C. DÉTECTION DU DEVICE (GPU / MPS / CPU)
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
#       D. GESTION DES PHRASES TROP LONGUES -> CSV D'ERREUR
##############################################################################
def check_text_exceeds_length_limit(text, tokenizer, max_length=512):
    """
    Vérifie si la séquence tokenisée dépasse la limite imposée (512).
    Retourne True si le texte dépasse 512 tokens (donc potentiellement
    un message d'avertissement).
    """
    # On désactive le truncation pour mesurer la taille réelle
    # add_special_tokens=True => inclut [CLS], [SEP]
    encoded = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return (len(encoded) > max_length)


##############################################################################
# E. PRÉPARATION DES CSV POUR ERREURS ET POUR LES MÉTRIQUES
##############################################################################
def init_error_csv(error_csv_path):
    """
    Initialise (ou ouvre en mode append) le CSV qui contiendra
    les phrases posant problème de longueur.
    """
    if not os.path.exists(os.path.dirname(error_csv_path)):
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)

    if not os.path.exists(error_csv_path):
        # On crée le fichier avec un header
        pd.DataFrame(columns=["row_id", "lang", "category", "text"]).to_csv(error_csv_path, index=False)


def append_to_error_csv(error_csv_path, rows):
    """
    Ajoute dans le CSV d'erreur (en mode append) la liste de dictionnaires `rows`.
    Chaque élément de `rows` doit être un dict avec
    { "row_id":..., "lang":..., "category":..., "text":... }
    """
    if not rows:
        return
    df_err = pd.DataFrame(rows)
    df_err.to_csv(error_csv_path, mode='a', header=False, index=False)


def init_metrics_csv(metrics_csv_path):
    """
    Initialise le CSV qui contiendra les métriques de distribution.
    """
    if not os.path.exists(os.path.dirname(metrics_csv_path)):
        os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)

    if not os.path.exists(metrics_csv_path):
        cols = ["category", "lang", "label_value", "count"]
        pd.DataFrame(columns=cols).to_csv(metrics_csv_path, index=False)


def append_to_metrics_csv(metrics_csv_path, category, lang, value_counts):
    """
    value_counts est une Series => index=label_value (0.0,1.0,NaN), values=count
    On alimente un CSV "annotated_label_metrics.csv" en mode append,
    avec les colonnes = category, lang, label_value, count
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
#   F. FONCTION GÉNÉRIQUE D'ANNOTATION (PRÉDICTION) PAR LOTS + GESTION ERREURS
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
    batch_size=100
):
    """
    Cette fonction annote df[output_col] sur les lignes 'indices' en utilisant
    un modèle déjà entraîné (dont le dossier est model_path) via AugmentedSocialScientist.

    1) On récupère les textes dans df.loc[indices, text_column].
    2) On instancie un modèle Camembert() ou Bert() selon lang, sur device.
    3) Par batch, on encode => on vérifie la longueur pour détecter les "erreurs".
       - Si un texte dépasse 512 tokens, on l'inscrit dans le CSV d'erreur.
         On continue tout de même la prédiction (car huggingface va tronquer).
    4) On fait 'predict_with_model' pour obtenir les probabilités, puis argmax => label (0/1).
    5) On met à jour df[output_col] pour les lignes en question.
    """

    if len(indices) == 0:
        return

    # Sélection des textes
    texts = df.loc[indices, text_column].tolist()

    # Instanciation du modèle
    if lang == 'FR':
        model = Camembert(device=device)
    else:
        model = Bert(device=device)

    # On prépare un itérateur sur les indices => pour logguer correctement l'erreur
    indices_list = list(indices)  # afin de pouvoir itérer en même temps que texts

    # Boucle par batch
    predictions = []
    with tqdm(total=len(texts), desc=f"Annot '{output_col}'", unit="txt") as pbar:
        for start_i in range(0, len(texts), batch_size):
            batch_texts = texts[start_i:start_i + batch_size]
            batch_idx = indices_list[start_i:start_i + batch_size]

            # On vérifie la longueur de chaque phrase avant encodage
            # pour détecter celles qui dépasseront 512 tokens
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

            # On append ces éventuelles erreurs dans le CSV
            if error_rows:
                append_to_error_csv(error_csv_path, error_rows)

            # Encodage effectif
            # (AugmentedSocialScientist sait gérer la truncation automatique,
            # donc même si >512, ça ne crash pas, ça tronque. On a juste loggé.)
            batch_loader = model.encode(
                batch_texts,
                labels=None,         # on n'a pas de labels
                batch_size=len(batch_texts),
                progress_bar=False
            )

            try:
                # Prediction => shape (N,2) si binaire
                probs = model.predict_with_model(
                    batch_loader,
                    model_path=model_path,
                    proba=True,         # on veut des proba
                    progress_bar=False
                )
                batch_preds = np.argmax(probs, axis=1).tolist()

            except Exception as e:
                # En cas d'exception, on loggue et on met NaN
                print(f"   [ERROR] Échec de predict_with_model sur '{model_path}'. Raison: {e}")
                batch_preds = [np.nan] * len(batch_texts)

            predictions.extend(batch_preds)

            pbar.update(len(batch_texts))

    # Mise à jour du DF
    df.loc[indices, output_col] = predictions


##############################################################################
#           G. LOGIQUE PRINCIPALE D'ANNOTATION (EN 3 ÉTAPES)
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path, error_csv_path, metrics_csv_path):
    """
    - On repère d'abord les catégories "Detection", "SUB", "Other"
      via leur suffixe strict (_Detection, _SUB, sinon Other).
    - On annote dans l'ordre : Detection => SUB => Other.
    - Après chaque catégorie/langue, on sauvegarde le CSV + on met à jour
      le CSV des métriques (annotated_label_metrics.csv).
    """

    text_col = "sentences"
    lang_col = "language"

    # 1) Classification des catégories selon suffixe
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
        # Si la colonne n'existe pas déjà dans le DF, on la crée
        if cat_det not in df.columns:
            df[cat_det] = np.nan

        # On annote pour chaque langue où un modèle existe
        for lang, model_path in model_dict[cat_det].items():
            print(f"\n -> Now annotating '{cat_det}' (Detection) for lang='{lang}' with model='{model_path}'")

            # On sélectionne les lignes qui n'ont pas encore de label (isna)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())
            ].index

            if len(idx) == 0:
                print(f"   => Pas de lignes à annoter pour '{cat_det}' / lang={lang}.")
                continue

            # Prédiction
            predict_labels(
                df, idx, text_col, 
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_det,
                error_csv_path=error_csv_path,
                batch_size=100
            )
            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

            # Distribution partielle pour la langue
            dist_lang = df.loc[df[lang_col] == lang, cat_det].value_counts(dropna=False)
            print(f"Distribution pour '{cat_det}' (lang={lang}):\n{dist_lang}")

            # On alimente le CSV des métriques
            append_to_metrics_csv(metrics_csv_path, cat_det, lang, dist_lang)

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

        # On détermine la catégorie parente
        # (ex: "Cult_1_SUB" => parent = "Cult_Detection")
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # retire '_1_SUB', '_2_SUB', ...
        main_category += '_Detection'

        if main_category not in df.columns:
            print(f"   [WARN] Catégorie parent '{main_category}' manquante pour '{cat_sub}'. On skip.")
            continue

        # Pour chaque langue
        for lang, model_path in model_dict[cat_sub].items():
            print(f"\n -> Now annotating '{cat_sub}' (SUB) for lang='{lang}' with model='{model_path}'")

            # Contrairement aux "Detection", on n’annote que là où:
            # - language = lang
            # - sentences notna
            # - la parent detection = 1
            # - cat_sub est NaN (pas encore annoté)
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &
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
                output_col=cat_sub,
                error_csv_path=error_csv_path,
                batch_size=100
            )
            df.to_csv(output_path, index=False)

            dist_lang = df.loc[df[lang_col] == lang, cat_sub].value_counts(dropna=False)
            print(f"Distribution pour '{cat_sub}' (lang={lang}):\n{dist_lang}")

            # Ajout au CSV de métriques
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
                print(f"   => Pas de lignes à annoter pour '{cat_other}' / lang={lang}.")
                continue

            predict_labels(
                df, idx, text_col,
                model_path=model_path,
                lang=lang,
                device=device,
                output_col=cat_other,
                error_csv_path=error_csv_path,
                batch_size=100
            )
            df.to_csv(output_path, index=False)

            dist_lang = df.loc[df[lang_col] == lang, cat_other].value_counts(dropna=False)
            print(f"Distribution pour '{cat_other}' (lang={lang}):\n{dist_lang}")

            # Ajout au CSV de métriques
            append_to_metrics_csv(metrics_csv_path, cat_other, lang, dist_lang)

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
#                           H. FONCTION PRINCIPALE
##############################################################################
def main():
    """
    1) Définit les chemins (data, models, output).
    2) Charge ou reprend le CSV si déjà annoté.
    3) Scanne le répertoire de modèles (suffixes stricts).
    4) Annote !
    5) Sauvegarde finale.
    6) Les éventuelles phrases >512 tokens sont loggées dans sentences_annotation_error.csv
    7) Les distributions de labels sont loggées au fil de l'eau dans annotated_label_metrics.csv
    """

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Fichiers d'entrée / sortie
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # Répertoire des modèles
    models_dir = os.path.join(base_path, "..", "..", "models")

    # CSV d'erreurs (token length)
    error_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "sentences_annotation_error.csv")

    # CSV de métriques
    metrics_csv_path = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotated_label_metrics.csv")

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

    # Initialisation CSV d'erreurs & métriques
    init_error_csv(error_csv_path)
    init_metrics_csv(metrics_csv_path)

    # Annotation principale
    print("[main] Démarrage de l'annotation...")
    df_annotated = annotate_dataframe(
        df=df,
        model_dict=model_dict,
        device=device,
        output_path=output_path,
        error_csv_path=error_csv_path,
        metrics_csv_path=metrics_csv_path
    )

    # Sauvegarde finale
    print("[main] Sauvegarde du DataFrame annoté final...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation terminée. Fichier final : {output_path}")


if __name__ == "__main__":
    main()
