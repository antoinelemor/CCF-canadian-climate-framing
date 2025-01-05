"""
6_Annotate.py

Ce script charge la base de données CCF.media_processed_texts.csv et annote chaque
phrase selon les modèles entraînés (anglais et français).

Règles principales :
1) Les modèles se terminant par "_Detection_EN" ou "_Detection_FR" annotent l'ensemble
   des lignes de la langue correspondante dans la base (catégories principales).
2) Les modèles se terminant par "_SUB_EN" ou "_SUB_FR" annotent uniquement les lignes
   de la langue correspondante qui ont été prédites positives (=1) par la catégorie
   principale associée.
3) Les modèles restants ("Other") annotent toutes les lignes de la langue correspondante.
4) Pour chaque catégorie (ex. "Event_Detection" ou "Cult_1_SUB"), on fusionne les prédictions
   EN/FR dans une seule et unique colonne du DataFrame (p. ex. "Event_Detection" ou "Cult_1_SUB").

Modifications incluses :
- Analyse de proximité pour reconnaître un suffixe "_Detection" ou "_SUB" même s'il existe
  de légères différences (ex : "Solutio_detection" sera compris comme "_Detection" s'il y a
  au moins 95% de similarité).
- Le script sauvegarde et reprend l'annotation là où elle s'est arrêtée si le script s'interrompt
  ou échoue en cours de route.

Auteur: Antoine Lemor
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

from difflib import SequenceMatcher  # Pour la similarité de chaînes


##############################################################################
#                          A. UTILITAIRES DE SIMILARITÉ
##############################################################################
def approximate_match(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Retourne True si la similarité entre a et b (insensible à la casse)
    est >= threshold, selon le ratio de SequenceMatcher.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def approximate_endswith(name: str, suffix: str, threshold: float = 0.95) -> bool:
    """
    Vérifie si la fin de `name` est "proche" de `suffix`, selon un certain threshold
    de similarité. Ex : approximate_endswith('Solution_detection', '_Detection') => True
    si le ratio est >= threshold.
    """
    # On ne regarde que la partie de name de longueur similaire à suffix
    if len(name) < len(suffix):
        return False
    end_part = name[-len(suffix):]
    return approximate_match(end_part, suffix, threshold=threshold)


##############################################################################
#                      B. PARSING DU NOM DE FICHIER DE MODÈLE
##############################################################################
def parse_model_filename(filename):
    """
    Extrait :
      - la 'base_category' (par ex. "Event_Detection" ou "Cult_1_SUB")
      - la langue 'EN' ou 'FR'
      - le type ('Detection', 'SUB', ou 'Other') juste pour info.

    Utilise de la similarité pour tolérer les petites fautes : "Solution_detectio" ≈ "_Detection"
    ou "Solutons_2_SUB" ≈ "_SUB", etc.
    """

    # Retirer l'extension
    name = filename.replace('.jsonl.model', '').replace('.model', '')

    # 1) Détecter la langue (approx) : on teste si la fin est proche de "_EN" ou "_FR"
    lang = None
    if approximate_endswith(name, '_EN', 0.95):
        lang = 'EN'
        # On enlève la partie de fin correspondant à _EN
        name = name[:-3]  # on enlève exactement 3 caractères
    elif approximate_endswith(name, '_FR', 0.95):
        lang = 'FR'
        name = name[:-3]

    # 2) Déterminer si c'est un Detection, un SUB, ou un Other, en regardant la fin
    #    de la chaîne (approx)
    if approximate_endswith(name, '_Detection', 0.95):
        model_type = 'Detection'
        # on enlève le suffixe (longueur len('_Detection')=10)
        base_category = name  # la base category inclura la fin "_Detection" (ou approx)
    elif approximate_endswith(name, '_SUB', 0.95):
        model_type = 'SUB'
        base_category = name
    else:
        model_type = 'Other'
        base_category = name

    return base_category, lang, model_type


##############################################################################
#         C. CRÉATION D'UNE STRUCTURE DE DICTIONNAIRES POUR LES MODÈLES
##############################################################################
def load_all_models(models_dir):
    """
    Parcourt tous les fichiers *.model du dossier, parse leur nom, et construit
    un dict de la forme :
    {
      "Event_Detection": {
          "EN": "/chemin/Event_Detection_EN.model",
          "FR": "/chemin/Event_Detection_FR.model"
      },
      "Event_1_SUB": {
          "EN": "/chemin/Event_1_SUB_EN.model",
          "FR": "/chemin/Event_1_SUB_FR.model"
      },
      "Emotion:_Positive": {
          "EN": "...",
          "FR": "..."
      },
      ...
    }

    Ce dictionnaire unifie EN/FR sous la même clé (base_category).
    """
    model_files = glob.glob(os.path.join(models_dir, "*.model"))
    model_dict = {}

    for filepath in model_files:
        filename = os.path.basename(filepath)
        base_cat, lang, _ = parse_model_filename(filename)

        # Si on n'a pas détecté la langue (None), on l'ignore.
        if lang is None:
            print(f"[AVERTISSEMENT] Fichier {filename} ignoré (langue introuvable).")
            continue

        if base_cat not in model_dict:
            model_dict[base_cat] = {}

        model_dict[base_cat][lang] = filepath

    return model_dict


##############################################################################
#                 D. DÉTECTION DU DEVICE (GPU / CPU)
##############################################################################
def get_device():
    """
    Détecte si un GPU (CUDA) ou MPS (Mac Silicon) est dispo, sinon CPU.
    """
    if torch.cuda.is_available():
        print("Utilisation du GPU CUDA pour les calculs.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Utilisation du GPU (MPS) pour les calculs.")
        return torch.device("mps")
    else:
        print("Utilisation du CPU pour les calculs.")
        return torch.device("cpu")


##############################################################################
#        E. CHARGEMENT DU MODÈLE + TOKENIZER À PARTIR DE SON CHEMIN
##############################################################################
def load_model_and_tokenizer(model_path, lang, device):
    """
    Charge le modèle + tokenizer à partir de model_path.
    """
    if lang == 'FR':
        base_model_name = 'camembert-base'
    else:
        base_model_name = 'bert-base-uncased'

    # Charger le tokenizer de base
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512

    # Charger le modèle fine-tuné
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Ajouter des logs pour vérifier le device du modèle
    print(f"[DEBUG] Modèle chargé sur le device : {next(model.parameters()).device}")

    return model, tokenizer


##############################################################################
#           F. FONCTION GÉNÉRIQUE D'ANNOTATION AVEC UN MODÈLE
##############################################################################
def predict_labels(df, indices, text_column, model, tokenizer, device, output_col):
    """
    Annote (avec tqdm) le DataFrame df[output_col] sur les lignes 'indices'
    en utilisant 'model'/'tokenizer'. On fait de la batch-inference (16).
    """
    if len(indices) == 0:
        return

    batch_size = 16
    texts = df.loc[indices, text_column].tolist()

    predictions = []
    with tqdm(total=len(texts), desc=f"Annotation '{output_col}'", unit="txt") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
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

    # Mise à jour du df
    df.loc[indices, output_col] = predictions


##############################################################################
#                    G. LOGIQUE PRINCIPALE D'ANNOTATION
##############################################################################
def annotate_dataframe(df, model_dict, device, output_path):
    """
    1) Pour chaque clé de model_dict (ex: "Event_Detection", "Cult_1_SUB", ...),
       on détermine si c'est un Detection, SUB ou Other, et on annote en conséquence.
    2) On crée une seule colonne (base_category) pour fusionner EN/FR.
    3) Après chaque catégorie (ou sous-catégorie), on sauvegarde le CSV pour
       pouvoir reprendre en cas d'interruption.
    """

    text_col = "sentences"
    lang_col = "language"

    # -- Lister l'ordre (Detection, SUB, Other) pour annoter d'abord
    #    les catégories principales, puis les sous-catégories, puis le reste.
    categories_detection = []
    categories_sub = []
    categories_other = []

    # On veut trier les clés pour éviter un ordre aléatoire
    sorted_categories = sorted(model_dict.keys())

    for base_cat in sorted_categories:
        if approximate_endswith(base_cat, '_Detection', 0.95):
            categories_detection.append(base_cat)
        elif approximate_endswith(base_cat, '_SUB', 0.95):
            categories_sub.append(base_cat)
        else:
            categories_other.append(base_cat)

    # --- 1) Catégories principales : Annote toutes les lignes
    print("\n[ANNOTATION] Étape 1 : Catégories principales (Detection)...")
    for cat_det in categories_detection:
        if cat_det not in df.columns:
            df[cat_det] = np.nan  # on crée la colonne si elle n'existe pas

        # On annote langue par langue
        for lang, model_path in model_dict[cat_det].items():
            # Sélection des lignes non encore annotées
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_det].isna())  # On ne ré-annote pas les lignes déjà remplies
            ].index

            if len(idx) == 0:
                print(f" - Aucune ligne à annoter pour '{cat_det}' / lang={lang}. (Déjà fait ou pas de donnée)")
                continue

            print(f" - Annotation de la catégorie {cat_det} (lang={lang}) sur {len(idx)} lignes.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERREUR] Impossible de charger le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_det)
            except Exception as e:
                print(f" [ERREUR] Échec de l'annotation pour le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            # Sauvegarde intermédiaire pour reprendre en cas d'arrêt brutal
            df.to_csv(output_path, index=False)

    # --- 2) Sous-catégories (SUB) : Annote uniquement les lignes positives (==1)
    print("\n[ANNOTATION] Étape 2 : Sous-catégories (SUB)...")
    for cat_sub in categories_sub:
        if cat_sub not in df.columns:
            df[cat_sub] = np.nan

        # Reconstitution de la catégorie principale : s'il s'appelle "Xxx_1_SUB",
        # sa catégorie principale est "Xxx_Detection" (on retire la partie "_1_SUB" ou "_SUB"
        # et on la remplace par "_Detection").
        # Mais attention aux cas "Solutions_2_SUB" => "Solutions_Detection"
        # On fait un simple re.sub :
        main_category = re.sub(r'_?\d*_SUB$', '', cat_sub)  # retire "_1_SUB" ou "_2_SUB", ...
        main_category = main_category + '_Detection'

        if main_category not in df.columns:
            print(f" - Avertissement : la catégorie principale '{main_category}' n'existe pas pour '{cat_sub}'. On saute.")
            continue

        for lang, model_path in model_dict[cat_sub].items():
            # Ne pas réannoter si c'est déjà rempli
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[main_category] == 1) &    # on veut que la catégorie principale soit 1
                (df[cat_sub].isna())         # non déjà annoté
            ].index

            if len(idx) == 0:
                print(f" - Aucune ligne à annoter pour '{cat_sub}' / lang={lang} (soit pas de positif, soit déjà annoté).")
                continue

            print(f" - Annotation de la sous-catégorie {cat_sub} (lang={lang}), sur {len(idx)} lignes.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERREUR] Impossible de charger le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_sub)
            except Exception as e:
                print(f" [ERREUR] Échec de l'annotation pour le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

    # --- 3) Autres modèles ("Other") : Annote toutes les lignes
    print("\n[ANNOTATION] Étape 3 : Autres modèles (ni Detection ni SUB)...")
    for cat_other in categories_other:
        if cat_other not in df.columns:
            df[cat_other] = np.nan

        for lang, model_path in model_dict[cat_other].items():
            idx = df[
                (df[lang_col] == lang) &
                (df[text_col].notna()) &
                (df[cat_other].isna())
            ].index

            if len(idx) == 0:
                print(f" - Aucune ligne à annoter pour '{cat_other}' / lang={lang} (déjà fait ou pas de donnée).")
                continue

            print(f" - Annotation de la catégorie {cat_other} (lang={lang}) sur {len(idx)} lignes.")
            try:
                model, tokenizer = load_model_and_tokenizer(model_path, lang, device)
            except Exception as e:
                print(f" [ERREUR] Impossible de charger le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            try:
                predict_labels(df, idx, text_col, model, tokenizer, device, cat_other)
            except Exception as e:
                print(f" [ERREUR] Échec de l'annotation pour le modèle '{model_path}'. On saute ce modèle.\n   Raison : {e}")
                continue

            # Sauvegarde intermédiaire
            df.to_csv(output_path, index=False)

    return df


##############################################################################
#                   H. FONCTION PRINCIPALE (main)
##############################################################################
def main():
    base_path = os.path.dirname(os.path.abspath(__file__))

    # 1) Définir les chemins
    data_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts.csv")
    models_dir = os.path.join(base_path, "..", "..", "..", "models")
    output_path = os.path.join(base_path, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv")

    # 2) Vérifier si un fichier d'annotations existe déjà
    if os.path.exists(output_path):
        print(f"[main] Fichier annoté existant détecté : '{output_path}'. Reprise de l'annotation...")
        df = pd.read_csv(output_path, low_memory=False)
    else:
        print("[main] Aucun fichier annoté existant. Chargement du CSV initial...")
        df = pd.read_csv(data_path, low_memory=False)

    print(f"[main] {len(df)} lignes dans le DataFrame chargé.")

    # 3) Charger la liste des modèles
    print("[main] Chargement des fichiers de modèles...")
    model_dict = load_all_models(models_dir)
    print(f"[main] Nombre de catégories détectées : {len(model_dict)}")

    # 4) Détecter le device
    device = get_device()

    # 5) Annoter
    print("[main] Début de l'annotation...")
    df_annotated = annotate_dataframe(df, model_dict, device, output_path)

    # 6) Sauvegarder le résultat final
    print("[main] Sauvegarde finale du DataFrame annoté...")
    df_annotated.to_csv(output_path, index=False)
    print(f"[main] Annotation terminée. Fichier de sortie : {output_path}")


if __name__ == "__main__":
    main()
