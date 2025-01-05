#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
7_Produce_JSON_for_Recheck.py

Ce script génère un nouveau jeu de données JSONL pour revalider (manuellement) 
l'efficacité des annotations après passage des modèles. Il cible notamment 
les catégories sous-représentées, afin de vérifier la qualité des modèles 
dans tous les cas de figure.

Fonctionnalités principales :
1) Lecture de la base annotée (ex. CCF.media_processed_texts_annotated.csv).
2) Détection et filtrage des phrases déjà utilisées dans les précédentes 
   annotations manuelles (sentences_to_annotate_EN.jsonl, sentences_to_annotate_FR.jsonl).
3) Sélection aléatoire avec sur-échantillonnage (ou "pondération") 
   des catégories sous-représentées. Le but est d'avoir une couverture 
   plus équitable de toutes les classes (detection, sub et autres).
4) Respect d'une répartition 50/50 entre langue anglaise et française.
5) Production d'un fichier JSONL (multilingue), où chaque entrée contient :
   - "text" : la phrase elle-même,
   - "label" : la liste des catégories actives (==1) pour cette phrase,
   - "meta" : un dictionnaire contenant toutes les métadonnées de l'article
              (par exemple : title, source, date, etc.).

Auteur : Antoine Lemor
Date  : 2025-01-03
"""

import os
import json
import random
import pandas as pd


##############################################################################
#                      A. CONSTANTES ET CHEMINS D'ACCÈS
##############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fichier CSV annoté en sortie de 6_Annotate.py
ANNOTATED_CSV = os.path.join(
    BASE_DIR, "..", "..", "Database", "Database", "CCF.media_processed_texts_annotated.csv"
)

# Dossier contenant les anciennes annotations manuelles
MANUAL_ANNOTATIONS_DIR = os.path.join(
    BASE_DIR, "..", "..", "Database", "Training_data", "manual_annotations_JSONL"
)

# Fichiers JSONL déjà annotés manuellement, qu'on souhaite exclure
MANUAL_ANNOTATIONS_EN = os.path.join(MANUAL_ANNOTATIONS_DIR, "sentences_to_annotate_EN.jsonl")
MANUAL_ANNOTATIONS_FR = os.path.join(MANUAL_ANNOTATIONS_DIR, "sentences_to_annotate_FR.jsonl")

# Fichier de sortie (un unique JSONL multilingue, 50/50 EN/FR)
OUTPUT_JSONL = os.path.join(
    MANUAL_ANNOTATIONS_DIR, "sentences_to_recheck_multiling.jsonl"
)

# Nombre total de phrases que l’on souhaite annoter (vous pouvez ajuster)
NB_SENTENCES_TOTAL = 400  # 200 EN + 200 FR, par exemple


##############################################################################
#                      B. FONCTIONS UTILITAIRES
##############################################################################
def load_already_annotated_texts(jsonl_path):
    """
    Charge un fichier JSONL déjà utilisé pour les annotations manuelles
    et retourne l'ensemble des 'text' (phrases) qui y figurent, 
    afin de les exclure.
    """
    if not os.path.exists(jsonl_path):
        return set()

    texts = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            txt = data.get("text", "")
            texts.add(txt)
    return texts


def identify_annotation_columns(df):
    """
    Identifie les colonnes correspondant à des catégories d'annotations.
    Typiquement, on peut prendre toutes les colonnes binaires (0/1) 
    à l'exclusion de colonnes clairement métadonnées (ex. 'language', 'sentences', etc.).
    À ajuster selon la structure réelle du CSV.
    """
    excluded_cols = {"language", "sentences", "id_article", "Unnamed: 0"}
    annotation_cols = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        # On vérifie si le contenu est 0/1 ou NaN/1, etc.
        # Simple heuristique : type numérique + au moins un '1' dedans
        if pd.api.types.is_numeric_dtype(df[col]):
            nb_ones = df[col].sum(skipna=True)
            if nb_ones > 0:
                annotation_cols.append(col)

    return annotation_cols


def get_underrepresented_categories(df, annotation_cols, threshold=50):
    """
    Repère les catégories sous-représentées, par exemple celles 
    qui ont moins de 'threshold' lignes positives (==1).
    Retourne une liste de ces catégories.
    """
    underrepresented = []
    for col in annotation_cols:
        nb_positives = df[col].sum(skipna=True)
        if nb_positives < threshold:
            underrepresented.append(col)
    return underrepresented


def build_doccano_jsonl_entry(row, annotation_cols):
    """
    Construit l'entrée JSONL conforme à Doccano :
      {
        "text": ...,
        "label": [...],
        "meta": {...}
      }
    où "label" est la liste des colonnes (catégories) pour lesquelles row[col] == 1.
    Toutes les métadonnées (sauf 'sentences' et colonnes d'annotations) 
    sont placées dans 'meta'.
    """
    # 1) Le texte
    text = row["sentences"]

    # 2) Les catégories actives
    active_labels = []
    for col in annotation_cols:
        val = row[col]
        if pd.notna(val) and val == 1:
            active_labels.append(col)

    # 3) Métadonnées : on inclut toutes les colonnes de row sauf le texte et les labels
    #    Les colonnes d'annotations sont exclues pour éviter la redondance 
    #    (elles sont déjà dans label).
    meta = {}
    for col in row.index:
        if col == "sentences":
            continue
        if col in annotation_cols:
            continue
        meta[col] = row[col]

    entry = {
        "text": text,
        "label": active_labels,
        "meta": meta
    }
    return entry


##############################################################################
#                    C. FONCTION PRINCIPALE DE GÉNÉRATION
##############################################################################
def main():
    # ----------------------------------------------------------------------
    # 1) Chargement du CSV annoté
    # ----------------------------------------------------------------------
    print("[INFO] Chargement du CSV annoté...")
    if not os.path.exists(ANNOTATED_CSV):
        raise FileNotFoundError(f"Fichier introuvable : {ANNOTATED_CSV}")

    df = pd.read_csv(ANNOTATED_CSV, low_memory=False)
    print(f" -> {len(df)} lignes chargées.")

    # ----------------------------------------------------------------------
    # 2) Identification des colonnes d'annotations
    # ----------------------------------------------------------------------
    annotation_cols = identify_annotation_columns(df)
    print(f"[INFO] Colonnes d'annotations détectées : {annotation_cols}")

    # ----------------------------------------------------------------------
    # 3) Exclure les phrases déjà annotées manuellement
    # ----------------------------------------------------------------------
    print("[INFO] Exclusion des phrases déjà annotées manuellement...")
    en_annotated = load_already_annotated_texts(MANUAL_ANNOTATIONS_EN)
    fr_annotated = load_already_annotated_texts(MANUAL_ANNOTATIONS_FR)

    initial_len = len(df)
    df = df[~(
        ((df["language"] == "EN") & (df["sentences"].isin(en_annotated))) |
        ((df["language"] == "FR") & (df["sentences"].isin(fr_annotated)))
    )].copy()
    print(f" -> {initial_len - len(df)} lignes exclues. Il reste {len(df)} lignes.")

    # ----------------------------------------------------------------------
    # 4) Identification des catégories sous-représentées
    # ----------------------------------------------------------------------
    # Par exemple, on fixe un seuil "threshold=50" (à ajuster). 
    # On peut ensuite sur-échantillonner (ou forcer la prise en compte) 
    # d'un plus grand nombre de phrases de ces catégories.
    under_cat = get_underrepresented_categories(df, annotation_cols, threshold=50)
    print(f"[INFO] Catégories sous-représentées (moins de 50 occurrences) : {under_cat}")

    # ----------------------------------------------------------------------
    # 5) Constitution de l'échantillon en deux étapes :
    #    a) Sur-échantillonner des phrases positives pour under_cat
    #    b) Compléter le reste pour atteindre NB_SENTENCES_TOTAL, 
    #       tout en préservant 50% EN, 50% FR
    # ----------------------------------------------------------------------

    # a) Sur-échantillonner en priorité les catégories sous-représentées
    #    On récupère toutes les phrases qui possèdent au moins 
    #    une catégorie sous-représentée = 1 (union).
    #    Ensuite, on pourra en prendre un certain nombre au hasard.
    df_under = df.copy()
    mask_under = False
    for cat in under_cat:
        mask_under |= (df_under[cat] == 1)
    df_candidates_under = df_under[mask_under]

    # Pour éviter de prendre que des sous-représentées, on va limiter 
    # cette prise si c'est trop grand. On peut par exemple prendre 
    # min(len(df_candidates_under), 200) si on veut limiter à 200 
    # (ou un certain ratio).
    random_under_limit = max(50, int(0.5 * NB_SENTENCES_TOTAL))  # 50% du total, par ex.
    if len(df_candidates_under) > random_under_limit:
        df_candidates_under = df_candidates_under.sample(random_under_limit, random_state=42)
    # On a maintenant un premier set surreprésentant (relativement) les under_cat
    # sans excéder la moitié de l'échantillon final visé.

    # b) Compléter l'échantillon (en plus de df_candidates_under) 
    #    depuis le reste du DF. 
    #    On veut NB_SENTENCES_TOTAL - len(df_candidates_under) phrases en tout.
    df_rest = df.drop(df_candidates_under.index, errors="ignore")
    nb_rest_needed = NB_SENTENCES_TOTAL - len(df_candidates_under)
    if nb_rest_needed < 0:
        nb_rest_needed = 0

    if len(df_rest) > nb_rest_needed:
        df_rest = df_rest.sample(nb_rest_needed, random_state=42)

    # c) Combiner
    df_final = pd.concat([df_candidates_under, df_rest], axis=0)
    # On mélange l'ensemble pour ne pas avoir d'ordre particulier
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # 6) Assurer le 50/50 EN/FR
    #    - Stratégie : on sépare en deux dataframes (EN et FR) et on 
    #      équilibre au plus proche 50/50, en random.
    # ----------------------------------------------------------------------
    df_en = df_final[df_final["language"] == "EN"].copy()
    df_fr = df_final[df_final["language"] == "FR"].copy()

    # Nombre maximum par langue : NB_SENTENCES_TOTAL // 2 (par ex. 200)
    half_target = NB_SENTENCES_TOTAL // 2

    if len(df_en) > half_target:
        df_en = df_en.sample(half_target, random_state=42)
    if len(df_fr) > half_target:
        df_fr = df_fr.sample(half_target, random_state=42)

    # Si l'un des deux est plus petit que half_target, 
    # on peut le prendre en entier et ajuster l'autre
    final_en = len(df_en)
    final_fr = len(df_fr)
    # On peut laisser tel quel si on veut impérativement NB_SENTENCES_TOTAL 
    # ou on peut accepter un total un peu plus faible si l'un des deux 
    # n'atteint pas half_target.
    # Ici on choisit la flexibilité : on prend tout ce qui est possible 
    # jusqu'à 50/50. 
    # => On réadapte l’autre groupe pour qu’il ait la même taille.
    if final_en < half_target:
        # On redescend FR au même nombre
        df_fr = df_fr.sample(final_en, random_state=42)
    elif final_fr < half_target:
        # On redescend EN au même nombre
        df_en = df_en.sample(final_fr, random_state=42)

    # Reconstruction finale
    df_final_balanced = pd.concat([df_en, df_fr], axis=0)
    # Mélange final
    df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[INFO] Échantillon final : {len(df_final_balanced)} lignes.")
    print(f"       -> EN : {(df_final_balanced['language'] == 'EN').sum()}")
    print(f"       -> FR : {(df_final_balanced['language'] == 'FR').sum()}")

    # ----------------------------------------------------------------------
    # 7) Production du JSONL
    # ----------------------------------------------------------------------
    print("[INFO] Écriture du JSONL final multilingue...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for idx, row in df_final_balanced.iterrows():
            entry = build_doccano_jsonl_entry(row, annotation_cols)
            json_line = json.dumps(entry, ensure_ascii=False)
            out_f.write(json_line + "\n")

    print(f"[INFO] Fichier JSONL créé : {OUTPUT_JSONL}")
    print("[INFO] Fin du script.")


if __name__ == "__main__":
    main()
