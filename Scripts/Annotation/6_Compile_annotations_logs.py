#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script : 8_Compile_annotations_logs.py

Objectif scientifique :
-----------------------
Ce script illustre une version plus "sophistiqué" et automatisé pour la sélection
de la meilleure époque (epoch), avec :

1) Utilisation conjointe de :
   - macro-F1 (importance de l'équilibre inter-classes),
   - weighted-F1 (prise en compte du déséquilibre),
   - recall de la classe "1" (prévention du sous-apprentissage).

2) Validation croisée (k folds) :
   - Agrégation des métriques par époque sur l'ensemble des folds.
   - Exclusion des époques si overfit détecté dans au moins un fold.

3) Score final pour chaque époque :
   score_epoch = alpha * macro_F1 + beta * weighted_F1 + gamma * recall_classe_1

4) En cas d'égalité de score, on compare test_loss moyen, puis train_loss moyen.

5) Dans le CSV final, on ajoute (entre autres) :
   - Le support de la classe 0 et 1
   - Les précisions/recalls/F1-scores
   - La distribution train/test lue dans les logs

Dépendances :
-------------
- Python >= 3.7
- pandas, re, os, numpy

Auteur :
--------
Antoine Lemor
"""

import os
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
#                Paramètres de sélection et de scoring
# --------------------------------------------------------------------------
overfit_threshold = 0.65  # Si (test_loss - train_loss) > 0.05 => overfit
alpha = 0.6               # Pondération du macro-F1
beta = 0.2                # Pondération du weighted-F1
gamma = 3.2               # Pondération du recall de la classe "1"

# Optionnel : Rappel minimal de la classe positive, pour ne pas la négliger
min_positive_recall = 0.0  # ex. 0.05 pour un recall min

# --------------------------------------------------------------------------
#    Définition des chemins (à adapter à votre structure de répertoires)
# --------------------------------------------------------------------------
base_path = os.path.dirname(os.path.abspath(__file__))

# Dossier où se trouvent les logs
log_output_dir = os.path.join(
    base_path, "..", "..", "Database", "Training_data", "Annotation_logs"
)

csv_output_dir = os.path.join(
    base_path, "..", "..", "Database", "Training_data"
)

# Fichier de sortie
csv_output_path = os.path.join(csv_output_dir, "models_metrics_summary_advanced_2.csv")

# --------------------------------------------------------------------------
#     Reconnaissance automatique des logs & groupement par modèle
# --------------------------------------------------------------------------
log_files = [f for f in os.listdir(log_output_dir) if f.endswith("_training_log.txt")]

fold_pattern = re.compile(r"^(.*)_fold(\d+)_training_log\.txt$")
model_to_folds = {}
for lf in log_files:
    match = fold_pattern.match(lf)
    if match:
        model_name, fold_num = match.groups()
        if model_name not in model_to_folds:
            model_to_folds[model_name] = []
        model_to_folds[model_name].append(lf)
    else:
        # Cas sans fold => on le traite comme un fold unique
        model_name_alt = lf.replace("_training_log.txt", "")
        if model_name_alt not in model_to_folds:
            model_to_folds[model_name_alt] = []
        model_to_folds[model_name_alt].append(lf)

# --------------------------------------------------------------------------
#    Fonctions utilitaires : parsing du rapport de classification
# --------------------------------------------------------------------------
def parse_classification_report(report_str):
    """
    Analyse un rapport de classification brut et renvoie un dict :
      {
        'accuracy': float,
        '0': {'precision':..., 'recall':..., 'f1-score':..., 'support':...},
        '1': {...},
        'macro avg': {...},
        'weighted avg': {...}
      }
    """
    lines = [line.strip() for line in report_str.strip().split("\n")]
    report_data = {}
    if len(lines) < 2:
        return report_data

    for line in lines:
        if not line:
            continue
        tokens = line.split()

        # Ex: "accuracy                           0.95       286"
        if tokens[0] == "accuracy":
            try:
                report_data["accuracy"] = float(tokens[-2])
            except:
                pass
            continue

        # Ex : "macro avg" ou "weighted avg"
        joined_first_two = " ".join(tokens[:2])
        if joined_first_two in ["macro avg", "weighted avg"]:
            label = joined_first_two
            try:
                precision, recall, f1_score, support = map(float, tokens[2:])
                report_data[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "support": support,
                }
            except:
                pass
            continue

        # Sinon, on suppose que c’est une classe (e.g. '0', '1')
        # Vérif qu'on a au moins 5 tokens => label + precision + recall + f1 + support
        if len(tokens) >= 5:
            label_candidate = tokens[0]
            if label_candidate.isdigit() or label_candidate in ["0","1","2"]:
                try:
                    precision, recall, f1_score, support = map(float, tokens[1:5])
                    report_data[label_candidate] = {
                        "precision": precision,
                        "recall": recall,
                        "f1-score": f1_score,
                        "support": support,
                    }
                except:
                    pass

    return report_data

# --------------------------------------------------------------------------
#   parse_single_log : récupération des métriques + distribution
# --------------------------------------------------------------------------
def parse_single_log(full_log_path):
    """
    Retourne un dict contenant :
    - "epochs_data" : dict (clé = epoch_num) => métriques de chaque époque
    - "train_class_0_count", "train_class_1_count", "test_class_0_count", "test_class_1_count"
    """
    if not os.path.exists(full_log_path) or os.path.getsize(full_log_path) == 0:
        return {}

    with open(full_log_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return {}

    # ----------------------------------------------------------------------
    # 1) Récupération de la distribution Train/Test depuis le log
    # ----------------------------------------------------------------------
    train_class_0_count = 0
    train_class_1_count = 0
    test_class_0_count = 0
    test_class_1_count = 0

    lines = content.split("\n")
    nb_lines = len(lines)
    
    i = 0
    while i < nb_lines:
        line = lines[i]
        # ------------------------------------------------------------------
        # Training distribution
        # ------------------------------------------------------------------
        if "Training label distribution" in line:
            dist_train = {}
            j = i + 1
            # On boucle jusqu'à tomber sur "Name: count" ou la fin
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                # Cherche un pattern : "0    29" ou "1    26", etc.
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_train[label_found] = count_found
                j += 1
            # On assigne
            train_class_0_count = dist_train.get('0', 0)
            train_class_1_count = dist_train.get('1', 0)
            i = j
            continue

        # ------------------------------------------------------------------
        # Validation distribution
        # ------------------------------------------------------------------
        if "Validation label distribution" in line:
            dist_test = {}
            j = i + 1
            while j < nb_lines:
                subline = lines[j].strip()
                if subline.startswith("Name: count"):
                    break
                m = re.match(r"(\d+)\s+(\d+)", subline)
                if m:
                    label_found = m.group(1)
                    count_found = int(m.group(2))
                    dist_test[label_found] = count_found
                j += 1
            test_class_0_count = dist_test.get('0', 0)
            test_class_1_count = dist_test.get('1', 0)
            i = j
            continue

        i += 1

    # ----------------------------------------------------------------------
    # 2) Récupération des métriques par epoch
    # ----------------------------------------------------------------------
    epoch_pattern = re.compile(r"^[=]{4,}\s*Epoch\s+(\d+)\s*/\s*(\d+)\s*[=]{4,}", re.MULTILINE)
    matches = list(epoch_pattern.finditer(content))
    if not matches:
        # Pas d'epoch trouvé => on renvoie juste la distribution
        return {
            "epochs_data": {},
            "train_class_0_count": train_class_0_count,
            "train_class_1_count": train_class_1_count,
            "test_class_0_count": test_class_0_count,
            "test_class_1_count": test_class_1_count,
        }

    epoch_data = {}
    for i, match in enumerate(matches):
        epoch_num = int(match.group(1))
        start_pos = match.end()
        end_pos = matches[i+1].start() if (i + 1) < len(matches) else len(content)
        block = content[start_pos:end_pos]

        # Extraire train_loss
        m_train = re.search(r"Average training loss:\s*([\d.]+)", block)
        train_loss = float(m_train.group(1)) if m_train else float("inf")

        # Extraire test_loss
        m_test = re.search(r"Average test loss:\s*([\d.]+)", block)
        test_loss = float(m_test.group(1)) if m_test else float("inf")

        # Repérer le rapport de classification
        classif_start = re.search(
            r"(?:^|\n)\s*precision\s+recall\s+f1-score\s+support\s*\n", block
        )
        if not classif_start:
            # Pas de rapport => on skip
            continue

        start_idx = classif_start.end()
        classif_content = block[start_idx:].strip()
        next_epoch_marker = re.search(r"^[=]{4,}\s*Epoch\s", classif_content, re.MULTILINE)
        if next_epoch_marker:
            classif_content = classif_content[:next_epoch_marker.start()].strip()

        report_data = parse_classification_report(classif_content)
        if "macro avg" not in report_data:
            continue

        macro_f1 = report_data["macro avg"]["f1-score"]
        weighted_f1 = report_data.get("weighted avg", {}).get("f1-score", 0.0)

        # Récupération des valeurs de la classe 1
        class_1_precision = 0.0
        class_1_recall = 0.0
        class_1_f1_score = 0.0
        class_1_support = 0.0
        if "1" in report_data:
            class_1_precision = report_data["1"]["precision"]
            class_1_recall = report_data["1"]["recall"]
            class_1_f1_score = report_data["1"]["f1-score"]
            class_1_support = report_data["1"]["support"]

        # Récupération du support de la classe 0
        class_0_support = 0.0
        if "0" in report_data:
            class_0_support = report_data["0"]["support"]

        epoch_data[epoch_num] = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "recall_1": class_1_recall,  # pour le scoring
            "class_0_support": class_0_support,
            "class_1_support": class_1_support,
            "class_1_precision": class_1_precision,
            "class_1_recall": class_1_recall,
            "class_1_f1_score": class_1_f1_score,
        }

    return {
        "epochs_data": epoch_data,
        "train_class_0_count": train_class_0_count,
        "train_class_1_count": train_class_1_count,
        "test_class_0_count": test_class_0_count,
        "test_class_1_count": test_class_1_count,
    }

# --------------------------------------------------------------------------
#   Agrégation cross-validation : combine les métriques de chaque fold
# --------------------------------------------------------------------------
def aggregate_folds_metrics(list_of_epoch_dicts):
    """
    Agrège les données de plusieurs folds pour chaque epoch.
    On calcule la moyenne de la plupart des métriques, SAUF pour le 'support'
    où on fait la somme (puisque c'est un effectif).
    Retourne un dict { epoch_num: { ... } }.
    """
    if not list_of_epoch_dicts:
        return {}

    # 1) Époques communes à tous les folds
    common_epochs = set(list_of_epoch_dicts[0].keys())
    for d in list_of_epoch_dicts[1:]:
        common_epochs = common_epochs.intersection(d.keys())

    aggregated = {}
    for ep in sorted(common_epochs):
        macro_f1_vals = []
        weighted_f1_vals = []
        test_loss_vals = []
        train_loss_vals = []
        recall_1_vals = []
        class_1_precision_vals = []
        class_1_f1_vals = []
        class_0_support_total = 0.0
        class_1_support_total = 0.0

        overfit_detected = False

        for fold_dict in list_of_epoch_dicts:
            ep_metrics = fold_dict[ep]

            # Vérif overfit
            if (ep_metrics["test_loss"] - ep_metrics["train_loss"]) > overfit_threshold:
                overfit_detected = True
                break

            # Accumuler
            macro_f1_vals.append(ep_metrics["macro_f1"])
            weighted_f1_vals.append(ep_metrics["weighted_f1"])
            test_loss_vals.append(ep_metrics["test_loss"])
            train_loss_vals.append(ep_metrics["train_loss"])
            recall_1_vals.append(ep_metrics["recall_1"])

            class_1_precision_vals.append(ep_metrics["class_1_precision"])
            class_1_f1_vals.append(ep_metrics["class_1_f1_score"])

            class_0_support_total += ep_metrics["class_0_support"]
            class_1_support_total += ep_metrics["class_1_support"]

        if overfit_detected:
            # On exclut complètement cette époque
            continue

        aggregated[ep] = {
            "macro_f1": np.mean(macro_f1_vals),
            "weighted_f1": np.mean(weighted_f1_vals),
            "test_loss": np.mean(test_loss_vals),
            "train_loss": np.mean(train_loss_vals),
            "recall_1": np.mean(recall_1_vals),
            "class_1_precision": np.mean(class_1_precision_vals),
            "class_1_f1_score": np.mean(class_1_f1_vals),
            "class_0_support": class_0_support_total,
            "class_1_support": class_1_support_total,
        }

    return aggregated

# --------------------------------------------------------------------------
#   Calcul d'un score global (pour prioriser l'époque)
# --------------------------------------------------------------------------
def compute_epoch_score(macro_f1, weighted_f1, recall_1):
    """
    Combine macro_f1, weighted_f1 et recall_1 en un seul score, selon
    alpha, beta, gamma.
    """
    return alpha * macro_f1 + beta * weighted_f1 + gamma * recall_1

# --------------------------------------------------------------------------
#   Sélection de la "meilleure" époque
# --------------------------------------------------------------------------
def select_best_epoch(epochs_metrics):
    """
    epochs_metrics : dict { epoch_num: { 'macro_f1', 'weighted_f1', 'recall_1', 'test_loss', ... } }
    Retourne (best_epoch, best_data) après calcul du score 
    et prise en compte test_loss/train_loss en cas d'ex-aequo.
    """
    best_epoch = None
    best_data = None
    best_score = -1.0

    for ep, vals in epochs_metrics.items():
        macro_f1 = vals["macro_f1"]
        weighted_f1 = vals["weighted_f1"]
        recall_1 = vals["recall_1"]
        test_loss = vals["test_loss"]
        train_loss = vals["train_loss"]

        if recall_1 < min_positive_recall:
            continue

        current_score = compute_epoch_score(macro_f1, weighted_f1, recall_1)

        if current_score > best_score:
            best_epoch = ep
            best_data = vals
            best_score = current_score
        elif abs(current_score - best_score) < 1e-9:
            # Ex æquo => compare test_loss, puis train_loss
            if test_loss < best_data["test_loss"]:
                best_epoch = ep
                best_data = vals
                best_score = current_score
            elif abs(test_loss - best_data["test_loss"]) < 1e-9:
                if train_loss < best_data["train_loss"]:
                    best_epoch = ep
                    best_data = vals
                    best_score = current_score

    if best_epoch is None:
        return None, None
    return best_epoch, best_data

# --------------------------------------------------------------------------
#                         MAIN : Boucle globale
# --------------------------------------------------------------------------
final_rows = []

for model_name, fold_logs in model_to_folds.items():
    # 1) Parser chaque log => on stocke :
    #    - un dict "epochs_data" dans une liste (pour agrégation cross-fold)
    #    - la distribution train/test
    fold_epoch_dicts = []
    train_class_0_counts = []
    train_class_1_counts = []
    test_class_0_counts = []
    test_class_1_counts = []

    for lf in fold_logs:
        path_log = os.path.join(log_output_dir, lf)
        parsed = parse_single_log(path_log)
        # On ne considère que s'il y a un "epochs_data"
        if "epochs_data" in parsed and parsed["epochs_data"]:
            fold_epoch_dicts.append(parsed["epochs_data"])
            train_class_0_counts.append(parsed.get("train_class_0_count", 0))
            train_class_1_counts.append(parsed.get("train_class_1_count", 0))
            test_class_0_counts.append(parsed.get("test_class_0_count", 0))
            test_class_1_counts.append(parsed.get("test_class_1_count", 0))

    # Si aucun log exploitable
    if not fold_epoch_dicts:
        continue

    # 2) Agréger si plusieurs folds, sinon on récupère direct
    if len(fold_epoch_dicts) == 1:
        single_dict = fold_epoch_dicts[0]
        aggregated_metrics = {}
        for ep, metrics in single_dict.items():
            # Vérif overfit
            if (metrics["test_loss"] - metrics["train_loss"]) > overfit_threshold:
                continue
            aggregated_metrics[ep] = {
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "test_loss": metrics["test_loss"],
                "train_loss": metrics["train_loss"],
                "recall_1": metrics["recall_1"],
                "class_1_precision": metrics["class_1_precision"],
                "class_1_f1_score": metrics["class_1_f1_score"],
                "class_0_support": metrics["class_0_support"],
                "class_1_support": metrics["class_1_support"],
            }
        # Distribution = celles du (seul) fold
        total_train_class_0_count = train_class_0_counts[0]
        total_train_class_1_count = train_class_1_counts[0]
        total_test_class_0_count = test_class_0_counts[0]
        total_test_class_1_count = test_class_1_counts[0]

    else:
        # Agrégation cross-fold
        aggregated_metrics = aggregate_folds_metrics(fold_epoch_dicts)
        # On somme la distribution sur les folds
        total_train_class_0_count = sum(train_class_0_counts)
        total_train_class_1_count = sum(train_class_1_counts)
        total_test_class_0_count = sum(test_class_0_counts)
        total_test_class_1_count = sum(test_class_1_counts)

    if not aggregated_metrics:
        # Aucune époque valide
        continue

    # 3) Sélection de la meilleure époque
    best_ep, best_vals = select_best_epoch(aggregated_metrics)
    if best_ep is None:
        continue

    # 4) Confection de la ligne CSV
    row = {
        "model_name": model_name,
        "best_epoch": best_ep,
        "score": compute_epoch_score(
            best_vals["macro_f1"],
            best_vals["weighted_f1"],
            best_vals["recall_1"],
        ),
        "macro_f1": best_vals["macro_f1"],
        "weighted_f1": best_vals["weighted_f1"],
        "test_loss": best_vals["test_loss"],
        "train_loss": best_vals["train_loss"],
        "class_1_precision": best_vals["class_1_precision"],
        "class_1_recall": best_vals["recall_1"],  # rappel de la classe 1
        "class_1_f1_score": best_vals["class_1_f1_score"],
        "class_1_support": best_vals["class_1_support"],
        "class_0_support": best_vals["class_0_support"],
        # Nouveaux champs : distribution train/test
        "train_class_0_count": total_train_class_0_count,
        "train_class_1_count": total_train_class_1_count,
        "test_class_0_count": total_test_class_0_count,
        "test_class_1_count": total_test_class_1_count,
    }
    final_rows.append(row)

# --------------------------------------------------------------------------
#  Export du DataFrame final en CSV
# --------------------------------------------------------------------------
df = pd.DataFrame(final_rows)
df.to_csv(csv_output_path, index=False)
print(f"[FIN] Résumé des métriques avancées sauvegardé dans : {csv_output_path}")
