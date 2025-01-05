import json
import sys
import os
import glob
import shutil  # <-- pour pouvoir supprimer des dossiers
import pandas as pd
import torch

from AugmentedSocialScientist.models import Camembert, Bert

# --------------------------------------------------------------------
# FONCTION POUR RÉCUPÉRER LE DEVICE (CUDA, MPS OU CPU)
# --------------------------------------------------------------------
def get_device():
    """
    Détecte le GPU si disponible (CUDA ou MPS), sinon bascule sur CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Utilisation du GPU (CUDA) pour les calculs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Utilisation du GPU (MPS) pour les calculs.")
    else:
        device = torch.device("cpu")
        print("Utilisation du CPU pour les calculs.")
    return device

# Debugging print pour vérifier l'exécution du script
print("Script started.")

# Récupère le chemin de base du script
base_path = os.path.dirname(os.path.abspath(__file__))

# Chemins (relatifs)
annotation_base_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "annotation_bases")
model_output_dir = os.path.join(base_path, "..", "..", "models")  # Point directement vers 'models'
log_output_dir = os.path.join(base_path, "..", "..", "Database", "Training_data", "Annotation_logs")
training_data_dir = os.path.join(base_path, "..", "..", "Database", "Training_data")  # Pour y enregistrer le CSV final

print(f"Model output directory: {model_output_dir}")  # Debug

# Vérifie l'existence du répertoire des logs
if not os.path.exists(log_output_dir):
    os.makedirs(log_output_dir)

# Vérifie l'existence du répertoire de sortie du modèle
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

# --------------------------------------------------------------------
# LOGGER CLASS POUR L'ENREGISTREMENT DES LOGS
# --------------------------------------------------------------------
class Logger(object):
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
# FONCTION DE CHARGEMENT JSONL -> PANDAS
# --------------------------------------------------------------------
def load_jsonl_to_dataframe(filepath):
    """Charge un fichier JSONL dans un DataFrame pandas."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --------------------------------------------------------------------
# FONCTION POUR DÉTERMINER LES FICHIERS INDISPENSABLES
# SELON LA LANGUE
# --------------------------------------------------------------------
def get_required_files_for_language(language: str):
    """
    Retourne la liste des fichiers indispensables pour un modèle de la langue donnée.
    - Pour FR (Camembert), c'est: ['config.json', 'pytorch_model.bin', 'sentencepiece.bpe.model']
    - Pour EN (Bert), c'est:      ['config.json', 'pytorch_model.bin', 'vocab.txt']
    """
    if language == "FR":
        return ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model"]
    else:
        return ["config.json", "pytorch_model.bin", "vocab.txt"]

# --------------------------------------------------------------------
# FONCTION POUR VÉRIFIER COMBIEN DE FICHIERS D'UN MODÈLE SONT PRÉSENTS
# POUR UNE LANGUE DONNÉE
# --------------------------------------------------------------------
def get_model_file_count(model_dir, language):
    """
    Retourne le nombre de fichiers requis présents dans 'model_dir',
    en se basant sur la langue pour déterminer lesquels sont requis.
    """
    required_files = get_required_files_for_language(language)
    present_count = 0

    if not os.path.exists(model_dir):
        # Le dossier n'existe même pas
        return 0

    for file_name in required_files:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            present_count += 1

    return present_count

# --------------------------------------------------------------------
# EXCEPTION PERSONNALISÉE POUR SAUTER L'ENTRAÎNEMENT
# --------------------------------------------------------------------
class SkipTrainingException(Exception):
    """Exception levée pour indiquer que l'entraînement doit être sauté."""
    pass

# --------------------------------------------------------------------
# FONCTION PRINCIPALE DE TRAINING
# --------------------------------------------------------------------
def train_models(base_dir, model_output_dir, log_output_dir):
    # Récupère le device (GPU ou CPU)
    device = get_device()

    # Pour le reporting global
    fully_trained_count = 0
    partial_count = 0
    not_started_count = 0
    skipped_count = 0  # Compteur pour les modèles sautés

    # Liste pour stocker les modèles non entraînés (pour cause d'annotations insuffisantes ou autres)
    # et leur distribution
    non_trained_models = []

    # ----------------------------------------------------------------
    # 1) RÉCUPÉRER LA LISTE DE TOUS LES MODÈLES ET DÉTERMINER LEUR ÉTAT
    # ----------------------------------------------------------------
    models_info = []

    for label in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path):
            continue

        # On traite pour chaque langue
        for language in ['EN', 'FR']:
            # Recherche des fichiers pour l'entraînement et la validation
            train_filepath_pattern = os.path.join(label_path, 'train', language, f"*train*_{language}.jsonl")
            test_filepath_pattern = os.path.join(label_path, 'validation', language, f"*validation*_{language}.jsonl")

            train_files = glob.glob(train_filepath_pattern)
            test_files = glob.glob(test_filepath_pattern)

            if not train_files:
                continue
            if not test_files:
                continue

            # Associe les fichiers d'entraînement aux fichiers de validation correspondants
            for train_file in train_files:
                base_name = os.path.basename(train_file).replace('_train_', '_validation_')
                matching_test_file = None
                for test_file in test_files:
                    if base_name in test_file:
                        matching_test_file = test_file
                        break

                if not matching_test_file:
                    continue

                # Détermine le nom du modèle (pour logs et répertoire)
                model_name = os.path.basename(train_file).replace('_train_', '_')

                # Le répertoire final du modèle : ex. "Cult_4_SUB_FR.jsonl.model"
                model_dir = os.path.join(model_output_dir, f"{model_name}.model")

                # Calcul du statut
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

                # Stocke pour l'étape d'entraînement
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
    # 2) AFFICHER UN RÉCAPITULATIF EN TERMINAL
    # ----------------------------------------------------------------
    print("===== RÉCAPITULATIF DE L'ÉTAT DES MODÈLES =====")
    print(f"Modèles entièrement entraînés : {fully_trained_count}")
    print(f"Modèles non démarrés : {not_started_count}")
    print(f"Modèles arrêtés en cours (partiels) : {partial_count}")
    print("===============================================")

    # ----------------------------------------------------------------
    # 3) BOUCLE D'ENTRAÎNEMENT POUR LES MODÈLES NON COMPLETS
    # ----------------------------------------------------------------
    for info in models_info:
        if info["status"] == "fully_trained":
            # On ignore les modèles déjà complets
            print(f"[INFO] Modèle déjà entièrement entraîné : {info['model_name']}")
            continue

        label = info["label"]
        language = info["language"]
        train_file = info["train_file"]
        test_file = info["test_file"]
        model_name = info["model_name"]
        model_dir = info["model_dir"]

        print(f"[TRAIN] Starting training for label: {label}, language: {language} -> {model_name}")

        # Instancie le modèle en fonction de la langue
        if language == 'FR':
            print("Instantiating Camembert model for French.")
            model = Camembert(device=device)
        else:
            print("Instantiating Bert model for English.")
            model = Bert(device=device)

        # Tente d'envoyer le modèle sur le device (si possible)
        try:
            model.to(device)
        except AttributeError:
            print("Attention : Votre classe de modèle ne supporte peut-être pas .to(device).")
            print("Veuillez vérifier l'API de vos classes Camembert / Bert.")

        # Configure le Logger
        log_filepath = os.path.join(log_output_dir, f"{model_name}_training_log.txt")
        print(f"Setting up logging to: {log_filepath}")
        logger = Logger(log_filepath)
        sys.stdout = logger
        print(f"[LOG] Logging started for {label} in {language}")

        scores = None  # Initialisation des scores
        train_label_counts = None
        test_label_counts = None

        try:
            # Chargement des données d'entraînement et de validation
            train_data = load_jsonl_to_dataframe(train_file)
            test_data = load_jsonl_to_dataframe(test_file)
            print(f"Data loaded successfully for label: {label}, language: {language}")

            if train_data.empty or test_data.empty:
                print(f"Training or test data is empty for {label} in {language}")
                raise ValueError("Training or test data is empty")

            # Distribution des labels
            train_label_counts = train_data['label'].value_counts()
            test_label_counts = test_data['label'].value_counts()
            print(f"Training label distribution for {label} in {language}:")
            print(train_label_counts)
            print(f"Validation label distribution for {label} in {language}:")
            print(test_label_counts)

            # Vérifie s'il y a des valeurs positives dans les labels
            train_has_positive = (train_data['label'] > 0).any()
            test_has_positive = (test_data['label'] > 0).any()

            if not train_has_positive or not test_has_positive:
                print(f"[SKIP] Les données d'entraînement ou de validation pour {label} en {language} contiennent uniquement des 0. Entraînement sauté.")
                skipped_count += 1
                raise SkipTrainingException("Données contenant uniquement des 0.")

            # Vérifie s'il y a assez d'annotations
            min_annotations = 4
            if len(train_data) < min_annotations or len(test_data) < min_annotations:
                print(f"Not enough annotations for {label} in {language}. Need at least {min_annotations} samples.")
                raise ValueError("Not enough annotations")

            # Prépare les DataLoader (ou équivalent) via la méthode encode() du modèle
            train_loader = model.encode(
                train_data.text.values,
                train_data.label.values.astype(int)  # Conversion explicite en int
            )
            test_loader = model.encode(
                test_data.text.values,
                test_data.label.values.astype(int)  # Conversion explicite en int
            )
            print(f"Data encoding completed for label: {label}, language: {language}")

            # Chemin relatif (par rapport à model_output_dir) pour la sauvegarde
            relative_model_output_path = f"{model_name}.model"
            print(f"Saving model to (relative path): {relative_model_output_path}")

            # Entraîne et sauvegarde le modèle
            scores = model.run_training(
                train_loader,
                test_loader,
                lr=5e-5,
                n_epochs=1,
                random_state=42,
                save_model_as=relative_model_output_path
            )
            print(f"Training completed successfully for {model_name}")

        except SkipTrainingException as ste:
            print(f"[INFO] {ste}")
            # On considère ce modèle comme non entraîné
            non_trained_models.append({
                "model_name": model_name,
                "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                "test_distribution": test_label_counts.to_dict() if test_label_counts is not None else {}
            })
            # Supprime le dossier du modèle s'il existe
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"Dossier supprimé pour le modèle non entraîné : {model_dir}")

        except Exception as e:
            print(f"Error during training for {model_name}: {e}")
            # On considère ce modèle comme non entraîné aussi (pour "toute autre raison")
            non_trained_models.append({
                "model_name": model_name,
                "train_distribution": train_label_counts.to_dict() if train_label_counts is not None else {},
                "test_distribution": test_label_counts.to_dict() if test_label_counts is not None else {}
            })
            # Supprime le dossier du modèle s'il existe
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"Dossier supprimé pour le modèle non entraîné : {model_dir}")

        finally:
            # Ferme le logger et restaure stdout
            sys.stdout = sys.__stdout__
            logger.close()
            print(f"[LOG] Logging closed for {label} in {language}")

        if scores is not None:
            print(f"[TRAIN] Training completed for {model_name}, scores: {scores}")
        else:
            # Si pas de scores, entraînement non effectué ou abandonné
            if train_label_counts is not None and test_label_counts is not None:
                print(f"[TRAIN] Training skipped or failed for {model_name} (see logs).")
            else:
                print(f"[TRAIN] Training not started for {model_name} (no data or exception before loading).")

    # ----------------------------------------------------------------
    # 4) CRÉATION DU FICHIER CSV POUR LES MODÈLES NON ENTRAÎNÉS
    # ----------------------------------------------------------------
    if non_trained_models:
        non_trained_csv_path = os.path.join(training_data_dir, "non_trained_models.csv")
        df_non_trained = pd.DataFrame(non_trained_models)
        df_non_trained.to_csv(non_trained_csv_path, index=False, encoding='utf-8')
        print(f"[INFO] Fichier 'non_trained_models.csv' créé dans : {non_trained_csv_path}")

    # ----------------------------------------------------------------
    # 5) AFFICHER UN RÉCAPITULATIF FINAL
    # ----------------------------------------------------------------
    print("===== RÉCAPITULATIF FINAL =====")
    print(f"Modèles entièrement entraînés : {fully_trained_count}")
    print(f"Modèles non démarrés : {not_started_count}")
    print(f"Modèles arrêtés en cours (partiels) : {partial_count}")
    print(f"Modèles sautés (données uniquement 0) : {skipped_count}")
    print("================================")

# --------------------------------------------------------------------
# DÉMARRAGE DU PROCESSUS D'ENTRAÎNEMENT
# --------------------------------------------------------------------
train_models(annotation_base_dir, model_output_dir, log_output_dir)

# Debugging print pour vérifier la fin de l'exécution
print("Script ended.")
