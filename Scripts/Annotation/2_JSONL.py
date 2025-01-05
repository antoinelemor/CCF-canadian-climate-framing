import os
import pandas as pd
import json
from sklearn.utils import shuffle

# Chemin relatif vers le dossier contenant le script
script_dir = os.path.dirname(__file__)

# Chemin relatif vers le fichier CSV source
csv_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_processed_texts.csv')

# Chemin du fichier JSONL existant pour éviter les doublons
existing_jsonl_path = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Annotated_sentences.jsonl')

# Charger les combinaisons doc_ID et sentence_id déjà utilisées dans le fichier JSONL existant
existing_entries = set()
if os.path.exists(existing_jsonl_path):
    with open(existing_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            doc_ID = entry['meta'].get('doc_ID')
            sentence_id = entry['meta'].get('sentence_id')
            if doc_ID and sentence_id:
                existing_entries.add((doc_ID, sentence_id))

# Chargement du fichier CSV
df = pd.read_csv(csv_path)

# Remplacement de 'English' par 'EN' dans la colonne 'language'
df['language'] = df['language'].replace('English', 'EN')

# Vérification que 'language' ne contient que deux valeurs possibles : 'FR' ou 'EN'
valeurs_possibles = {'FR', 'EN'}
if not df['language'].isin(valeurs_possibles).all():
    lignes_invalides = df[~df['language'].isin(valeurs_possibles)]
    print(f"Attention : Certaines lignes ont une valeur incorrecte pour 'language'. Lignes concernées :\n{lignes_invalides}")
else:
    print("La colonne 'language' contient uniquement des valeurs valides ('FR' ou 'EN').")

# Mise à jour de la colonne 'date' en fonction du 'doc_ID'
dates_specifiques = {
    16807: '2005-03-11',
    13366: '2005-11-05',
    9740: '2013-07-04',
    5187: '2002-11-27'
}

# Appliquer les dates spécifiées aux lignes correspondantes
df['date'] = df.apply(lambda row: dates_specifiques.get(row['doc_ID'], row['date']), axis=1)

# Filtrer les phrases déjà utilisées en croisant 'doc_ID' et 'sentence_id'
df_filtered = df[~df.apply(lambda row: (row['doc_ID'], row['sentence_id']) in existing_entries, axis=1)]

# Séparation du DataFrame en phrases françaises et anglaises
df_fr = df_filtered[df_filtered['language'] == 'FR']
df_en = df_filtered[df_filtered['language'] == 'EN']

# Mélange et sélection de 2000 phrases pour chaque langue
df_fr = shuffle(df_fr).head(2000)
df_en = shuffle(df_en).head(2000)

# Fonction pour créer les données JSONL à partir d'un DataFrame
def create_jsonl_data(df_subset):
    jsonl_data = []
    for _, row in df_subset.iterrows():
        if pd.notna(row['sentences']) and row['sentences'].strip():  # Vérifie que la colonne 'sentences' n'est pas vide
            jsonl_data.append({
                'text': row['sentences'],
                'label': [],  # Colonne label vide
                'meta': {
                    'media': row['media'] if pd.notna(row['media']) else 'media : non-renseigné',
                    'date': row['date'] if pd.notna(row['date']) else 'date : non-renseignée',
                    'title': row['title'] if pd.notna(row['title']) else 'title : non-renseigné',
                    'page number': row['page_number'] if pd.notna(row['page_number']) else 'page number : non-renseigné',
                    'sentence_id': row['sentence_id'] if pd.notna(row['sentence_id']) else 'sentence_id : non-renseigné',
                    'doc_ID': row['doc_ID'] if pd.notna(row['doc_ID']) else 'doc_ID : non-renseigné',
                    'language': row['language']
                }
            })
        else:
            print(f"Ligne ignorée : '{row}' car 'sentences' est manquant ou vide.")
    return jsonl_data

# Création des données JSONL pour le français et l'anglais
jsonl_data_fr = create_jsonl_data(df_fr)
jsonl_data_en = create_jsonl_data(df_en)

# Chemins pour les fichiers JSONL de sortie
jsonl_output_path_fr = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Sentences_to_annotate_FR.jsonl')
jsonl_output_path_en = os.path.join(script_dir, '..', '..', 'Database', 'Training_data', 'manual_annotations_JSONL', 'Sentences_to_annotate_EN.jsonl')

# Écriture des données dans les fichiers JSONL
with open(jsonl_output_path_fr, 'w', encoding='utf-8') as jsonl_file_fr:
    for entry in jsonl_data_fr:
        jsonl_file_fr.write(json.dumps(entry, ensure_ascii=False) + '\n')

with open(jsonl_output_path_en, 'w', encoding='utf-8') as jsonl_file_en:
    for entry in jsonl_data_en:
        jsonl_file_en.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Fichier JSONL en français créé avec succès : {jsonl_output_path_fr}")
print(f"Fichier JSONL en anglais créé avec succès : {jsonl_output_path_en}")
