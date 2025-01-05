import os
import pandas as pd
import spacy
from datetime import datetime
import locale

# Chemin relatif vers le dossier contenant le script
script_dir = os.path.dirname(__file__)

# Chemin relatif vers le fichier CSV dans le dossier Database
csv_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_database.csv')

# Chargement des modèles spaCy pour l'anglais et le français
nlp_fr = spacy.load('fr_dep_news_trf')
nlp_en = spacy.load('en_core_web_lg')

# Fonction pour compter le nombre de mots dans le texte
def count_words(text):
    return len(text.split())

# Fonction pour tokéniser et créer le contexte de 2 phrases
def tokenize_and_context(text, language):
    nlp = nlp_fr if language == 'FR' else nlp_en
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    contexts = []
    for i in range(len(sentences)):
        if i > 0:  # S'assurer qu'il y a une phrase précédente
            context = ' '.join(sentences[i-1:i+1])
            contexts.append(context)
        else:
            contexts.append(sentences[i])
    return contexts if contexts else ['']

locale.setlocale(locale.LC_TIME, 'fr_FR' if os.name != 'nt' else 'French_France')

def convert_date(date_str):
    # Liste des formats de date possibles
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d %b %Y', '%d %B %Y']
    
    for date_format in date_formats:
        try:
            date_obj = datetime.strptime(date_str, date_format)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    print(f"Erreur de format pour la date: {date_str}")
    return 'date : non-renseignée'

def verify_dates_format(dates):
    for date in dates:
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print(f"Date incorrecte détectée : {date}")
            return False
    return True

# Chargement du fichier CSV
df = pd.read_csv(csv_path)

# Ajout de la colonne 'words_count_updated'
df['words_count_updated'] = df['text'].apply(count_words)

# Conversion des dates
df['date'] = df['date'].apply(convert_date)

# Vérification des formats de date
if verify_dates_format(df['date']):
    print("Toutes les dates sont au format 'yyyy-mm-dd'.")
else:
    print("Des dates sont dans un format incorrect.")

# Création d'une nouvelle DataFrame pour les textes traités
processed_texts = []

doc_id = 0  # Initialisation de l'identifiant unique de document
for _, row in df.iterrows():
    doc_id += 1  # Incrémentation de l'identifiant pour chaque nouvel article
    contexts = tokenize_and_context(row['text'], row['language'])
    for sentence_id, context in enumerate(contexts):
        processed_texts.append({
            'doc_ID': doc_id,
            'sentence_id': sentence_id,
            'news_type': row['news_type'],
            'title': row['title'],
            'author': row['author'],
            'media': row['media'] if pd.notna(row['media']) else 'media : non-renseigné',
            'words_count': row['words_count'],
            'words_count_updated': row['words_count_updated'],
            'date': row['date'],
            'language': row['language'],
            'page_number': row['page_number'],
            'sentences': context
        })

        if sentence_id == 0 and doc_id <= 10:  # Limite d'impression pour les premiers documents
            print("Exemple d'entrée dans processed_texts:", processed_texts[-1])

processed_df = pd.DataFrame(processed_texts)

# Chemin relatif pour l'enregistrement du nouveau DataFrame
output_path = os.path.join(script_dir, '..', '..', 'Database', 'Database', 'CCF.media_processed_texts.csv')

# Enregistrement du nouveau dataframe
processed_df.to_csv(output_path, index=False, header=True)
