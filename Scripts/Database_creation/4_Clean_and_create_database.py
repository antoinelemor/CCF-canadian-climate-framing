import pandas as pd
import re
from datetime import datetime
import locale

# Définir la locale pour le format de date en français
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
except locale.Error:
    print("La locale 'fr_FR.UTF-8' n'est pas disponible sur votre système.")
    # Vous pouvez définir une autre locale ou gérer l'erreur selon vos besoins

# Chemin vers la base de données à nettoyer
chemin_db = '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Database/CCF.media_database.csv'
chemin_export_csv = '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Database/Database_media_count.csv'

# Lire le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_db)

# --------------------------------------------
# Étape 1 : Inspection des formats de date
# --------------------------------------------

def identifier_formats_dates(series_dates):
    """
    Identifie les différents formats de date présents dans une série pandas.
    Retourne un ensemble des formats détectés.
    """
    formats_detectes = set()
    exemples_par_format = {}
    
    for date_str in series_dates.dropna().unique():
        date_str = str(date_str).strip()
        # Liste des formats à tester
        formats_a_tester = [
            '%d %B %Y',    # Exemple : 25 décembre 2023
            '%d/%m/%Y',    # Exemple : 25/12/2023
            '%Y-%m-%d',    # Exemple : 2023-12-25
            '%m/%d/%Y',    # Exemple : 12/25/2023
            '%d-%m-%Y',    # Exemple : 25-12-2023
            '%B %d, %Y',   # Exemple : décembre 25, 2023
            '%d %b %Y',    # Exemple : 25 déc. 2023
            '%d.%m.%Y',    # Exemple : 25.12.2023
        ]
        for fmt in formats_a_tester:
            try:
                datetime.strptime(date_str, fmt)
                formats_detectes.add(fmt)
                if fmt not in exemples_par_format:
                    exemples_par_format[fmt] = date_str
                break  # Arrêter après avoir trouvé un format correspondant
            except ValueError:
                continue
    return formats_detectes, exemples_par_format

formats_dates, exemples_formats = identifier_formats_dates(df['date'])

print("=== Formats de date détectés ===")
for fmt in formats_dates:
    exemple = exemples_formats.get(fmt, '')
    print(f"Format : {fmt} | Exemple : {exemple}")

# --------------------------------------------
# Étape 2 : Inspection des valeurs de 'language'
# --------------------------------------------

valeurs_language = df['language'].dropna().unique()

print("\n=== Valeurs uniques dans la colonne 'language' ===")
for valeur in valeurs_language:
    print(valeur)

# --------------------------------------------
# Étape 3 : Nettoyage de la base de données
# --------------------------------------------

# Suppression des lignes pour lesquelles 'text' contient moins de 100 mots
df = df[df['text'].apply(lambda x: len(str(x).split()) >= 100)]

# Suppression de la colonne 'URL'
df = df.drop(columns=['URL'])

# Filtrage basé sur la similarité des titres
def similarite_titres(titre1, titre2):
    titre1, titre2 = str(titre1), str(titre2)
    mots_titre1, mots_titre2 = set(titre1.lower().split()), set(titre2.lower().split())
    intersection, union = mots_titre1.intersection(mots_titre2), mots_titre1.union(mots_titre2)
    return len(intersection) / len(union) if union else 0

indices_a_conserver = []
titres_conserves = []

for i, titre in enumerate(df['title']):
    if all(similarite_titres(titre, titre_conserve) < 0.90 for titre_conserve in titres_conserves):
        indices_a_conserver.append(i)
        titres_conserves.append(titre)

# Filtrer le DataFrame pour ne conserver que les lignes uniques en termes de similarité
df = df.iloc[indices_a_conserver].reset_index(drop=True)

# Nettoyage de la colonne 'author'
mots_a_remplacer = [
    'From', ',', 'BY', 'Review by', 'Sources:', 
    'Compiled and edited by', 'By', 'by', 
    'SPECIAL TO THE STAR', 'Toronto Star'
]
pattern = '|'.join(map(re.escape, mots_a_remplacer))
df['author'] = df['author'].replace(pattern, '', regex=True)
df.loc[df['author'].str.contains('mots', case=False, na=False), 'author'] = ''

# Conversion des dates
def convertir_date(date_texte):
    formats_a_tester = [
        '%d %B %Y',    # Exemple : 14 avril 2009
        '%Y-%m-%d',    # Exemple : 1989-03-11
    ]
    for fmt in formats_a_tester:
        try:
            return datetime.strptime(date_texte, fmt).strftime('%m/%d/%Y')
        except ValueError:
            continue
    return date_texte  # Retourner la date originale si aucun format ne correspond

df['date'] = df['date'].apply(convertir_date)

# Sauvegarde des modifications dans le fichier CSV
df.to_csv(chemin_db, index=False)

# Fonction pour imprimer et exporter le nombre d'articles par média
def imprimer_et_exporter_nombre_articles_par_media(df):
    articles_par_media = df['media'].value_counts()
    total_articles = articles_par_media.sum()
    
    # Ajouter une ligne pour le total d'articles
    articles_par_media['Total'] = total_articles
    
    # Afficher les comptes
    print("\n=== Nombre d'articles par média ===")
    print(articles_par_media)
    
    # Exporter le nombre d'articles par média dans un fichier CSV
    articles_par_media.to_csv(chemin_export_csv, header=["Nombre d'articles"])

# Appel de la fonction pour afficher et exporter le nombre d'articles par média
imprimer_et_exporter_nombre_articles_par_media(df)

print(f"\nLa base de données a été nettoyée et sauvegardée, et le fichier '{chemin_export_csv}' a été créé.")
