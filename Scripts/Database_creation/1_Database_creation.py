import pandas as pd
import re
from unidecode import unidecode
from pathlib import Path

# Définir le répertoire de base (racine du projet)
base_dir = Path(__file__).resolve().parent.parent.parent  # Ajustez en fonction de la profondeur du script

# Chemins vers vos fichiers CSV relatifs au répertoire de base
chemins = {
    'CH': base_dir / 'Database/Raw_data/Calgary_Herald/clean_CH_data/articles_extraits.csv',
    'GM': base_dir / 'Database/Raw_data/Global_and_Mail/clean_GM_data/articles_extraits.csv',
    'TS': base_dir / 'Database/Raw_data/Toronto_Star/clean_TS_data/articles_extraits.csv',
    'NP': base_dir / 'Database/Raw_data/National_Post/clean_NP_data/articles_extraits.csv',
    'LP': base_dir / 'Database/Raw_data/La_Presse_ris/clean_LP_data/articles_extraits.csv',
    'LD': base_dir / 'Database/Raw_data/Le_Devoir/clean_LD_data/articles_extraits.csv',
    'JdM': base_dir / 'Database/Raw_data/Journal_de_Montreal/clean_JDM_data/articles_extraits.csv',
    'LPP': base_dir / 'Database/Raw_data/La_Presse_Plus/clean_LPP_data/articles_extraits.csv',
    'CHH': base_dir / 'Database/Raw_data/Chronicle_Herald/Clean_Chronicle_Herald_data/articles_extraits.csv',
    'VS': base_dir / 'Database/Raw_data/Vancouver_Sun/clean_VS_data/articles_extraits.csv',
    'WHD': base_dir / 'Database/Raw_data/Whitehorse_Daily_Star/Clean_White_Horse_data/articles_extraits.csv',
    'AcN': base_dir / 'Database/Raw_data/Acadie_Nouvelle/clean_Acadie_Nouvelle_data/articles_extraits.csv',
    'LeDr': base_dir / 'Database/Raw_data/Le_Droit/Le_Droit_clean_data/articles_extraits.csv',
    'Tel': base_dir / 'Database/Raw_data/Telegram/Telegram_clean_data/articles_extraits.csv',
    'EdJo': base_dir / 'Database/Raw_data/Edmonton_Journal/ED_clean_data/articles_extraits.csv',
    'MoGa': base_dir / 'Database/Raw_data/Montreal_Gazette/clean_Montreal_Gazette_data/articles_extraits.csv',
    'StPh': base_dir / 'Database/Raw_data/Star_Phoenix/clean_Star_Phoenix_data/articles_extraits.csv',
    'TorSun': base_dir / 'Database/Raw_data/Toronto_Sun/clean_Toronto_Sun_data/articles_extraits.csv',
    'TiCo': base_dir / 'Database/Raw_data/Times_Colonist/Times_Colonist_clean_data/articles_extraits.csv',
    'WiFrPr': base_dir / 'Database/Raw_data/Winnipeg_Free_Press/Winnipeg_Free_Press_clean_data/articles_extraits.csv'
}

# Dictionnaire de mapping des clés à leurs noms complets de médias
media_mapping = {
    'CH': 'Calgary Herald',
    'GM': 'Globe and Mail',
    'TS': 'Toronto Star',
    'NP': 'National Post',
    'LP': 'La Presse',
    'LD': 'Le Devoir',
    'JdM': 'Journal de Montreal',
    'LPP': 'La Presse Plus',
    'CHH': 'Chronicle Herald',
    'VS': 'Vancouver Sun',
    'WHD': 'Whitehorse Daily Star',
    'AcN': 'Acadie Nouvelle',
    'LeDr': 'Le Droit',
    'Tel': 'The Telegram',
    'EdJo': 'Edmonton Journal',
    'MoGa': 'Montreal Gazette',
    'StPh': 'Star Phoenix',
    'TorSun': 'Toronto Sun',
    'TiCo': 'Times Colonist',
    'WiFrPr': 'Winnipeg Free Press'
}

# Vérifier que tous les clés de chemins sont dans media_mapping
missing_keys = set(chemins.keys()) - set(media_mapping.keys())
if missing_keys:
    print(f"Attention : Les clés suivantes ne sont pas dans le dictionnaire de mapping des médias : {missing_keys}")
    # Vous pouvez décider d'ajouter les mappings manquants ou de gérer autrement
    # Par exemple :
    # media_mapping['NouvelleClé'] = 'Nom du Média'
    # Ou lever une erreur
    # raise ValueError(f"Mappings manquants pour les clés : {missing_keys}")

# Lire les fichiers CSV dans des DataFrames pandas
dataframes = {}
for key, chemin in chemins.items():
    try:
        df = pd.read_csv(chemin)
        # Assigner la colonne 'Média' en utilisant le dictionnaire de mapping
        media_nom = media_mapping.get(key, 'Média Inconnu')  # 'Média Inconnu' si la clé n'est pas trouvée
        df['Média'] = media_nom
        dataframes[key] = df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin} n'a pas été trouvé.")
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {chemin} est vide.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {chemin} : {e}")

# Combiner les DataFrames en un seul
if dataframes:
    df_combiné = pd.concat(dataframes.values(), ignore_index=True)
else:
    print("Aucun DataFrame à combiner. Vérifiez les erreurs précédentes.")
    exit(1)

# Renommer les colonnes selon les nouvelles spécifications
nouveaux_noms = {
    'Type de nouvelle': 'news_type',
    'Titre': 'title',
    'Auteur': 'author',
    'Média': 'media',
    'Nombre de mots': 'words_count',
    'Date': 'date',
    'Langue': 'language',
    'Numéro de page': 'page_number',
    'Texte': 'text'
}

df_combiné.rename(columns=nouveaux_noms, inplace=True)

# --------------------------------------------
# Étape : Assignation forcée de la langue
# --------------------------------------------

# Définir un dictionnaire de mapping des médias à la langue (avec clés normalisées)
mapping_language = {
    'calgary herald': 'EN',
    'globe and mail': 'EN',
    'toronto star': 'EN',
    'national post': 'EN',
    'chronicle herald': 'EN',
    'vancouver sun': 'EN',
    'whitehorse daily star': 'EN',
    'the telegram': 'EN',
    'edmonton journal': 'EN',
    'montreal gazette': 'EN',
    'star phoenix': 'EN',
    'toronto sun': 'EN',
    'times colonist': 'EN',
    'winnipeg free press': 'EN',
    'la presse': 'FR',
    'le devoir': 'FR',
    'journal de montreal': 'FR',         
    'la presse+': 'FR',
    'la presse plus': 'FR',
    'acadie nouvelle': 'FR',
    'le droit': 'FR'
}

# Fonction de normalisation des noms de médias
def normalize_media(name):
    if pd.isnull(name):
        return ''
    name = unidecode(name.lower().strip())
    name = re.sub(r'\s+', ' ', name)  # Remplacer multiples espaces par un seul
    return name

# Appliquer la normalisation
df_combiné['media_normalized'] = df_combiné['media'].apply(normalize_media)

# Assigner la langue en fonction du mapping
df_combiné['language'] = df_combiné['media_normalized'].map(mapping_language).fillna('Unknown')

# Optionnel : Vérifier les assignations
print("\n=== Exemple d'assignation de la langue ===")
print(df_combiné[['media', 'language']].drop_duplicates())

# Identifier et afficher les médias non assignés (optionnel)
medias_unknown = df_combiné[df_combiné['language'] == 'Unknown']['media'].unique()
if len(medias_unknown) > 0:
    print("\n=== Médias non assignés ===")
    for media in medias_unknown:
        print(media)

# Supprimer la colonne temporaire de normalisation
df_combiné.drop(columns=['media_normalized'], inplace=True)

# Chemin où sauvegarder la base de données combinée (chemin relatif)
chemin_sauvegarde = base_dir / 'Database/Database/CCF.media_database.csv'

# Sauvegarder le DataFrame combiné dans un nouveau fichier CSV
try:
    df_combiné.to_csv(chemin_sauvegarde, index=False)
    print("\nLa base de données a été créée et sauvegardée à l'emplacement suivant :", chemin_sauvegarde)
except Exception as e:
    print(f"Erreur lors de la sauvegarde du fichier {chemin_sauvegarde} : {e}")
