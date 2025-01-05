import pandas as pd
import re
from unidecode import unidecode

# Chemins vers vos fichiers CSV
chemins = {
    'CH': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Calgary_Herald/clean_CH_data/articles_extraits.csv',
    'GM': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Global_and_Mail/clean_GM_data/articles_extraits.csv',
    'TS': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Toronto_Star/clean_TS_data/articles_extraits.csv',
    'NP': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/National_Post/clean_NP_data/articles_extraits.csv',
    'LP': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/La_Presse_ris/clean_LP_data/articles_extraits.csv',
    'LD': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Le_Devoir/clean_LD_data/articles_extraits.csv',
    'JdM': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Journal_de_Montreal/clean_JDM_data/articles_extraits.csv',
    'LPP': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/La_Presse_Plus/clean_LPP_data/articles_extraits.csv',
    'CHH': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Chronicle_Herald/Clean_Chronicle_Herald_data/articles_extraits.csv',
    'VS': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Vancouver_Sun/clean_VS_data/articles_extraits.csv',
    'WHD': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Whitehorse_Daily_Star/Clean_White_Horse_data/articles_extraits.csv',
    'AcN': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Acadie_Nouvelle/clean_Acadie_Nouvelle_data/articles_extraits.csv',
    'LeDr': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Le_Droit/Le_Droit_clean_data/articles_extraits.csv',
    'Tel': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Telegram/Telegram_clean_data/articles_extraits.csv',
    'EdJo': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Edmonton_Journal/ED_clean_data/articles_extraits.csv',
    'MoGa': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Montreal_Gazette/clean_Montreal_Gazette_data/articles_extraits.csv',
    'StPh': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Star_Phoenix/clean_Star_Phoenix_data/articles_extraits.csv',
    'TorSun': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Toronto_Sun/clean_Toronto_Sun_data/articles_extraits.csv',
    'TiCo': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Times_Colonist/Times_Colonist_clean_data/articles_extraits.csv',
    'WiFrPr': '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Raw_data/Winnipeg_Free_Press/Winnipeg_Free_Press_clean_data/articles_extraits.csv'
}

# Lire les fichiers CSV dans des DataFrames pandas
dataframes = {}
for key, chemin in chemins.items():
    try:
        df = pd.read_csv(chemin)
        # Modifier la colonne 'Média' pour certains DataFrames
        if key == 'WHD':
            df['Média'] = 'Whitehorse Daily Star'
        elif key == 'CHH':
            df['Média'] = 'Chronicle Herald'
        elif key == 'AcN':
            df['Média'] = 'Acadie Nouvelle'
        elif key == 'LeDr':
            df['Média'] = 'Le Droit'
        elif key == 'Tel':
            df['Média'] = 'The Telegram'
        elif key == 'EdJo':
            df['Média'] = 'Edmonton Journal'
        elif key == 'MoGa':
            df['Média'] = 'Montreal Gazette'
        elif key == 'StPh':
            df['Média'] = 'Star Phoenix'
        elif key == 'TorSun':
            df['Média'] = 'Toronto Sun'  # Correction du nom de 'Toronton Sun' à 'Toronto Sun'
        elif key == 'TiCo':
            df['Média'] = 'Times Colonist'
        elif key == 'WiFrPr':
            df['Média'] = 'Winnipeg Free Press'
        # Ajoutez d'autres modifications si nécessaire
        dataframes[key] = df
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {chemin} : {e}")

# Combiner les DataFrames en un seul
df_combiné = pd.concat(dataframes.values(), ignore_index=True)

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
    'le journal de montreal': 'FR',          # Correction ici
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

# Chemin où sauvegarder la base de données combinée
chemin_sauvegarde = '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Database/CCF.media_database.csv'

# Sauvegarder le DataFrame combiné dans un nouveau fichier CSV
df_combiné.to_csv(chemin_sauvegarde, index=False)

print("\nLa base de données a été créée et sauvegardée à l'emplacement suivant :", chemin_sauvegarde)
