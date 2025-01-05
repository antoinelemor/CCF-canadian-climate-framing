import os
import pandas as pd

def lire_ris(fichier_ris):
    """
    Lit un fichier .ris et extrait les informations requises pour chaque article.
    """
    with open(fichier_ris, 'r', encoding='utf-8') as fichier:
        lines = fichier.readlines()
    
    articles = []
    article_actuel = {}
    
    for line in lines:
        if line.startswith('TY  - '):
            if article_actuel:
                articles.append(article_actuel)
            article_actuel = {'Type de nouvelle': line[6:].strip(), 'Auteur': '', 'URL': ''}
        elif line.startswith('TI  - ') or line.startswith('T1  - '):
            article_actuel['Titre'] = line[6:].strip()
        elif line.startswith('AU  - '):
            if article_actuel['Auteur']:
                article_actuel['Auteur'] += '; ' + line[6:].strip()
            else:
                article_actuel['Auteur'] = line[6:].strip()
        elif line.startswith('JF  - '):
            article_actuel['Média'] = line[6:].strip()
        elif line.startswith('LA  - '):
            article_actuel['Langue'] = line[6:].strip()
        elif line.startswith('DA  - '):
            article_actuel['Date'] = line[6:].strip()
        elif line.startswith('SP  - '):
            article_actuel['Numéro de page'] = line[6:].strip()
        elif line.startswith('UR  - '):
            article_actuel['URL'] = line[6:].strip()
    if article_actuel:
        articles.append(article_actuel)
    
    return articles

# Fichiers .ris 
repertoire_ris = '/Volumes/CLIMATE.FRAME/Raw_data/Acadie_Nouvelle' 

# Compilation des données de tous les fichiers .ris
donnees_tous_les_fichiers = []

# Itérer sur chaque fichier .ris dans le répertoire spécifié
for fichier in os.listdir(repertoire_ris):
    if fichier.endswith('.ris'):
        chemin_complet = os.path.join(repertoire_ris, fichier)
        donnees_fichier = lire_ris(chemin_complet)
        donnees_tous_les_fichiers.extend(donnees_fichier)

# Créer un DataFrame avec les données compilées
df = pd.DataFrame(donnees_tous_les_fichiers)

# CSV de sortie 
chemin_csv = '/Volumes/CLIMATE.FRAME/Raw_data/Acadie_Nouvelle/clean_Acadie_Nouvelle_data/Clean_AN.csv'  
df.to_csv(chemin_csv, index=False, encoding='utf-8')

print(f'Fichier CSV créé avec succès. Chemin : {chemin_csv}')