import pandas as pd
import re

def extraire_donnees_articles(chemin_fichier):
    donnees_articles = []
    lignes_article = []
    collecter = False  # Indique si on est en train de collecter les lignes d'un article

    with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            
            if ligne.startswith('Document'):
                if lignes_article:  # Si on a déjà collecté des lignes, traiter l'article actuel
                    traiter_article(lignes_article, donnees_articles)
                lignes_article = []  # Préparation pour le prochain article
                collecter = False  # Réinitialisation de la collecte
            elif ligne and not collecter:  # Si on trouve du texte après 'Document', commencer la collecte
                collecter = True
            
            if collecter:
                lignes_article.append(ligne)

    if lignes_article:  # Traitement du dernier article
        traiter_article(lignes_article, donnees_articles)

    return donnees_articles

def traiter_article(lignes_article, donnees_articles):
    # Trouver l'index de la première ligne vide qui sert de séparateur
    try:
        index_ligne_vide = lignes_article.index('')
    except ValueError:
        index_ligne_vide = None

    if index_ligne_vide is not None and index_ligne_vide > 1:
        # 'Type de nouvelle' contient toutes les lignes jusqu'à l'avant-dernière ligne avant la ligne vide
        type_nouvelle = ' '.join(lignes_article[:index_ligne_vide - 1])
        titre = lignes_article[index_ligne_vide - 1]  # Le titre est la dernière ligne avant la ligne vide
        auteur = lignes_article[index_ligne_vide + 1]  # L'auteur est la ligne suivant immédiatement la ligne vide
    else:
        type_nouvelle, titre, auteur = lignes_article[0], lignes_article[1], lignes_article[3]

    # Extraction des autres informations comme précédemment
    nombre_mots_match = re.search(r'(\d{1,3}(?:,\d{3})?) mots', ' '.join(lignes_article))
    nombre_mots = nombre_mots_match.group(1).replace(',', '') if nombre_mots_match else 'Inconnu'
    
    date_match = re.search(r'\d{1,2} \w+ \d{4}', ' '.join(lignes_article))
    date = date_match.group() if date_match else 'Inconnue'
    
    langue_match = re.search(r'(Anglais|Français)', ' '.join(lignes_article))
    langue = langue_match.group(1) if langue_match else 'Inconnue'
    
    indice_fin = next((i for i, ligne in enumerate(lignes_article) if "all rights reserved" in ligne.lower()), None)
    numero_page = lignes_article[indice_fin - 2] if indice_fin else 'Inconnu'
    texte_start_index = indice_fin + 1 if indice_fin else index_ligne_vide + 2 if index_ligne_vide else 4
    texte = ' '.join(lignes_article[texte_start_index:])

    donnees_articles.append({
        'Type de nouvelle': type_nouvelle,
        'Titre': titre,
        'Auteur': auteur,
        'Média': 'Globe and Mail',
        'Nombre de mots': nombre_mots,
        'Date': date,
        'Langue': langue,
        'Numéro de page': numero_page,
        'Texte': texte
    })


# Chemin vers le fichier de texte
chemin_fichier = '/Volumes/CLIMATE.FRAME/Raw_data/Global_and_Mail/clean_GM_data/combined_text.txt'

# Exécution de la fonction d'extraction
donnees_articles = extraire_donnees_articles(chemin_fichier)

# Conversion des données en DataFrame et sauvegarde en CSV
df_articles = pd.DataFrame(donnees_articles)
chemin_csv = '/Volumes/CLIMATE.FRAME/Raw_data/Global_and_Mail/clean_GM_data/articles_extraits.csv'
df_articles.to_csv(chemin_csv, index=False)

# Affichage des premières lignes du DataFrame pour vérification
df_articles.head(), chemin_csv