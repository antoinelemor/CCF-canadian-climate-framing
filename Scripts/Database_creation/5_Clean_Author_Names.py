import pandas as pd
import spacy

# Charger les modèles NLP pré-entraînés avec des pipelines NER pour le français et l'anglais
nlp_en = spacy.load("en_core_web_trf")  # Modèle de transformateur pour l'anglais
nlp_fr = spacy.load("fr_core_news_lg")  # Modèle de transformateur pour le français

# Chemin vers la base de données à nettoyer
chemin_db = '/Volumes/CLIMATE.FRAME/CLIMATE.FRAME/Database/Database/CCF.media_database.csv'

# Lire le fichier CSV dans un DataFrame
df = pd.read_csv(chemin_db)

# Fonction pour extraire les noms d'auteurs en utilisant le NER adapté à la langue
def extraire_auteur(auteur, langue):
    if langue == 'EN':
        doc = nlp_en(str(auteur))
    elif langue == 'FR':
        doc = nlp_fr(str(auteur))
    else:
        return auteur  # Retourner l'original si la langue est inconnue
    
    # Extraction des entités nommées de type PERSON
    personnes = [ent.text for ent in doc.ents if ent.label_ == "PER" or ent.label_ == "PERSON"]
    return ' '.join(personnes) if personnes else auteur

# Application de la fonction de nettoyage sur la colonne 'author' en fonction de la langue
df['author'] = df.apply(lambda row: extraire_auteur(row['author'], row['language']), axis=1)

# Sauvegarde des modifications dans le fichier CSV
df.to_csv(chemin_db, index=False)

print("Le nettoyage de la variable 'author' pour les deux langues est terminé et les modifications ont été sauvegardées.")
