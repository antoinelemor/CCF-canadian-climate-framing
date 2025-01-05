import os
from striprtf.striprtf import rtf_to_text

def convert_and_combine_rtf_to_text(source_dir, target_file):
    files_processed = 0
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    with open(target_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(source_dir):
            if filename.endswith('.rtf') and not filename.startswith('~$'):
                try:
                    file_path = os.path.join(source_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        rtf_content = file.read()
                        text_content = rtf_to_text(rtf_content)
                        # Suppression du sommaire de la recherche
                        summary_start = text_content.find('Sommaire de la recherche')
                        if summary_start != -1:
                            text_content = text_content[:summary_start]
                    outfile.write(text_content + '\n\n')
                    files_processed += 1
                    print(f'Fichier traité : {filename}')
                except Exception as e:
                    print(f'Erreur lors du traitement du fichier {filename}: {e}')

    print(f'Nombre de fichiers traités : {files_processed}')


# Chemins relatifs ajustés
source_dir_rel = '../../Raw_data/Calgary_Herald'
target_dir_rel = '../../Raw_data/Calgary_Herald/clean_CH_data'

# Construction du chemin absolu basé sur le chemin du script
script_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(script_dir, source_dir_rel)
target_file = os.path.join(script_dir, target_dir_rel, 'combined_text.txt')

# Exécuter la fonction
convert_and_combine_rtf_to_text(source_dir, target_file)
