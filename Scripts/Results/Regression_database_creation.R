library(dplyr)
library(readr)
library(dplyr)
library(readr)
library(readxl)

# Chemin de base pour les importations
import_data_path <- "/Volumes/CLIMATE.FRAME/Database"
export_path <- "/Volumes/CLIMATE.FRAME/Results"

# Charger la base de données textuelle
db_file <- file.path(import_data_path, "CCF_media_full_annotated.csv")
db <- read_csv(db_file, col_types = cols())

# Conversion des dates en datetime pour harmoniser les formats
db$date <- as.Date(db$date)

# Calculer les agrégats avec la nouvelle variable 'evidence_full'
text_aggregates <- db %>%
  group_by(doc_ID, language, media, npn_ajuste, npn_normalized_ajuste, npn_normalized_global_adjusted_ajuste, date) %>%
  summarise(
    detect_event = mean(detect_event == 1, na.rm = TRUE) * 100,
    detect_location = mean(detect_location == 1, na.rm = TRUE) * 100,
    detect_solutions = mean(detect_solutions == 1, na.rm = TRUE) * 100,
    detect_PBH = mean(detect_PBH == 1, na.rm = TRUE) * 100,
    detect_ECO = mean(detect_ECO == 1, na.rm = TRUE) * 100,
    detect_SECU = mean(detect_SECU == 1, na.rm = TRUE) * 100,
    detect_JUST = mean(detect_JUST == 1, na.rm = TRUE) * 100,
    detect_CULT = mean(detect_CULT == 1, na.rm = TRUE) * 100,
    detect_ENVT = mean(detect_ENVT == 1, na.rm = TRUE) * 100,
    detect_SCI = mean(detect_SCI == 1, na.rm = TRUE) * 100,
    negative = mean(detect_negative == 1, na.rm = TRUE) * 100,
    positive = mean(detect_positive == 1, na.rm = TRUE) * 100,
    .groups = "drop" # drop the grouping for the resulting dataframe
  )


# Corriger l'erreur dans la ligne de commande pour enregistrer le fichier correctement
write_csv(text_aggregates, file.path(export_path, "CCF.media_regression_database.csv"))

