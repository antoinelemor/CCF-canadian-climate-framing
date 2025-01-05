# Chargement des packages nécessaires
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyverse)
library(broom)
library(ggplot2)
library(tidyr)
library(dplyr)
library(forcats)
library(RColorBrewer)
library(tools) 



# Base path
import_data_path <- "/Volumes/CLIMATE.FRAME/Database"
export_path <- "/Volumes/CLIMATE.FRAME/Results"



## BASE DE DONNÉES FRANÇAISE ##

# Chargement des données 
data <- read.csv(file.path(import_data_path, "CCF.sample.annotated_texts_FR.csv"))

data <- data %>%
  mutate(language = if_else(is.na(language), 'FR', language))


data <- data %>%
  mutate(
    language = if_else(is.na(language), 'FR', language),
  )


## BASE DE DONNÉES ANGLAISE ##

data_EN <- read.csv(file.path(import_data_path, "CCF.sample.annotated_texts_EN.csv"))

data_EN <- data_EN %>%
  mutate(language = if_else(is.na(language), 'EN', language))


data_EN <- data_EN %>%
  mutate(
    language = if_else(is.na(language), 'EN', language),
  )


## BASE DE DONNÉES COMPLÈTE ##

# Définir une fonction pour convertir les réponses des catégories principales et spécifiques
convert_responses <- function(df) {
  cols_to_convert <- c('detect_location', 'detect_event', 'detect_messenger', 'detect_solutions', 
                       'detect_PBH', 'detect_ECO', 'detect_SECU', 'detect_JUST', 'detect_CULT', 
                       'detect_ENVT', 'detect_SCI', 'detect_risks', 'detect_EQUIPOS', 'detect_EQUINEG', 
                       'detect_EQUI_NEU_frame', 'detect_POL', 'event_1', 'event_2', 'event_3', 'event_4',
                       'messenger_1', 'messenger_2', 'messenger_3', 'messenger_4', 'messenger_5', 'messenger_6', 'messenger_7', 'messenger_8',
                       'solution_1', 'solution_2', 'PBH_1', 'PBH_2', 'PBH_3', 'PBH_4', 'ECO_1', 'ECO_2', 
                       'ECO_3', 'ECO_4', 'ECO_5', 'SECU_1', 'SECU_2', 'SECU_3', 'SECU_4', 'SECU_5', 'JUST_1', 'JUST_2', 
                       'JUST_3', 'JUST_4', 'CULT_1', 'CULT_2', 'CULT_3', 'CULT_4', 'ENVT_1', 'ENVT_2', 'SCI_1', 'SCI_2', 'POL_1', 'POL_2')
  
  df[cols_to_convert] <- lapply(df[cols_to_convert], function(x) ifelse(x %in% c('oui', 'yes'), 1, ifelse(x %in% c('non', 'no'), 0, NA)))
  return(df)
}

# Appliquer la fonction convert_responses aux dataframes
data <- convert_responses(data)
data_EN <- convert_responses(data_EN)

# Fusionner les dataframes
CCF.media_full_annotated <- bind_rows(data, data_EN)

# Ajouter des colonnes pour les réponses émotionnelles
CCF.media_full_annotated <- CCF.media_full_annotated %>%
  mutate(
    detect_negative = if_else(emotion %in% c('négatif', 'negative'), 1, 0),
    detect_positive = if_else(emotion %in% c('positif', 'positive'), 1, 0),
    detect_neutral = if_else(emotion %in% c('neutre', 'neutral'), 1, 0)
  )

# Supprimer la colonne originale 'emotion' si nécessaire
CCF.media_full_annotated <- select(CCF.media_full_annotated, -emotion)

# Chemin complet du fichier de sortie
output_file_path <- file.path(import_data_path, "CCF.fullsample.annotated_media.csv")

# Enregistrer le dataframe au format CSV
write.csv(CCF.media_full_annotated, file = output_file_path, row.names = FALSE)

# Enregistrer le dataframe light (sans phrases ni titres) au format CSV
CCF.media_full_annotated_without_context <- select(CCF.media_full_annotated, -context)
CCF.media_full_annotated_without_context <- select(CCF.media_full_annotated, -title)
output_file_path <- file.path(import_data_path, "CCF.lightsample.annotated_media.csv")
write.csv(CCF.media_full_annotated_without_context, file = output_file_path, row.names = FALSE)
