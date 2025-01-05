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







## DISTRIBUTION DES VARIABLES DANS LE TEMPS ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Transformation des données en format long pour une meilleure visualisation avec ggplot2
data_long <- data %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1")

ggplot(data_long, aes(x = variable, fill = presence)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution des variables 'detect'", x = "Variables 'detect'", y = "Nombre de phrases")

ggsave(filename = "Distribution_detect.pdf", path=export_path, width = 10, height = 8, units = "in")


## DISTRIBUTION DES VARIABLES DANS LE TEMPS ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Filtrer
pattern <- "_[0-9]+$" 
cols_of_interest <- names(data)[grepl(pattern, names(data))]

data_long <- data %>%
  pivot_longer(cols = cols_of_interest, names_to = "variable", values_to = "value") %>%
  filter(value == "1") 

# Visualiser
ggplot(data_long, aes(x = variable, fill = value)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution des sous variables",
       x = "Variables 'sub'",
       y = "Nombre de phrases")

ggsave(filename = "Distribution_sub.pdf", path=export_path, width = 10, height = 8, units = "in")






## PROPORTIONS DES DETECT ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Préparation des données avec total_sentences conservé
data_prepared <- data %>%
  group_by(doc_ID) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Calcul de la proportion de chaque thématique par rapport au nombre total de phrases par discours
data_long <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  group_by(doc_ID, variable) %>%
  summarise(count = n(), .groups = "keep") %>%
  ungroup() %>%
  left_join(data_prepared %>% select(doc_ID, total_sentences) %>% distinct(doc_ID, .keep_all = TRUE), by = "doc_ID") %>%
  mutate(proportion = count / total_sentences) %>%
  group_by(variable) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  arrange(mean_proportion) %>%
  mutate(variable = factor(variable, levels = unique(variable))) %>%
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = tools::toTitleCase(variable_clean))

# Assurer que la commande 'arrange' a bien été appliquée pour ordonner 'mean_proportion'
data_long <- data_long %>%
  arrange(mean_proportion)

# Mise à jour de 'variable_clean' pour être un facteur avec les niveaux dans l'ordre souhaité
data_long$variable_clean <- factor(data_long$variable_clean, levels = unique(data_long$variable_clean))

# Visualiser les proportions moyennes avec une seule couleur grise pour toutes les barres
ggplot(data_long, aes(x = variable_clean, y = mean_proportion)) +
  geom_bar(stat = "identity", fill = "grey") +  # Utilisation d'une couleur grise fixe pour toutes les barres
  theme_minimal() +
  labs(title = "Proportion moyenne du nombre de phrases positives pour les variables 'detect' par article",
       x = "Variables 'detect'",
       y = "Proportion moyenne") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ggsave(filename = "Proportions_detect_articles.pdf", path=export_path, width = 10, height = 8, units = "in")








# Charger les données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Préparation des données avec total_sentences conservé et distinction des langues
data_prepared <- data %>%
  group_by(doc_ID, language) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Calcul de la proportion de chaque thématique par rapport au nombre total de phrases par discours et par langue
data_long <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == 1) %>%
  group_by(doc_ID, variable, language) %>%
  summarise(count = n(), .groups = "keep") %>%
  ungroup() %>%
  left_join(data_prepared %>% select(doc_ID, language, total_sentences) %>% distinct(doc_ID, language, .keep_all = TRUE), by = c("doc_ID", "language")) %>%
  mutate(proportion = count / total_sentences) %>%
  group_by(variable, language) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  arrange(desc(mean_proportion)) %>%
  mutate(variable = factor(variable, levels = unique(variable))) %>%
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = tools::toTitleCase(variable_clean))

# Mise à jour de 'variable_clean' pour être un facteur avec les niveaux dans l'ordre souhaité
data_long$variable_clean <- factor(data_long$variable_clean, levels = unique(data_long$variable_clean))

# Visualiser les proportions moyennes avec distinction des langues
ggplot(data_long, aes(x = variable_clean, y = mean_proportion, fill = language)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("FR" = "#22427C", "EN" = "#ad1328")) +  # Définir des couleurs manuellement
  theme_minimal() +
  labs(title = "Proportion moyenne du nombre de phrases positives pour les variables 'detect' par article et par langue",
       x = "Variables 'detect'",
       y = "Proportion moyenne") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Proportions_detect_articles_séparées.pdf", path=export_path, width = 10, height = 8, units = "in")







## PROPORTIONS DES SUB ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Préparation des données avec total_sentences conservé
data_prepared <- data %>%
  group_by(doc_ID) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Filtering columns that end with patterns like _1, _2, _3, etc.
pattern <- "_[0-9]+$"  # Regex to match _<number>
cols_of_interest <- names(data)[grepl(pattern, names(data))]

# Pivot data to long format and calculate proportions
data_long <- data_prepared %>%
  pivot_longer(cols = cols_of_interest, names_to = "variable", values_to = "value") %>%
  filter(value == "1") %>%
  group_by(doc_ID, variable) %>%
  summarise(count = n(), .groups = "keep") %>%
  ungroup() %>%
  left_join(data_prepared %>% select(doc_ID, total_sentences) %>% distinct(doc_ID, .keep_all = TRUE), by = "doc_ID") %>%
  mutate(proportion = count / total_sentences) %>%
  group_by(variable) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  arrange(desc(mean_proportion)) %>%
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = toTitleCase(variable_clean)) %>%
  mutate(variable_clean = factor(variable_clean, levels = unique(variable_clean)))

# Visualiser
ggplot(data_long, aes(x = variable_clean, y = mean_proportion)) +
  geom_bar(stat = "identity", fill = "grey") +  # Fixed grey color for all bars
  theme_minimal() +
  labs(title = "Proportion moyenne du nombre de phrases positives pour les variables 'sub' par article",
       x = "Variables 'detect'",
       y = "Proportion moyenne") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Enregistrer
ggsave(filename = "Proportions_sub_articles.pdf", path=export_path, width = 10, height = 8, units = "in")








# Lire les données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Préparation des données avec total_sentences conservé
data_prepared <- data %>%
  group_by(doc_ID, language) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Filtrer
pattern <- "_[0-9]+$"  
cols_of_interest <- names(data)[grepl(pattern, names(data))]

# visualisation
data_long <- data_prepared %>%
  pivot_longer(cols = cols_of_interest, names_to = "variable", values_to = "value") %>%
  filter(value == 1) %>%
  group_by(doc_ID, variable, language) %>%
  summarise(count = n(), .groups = "drop") %>%
  left_join(data_prepared %>% select(doc_ID, language, total_sentences) %>% distinct(), by = c("doc_ID", "language")) %>%
  mutate(proportion = count / total_sentences) %>%
  group_by(variable, language) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  arrange(desc(mean_proportion)) %>%
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = toTitleCase(variable_clean)) %>%
  mutate(variable_clean = factor(variable_clean, levels = unique(variable_clean)))

# visualisation
ggplot(data_long, aes(x = variable_clean, y = mean_proportion, fill = language)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("FR" = "#22427C", "EN" = "#ad1328")) +  # Define colors manually for 'EN' and 'FR'
  theme_minimal() +
  labs(title = "Proportion moyenne du nombre de phrases positives pour les variables 'sub' par article, distinguée par la langue",
       x = "Variables 'detect'",
       y = "Proportion moyenne") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Adjust text angle for x-axis labels

# enregistrer
ggsave(filename = "Proportions_sub_articles_séparées.pdf", path=export_path, width = 10, height = 8, units = "in")








## THÉMATIQUES DANS LE TEMPS ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Format date
data$date <- as.Date(data$date, "%Y-%m-%d")

# Préparer les données du nb de phrases / art.
data_prepared <- data %>%
  group_by(doc_ID, year = year(date), media) %>%  # Using 'year' extracted from 'date'
  mutate(total_sentences = n()) %>%
  ungroup()

# Calcul des proportions
data_time_series <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  group_by(year, variable, media) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  mutate(proportion = count / total_sentences) %>%
  ungroup() %>%
  # Nettoyer les variables
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = tools::toTitleCase(variable_clean)) %>%
  group_by(year, variable_clean) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  ungroup()

# Visualisation
ggplot(data_time_series, aes(x = year, y = mean_proportion, color = variable_clean)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Évolution de la proportion moyenne de phrase positive par variables par année",
       x = "Année",
       y = "Proportion moyenne",
       color = "Variables") +
  theme(axis.text.x = element_text(angle=45, hjust=1), legend.position = "right")



# Visualisation avec ajustements pour les labels
ggplot(data_time_series, aes(x = year, y = mean_proportion, color = variable_clean)) +
  geom_line(alpha = 0.3) +  # Lignes originales plus transparentes
  geom_smooth(se = FALSE, method = "loess", span = 1) +  # Lissage
  geom_point(alpha = 0.3) +  # Points originaux plus transparents
  theme_minimal() +
  labs(title = "Évolution de la proportion moyenne de phrase positive par variables par année",
       x = "Année",
       y = "Proportion moyenne",
       color = "Variables") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
  guides(color = guide_legend(override.aes = list(alpha = 1)))  # Lignes pleines dans la légende
warnings()

ggsave(filename = "All_through_time.pdf", path=export_path, width = 10, height = 8, units = "in")









data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Format date
data$date <- as.Date(data$date, "%Y-%m-%d")

# Préparer les données avec le total des phrases / art. + date
data_prepared <- data %>%
  group_by(doc_ID, year = year(date), media) %>%  
  mutate(total_sentences = n()) %>%
  ungroup()

# Calcul de la proportion de chaque thématique par rapport au nombre total de phrases par discours, groupé par date
data_time_series <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  # Exclure les variables spécifiques ici
  filter(!(variable %in% c('detect_neutral', 'detect_location',  'detect_messenger',  'detect_EQUIPOS', 'detect_EQUINEG', 'detect_EQUI_NEU_frame', 'detect_POL', 'POL_1', 'POL_2'))) %>%
  group_by(year, variable, media) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  mutate(proportion = count / total_sentences) %>%
  ungroup() %>%
  # Nettoyer variables
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = tools::toTitleCase(variable_clean)) %>%
  group_by(year, variable_clean) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  ungroup()


# Visualisation avec lissage et étiquettes pour une thématique spécifique
ggplot(data_time_series, aes(x = year, y = mean_proportion, color = variable_clean)) +
  geom_line(alpha = 0.3) +  # Rendre les lignes originales plus transparentes
  geom_smooth(se = FALSE, method = "loess") +  # Appliquer le lissage
  geom_point(alpha = 0.3) +  # Points originaux plus transparents
  theme_minimal() +
  labs(title = "Évolution de la proportion moyenne de phrase positive par variables par année",
       x = "Date",
       y = "Proportion",
       color = "Variables") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
  guides(color = guide_legend(override.aes = list(alpha = 1)))  # Assurer que la légende affiche des lignes pleines

ggsave(filename = "Detect_1_through_time.pdf", path=export_path, width = 10, height = 8, units = "in")










# Calcul de la proportion de chaque thématique par rapport au nombre total de phrases par discours, groupé par date
data_time_series <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  # Exclure les variables spécifiques ici
  filter(!(variable %in% c('detect_CULT', 'detect_ECO', 'detect_ENVT', 'detect_JUST', 'detect_PBH', 'detect_SECU'))) %>%
  group_by(year, variable, media) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  mutate(proportion = count / total_sentences) %>%
  ungroup() %>%
  # Nettoyer variables
  mutate(variable_clean = gsub("^detect_", "", variable),
         variable_clean = gsub("_", " ", variable_clean),
         variable_clean = tools::toTitleCase(variable_clean)) %>%
  group_by(year, variable_clean) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  ungroup()


# Visualisation avec lissage et étiquettes pour une thématique spécifique
ggplot(data_time_series, aes(x = year, y = mean_proportion, color = variable_clean)) +
  geom_line(alpha = 0.3) +  # Rendre les lignes originales plus transparentes
  geom_smooth(se = FALSE, method = "loess", span = 1) +  # Appliquer le lissage
  geom_point(alpha = 0.3) +  # Points originaux plus transparents
  theme_minimal() +
  labs(title = "Évolution de la proportion moyenne de phrase positive par variables par année",
       x = "Date",
       y = "Proportion",
       color = "Thématique") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
  guides(color = guide_legend(override.aes = list(alpha = 1)))  # Assurer que la légende affiche des lignes pleines

ggsave(filename = "Detect_2_through_time.pdf", path=export_path, width = 10, height = 8, units = "in")






#### DEMANDE VISUALISATIONS ALIZÉE ####


#### TOUS LES CADRES DANS LE TEMPS ####

data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Chargement et préparation des données
data$date <- as.Date(data$date, "%Y-%m-%d")
data_prepared <- data %>%
  group_by(doc_ID, year = year(date), media) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Calculer les proportions
data_time_series <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  filter(variable %in% c('detect_ECO', 'detect_PBH', 'detect_ENVT', 'detect_SECU', 'detect_CULT', 'detect_SCI', 'detect_JUST', 'detect_POL')) %>%
  group_by(year, variable, media) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  mutate(proportion = count / total_sentences) %>%
  ungroup() %>%
  mutate(variable_clean = recode(variable,
                                 'detect_ECO' = 'Economy',
                                 'detect_PBH' = 'Health',
                                 'detect_ENVT' = 'Environment',
                                 'detect_SECU' = 'Security',
                                 'detect_CULT' = 'Culture',
                                 'detect_SCI' = 'Science',
                                 'detect_JUST' = 'Justice',
                                 'detect_POL' = 'Politics'))

# Grouper et calculer la moyenne des proportions
data_time_series <- data_time_series %>%
  group_by(year, variable_clean) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  ungroup()

# Trouver les points les plus hauts pour chaque année
label_data <- data_time_series %>%
  group_by(year) %>%
  top_n(1, mean_proportion) %>%
  ungroup()

# Créer le graphique
ggplot(data_time_series, aes(x = year, y = mean_proportion, group = variable_clean, color = variable_clean)) +
  geom_line() +
  geom_point() +
  geom_text(data = label_data, aes(label = year), vjust = -1, nudge_y = 0.01, check_overlap = TRUE, color = "black") +
  theme_minimal() +
  labs(title = "Evolution of the Average Proportion of Positive Sentences Segments per Article by Frame per Year",
       x = "Year",
       y = "Average Proportion per Article",
       color = "Frames") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right")

ggsave(filename = "Frame_through_time_without_smooth.pdf", path=export_path, width = 10, height = 8, units = "in")

# Visualisation avec lissage et étiquettes pour une thématique spécifique
ggplot(data_time_series, aes(x = year, y = mean_proportion, , group = variable_clean, color = variable_clean)) +
  geom_line(alpha = 0.3) +  # Rendre les lignes originales plus transparentes
  geom_smooth(se = FALSE, method = "loess", span = 0.45) +  # Appliquer le lissage
  geom_point(alpha = 0.3) +  # Points originaux plus transparents
  geom_text(data = label_data, aes(label = year), vjust = -1, nudge_y = 0.01, check_overlap = TRUE, color = "black") +  # Add slight nudge to avoid overlap, force labels to be black
  theme_minimal() +
  labs(title = "Evolution of the Average Proportion of Positive Sentences Segments per Article by Frame per Year",
       x = "Year",
       y = "Average Proportion per Article",
       color = "Frames") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
  guides(color = guide_legend(override.aes = list(alpha = 1)))  # Assurer que la légende affiche des lignes pleines

ggsave(filename = "Frame_through_time_with_smooth.pdf", path=export_path, width = 10, height = 8, units = "in")




#### TON DANS LE TEMPS ####

data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Chargement et préparation des données
data$date <- as.Date(data$date, "%Y-%m-%d")
data_prepared <- data %>%
  group_by(doc_ID, year = year(date), media) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Calculer les proportions
data_time_series <- data_prepared %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  filter(variable %in% c('detect_negative', 'detect_positive')) %>%
  group_by(year, variable, media) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  mutate(proportion = count / total_sentences) %>%
  ungroup() %>%
  mutate(variable_clean = recode(variable,
                                 'detect_negative' = 'Negative',
                                 'detect_positive' = 'Positive'))

# Grouper et calculer la moyenne des proportions
data_time_series <- data_time_series %>%
  group_by(year, variable_clean) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  ungroup()

# Trouver les points les plus hauts pour chaque année
label_data <- data_time_series %>%
  group_by(year) %>%
  top_n(1, mean_proportion) %>%
  ungroup()

# Créer le graphique
ggplot(data_time_series, aes(x = year, y = mean_proportion, group = variable_clean, color = variable_clean)) +
  geom_line() +
  geom_point() +
  geom_text(data = label_data, aes(label = year), vjust = -1, nudge_y = 0.01, check_overlap = TRUE, color = "black") +
  theme_minimal() +
  labs(title = "Evolution of the Average Proportion of Tone per Article per Year",
       x = "Year",
       y = "Average Proportion per Article",
       color = "Tone") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right")

ggsave(filename = "Tone_through_time_without_smooth.pdf", path=export_path, width = 10, height = 8, units = "in")

# Visualisation avec lissage et étiquettes pour une thématique spécifique
ggplot(data_time_series, aes(x = year, y = mean_proportion, , group = variable_clean, color = variable_clean)) +
  geom_line(alpha = 0.3) +  # Rendre les lignes originales plus transparentes
  geom_smooth(se = FALSE, method = "loess", span = 0.45) +  # Appliquer le lissage
  geom_point(alpha = 0.3) +  # Points originaux plus transparents
  geom_text(data = label_data, aes(label = year), vjust = -1, nudge_y = 0.01, check_overlap = TRUE, color = "black") +  # Add slight nudge to avoid overlap, force labels to be black
  theme_minimal() +
  labs(title = "Evolution of the Average Proportion of Tone per Article per Year",
       x = "Year",
       y = "Average Proportion per Article",
       color = "Frames") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right") +
  guides(color = guide_legend(override.aes = list(alpha = 1)))  # Assurer que la légende affiche des lignes pleines

ggsave(filename = "Tone_through_time_with_smooth.pdf", path=export_path, width = 10, height = 8, units = "in")

#### TON GRAPHIQUE À BARRES ####

# Count total sentences per article
data <- data %>%
  group_by(doc_ID) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Filter and pivot data for 'detect_positive' and 'detect_negative'
data_summary <- data %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  filter(variable %in% c('detect_negative', 'detect_positive'))

# Calculate total counts and proportions
data_summary <- data_summary %>%
  group_by(doc_ID, variable) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  group_by(variable) %>%
  summarise(
    mean_positive_phrases = mean(count / total_sentences, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(variable_clean = recode(variable,
                                 'detect_negative' = 'Negative',
                                 'detect_positive' = 'Positive'))

# Create the bar chart
ggplot(data_summary, aes(x = variable_clean, y = mean_positive_phrases, fill = variable_clean)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Average Proportion of Positive Sentences per Article by Tone",
       x = "Tone",
       y = "Average Proportion of Sentences") +
  theme_minimal() +
  scale_fill_manual(values = c("Negative" = "#E57373", "Positive" = "#87ceeb"))  # Adjusted colors

# Save the plot
ggsave(filename = "Average_Tone_Per_Article.pdf", path=export_path, width = 10, height = 8, units = "in")




#### CRÉATION DU DATAFRAME POUR LES MOYENNES PAR CADRES PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('detect_ECO', 'detect_PBH', 'detect_ENVT', 'detect_SECU', 'detect_CULT', 'detect_SCI', 'detect_JUST', 'detect_POL')

# Calculer la moyenne et le nombre total de phrases positives par document, puis par média et langue
CCF.media_frame_means <- data %>%
  group_by(doc_ID, media, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(across(all_of(variables), list(mean = ~mean(., na.rm = TRUE), N = ~sum(. > 0, na.rm = TRUE)), .names = "{.col}_{.fn}"), .groups = 'drop') %>%
  ungroup()

# Ajouter des résumés pour chaque langue
language_summary <- data %>%
  group_by(language) %>%
  summarise(across(all_of(variables), list(mean = ~mean(. > 0, na.rm = TRUE), N = ~sum(. > 0, na.rm = TRUE)), .names = "{.col}_{.fn}"), .groups = 'drop') %>%
  mutate(media = language) %>%
  mutate(across(contains("_mean"), ~round(., 2)))

# Combiner les résumés par média et les résumés par langue
CCF.media_frame_means <- bind_rows(CCF.media_frame_means, language_summary)

# Arrondir les valeurs moyennes à deux décimales pour les données combinées
CCF.media_frame_means <- CCF.media_frame_means %>%
  mutate(across(ends_with("_mean"), ~round(., 2)))

# Enregistrer le DataFrame en CSV avec des valeurs arrondies
write_csv(CCF.media_frame_means, file.path(export_path, "CCF_media_frame_means.csv"))



#### CRÉATION DU DATAFRAME POUR LES PROPORTIONS PAR CADRES PAR MÉDIA ####

# Charger les données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('detect_ECO', 'detect_PBH', 'detect_ENVT', 'detect_SECU', 'detect_CULT', 'detect_SCI', 'detect_JUST', 'detect_POL')

# Définir les variables à analyser
variables <- c('detect_ECO', 'detect_PBH', 'detect_ENVT', 'detect_SECU', 'detect_CULT', 'detect_SCI', 'detect_JUST', 'detect_POL')

# Calculer la proportion et le nombre total de documents ayant au moins une occurrence positive par document
CCF.media_frame_proportions <- data %>%
  group_by(doc_ID, media, language) %>%
  summarise(across(all_of(variables), ~sum(. > 0) > 0, .names = "{.col}_positive")) %>%
  group_by(media, language) %>%
  summarise(across(contains("_positive"), list(proportion = ~mean(. == TRUE, na.rm = TRUE), count = ~sum(. == TRUE, na.rm = TRUE)), .names = "{.col}_{.fn}"), .groups = 'drop') %>%
  ungroup()

# Ajouter des résumés pour chaque langue, en calculant également la proportion par article
language_summary <- data %>%
  group_by(doc_ID, language) %>%
  summarise(across(all_of(variables), ~sum(. > 0) > 0, .names = "{.col}_positive")) %>%
  group_by(language) %>%
  summarise(across(contains("_positive"), list(proportion = ~mean(. == TRUE, na.rm = TRUE), count = ~sum(. == TRUE, na.rm = TRUE)), .names = "{.col}_{.fn}"), .groups = 'drop') %>%
  mutate(media = language) %>%
  mutate(across(contains("_proportion"), ~round(., 2)))

# Combiner les résumés par média et les résumés par langue
CCF.media_frame_proportions <- bind_rows(CCF.media_frame_proportions, language_summary)

# Arrondir les proportions à deux décimales pour les données combinées
CCF.media_frame_proportions <- CCF.media_frame_proportions %>%
  mutate(across(ends_with("_proportion"), ~round(., 2)))

# Enregistrer le DataFrame en CSV
write_csv(CCF.media_frame_proportions, file.path(export_path, "CCF_media_frame_proportions.csv"))




### MOYENNES DES ÉVÈNEMENTS ####


# Calcul de la moyenne et de N pour detect_event sur l'ensemble des données
detect_event_data <- data %>%
  summarise(
    detect_event_mean = mean(detect_event, na.rm = TRUE),
    detect_event_N = sum(detect_event > 0, na.rm = TRUE)
  )

# Filtrer les données où detect_event est 1 pour les sous-événements
filtered_data <- data %>%
  filter(detect_event == 1) %>%
  select(doc_ID, starts_with("event"))

# Liste des variables d'événement à analyser pour les sous-événements
event_variables <- c('event_1', 'event_2', 'event_3', 'event_4')

# Calcul de la moyenne et de N pour chaque sous-variable d'événement
event_data <- filtered_data %>%
  pivot_longer(cols = event_variables, names_to = "event", values_to = "value") %>%
  group_by(event) %>%
  summarise(
    mean_positive_phrases = mean(value, na.rm = TRUE),
    N = sum(value > 0, na.rm = TRUE),
    .groups = 'drop'
  )

# Ajouter les données de detect_event
event_data <- event_data %>%
  add_row(event = 'detect_event', mean_positive_phrases = detect_event_data$detect_event_mean, N = detect_event_data$detect_event_N)

# Rename the events for the graph
event_data <- event_data %>%
  mutate(event = recode(event,
                        'event_1' = 'Natural Disasters', 
                        'event_2' = 'Conferences and Summits', 
                        'event_3' = 'Publications', 
                        'event_4' = 'Elections',
                        'detect_event' = 'Events'))

# Sort the data to order the events
event_data <- event_data %>%
  mutate(event = factor(event, levels = unique(event[order(-mean_positive_phrases)])))

# Force "Events" to be the last in the order
event_order <- c("Conferences and Summits", "Publications", "Natural Disasters",  "Elections", "Events")
event_data$event <- factor(event_data$event, levels = event_order)

# Create the bar chart with specified colors
p <- ggplot(event_data, aes(x = event, y = mean_positive_phrases, fill = event)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = N), vjust = -0.5) +
  labs(title = "Average Number of Positive Sentences per Article by Event",
       x = "Events",
       y = "Average Number of Positive Sentences") +
  theme_minimal() +
  scale_fill_manual(values = c("Natural Disasters" = "#add8e6",
                               "Conferences and Summits" = "#87ceeb",
                               "Publications" = "#4682b4",
                               "Elections" = "#0f52ba",
                               "Events" = "black"))  # Gradient of blues

# Add a dashed line before "Events"
last_non_event_position <- which(levels(event_data$event) == "Elections")
p + geom_vline(xintercept = last_non_event_position + 0.5, linetype = "dashed", color = "black")

# Enregistrer le graphique si nécessaire
ggsave(filename = "Proportion_events.pdf", path=export_path, width = 10, height = 6)




### MOYENNES DES SOLUTIONS ####


# Calculate the mean and total number of detections for 'detect_solutions' over the entire dataset
detect_solutions_data <- data %>%
  summarise(
    detect_solutions_mean = mean(detect_solutions, na.rm = TRUE),
    detect_solutions_N = sum(detect_solutions > 0, na.rm = TRUE)
  )

# Filter data where 'detect_solutions' is 1 for sub-solutions
filtered_data <- data %>%
  filter(detect_solutions == 1) %>%
  select(doc_ID, starts_with("solution"))

# List of solution variables to analyze
solution_variables <- c('solution_1', 'solution_2')

# Calculate the mean and total number for each sub-solution variable
solution_data <- filtered_data %>%
  pivot_longer(cols = solution_variables, names_to = "solution", values_to = "value") %>%
  group_by(solution) %>%
  summarise(
    mean_positive_phrases = mean(value, na.rm = TRUE),
    N = sum(value > 0, na.rm = TRUE),
    .groups = 'drop'
  )

# Add the data for 'detect_solutions'
solution_data <- solution_data %>%
  add_row(solution = 'detect_solutions', mean_positive_phrases = detect_solutions_data$detect_solutions_mean, N = detect_solutions_data$detect_solutions_N)

# Rename the solutions for the graph
solution_data <- solution_data %>%
  mutate(solution = recode(solution,
                           'solution_1' = 'Mitigation', 
                           'solution_2' = 'Adaptation',
                           'detect_solutions' = 'Solutions'))

# Manually set the order of the levels to place 'Solutions' on the right
solution_levels <- c("Mitigation", "Adaptation", "Solutions")  # Specific order

solution_data <- solution_data %>%
  mutate(solution = factor(solution, levels = solution_levels))

# Create the bar chart with specified colors
p <- ggplot(solution_data, aes(x = solution, y = mean_positive_phrases, fill = solution)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = N), vjust = -0.5) +
  labs(title = "Average Number of Positive Sentences per Article by Solution",
       x = "Solutions",
       y = "Average Number of Positive Sentences Segments") +
  theme_minimal() +
  scale_fill_manual(values = c("Mitigation" = "#0f52ba", "Adaptation" = "#4682b4", "Solutions" = "black"))  # Adjusted colors

# Add a dashed line before 'Solutions'
last_non_solution_position <- which(levels(solution_data$solution) == "Adaptation")
p + geom_vline(xintercept = last_non_solution_position + 0.5, linetype = "dashed", color = "black")


# Enregistrer le graphique si nécessaire
ggsave(filename = "Proportion_solutions.pdf", path=export_path, width = 10, height = 6)






#### CRÉATION DU DATAFRAME POUR LES MOYENNES MESSENGER PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('messenger_1', 'messenger_2', 'messenger_3', 'messenger_4', 'messenger_5', 'messenger_6')

# Filtrer les données pour garder seulement les lignes où detect_messenger est 1
filtered_data <- data %>%
  filter(detect_messenger == 1)

# Calculer les statistiques par média
messenger_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
messenger_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  messenger_stats_media %>% select(-media),
  messenger_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("messenger_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% messenger_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

# Écrire le résultat dans un fichier CSV
write_csv(combined_stats, file.path(export_path, "CCF.messenger_frame_means.csv"))








#### CRÉATION DU DATAFRAME POUR LES MOYENNES PBH PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('PBH_1', 'PBH_2', 'PBH_3', 'PBH_4')

# Filtrer les données pour garder seulement les lignes où detect_PBH est 1
filtered_data <- data %>%
  filter(detect_PBH == 1)

# Calculer les statistiques par média
PBH_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
PBH_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  PBH_stats_media %>% select(-media),
  PBH_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("PBH_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% PBH_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)


write_csv(combined_stats, file.path(export_path, "CCF.PBH_frame_means.csv"))






#### CRÉATION DU DATAFRAME POUR LES MOYENNES ECO PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('ECO_1', 'ECO_2', 'ECO_3', 'ECO_4')

# Filtrer les données pour garder seulement les lignes où detect_ECO est 1
filtered_data <- data %>%
  filter(detect_ECO == 1)

# Calculer les statistiques par média
ECO_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
ECO_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  ECO_stats_media %>% select(-media),
  ECO_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("ECO_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% ECO_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.ECO_frame_means.csv"))








#### CRÉATION DU DATAFRAME POUR LES MOYENNES ENVT PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('ENVT_1', 'ENVT_2')

# Filtrer les données pour garder seulement les lignes où detect_ENVT est 1
filtered_data <- data %>%
  filter(detect_ENVT == 1)

# Calculer les statistiques par média
ENVT_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
ENVT_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  ENVT_stats_media %>% select(-media),
  ENVT_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("ENVT_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% ENVT_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.ENVT_frame_means.csv"))








#### CRÉATION DU DATAFRAME POUR LES MOYENNES SECU PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

variables <- c('SECU_1', 'SECU_2', 'SECU_3', 'SECU_4', 'SECU_5')

# Filtrer les données pour garder seulement les lignes où detect_SECU est 1
filtered_data <- data %>%
  filter(detect_SECU == 1)

# Calculer les statistiques par média
SECU_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
SECU_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  SECU_stats_media %>% select(-media),
  SECU_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("SECU_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% SECU_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.SECU_frame_means.csv"))






#### CRÉATION DU DATAFRAME POUR LES MOYENNES CULT PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('CULT_1', 'CULT_2', 'CULT_3', 'CULT_4')

# Filtrer les données pour garder seulement les lignes où detect_CULT est 1
filtered_data <- data %>%
  filter(detect_CULT == 1)

# Calculer les statistiques par média
CULT_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
CULT_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  CULT_stats_media %>% select(-media),
  CULT_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("CULT_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% CULT_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)


write_csv(combined_stats, file.path(export_path, "CCF.CULT_frame_means.csv"))







#### CRÉATION DU DATAFRAME POUR LES MOYENNES SCI PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('SCI_1', 'SCI_2')

# Filtrer les données pour garder seulement les lignes où detect_SCI est 1
filtered_data <- data %>%
  filter(detect_SCI == 1)

# Calculer les statistiques par média
SCI_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
SCI_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  SCI_stats_media %>% select(-media),
  SCI_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("SCI_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% SCI_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.SCI_frame_means.csv"))







#### CRÉATION DU DATAFRAME POUR LES MOYENNES JUST PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('JUST_1', 'JUST_2', 'JUST_3', 'JUST_4')

# Filtrer les données pour garder seulement les lignes où detect_JUST est 1
filtered_data <- data %>%
  filter(detect_JUST == 1)

# Calculer les statistiques par média
JUST_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
JUST_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  JUST_stats_media %>% select(-media),
  JUST_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("JUST_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% JUST_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.JUST_frame_means.csv"))







#### CRÉATION DU DATAFRAME POUR LES MOYENNES POLI PAR MÉDIA ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('POL_1', 'POL_2')

# Filtrer les données pour garder seulement les lignes où detect_POL est 1
filtered_data <- data %>%
  filter(detect_POL == 1)

# Calculer les statistiques par média
POL_stats_media <- filtered_data %>%
  group_by(media, doc_ID, language) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(media, language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = media) %>%
  select(category, language, everything())

# Calculer les statistiques par langue
POL_stats_language <- filtered_data %>%
  group_by(language, doc_ID) %>%
  summarise(across(all_of(variables), ~mean(. > 0, na.rm = TRUE)), .groups = 'drop') %>%
  group_by(language) %>%
  summarise(
    across(all_of(variables), list(
      avg_positive_per_doc = ~mean(., na.rm = TRUE),
      prop_docs_with_positive = ~mean(. > 0, na.rm = TRUE),
      num_docs_with_positive = ~sum(. > 0, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  ) %>%
  mutate(category = language) %>%
  select(category, language, everything())

# Fusionner les données
combined_stats <- bind_rows(
  POL_stats_media %>% select(-media),
  POL_stats_language
)

# Arrondir les valeurs moyennes à deux décimales
combined_stats <- combined_stats %>%
  mutate(across(starts_with("POL_"), ~round(., 2)))

# Ordonner les données par 'category' pour afficher d'abord les médias, suivis des langues
combined_stats <- combined_stats %>%
  arrange(case_when(category %in% POL_stats_language$category ~ 1, TRUE ~ 0), category)

combined_stats <- combined_stats %>%
  select(-media)

write_csv(combined_stats, file.path(export_path, "CCF.POLI_frame_means.csv"))




#### DISTRIBUTION ECHANTILLON ####

data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))
#### Calculer le nombre d'articles par média grâce à 'doc_ID' ####
articles_par_media <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  group_by(media) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

# Calculer le nombre total d'articles
total_articles <- sum(articles_par_media$nombre_articles)

# Ajouter une colonne pour le pourcentage d'articles par média
articles_par_media <- articles_par_media %>%
  mutate(percentage_of_total = (nombre_articles / total_articles) * 100)

write_csv(articles_par_media, file.path(export_path, "CCF.SAMPLE.N_art_media.csv"))


#### Calculer le nombre d'articles par média grâce à 'doc_ID' par année ####
data$date <- as.Date(data$date)

articles_par_media_annee <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  mutate(annee = year(date)) %>%
  group_by(media, annee) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

write_csv(articles_par_media_annee, file.path(export_path, "CCF.SAMPLE.N_art_media_year.csv"))

#### Calculer le nombre d'articles par année grâce à 'doc_ID' ####
data$date <- as.Date(data$date)

articles_par_annee <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  mutate(annee = year(date)) %>%
  group_by(annee) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

# Remove NA values from the dataframe
articles_par_annee <- articles_par_annee[complete.cases(articles_par_annee), ]

write_csv(articles_par_annee, file.path(export_path, "CCF.SAMPLE.N_art_year.csv"))


ggplot(articles_par_annee, aes(x = as.factor(annee), y = nombre_articles)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Number of Articles per Year",
       x = "Year",
       y = "Number of Articles") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Distribution_sample_database.pdf", path=export_path, width = 10, height = 6)


#### DISTRIBUTION BASE DE DONNÉES TOTALE ####

data <- read.csv(file.path(import_data_path, "CCF.media_instructions_texts.csv"))

data$date <- as.Date(data$date)

articles_par_annee <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  mutate(annee = year(date)) %>%
  group_by(annee) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

# Remove NA values from the dataframe
articles_par_annee <- articles_par_annee[complete.cases(articles_par_annee), ]

# Write the updated dataframe to a CSV file
write_csv(articles_par_annee, file.path(export_path, "CCF.TOTAL_N_art_year.csv"))

# Create the bar plot
ggplot(articles_par_annee, aes(x = as.factor(annee), y = nombre_articles)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Number of Articles per Year",
       x = "Year",
       y = "Number of Articles") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Distribution_full_database.pdf", path=export_path, width = 10, height = 6)


#### Calculer le nombre d'articles par média grâce à 'doc_ID' TOTAL ####
articles_par_media <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  group_by(media) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

# Calculer le nombre total d'articles
total_articles <- sum(articles_par_media$nombre_articles)

# Ajouter une colonne pour le pourcentage d'articles par média
articles_par_media <- articles_par_media %>%
  mutate(percentage_of_total = (nombre_articles / total_articles) * 100)

write_csv(articles_par_media, file.path(export_path, "CCF.TOTAL_N_art_media.csv"))


#### Calculer le nombre d'articles par média grâce à 'doc_ID' par année TOTAL ####
data$date <- as.Date(data$date)

articles_par_media_annee <- data %>%
  distinct(doc_ID, .keep_all = TRUE) %>%
  mutate(annee = year(date)) %>%
  group_by(media, annee) %>%
  summarise(nombre_articles = n_distinct(doc_ID))

write_csv(articles_par_media_annee, file.path(export_path, "CCF.TOTAL_N_art_media_year.csv"))



#### GRAPHIQUES CPSA ####

#### Proportion Moyenne des cadrages ####

## PROPORTIONS DES DETECT ##
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Préparation des données avec total_sentences conservé
data_prepared <- data %>%
  group_by(doc_ID) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Filtrage et renommage des variables 'detect'
detect_vars <- c("detect_CULT", "detect_SECU", "detect_JUST", "detect_ENVT", 
                 "detect_PBH", "detect_ECO", "detect_SCI", "detect_POL")
new_names <- c("Culture", "Security", "Justice", "Environment", 
               "Public health", "Economy", "Science", "Politics")

data_long <- data_prepared %>%
  pivot_longer(cols = all_of(detect_vars), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  mutate(variable = factor(variable, levels = detect_vars, labels = new_names)) %>%
  group_by(doc_ID, variable) %>%
  summarise(count = n(), .groups = "keep") %>%
  ungroup() %>%
  left_join(data_prepared %>% select(doc_ID, total_sentences) %>% distinct(doc_ID, .keep_all = TRUE), by = "doc_ID") %>%
  mutate(proportion = count / total_sentences) %>%
  group_by(variable) %>%
  summarise(mean_proportion = mean(proportion, na.rm = TRUE)) %>%
  arrange(mean_proportion)

# Visualisation avec des couleurs neutres
ggplot(data_long, aes(x = variable, y = mean_proportion, fill = variable)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Culture" = "#B4C7E7", "Security" = "#D9EAD3", "Justice" = "#F4CCCC", 
                               "Environment" = "#EAD1DC", "Public health" = "#FFF2CC", 
                               "Economy" = "#DDEBF7", "Science" = "#FCE4D6", "Politics" = "#E6B8B7")) +
  theme_minimal() +
  labs(title = "Average Proportion of Positive Sentences per Article by Frames",
       x = "Frames",
       y = "Average Proportion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(filename = "Proportions_detect_articles.pdf", path=export_path, width = 10, height = 8, units = "in")



####  Graphiques cadres par journaux ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Définir les variables à analyser
variables <- c('detect_ECO', 'detect_PBH', 'detect_ENVT', 'detect_SECU', 'detect_CULT', 'detect_SCI', 'detect_JUST', 'detect_POL')

# Créer un dataframe pour les proportions par journal et par cadre
library(dplyr)
library(tidyr)
library(ggplot2)

# Calculer la proportion d'articles avec au moins trois phrases positives pour chaque cadre par média
data_long <- data %>%
  pivot_longer(cols = all_of(variables), names_to = "variable", values_to = "presence") %>%
  group_by(media, doc_ID, variable) %>%
  summarise(count = sum(presence == 1), .groups = 'drop') %>%
  mutate(threshold_met = count >= 3) %>%
  group_by(media, variable) %>%
  summarise(proportion = mean(threshold_met), .groups = 'drop') %>%
  mutate(variable = factor(variable, levels = variables, 
                           labels = c("Economy", "Public health", "Environment", "Security", 
                                      "Culture", "Science", "Justice", "Politics")))

# Visualiser les proportions par média et par cadre avec des couleurs neutres
ggplot(data_long, aes(x = media, y = proportion, fill = variable, label = scales::percent(proportion, accuracy = 0.1))) +
  geom_bar(stat = "identity", position = position_stack()) +
  geom_text(position = position_stack(vjust = 0.5), size = 3) +
  scale_fill_manual(values = c("Culture" = "#B4C7E7", "Security" = "#D9EAD3", "Justice" = "#F4CCCC", 
                               "Environment" = "#EAD1DC", "Public health" = "#FFF2CC", 
                               "Economy" = "#DDEBF7", "Science" = "#FCE4D6", "Politics" = "#E6B8B7")) +
  theme_minimal() +
  labs(title = "Proportion of Articles with at Least Three Positive Sentences per Frame by Media",
       x = "Media",
       y = "Proportion of Articles",
       fill = "Frames") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Enregistrer le graphique
ggsave(filename = "Proportions_detect_articles_by_media.pdf", path = export_path, width = 10, height = 8, units = "in")


#### Évènements au Canada ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Créer un dataframe pour les proportions de 'In Canada' et 'Elsewhere'
library(dplyr)
library(ggplot2)

# Calculer les proportions
data_summary <- data %>%
  group_by(doc_ID) %>%
  summarise(
    in_canada = sum(detect_location == 1) >= 1,
    elsewhere = sum(detect_location == 0) == n()
  ) %>%
  summarise(
    in_canada = mean(in_canada) * 100,
    elsewhere = mean(elsewhere) * 100
  ) %>%
  pivot_longer(cols = everything(), names_to = "location", values_to = "percentage") %>%
  mutate(location = factor(location, levels = c("in_canada", "elsewhere"), labels = c("In Canada", "Elsewhere")))

# Visualiser les proportions
ggplot(data_summary, aes(x = location, y = percentage, fill = location)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +
  scale_fill_manual(values = c("In Canada" = "#B4C7E7", "Elsewhere" = "#D9EAD3")) +
  theme_minimal() +
  labs(title = "Proportion of Articles by Location",
       x = "Location",
       y = "Percentage of Articles",
       fill = "Location") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Enregistrer le graphique
ggsave(filename = "Proportion_articles_by_location.pdf", path = export_path, width = 10, height = 8, units = "in")



#### Proportion solutions ####

# Chargement des données
data <- read.csv(file.path(import_data_path, "CCF.fullsample.annotated_media.csv"))

# Créer un dataframe pour les proportions des solutions
library(dplyr)
library(ggplot2)

# Calculer les proportions
data_summary <- data %>%
  group_by(doc_ID) %>%
  summarise(
    solutions = sum(detect_solutions == 1) >= 2,
    mitigation = sum(detect_solutions == 1 & solution_1 == 1) >= 2,
    adaptation = sum(detect_solutions == 1 & solution_2 == 1) >= 2
  ) %>%
  summarise(
    solutions = mean(solutions) * 100,
    mitigation = mean(mitigation) * 100,
    adaptation = mean(adaptation) * 100
  ) %>%
  pivot_longer(cols = everything(), names_to = "category", values_to = "percentage") %>%
  mutate(category = factor(category, levels = c("solutions", "mitigation", "adaptation"), 
                           labels = c("Solutions", "Mitigation", "Adaptation")))

# Créer le graphique
ggplot(data_summary, aes(x = category, y = percentage, fill = category)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5) +
  scale_fill_manual(values = c("Solutions" = "#B4C7E7", "Mitigation" = "#D9EAD3", "Adaptation" = "#F4CCCC")) +
  theme_minimal() +
  labs(title = "Proportion of Articles Discussing Climate Change Solutions",
       x = "Category",
       y = "Percentage of Articles",
       fill = "Category") +
  geom_vline(xintercept = 1.5, linetype = "dotted") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Enregistrer le graphique
ggsave(filename = "Proportion_articles_by_solutions.pdf", path = export_path, width = 10, height = 8, units = "in")


#### Ton graphique à barres ####

# Count total sentences per article
data <- data %>%
  group_by(doc_ID) %>%
  mutate(total_sentences = n()) %>%
  ungroup()

# Filter and pivot data for 'detect_positive' and 'detect_negative'
data_summary <- data %>%
  pivot_longer(cols = starts_with("detect_"), names_to = "variable", values_to = "presence") %>%
  filter(presence == "1") %>%
  filter(variable %in% c('detect_negative', 'detect_positive'))

# Calculate total counts and proportions
data_summary <- data_summary %>%
  group_by(doc_ID, variable) %>%
  summarise(count = n(), total_sentences = first(total_sentences), .groups = "drop") %>%
  group_by(variable) %>%
  summarise(
    mean_positive_phrases = mean(count / total_sentences, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(variable_clean = recode(variable,
                                 'detect_negative' = 'Negative',
                                 'detect_positive' = 'Positive'))

# Create the bar chart
ggplot(data_summary, aes(x = variable_clean, y = mean_positive_phrases, fill = variable_clean)) +
  geom_col(show.legend = FALSE) +
  labs(title = "Average Proportion of Positive Sentences per Article by Tone",
       x = "Tone",
       y = "Average Proportion of Sentences") +
  theme_minimal() +
  scale_fill_manual(values = c("Negative" = "#E57373", "Positive" = "#87ceeb"))  # Adjusted colors

# Save the plot
ggsave(filename = "Average_Tone_Per_Article.pdf", path=export_path, width = 10, height = 8, units = "in")

