# Base path
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Database"
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Results"

# Importing the database 
input_file <- file.path(import_data_path, "CCF.media_regression_database.csv")
reg_data_daily <- read.csv(input_file, header = TRUE, sep=",")

# Packages
library(modelsummary)
library(flextable)
library(tidyverse)
library(officer)
library(knitr)
library(kableExtra)


## OLS MODELS ##

# Base path
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Database"
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Results"

# Importing the database 
input_file <- file.path(import_data_path, "CCF.media_regression_database.csv")
reg_data_daily <- read.csv(input_file, header = TRUE, sep=",")

# Models
models <- list()
models[['OLS1']] = lm(npn_normalized_global_adjusted_ajuste ~ detect_event + detect_location + detect_solutions + 
                        detect_PBH + detect_ECO + detect_SECU + detect_JUST + 
                        detect_CULT + detect_ENVT + detect_SCI + negative + lag(npn_normalized_global_adjusted_ajuste, 1), 
                      data = reg_data_daily)

# Updated Coefficient Map
cm <- c('(Intercept)' = '(Intercept)', 
        'lag(npn_normalized_global_adjusted_ajuste, 1)' = 'NPN - 1',
        'detect_event' = 'Event Detection',
        'detect_location' = 'Location Detection',
        'detect_solutions' = 'Solutions Detection',
        'detect_PBH' = 'Public Health Detection',
        'detect_ECO' = 'Economic Detection',
        'detect_SECU' = 'Security Detection',
        'detect_JUST' = 'Justice Detection',
        'detect_CULT' = 'Cultural Detection',
        'detect_ENVT' = 'Environmental Detection',
        'detect_SCI' = 'Scientific Detection',
        'negative' = 'Negative',
        'positive' = 'Positive')

# Caption for the table
cap <- 'Table 1. Effects of Detection Variables on Normalized NPN Adjustment'
tab<-modelsummary(models, output='flextable',  coef_map=cm, stars =TRUE, title=cap)

# Printing results
tab %>%autofit()

# Set the export file path for the regression table
table_file_name <- "CCF.media.results_OLS.docx"
table_full_path <- file.path(export_path, table_file_name)

# Create a Word document to store the table
doc <- read_docx() %>% 
  body_add_flextable(tab)

# Save the Word document
print(doc, target = table_full_path)








## OLS MODELS ##

# Base path
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Database"
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Results"

# Importing the database 
input_file <- file.path(import_data_path, "CCF.media_regression_database.csv")
reg_data_daily <- read.csv(input_file, header = TRUE, sep=",")

# Filter for rows where 'language' is 'FR'
reg_data_daily <- reg_data_daily %>%
  filter(language == "FR")

# Models
models <- list()
models[['OLS1']] = lm(npn_normalized_global_adjusted_ajuste ~ detect_event + detect_location + detect_solutions + 
                        detect_PBH + detect_ECO + detect_SECU + detect_JUST + 
                        detect_CULT + detect_ENVT + detect_SCI + negative + lag(npn_normalized_global_adjusted_ajuste, 1), 
                      data = reg_data_daily)

# Updated Coefficient Map
cm <- c('(Intercept)' = '(Intercept)', 
        'lag(npn_normalized_global_adjusted_ajuste, 1)' = 'NPN - 1',
        'detect_event' = 'Event Detection',
        'detect_location' = 'Location Detection',
        'detect_solutions' = 'Solutions Detection',
        'detect_PBH' = 'Public Health Detection',
        'detect_ECO' = 'Economic Detection',
        'detect_SECU' = 'Security Detection',
        'detect_JUST' = 'Justice Detection',
        'detect_CULT' = 'Cultural Detection',
        'detect_ENVT' = 'Environmental Detection',
        'detect_SCI' = 'Scientific Detection',
        'negative' = 'Negative',
        'positive' = 'Positive')

# Caption for the table
cap <- 'Table 1. Effects of Detection Variables on Normalized NPN Adjustment'
tab<-modelsummary(models, output='flextable',  coef_map=cm, stars =TRUE, title=cap)

# Printing results
tab %>%autofit()

# Set the export file path for the regression table
table_file_name <- "CCF.media.results_OLS_FR.docx"
table_full_path <- file.path(export_path, table_file_name)

# Create a Word document to store the table
doc <- read_docx() %>% 
  body_add_flextable(tab)

# Save the Word document
print(doc, target = table_full_path)





## OLS MODELS ##

# Base path
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Database"
import_data_path <- "/Users/antoine/Documents/GitHub/CLIMATE.FRAME/Data/Results"

# Importing the database 
input_file <- file.path(import_data_path, "CCF.media_regression_database.csv")
reg_data_daily <- read.csv(input_file, header = TRUE, sep=",")

# Filter for rows where 'language' is 'FR'
reg_data_daily <- reg_data_daily %>%
  filter(language == "EN")

# Models
models <- list()
models[['OLS1']] = lm(npn_normalized_global_adjusted_ajuste ~ detect_event + detect_location + detect_solutions + 
                        detect_PBH + detect_ECO + detect_SECU + detect_JUST + 
                        detect_CULT + detect_ENVT + detect_SCI + negative + lag(npn_normalized_global_adjusted_ajuste, 1), 
                      data = reg_data_daily)

# Updated Coefficient Map
cm <- c('(Intercept)' = '(Intercept)', 
        'lag(npn_normalized_global_adjusted_ajuste, 1)' = 'NPN - 1',
        'detect_event' = 'Event Detection',
        'detect_location' = 'Location Detection',
        'detect_solutions' = 'Solutions Detection',
        'detect_PBH' = 'Public Health Detection',
        'detect_ECO' = 'Economic Detection',
        'detect_SECU' = 'Security Detection',
        'detect_JUST' = 'Justice Detection',
        'detect_CULT' = 'Cultural Detection',
        'detect_ENVT' = 'Environmental Detection',
        'detect_SCI' = 'Scientific Detection',
        'negative' = 'Negative',
        'positive' = 'Positive')

# Caption for the table
cap <- 'Table 1. Effects of Detection Variables on Normalized NPN Adjustment'
tab<-modelsummary(models, output='flextable',  coef_map=cm, stars =TRUE, title=cap)

# Printing results
tab %>%autofit()

# Set the export file path for the regression table
table_file_name <- "CCF.media.results_OLS_EN.docx"
table_full_path <- file.path(export_path, table_file_name)

# Create a Word document to store the table
doc <- read_docx() %>% 
  body_add_flextable(tab)

# Save the Word document
print(doc, target = table_full_path)