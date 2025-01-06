# CCF-Canadian-Climate-Framing

## Introduction

Welcome to the **CCF-Canadian-Climate-Framing** repository. This project is dedicated to studying the media covering of climate change in the Canadian press. This project aims to understand how climate change narratives are constructed and communicated to the public since the first media article published on the subject in 1988. This repository contains all the scripts, data processing tools, and machine learning models necessary for conducting this study.

### The database

This repository includes a newly compiled database of climate change articles from 20 major Canadian newspapers (_non available in plain text at this time for copyright reasons_). The table below shows the distribution of articles per newspaper (_after filtering and preprocessing_).

| Media                     | Number of Articles |
|---------------------------|--------------------:|
| Toronto Star              | 2523              |
| Globe and Mail            | 1588              |
| Vancouver Sun             | 1456              |
| Edmonton Journal          | 1179              |
| Le Devoir                 | 1026              |
| National Post             | 1005              |
| Calgary Herald            | 1003              |
| Whitehorse Daily Star     | 970               |
| Montreal Gazette          | 900               |
| Chronicle Herald          | 866               |
| The Telegram              | 706               |
| Times Colonist            | 652               |
| La Presse Plus            | 641               |
| La Presse                 | 596               |
| Winnipeg Free Press       | 451               |
| Acadie Nouvelle           | 420               |
| Star Phoenix              | 342               |
| Le Droit                  | 332               |
| Toronto Sun               | 262               |
| Journal de Montreal       | 222               |
| **Total**                 | **17140**         |

## Table of Contents
- [Introduction](#introduction)
- [Members of the Project](#members-of-the-project)
- [Project Objectives](#project-objectives)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
  - [Database Creation Scripts](#database-creation-scripts)
    - [1_Database_Creation.py](#1_database_creationpy)
    - [2_Clean_and_Create_Database.py](#2_clean_and_create_databasepy)
  - [Annotation Scripts](#annotation-scripts)
    - [1_Preprocess.py](#1_preprocesspy)
    - [2_JSONL.py](#2_jsonlpy)
    - [3_Manual_annotations.py](#3_manual_annotationspy)
    - [4_JSONL_for_training.py](#4_jsonl_for_trainingpy)
    - [5_Training.py](#5_trainingpy)
    - [5bis_Training_best_model.py](#5bis_training_best_modelpy)
    - [6_Compile_annotations_logs.py](#6_compile_annotations_logspy)
    - [7_Annotation.py](#7_annotationpy)
    - [8_JSONL_for_Recheck.py](#8_jsonl_for_recheckpy)
- [Dependencies](#dependencies)
- [Contact](#contact)

## Members of the Project

- [**Alizée Pillod**, Université de Montréal](https://pol.umontreal.ca/repertoire-departement/professeurs/professeur/in/in35292/sg/Aliz%C3%A9e%20Pillod/)
- [**Antoine Lemor**, Université de Montréal](https://antoinelemor.github.io/)
- [**Matthew Taylor**, Université de Montréal](https://www.chairedemocratie.com/fr/members/taylor-matthew/)


## Project Objectives

The primary objectives of this project are:

1. **Data Collection and Preprocessing:** Gather media articles from Canadian press sources and preprocess the data for analysis.
2. **Manual Annotatio and Model Training:** Manually annotate sentences within the articles to identify key elements and train machine learning models to automatically detect and classify these elements in the articles/
3. **Evaluation and Refinement:** Evaluate models' performance, and ensure high accuracy in classifications.

## Repository Structure

```
CCF-Canadian-Climate-Framing/
├── Data/
│   ├── Database/
│   │   ├── Database/
│   │   │   ├── CCF.media_database.csv _absent from the repository due to file size_
│   │   │   └── CCF.media_processed_texts.csv _absent from the repository due to file size_
│   │   │   └── CCF.media_processed_texts_annotated.csv _absent from the repository due to file size_
│   │   │   └── Database_media_count.csv 
│   │   └── Training_data/
│   │       ├── manual_annotations_JSONL/ _excluded until our first publication_
│   │       │   ├── Annotated_sentences.jsonl _excluded until our first publication_
│   │       │   ├── label_config.json _excluded until our first publication_
│   │       │   ├── sentences_to_annotate_EN.jsonl _excluded until our first publication_
│   │       │   └── sentences_to_annotate_FR.jsonl _excluded until our first publication_
│   │       ├── annotation_bases/ _excluded until our first publication_
│   │       ├── Annotation_logs/
│   │       └── training_database_metrics.csv _contain the distribution of all annotations by language and labels_
│   │       └── model_metrics_summary_advanced.csv _contain the metrics of the best selected model_
│   │       └── non_trained_models.csv _contain the non-trained model due to a lack of annotations_
│   │       └── manual_annotations_metrics.csv _contain the distribution of instances among lables in the training and test databases_
├── Models/
│   ├── _Sorted by label names (absent from the repository due to file size)_
├── Scripts/
│   ├── Database_creation/
│   │   ├── 1_Database_Creation.py
│   │   └── 2_Clean_and_Create_Database.py
│   └── Annotation/
│       ├── 1_Preprocess.py
│       ├── 2_JSONL.py
│       ├── 3_Manual_annotations.py
│       ├── 4_JSONL_for_training.py
│       ├── 5_Training.py
│       ├── 5bis_Training_best_model.py
│       ├── 6_Compile_annotations_logs.py
│       ├── 7_Annotation.py
│       └── 8_JSONL_for_Recheck.py
├── 

README.md

└── requirements.txt
```

**Note:** The `Database_creation` scripts are based on articles that have been scraped and aggregated (codes used for this purpose were not included in this repo) into various `.txt` files within the `Raw_data` folder. Due to copyright restrictions, the `Raw_data` folder is not currently available.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CCF-Canadian-Climate-Framing.git
   cd CCF-Canadian-Climate-Framing
   ```

2. **Set Up a Virtual Environment**
   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy Models**
   ```bash
   python -m spacy download fr_dep_news_trf
   python -m spacy download en_core_web_lg
   ```

## Usage

The project is organized into several scripts, each responsible for different aspects of data processing and analysis. Below is an overview of how to use each script.

### Database Creation Scripts

These scripts are responsible for creating and cleaning the database from scraped articles. **Note:** The `Database_creation` scripts are based on articles that have been scraped and aggregated (codes used for this purpose were not included in this repo) into various `.txt` files within the `Raw_data` folder. Due to copyright restrictions, the `Raw_data` folder is not currently available.

1. **Database Creation**
   ```bash
   python Scripts/Database_creation/1_Database_Creation.py
   ```

2. **Clean and Create Database**
   ```bash
   python Scripts/Database_creation/2_Clean_and_Create_Database.py
   ```

### Annotation Scripts

These scripts handle data preprocessing, manual annotations, model training, and annotation processes.

1. **Preprocess Data**
   ```bash
   python Scripts/Annotation/1_Preprocess.py
   ```

2. **Generate JSONL Files**
   ```bash
   python Scripts/Annotation/2_JSONL.py
   ```

3. **Manual Annotations**
   ```bash
   python Scripts/Annotation/3_Manual_annotations.py
   ```

4. **Prepare JSONL for Training**
   ```bash
   python Scripts/Annotation/4_JSONL_for_training.py
   ```

5. **Train Models**
   ```bash
   python Scripts/Annotation/5_Training.py
   ```

6. **Train Best Models**
   ```bash
   python Scripts/Annotation/5bis_Training_best_model.py
   ```

7. **Compile Annotation Logs**
   ```bash
   python Scripts/Annotation/6_Compile_annotations_logs.py
   ```

8. **Annotation Process**
   ```bash
   python Scripts/Annotation/7_Annotation.py
   ```

9. **Generate JSONL for Rechecking**
    ```bash
    python Scripts/Annotation/8_JSONL_for_Recheck.py
    ```

## Scripts Overview

### Database Creation Scripts

#### 1_Database_Creation.py

**Purpose:**  
Creates a unified media database by aggregating articles from various Canadian press sources. This script reads cleaned article data from multiple CSV files, maps media codes to their full names, and consolidates the data into a single CSV file.

**Key Features:**
- Defines paths to raw article CSV files from different media outlets.
- Maps media codes to their full media names.
- Reads each CSV file into a pandas DataFrame and assigns the corresponding media name.
- Combines all DataFrames into a single consolidated DataFrame.
- Renames columns to standardized names.
- Assigns language labels based on media sources.
- Saves the combined database to `CCF.media_database.csv`.

**Dependencies:**  
`pandas`, `re`, `unidecode`, `pathlib`

#### 2_Clean_and_Create_Database.py

**Purpose:**  
Cleans the combined media database by standardizing date formats, removing irrelevant entries, and ensuring data consistency. The script also exports metrics related to the number of articles per media outlet.

**Key Features:**
- Identifies different date formats present in the dataset.
- Standardizes date formats to `MM/DD/YYYY`.
- Removes articles with less than 100 words.
- Drops the `URL` column.
- Filters out duplicate or highly similar titles to maintain data quality.
- Cleans the `author` column by removing unwanted text.
- Assigns language labels based on media sources.
- Saves the cleaned database and exports the count of articles per media.

**Dependencies:**  
`pandas`, `re`, `datetime`, `locale`, `pathlib`, `collections`

**Note:**  
These scripts utilize articles that were scraped and compiled into various `.txt` files within the `Raw_data` folder. Due to copyright restrictions,
the `Raw_data` folder is not currently available.

### Annotation Scripts

#### 1_Preprocess.py

**Purpose:**  
Preprocess the media database CSV by loading data, generating sentence contexts, counting words, converting and verifying date formats, and saving the processed data.

**Key Features:**
- Loads and preprocesses CSV data.
- Tokenizes text into two-sentence contexts.
- Counts words and updates relevant columns.
- Validates and standardizes date formats.
- Saves processed data into a new CSV file.

**Dependencies:**  
`os`, `pandas`, `spacy`, `datetime`, `locale`

#### 2_JSONL.py

**Purpose:**  
[Content not provided. Please refer to the script for details.]

#### 3_Manual_annotations.py

**Purpose:**  
Processes manual annotations from a JSONL file, counts label usage, and exports annotation metrics (counts and proportions) per language to a CSV file.

**Key Features:**
- Reads and parses manual annotations from a JSONL file.
- Counts label usage for English and French sentences.
- Calculates proportions of each label usage.
- Exports the results to a CSV file for further analysis.

**Dependencies:**  
`json`, `csv`, `os`, `collections`, `defaultdict`

#### 4_JSONL_for_training.py

**Purpose:**  
Processes manually annotated JSONL sentences, splits them into training and validation sets, and exports them as separate JSONL files for subsequent model training.

**Key Features:**
- Loads label configuration to determine main and sub-labels.
- Creates directory structures for training and validation outputs.
- Processes manually annotated lines from a JSONL file.
- Handles main labels, sub-labels, and exception labels.
- Ensures stratified splitting with a minimum presence of positive and negative samples.
- Aggregates annotation counts per label, language, and data split.
- Exports processed data and metrics.

**Dependencies:**  
`json`, `os`, `random`, `csv`, `collections.defaultdict`

#### 5_Training.py

**Purpose:**  
Manages the training process for Camembert (French) and Bert (English) models by loading data, detecting necessary files, and orchestrating training routines.

**Key Features:**
- Detects model training status (fully trained, partially trained, not started).
- Loads training and validation data from JSONL into pandas DataFrames.
- Instantiates and trains appropriate models based on language.
- Handles resumption of training in case of interruptions.
- Logs training progress and outcomes.
- Produces a CSV listing untrained models.

**Dependencies:**  
`os`, `json`, `sys`, `glob`, `shutil`, `pandas`, `torch`, `AugmentedSocialScientist (Camembert, Bert)`

#### 5bis_Training_best_model.py

**Purpose:**  
Manages training processes for selecting the best-performing Camembert/Bert models using labeled data. It checks for completed or partial trainings, handles directories/logs, and summarizes trained/untrained models.

**Key Features:**
- Detects available computing devices (CUDA/MPS/CPU).
- Loads and manages training data.
- Summarizes model training statuses.
- Logs processes and handles directory structures.
- Produces summaries of model performances.

**Dependencies:**  
`os`, `sys`, `glob`, `shutil`, `json`, `pandas`, `torch`, `AugmentedSocialScientist`

#### 6_Compile_annotations_logs.py

**Purpose:**  
Compiles logs from annotation processes to evaluate and select the best-performing epochs based on various metrics.

**Key Features:**
- Selects the best epoch using macro-F1, weighted-F1, and recall metrics.
- Performs cross-validation with k-folds to aggregate metrics.
- Calculates a final score for each epoch.
- Handles tie-breaking using test and train loss.
- Appends additional information to CSV outputs.

**Dependencies:**  
`pandas`, `re`, `os`, `numpy`

#### 7_Annotation.py

**Purpose:**  
Loads the annotated database and applies trained English and French models to annotate sentences. It supports saving and resuming progress to handle interruptions.

**Key Features:**
- Loads annotated data.
- Utilizes approximate string matching for suffix recognition.
- Manages model loading and device allocation.
- Predicts labels using trained models.
- Saves annotations and handles interruptions gracefully.

**Dependencies:**  
`transformers`, `tqdm`, `difflib`, `pandas`, `numpy`, `torch`

#### 8_JSONL_for_Recheck.py

**Purpose:**  
Generates a new JSONL dataset for manual revalidation of annotations post-model processing, focusing on underrepresented categories to ensure quality across all scenarios.

**Key Features:**
- Reads annotated data and filters out previously annotated sentences.
- Randomly selects samples with oversampling of underrepresented categories.
- Maintains a balanced distribution between English and French languages.
- Produces a multilingual JSONL file with necessary annotations and metadata.

**Dependencies:**  
`os`, `json`, `random`, `pandas`, `math`

## Dependencies

Ensure you have the following dependencies installed (_requirements.txt might not contain all the needed dependencies_). You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Contact

For any inquiries or feedback, please contact:

- **Alizée Pillod**
- Email : [alizee.pillod@umontreal.ca](mailto:alizee.pillod@umontreal.ca)
- **Antoine Lemor**
- Email: [antoine.lemor@umontreal.ca](mailto:antoine.lemor@umontreal.ca)
- **Matthew Taylor**
- Email : [matthew.taylor@umontreal.ca](mailto:matthew.taylor@umontreal.ca)