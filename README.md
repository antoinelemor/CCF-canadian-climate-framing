# CCF-canadian-climate-framing

## Introduction

Welcome to the **CCF-canadian-climate-framing** repository. This project is dedicated to studying the media coverage of climate change in the Canadian press. It aims to understand how climate change narratives are constructed and communicated to the public since the first media article published on the subject in 1978. This repository contains all the scripts, data processing tools, and machine learning models necessary for conducting this study.

![CCf_icone](CCF_icone.jpg)

### The database

This repository includes a newly compiled database of climate change articles from 20 major Canadian newspapers (n=266,271) (_not available in plain text at this time for copyright reasons_). The table below shows the distribution of articles per newspaper (_after filtering and preprocessing_):

| Toronto Star | Globe and Mail | Vancouver Sun | Edmonton Journal | Le Devoir | National Post | Calgary Herald | Whitehorse Daily Star | Montreal Gazette | Chronicle Herald | The Telegram | Times Colonist | La Presse Plus | La Presse | Winnipeg Free Press | Acadie Nouvelle | Star Phoenix | Le Droit | Toronto Sun | Journal de Montreal | **Total** |
|--------------|----------------|---------------|------------------|-----------|---------------|----------------|-----------------------|------------------|------------------|--------------|----------------|----------------|----------|----------------------|-----------------|--------------|----------|-------------|---------------------|-----------|
| 46980         | 29442           | 17871          | 18162             | 13685      | 20032          | 19336           | 7603                   | 9567              | 10770              | 5841          | 11800            | 9548            | 6917      | 12421                  | 5143             | 7794          | 4727      | 3174         | 5458                 | **266271** |

![Number of Articles Per Year](Database/Database/articles_per_year.png)

### Example of analysis

Below is an illustrative example of the analyses conducted in this project. The animated GIF shows how the **dominant climate-change frame** evolves from year to year across Canadian provinces. For each article, the proportion of sentences mentioning a given frame is calculated; the frame with the highest average proportion in each province for each year is designated as the **dominant frame**. Gray-hatched provinces indicate insufficient data for that year.

![Yearly Dominant Frames by Province](Database/Database/dominant_frames_yearly.gif)

---

## Table of contents

- [Introduction](#introduction)
- [Members of the project](#members-of-the-project)
- [Project objectives](#project-objectives)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts overview](#scripts-overview)
  - [Database creation scripts](#database-creation-scripts)
    - [1_Database_Creation.py](#1_database_creationpy)
    - [2_Clean_and_Create_Database.py](#2_clean_and_create_databasepy)
  - [Annotation scripts](#annotation-scripts)
    - [1_Preprocess.py](#1_preprocesspy)
    - [2_JSONL.py](#2_jsonlpy)
    - [3_Manual_annotations.py](#3_manual_annotationspy)
    - [4_JSONL_for_training.py](#4_jsonl_for_trainingpy)
    - [5_Training.py](#5_trainingpy)
    - [5bis_Training_best_model.py](#5bis_training_best_modelpy)
    - [5bis2_Personalised_retraining_model.py](#5bis2_personalised_retraining_modelpy)
    - [6_Compile_annotations_logs.py](#6_compile_annotations_logspy)
    - [7_Annotation.py](#7_annotationpy)
    - [7bis_Annotation.py](#7bis_annotationpy)
    - [8_JSONL_for_Recheck.py](#8_jsonl_for_recheckpy)
    - [8bis_JSONL_for_recheck.py](#8bis_jsonl_for_recheckpy)
    - [9_Annotation_metrics.py](#9_annotation_metricspy)
- [Dependencies](#dependencies)
- [Contact](#contact)

---

## Members of the project

- [**Alizée Pillod**, Université de Montréal](https://pol.umontreal.ca/repertoire-departement/professeurs/professeur/in/in35292/sg/Aliz%C3%A9e%20Pillod/)
- [**Antoine Lemor**, Université de Montréal](https://antoinelemor.github.io/)
- [**Matthew Taylor**, Université de Montréal](https://www.chairedemocratie.com/fr/members/taylor-matthew/)

---

## Project objectives

The primary objectives of this project are:

1. **Data collection and preprocessing:** gather media articles from Canadian press sources and preprocess the data for analysis.
2. **Manual annotation and model training:** manually annotate sentences within the articles to identify key elements and train machine learning models to automatically detect and classify these elements in the articles.
3. **Evaluation and refinement:** evaluate model performance, iteratively refine annotations, and ensure high accuracy in classifications.

---

## Repository structure

```
CCF-Canadian-Climate-Framing/
├── Database/
│   ├── Database/
│   │   ├── CCF.media_database.csv _absent from the repository due to file size_
│   │   ├── CCF.media_processed_texts.csv _absent from the repository due to file size_
│   │   ├── CCF.media_processed_texts_annotated.csv _absent from the repository due to file size_
│   │   ├── Database_media_count.csv
│   │   └── articles_per_year.png
│   └── Training_data/
│       ├── manual_annotations_JSONL/ _excluded until our first publication_
│       │   ├── Annotated_sentences.jsonl _excluded_
│       │   ├── label_config.json _excluded_
│       │   ├── sentences_to_annotate_EN.jsonl _excluded_
│       │   ├── sentences_to_annotate_FR.jsonl _excluded_
│       │   ├── sentences_to_recheck_multiling.jsonl _excluded_
│       │   └── sentences_to_recheck_multiling_done.jsonl _excluded_
│       ├── annotation_bases/ _excluded until our first publication_
│       ├── Annotation_logs/
│       ├── training_database_metrics.csv
│       ├── models_metrics_summary_advanced.csv
│       ├── non_trained_models.csv
│       ├── manual_annotations_metrics.csv
│       ├── sentences_annotation_error.csv
│       ├── annotated_label_metrics.csv
│       └── final_annotation_metrics.csv
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
│       ├── 5bis2_Personalised_retraining_model.py
│       ├── 6_Compile_annotations_logs.py
│       ├── 7_Annotation.py
│       ├── 7bis_Annotation.py
│       ├── 8_JSONL_for_Recheck.py
│       ├── 8bis_JSONL_for_recheck.py
│       └── 9_Annotation_metrics.py
└── requirements.txt

README.md
```

**Note:**  
The `Database_creation` scripts rely on articles that were originally scraped and aggregated into various `.txt` files in a `Raw_data` folder (not included here). Due to copyright restrictions, those raw files are not publicly available.

---

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CCF-Canadian-Climate-Framing.git
   cd CCF-Canadian-Climate-Framing
   ```

2. **Set Up a Virtual Environment**  
   It's recommended to use a virtual environment to manage dependencies:
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

---

## Usage

The project is organized into several scripts, each responsible for different aspects of data processing, annotation, and model training. Below is an overview of how to use them.

### Database creation scripts

These scripts are responsible for creating and cleaning the database from scraped articles. Note that the raw data collection process is not part of this repository due to copyright restrictions.

1. **Database creation**
   ```bash
   python Scripts/Database_creation/1_Database_Creation.py
   ```

2. **Clean and create database**
   ```bash
   python Scripts/Database_creation/2_Clean_and_Create_Database.py
   ```

### Annotation scripts

1. **Preprocess data**  
   ```bash
   python Scripts/Annotation/1_Preprocess.py
   ```

2. **Generate JSONL files**  
   ```bash
   python Scripts/Annotation/2_JSONL.py
   ```

3. **Manual annotations**  
   ```bash
   python Scripts/Annotation/3_Manual_annotations.py
   ```

4. **Prepare JSONL for training**  
   ```bash
   python Scripts/Annotation/4_JSONL_for_training.py
   ```

5. **Train models**  
   ```bash
   python Scripts/Annotation/5_Training.py
   ```

6. **Train Best models**  
   ```bash
   python Scripts/Annotation/5bis_Training_best_model.py
   ```

7. **Personalised retraining for a single model**  
   ```bash
   python Scripts/Annotation/5bis2_Personalised_retraining_model.py
   ```
   Allows you to specifically retrain one model with custom epoch settings without re-training all models.

8. **Compile annotation logs**  
   ```bash
   python Scripts/Annotation/6_Compile_annotations_logs.py
   ```

9. **Annotation process**  
   ```bash
   python Scripts/Annotation/7_Annotation.py
   ```

10. **Personalised annotation of selected models**  
    ```bash
    python Scripts/Annotation/7bis_Annotation.py
    ```
    Enables selective re-annotation of the final database with one or more *retrained* models, clearing old labels for those models before applying fresh predictions.

11. **Generate JSONL for rechecking**  
    ```bash
    python Scripts/Annotation/8_JSONL_for_Recheck.py
    ```

12. **Re-generate JSONL to compare updated annotations**  
    ```bash
    python Scripts/Annotation/8bis_JSONL_for_recheck.py
    ```
    Produces a JSONL containing the *same sentences* but with newly-updated annotations, facilitating comparison with previous annotation versions.

13. **Final annotation metrics**  
    ```bash
    python Scripts/Annotation/9_Annotation_metrics.py
    ```
    Compares predicted labels to manually verified labels to compute precision, recall, and F1-scores for each category (in both English and French). Outputs results in `final_annotation_metrics.csv`.

---

## Scripts overview

### Database creation scripts

#### 1_Database_Creation.py

**Purpose:**  
Creates a unified media database by aggregating articles from various Canadian press sources into a single CSV file.

**Key features:**
- Defines paths to raw article data from multiple CSVs.
- Maps media codes to their full names.
- Combines all DataFrames into a consolidated database.
- Standardizes column names and language labels.

**Dependencies:**  
`pandas`, `re`, `unidecode`, `pathlib`

#### 2_Clean_and_Create_Database.py

**Purpose:**  
Cleans and filters the combined media database.

**Key features:**
- Standardizes date formats.
- Removes short/duplicate articles.
- Assigns language based on media source.
- Saves final cleaned CSV with summary metrics.

**Dependencies:**  
`pandas`, `re`, `datetime`, `pathlib`

### Annotation scripts

#### 1_Preprocess.py

**Purpose:**  
Preprocesses the media database CSV by generating sentence contexts and verifying date formats.

**Key features:**
- Splits texts into two-sentence contexts.
- Counts words and updates relevant columns.
- Saves processed data to a new CSV.

**Dependencies:**  
`os`, `pandas`, `spacy`

#### 2_JSONL.py

**Purpose:**  
Converts processed text data into JSONL files for manual annotation, separating French and English sentences.

**Key features:**  
- Loads and cleans CSV data.
- Removes duplicates.
- Splits data by language.
- Creates JSONL with metadata fields.

**Dependencies:**  
`os`, `pandas`, `json`

#### 3_Manual_annotations.py

**Purpose:**  
Reads manual annotations from a JSONL file, counts label usage, and exports annotation metrics.

**Key features:**
- Calculates label usage distribution.
- Outputs CSV with label proportions.

**Dependencies:**  
`json`, `csv`, `os`

#### 4_JSONL_for_training.py

**Purpose:**  
Prepares manually annotated JSONL data for training/validation splits.

**Key features:**
- Splits data into train/validation sets.
- Handles stratification for main/sub labels.
- Exports annotation metrics to a CSV.

**Dependencies:**  
`json`, `os`, `random`, `csv`

#### 5_Training.py

**Purpose:**  
Trains Camembert and Bert models on prepared data.

**Key features:**
- Detects model status (fully trained, partial, etc.).
- Loads training/validation from JSONL.
- Handles logging and resumption of training.
- Outputs list of non-trained models if needed.

**Dependencies:**  
`os`, `json`, `sys`, `glob`, `shutil`, `pandas`, `torch`, `AugmentedSocialScientist`

#### 5bis_Training_best_model.py

**Purpose:**  
Trains selected best models using advanced metrics from cross-validation.

**Key features:**
- Loads best epoch from `models_metrics_summary_advanced.csv`.
- Summarizes fully trained/partial/not trained status.
- Logs results and error handling.

**Dependencies:**  
`os`, `sys`, `glob`, `shutil`, `json`, `pandas`, `torch`, `AugmentedSocialScientist`

#### 5bis2_Personalised_retraining_model.py

**Purpose:**  
Facilitates *selective retraining* of exactly one Camembert/Bert model without retraining all other models.

**Key features:**
- Lists all models in the annotation bases.
- Prompts user to select a single model and specify epoch count.
- Retrains the chosen model from scratch, removing its previous directory.

**Dependencies:**  
`os`, `sys`, `glob`, `shutil`, `json`, `pandas`, `torch`, `AugmentedSocialScientist`

#### 6_Compile_annotations_logs.py

**Purpose:**  
Compiles various logs from annotation/training sessions to identify best-performing epochs and aggregates metrics.

**Key features:**
- Merges logs from multiple training runs.
- Identifies best epoch via macro/weighted F1.
- Generates summary CSV outputs.

**Dependencies:**  
`pandas`, `re`, `os`, `numpy`

#### 7_Annotation.py

**Purpose:**  
Applies trained English and French models to annotate the main database, saving or resuming progress as needed.

**Key features:**
- Loads/updates existing annotation columns.
- Performs annotation for detection, sub-categories, etc.
- Logs and saves partial results to handle interruptions.

**Dependencies:**  
`torch`, `tqdm`, `pandas`, `numpy`

#### 7bis_Annotation.py

**Purpose:**  
Provides *selective re-annotation* of the main database using one or more retrained models, clearing old labels for those models before applying fresh predictions.

**Key features:**
- Interactive menu to pick models for annotation.
- Error logging for sentences exceeding token limits.
- Updates distribution metrics in `annotated_label_metrics.csv`.

**Dependencies:**  
`os`, `glob`, `re`, `sys`, `pandas`, `numpy`, `torch`, `tqdm`, `AugmentedSocialScientist`

#### 8_JSONL_for_Recheck.py

**Purpose:**  
Builds a JSONL dataset focusing on underrepresented categories for manual revalidation.

**Key features:**
- Filters out previously annotated sentences.
- Balances English/French distributions.
- Oversamples rare labels.
- Exports a final JSONL for manual checks.

**Dependencies:**  
`os`, `json`, `random`, `pandas`, `math`

#### 8bis_JSONL_for_recheck.py

**Purpose:**  
Regenerates a JSONL (`sentences_to_recheck_multiling_bis.jsonl`) with the *same text sentences* from the original re-check file, but updates labels to reflect newly retrained models.

**Key features:**
- Loads old JSONL to preserve exact text.
- Gathers updated annotations from the latest CSV.
- Produces a new JSONL with fresh labels for direct comparison.

**Dependencies:**  
`os`, `sys`, `json`, `pandas`, `numpy`

#### 9_Annotation_metrics.py

**Purpose:**  
Compares predicted labels to manually verified labels in order to compute final precision, recall, and F1-scores for each category. Outputs the results to `final_annotation_metrics.csv`.

**Key features:**
- Loads predicted vs. gold (manually verified) annotations.
- Computes confusion stats (TP, FP, FN).
- Derives precision, recall, F1 by label and language.
- Exports final metrics to CSV.

**Dependencies:**  
`os`, `json`, `csv`, `pandas`, `math`, `collections`

---

## Additional notice: iterative retraining & comparison

The scripts (`5bis2_Personalised_retraining_model.py`, `7bis_Annotation.py`, `8bis_JSONL_for_recheck.py`, and optionally `9_Annotation_metrics.py`) enable an **iterative process**:

1. **Retrain specific models** to improve precision (`5bis2_Personalised_retraining_model.py`).  
2. **Re-annotate** only those categories using the updated models (`7bis_Annotation.py`).  
3. **Regenerate JSONL** files containing the *same sentences* but with new labels for **direct comparison** to previous annotations (`8bis_JSONL_for_recheck.py`).  
4. Finally, **evaluate** updated model predictions against manually verified annotations (`9_Annotation_metrics.py`), producing comprehensive performance metrics in `Database/Training_data/final_annotation_metrics.csv`.

This iterative workflow ensures continuous refinement of model accuracy and traceable comparisons between old and new annotations.

---

## Dependencies

Make sure you have the following Python packages installed. You can install them via:

```bash
pip install -r requirements.txt
```

The project may also require additional packages such as `torch`, `transformers`, `AugmentedSocialScientist`, and spaCy models (`en_core_web_lg`, `fr_dep_news_trf`). Refer to the script headers for detailed dependency listings.

---

## Contact

For any inquiries or feedback, please contact:

- **Alizée Pillod**  
  Email: [alizee.pillod@umontreal.ca](mailto:alizee.pillod@umontreal.ca)

- **Antoine Lemor**  
  Email: [antoine.lemor@umontreal.ca](mailto:antoine.lemor@umontreal.ca)

- **Matthew Taylor**  
  Email: [matthew.taylor@umontreal.ca](mailto:matthew.taylor@umontreal.ca)
