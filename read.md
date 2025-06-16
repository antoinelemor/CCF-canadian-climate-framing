Voici les **sections mises à jour** (ou nouvelles) ; tu peux les copier-coller telles quelles dans ton README.

---

### 1) Table of contents – *Scripts overview* (compléter l’entrée existante)

```markdown
- [Scripts overview](#scripts-overview)
  - [Annotation scripts](#annotation-scripts)
    - [1_Preprocess.py](#1_preprocesspy)
    - [2_JSONL.py](#2_jsonlpy)
    - [3_Manual_annotations.py](#3_manual_annotationspy)
    - [4_JSONL_for_training.py](#4_jsonl_for_trainingpy)
    - [5_populate_SQL_database.py](#5_populate_sql_databasepy)
    - [6_Training_best_models.py](#6_training_best_modelspy)
    - [7_Annotation.py](#7_annotationpy)
    - [8_NER.py](#8_nerpy)
    - [9_JSONL_for_recheck.py](#9_jsonl_for_recheckpy)
    - [10_Annotation_metrics.py](#10_annotation_metricspy)
    - [11_Blind_verification.py](#11_blind_verificationpy)
```

---

### 2) Usage – *Annotation scripts* (remplacer la liste numérotée)

````markdown
### Annotation scripts

1. **Preprocess data**
   ```bash
   python Scripts/Annotation/1_Preprocess.py
````

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
5. **Populate SQL database**

   ```bash
   python Scripts/Annotation/5_populate_SQL_database.py
   ```
6. **Training best models**

   ```bash
   python Scripts/Annotation/6_Training_best_models.py
   ```
7. **Annotation process**

   ```bash
   python Scripts/Annotation/7_Annotation.py
   ```
8. **NER (Named Entity Recognition)**

   ```bash
   python Scripts/Annotation/8_NER.py
   ```
9. **Generate JSONL for rechecking**

   ```bash
   python Scripts/Annotation/9_JSONL_for_recheck.py
   ```
10. **Final annotation metrics**

    ```bash
    python Scripts/Annotation/10_Annotation_metrics.py
    ```
11. **Blind verification of manual annotations**

    ```bash
    python Scripts/Annotation/11_Blind_verification.py
    ```

````

---

### 3) Scripts overview – descriptions détaillées  
*(à insérer/mettre à jour dans la section « Scripts overview » juste après les anciennes descriptions)*

```markdown
#### 9_JSONL_for_recheck.py

**Purpose:**  
Builds a *multilingual* JSONL file to **re-check model annotations** directly from the PostgreSQL table `CCF_processed_data`, ensuring statistically robust sub-class evaluation.

Key features:

* **Root-inverse weighted sampling** (`weight ∝ 1/√frequency`) that favours rare labels while avoiding extreme oversampling.
* **Hard inequality constraints**: guarantees at least `MIN_PER_LABEL` positives per label and caps each label at `MAX_LABEL_PCT` of the sample.
* **Dynamic sample size**: automatically upsizes beyond the nominal `NB_SENTENCES_TOTAL` if needed to satisfy minimum quotas.
* **Iterative post-processing** that adds or trims rows until all constraints are met without breaking multi-label coherence.
* **Language balancing**: enforces a 50 %/50 % EN–FR split when possible, redistributing quotas if one language lacks sentences.
* Excludes sentences already present in previous manual-annotation JSONL files.
* Fully parameterised constants (`MIN_PER_LABEL`, `MAX_LABEL_PCT`, `BETA`, etc.) placed at the top of the script.

Dependencies: `pandas`, `psycopg2`, `tqdm`, `json`, `math`, `random`
````

```markdown
#### 11_Blind_verification.py

**Purpose:**  
Creates a *blind-verification* copy of any manual-annotation JSONL by **wiping all labels**, so annotators can re-label sentences without bias.

Key features:

* **Streaming I/O**: reads and writes line by line → constant memory footprint even on very large files.
* **Automatic directory creation** for the output path.
* **CLI arguments** `--input` and `--output` with sensible defaults pointing to the `manual_annotations_JSONL` directory.
* **Progress indicator**: uses `tqdm` if available; otherwise falls back to a lightweight counter.
* Robust error handling for invalid JSON lines or missing files.

Dependencies: Python standard library only (`argparse`, `json`, `pathlib`, `sys`) plus optional `tqdm`.
```

---

### 4) Methodology – *Validation and Quality Control* (ajouter la phrase en fin de paragraphe)

```markdown
The integrity and quality of the annotations are paramount. `Scripts/Annotation/9_JSONL_for_recheck.py` facilitates the creation of targeted subsets of data for manual re-verification, especially for under-represented or ambiguous categories. `Scripts/Annotation/11_Blind_verification.py` then produces a **label-free “blind” version** of these files so that annotators can re-label sentences independently, eliminating confirmation bias. Performance metrics, including precision, recall, and F1-scores for each annotated category, are systematically computed using `Scripts/Annotation/10_Annotation_metrics.py` to ensure transparency and research-grade reliability.
```

---

Ces quatre blocs couvrent toutes les parties du README affectées par l’ajout des scripts **9\_JSONL\_for\_recheck.py** (nouvelle version) et **11\_Blind\_verification.py**.
