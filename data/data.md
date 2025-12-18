# Data Download and Preparation Guide

This project uses the **Stanford Question Answering Dataset (SQuAD) 2.0** (Rajpurkar et al., 2018), a large-scale benchmark for **extractive question answering with unanswerable questions**.

Because the official SQuAD 2.0 **test set does not release ground-truth labels**, we construct our own train / validation / test splits from the publicly available data. This README explains how to **download the raw dataset** and **reproduce our splits**.

---

## 1. Dataset Overview

Each SQuAD 2.0 instance consists of:

* a **context paragraph** from Wikipedia,
* a **question**,
* one or more **answer spans** (text + character start index), or
* a Boolean flag **`is_impossible`** indicating the question is unanswerable.

Our final data splits are:

* **Train**: 88% of the official training set
* **Dev (Validation)**: official SQuAD 2.0 dev set
* **Test**: remaining 12% of the official training set

Splitting is performed **at the article level** to prevent context leakage across splits.

---

## 2. Downloading the Raw SQuAD 2.0 Data

The official SQuAD 2.0 data can be downloaded from Stanford NLP:

**Homepage:**
[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

**Direct download links:**

```text
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

You may download the files manually or using the command line:

```bash
mkdir -p data/raw
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O data/raw/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json   -O data/raw/dev-v2.0.json
```

---

## 3. Creating Project-Specific Splits

### Step 1: Flatten the Training Data

The official training JSON is first flattened into CSV format to enable article-level splitting and inspection.

```bash
python3 flatten_squad_v2.py \
  --input data/raw/train-v2.0.json \
  --output data/train_flat.csv
```

Each CSV row corresponds to one question-answer instance and includes:

* context
* question
* answer text(s)
* answer start index/indices
* `is_impossible` flag
* article identifier

---

### Step 2: Split at the Article Level

The flattened CSV is split into **train (88%)** and **test (12%)** subsets based on article IDs:

```bash
python3 split_by_article.py \
  --input data/train_flat.csv \
  --train_ratio 0.88 \
  --out_dir data/
```

This produces:

```text
data/train_flat.csv
data/test_flat.csv
```

---

### Step 3: Convert CSV Back to SQuAD v2.0 JSON

Each split is converted back into standard SQuAD 2.0 JSON format using the provided script:

```bash
python3 csv_to_squad_v2.py --csv data/train_flat.csv --out data/train_v2.json
python3 csv_to_squad_v2.py --csv data/test_flat.csv  --out data/test_v2.json
```

The conversion script:

* restores nested SQuAD structure,
* normalizes Boolean labels,
* parses multiple answer spans,
* ensures numeric answer start indices.

The official **dev set** is used directly as validation data:

```text
data/raw/dev-v2.0.json
```

---

---
