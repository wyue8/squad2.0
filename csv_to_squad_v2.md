# Data Description

## Dataset Overview
This project uses data derived from the **SQuAD 2.0** dataset (Rajpurkar et al., ACL 2018), which contains questions, context passages, and answer spans from Wikipedia articles.  
In addition to the original *development* set provided by SQuAD, we split the original `train-v2.0.json` file into **training** and **testing** portions to create our own held-out evaluation data.

- **Training set:** 88% of the original SQuAD 2.0 training data  
- **Test set:** 12% of the original SQuAD 2.0 training data  
- **Development set:** the official SQuAD 2.0 dev set (unchanged)

Each example consists of:
- a `context` paragraph (text passage)
- a `question`
- one or more `answer_text` and corresponding `answer_start` indices (for answerable questions)
- an `is_impossible` flag indicating whether the question is unanswerable
- a unique `id` field

---


## Conversion Script
We provide a Python script `csv_to_squad_v2.py` that converts the flattened CSV files into the official SQuAD v2.0 JSON structure required by the evaluation script (`score.py`).

### Usage
```bash
python3 csv_to_squad_v2.py --csv data/train_flat.csv --out data/train_v2.json
python3 csv_to_squad_v2.py --csv data/dev_flat.csv   --out data/dev_v2.json
python3 csv_to_squad_v2.py --csv data/test_flat.csv  --out data/test_v2.json
