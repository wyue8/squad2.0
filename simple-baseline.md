# simple-baseline.md

## Overview
This baseline predicts **NO-ANSWER** (empty string `""`) for **every question** in a SQuAD v2.0–formatted dataset.  
It serves as a trivial lower bound and characterizes the dataset’s difficulty, especially the proportion of unanswerable questions.

- **Input:** SQuAD v2.0 JSON (`data/dev_v2.json` or `data/test_v2.json`)
- **Output:** JSON mapping `{question_id: ""}` for all question IDs

## How to Run

### 1) Generate predictions
```bash
  python3 simple-baseline.py --gold data/dev_v2.json --out outputs/pred_dev_simple.json
  python3 simple-baseline.py --gold data/test_v2.json --out outputs/pred_test_simple.json
```
### 2) Sample Output
```json
{
  "56ddde6b9a695914005b9628": "",
  "56ddde6b9a695914005b9629": "",
  "56ddde6b9a695914005b962a": "",
  "56ddde6b9a695914005b962b": "",
  "56ddde6b9a695914005b962c": "",
  "5ad39d53604f3c001a3fe8d1": "",
  "5ad39d53604f3c001a3fe8d2": "",
    ...
  "5ad28ad0d7d075001a4299ce": "",
  "5ad28ad0d7d075001a4299cf": ""
}
```




