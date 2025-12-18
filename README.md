# Outputs & Evaluation

# Generate Outputs
Simple Baseline: 

```bash
  python3 simple-baseline.py --gold data/dev_v2.json --out outputs/pred_dev_simple.json
  python3 simple-baseline.py --gold data/test_v2.json --out outputs/pred_test_simple.json
```

Strong Baseline:
```bash
  python strong-baseline.py --gold data/dev_v2.json  --out outputs/pred_dev_strong.json
  python strong-baseline.py --gold data/test_v2.json --out outputs/pred_test_strong.json
```

Extended Models:
```bash
  python extension-baseline.py --gold data/dev_v2.json  --out outputs/pred_dev_glove.json
  python extension-baseline.py --gold data/test_v2.json --out outputs/pred_test_glove.json
```
We provide a lightweight evaluation script (`score.py`) that computes **SQuAD 2.0 EM/F1** (overall + HasAns/NoAns breakdown) given:

1. a gold SQuAD v2.0 JSON file, and
2. a prediction JSON mapping `qid -> predicted_answer_string`.

## Run on saved outputs

From the repo root:

```bash
# Evaluate on dev (gold = dev_v2.json / or your provided dev file)
python3 score.py \
  --gold data/dev_v2.json \
  --pred outputs/pred_dev_simple.json

python3 score.py \
  --gold data/dev_v2.json \
  --pred outputs/pred_dev_strong.json

python3 score.py \
  --gold data/dev_v2.json \
  --pred outputs/pred_dev_glove.json
```

For the held-out test split:

```bash
python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_simple.json

python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_strong.json

python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_glove.json
```

## Example output

Example (simple no-answer baseline):

```text
Overall EM: 50.08
Overall F1: 50.08
HasAns EM: 0.02
HasAns F1: 0.02
NoAns  EM: 100.00
NoAns  F1: 100.00
```

And on our held-out test split:

```text
Overall EM: 33.87
Overall F1: 33.87
HasAns EM: 0.00
HasAns F1: 0.00
NoAns  EM: 100.00
NoAns  F1: 100.00
```

**Prediction file format:** a JSON dictionary like:

```json
{"56be4db0acb8001400a502ec": "in the late 1990s", "56be4db0acb8001400a502ed": ""}
```
