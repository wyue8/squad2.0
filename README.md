Evaluation (Scoring Predictions)
We provide a lightweight evaluation script (score.py) that computes SQuAD 2.0 EM/F1 (overall + HasAns/NoAns breakdown) given:

a gold SQuAD v2.0 JSON file, and
a prediction JSON mapping qid -> predicted_answer_string.
Run on saved outputs
From the repo root:

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
For the held-out test split:

python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_simple.json

python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_strong.json

python3 score.py \
  --gold data/test_v2.json \
  --pred outputs/pred_test_glove.js
