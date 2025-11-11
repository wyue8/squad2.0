# scoring.md

##  Evaluation Metric Overview
This project follows the **official SQuAD v2.0** evaluation protocol, which uses two complementary metrics:

- **Exact Match (EM)** — the percentage of predictions that exactly match any gold answer after normalization.  
- **[F1 Score](https://en.wikipedia.org/wiki/F-score)** — the token-level harmonic mean of precision and recall, rewarding partial overlap between prediction and gold answers.

For **SQuAD 2.0**, which contains both answerable and unanswerable questions, a model must also decide when **no answer exists**.  
Predicting the empty string `""` is considered correct only for unanswerable items.

---

##  Formal Definitions

Let:

- A_p = set of tokens in the predicted answer  
- A_g = set of tokens in a gold (reference) answer  
- A_g\* = the best-matching gold answer among all references for that question  

Then:

- **Precision = |A_p ∩ A_g\*| / |A_p|**  
- **Recall = |A_p ∩ A_g\*| / |A_g\*|**  
- **F1 = 2 × Precision × Recall / (Precision + Recall)**  

**Exact Match (EM)** = 1 if the normalized prediction exactly matches any gold answer and 0 otherwise.

**Normalization steps:**
1. Lower-case all text  
2. Remove punctuation  
3. Remove articles (`a`, `an`, `the`)  
4. Collapse extra whitespace  

---

## ️ Evaluation Script Usage

The evaluation uses the official **SQuAD 2.0** script (`score.py`).

### Example Command
```bash
python3 score.py data/dev_v2.json outputs/pred_dev_simple.json
python3 score.py data/test_v2.json outputs/pred_test_simple.json
```


### Example Output
```json
{
  "exact": 50.08001347595385,
  "f1": 50.08001347595385,
  "total": 11873,
  "HasAns_exact": 0.016869095816464237,
  "HasAns_f1": 0.016869095816464237,
  "HasAns_total": 5928,
  "NoAns_exact": 100.0,
  "NoAns_f1": 100.0,
  "NoAns_total": 5945
}
```
```json
{
  "exact": 33.87199795605519,
  "f1": 33.87199795605519,
  "total": 15656,
  "HasAns_exact": 0.0,
  "HasAns_f1": 0.0,
  "HasAns_total": 10353,
  "NoAns_exact": 100.0,
  "NoAns_f1": 100.0,
  "NoAns_total": 5303
}
```

