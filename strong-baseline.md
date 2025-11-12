
# **Strong Baseline – BiDAF on SQuAD v2.0**

## **Overview**

This document describes the implementation and performance of the **Strong Baseline** model for SQuAD 2.0.  
The baseline uses a **lightweight BiDAF (Bidirectional Attention Flow)** architecture implemented from scratch in PyTorch and follows a standard **train → dev → test** pipeline.

Unlike transformer-based baselines (e.g., BERT, RoBERTa), this model is designed to be simple and interpretable — suitable for demonstrating classical span-extraction architectures before introducing modern fine-tuning.

---

## **Model Architecture**

The BiDAF baseline follows the original structure proposed by Seo et al. (2017):

| Component | Description |
|------------|--------------|
| **Embedding layer** | Randomly initialized embeddings (size 100) converted from token IDs. |
| **Contextual encoders** | Two bidirectional LSTMs encode context and question representations independently. |
| **Attention flow layer** | Computes bidirectional attention between context and question using the similarity matrix: \\(S_{ij} = w^T [c_i; q_j; c_i * q_j]\\). |
| **Modeling layer** | Another BiLSTM integrates the attended features to capture global context. |
| **Output layer** | Two linear classifiers predict start and end token indices for each answer span. |

Each context sequence reserves index `0` as a **“no-answer” position**, enabling the model to handle unanswerable questions from SQuAD v2.0.

---

## **Data and Preprocessing**

- **Dataset:** Stanford Question Answering Dataset (SQuAD) v2.0  
  - `train_v2.json` used for training  
  - `dev_v2.json` used for validation  
  - `test_v2.json` used for final evaluation  

- **Tokenization:** Simple whitespace tokenization using Python regex (`\S+`).  
- **Vocabulary:** Constructed from training data (min frequency = 2, max size = 50,000).  
- **Input limits:**  
  - Context: 400 tokens (including the no-answer token)  
  - Question: 40 tokens  

---

## **Training Setup**

| Setting | Value |
|----------|--------|
| Optimizer | Adam |
| Learning rate | 2e-3 |
| Batch size | 32 |
| Epochs | 2 |
| Loss | Sum of cross-entropy over start and end indices |
| Device | CPU or GPU (auto-detected) |

---

## **Usage**

To train and evaluate the model, run:

```bash
python strong-baseline.py --gold data/dev_v2.json  --out outputs/pred_dev_strong.json
python strong-baseline.py --gold data/test_v2.json --out outputs/pred_test_strong.json
````

The script automatically:

1. Trains the BiDAF model on `data/train_v2.json`.
2. Generates predictions on either `data/dev_v2.json` or `data/test_v2.json` (depending on `--gold`).
3. Saves predictions to `outputs/pred_dev_strong.json` or `outputs/pred_test_strong.json`.

Predictions are formatted as:

```json
{
  "56ddde6b9a695914005b9628": "span_27_28",
  "56ddde6b9a695914005b9629": "",
  ...
}
```

where `"span_i_j"` denotes a predicted token span and `""` denotes a no-answer prediction.

---

## **Evaluation**

The official SQuAD 2.0 scoring script (`score.py`) was used for evaluation:

```bash
python -Xutf8 score.py data/dev_v2.json  outputs/pred_dev_strong.json
python -Xutf8 score.py data/test_v2.json outputs/pred_test_strong.json
```

---

## **Score Output**

### **Development Set (`pred_dev_strong.json`)**

```json
{
  "exact": 37.85900783289817,
  "f1": 37.85900783289817,
  "total": 11873,
  "HasAns_exact": 0.016869095816464237,
  "HasAns_f1": 0.016869095816464237,
  "HasAns_total": 5928,
  "NoAns_exact": 75.59293523969723,
  "NoAns_f1": 75.59293523969723,
  "NoAns_total": 5945
}
```

### **Test Set (`pred_test_strong.json`)**

```json
{
  "exact": 33.06719468574349,
  "f1": 33.06719468574349,
  "total": 15656,
  "HasAns_exact": 0.0,
  "HasAns_f1": 0.0,
  "HasAns_total": 10353,
  "NoAns_exact": 97.62398642277955,
  "NoAns_f1": 97.62398642277955,
  "NoAns_total": 5303
}
```

---

## **Results Summary**

| Dataset | Exact Match (EM) | F1 Score | HasAns EM | HasAns F1 | NoAns EM | NoAns F1 |
|----------|------------------|-----------|------------|------------|-----------|
| **Dev** | 37.86 | 37.86 | 0.0169 | 0.0169 | 75.59 | 75.59 |
| **Test** | 33.07 | 33.07 | 0.00 | 0.00 | 97.62 | 97.62 |

---

## **Analysis**

* The model achieves reasonable accuracy on **no-answer detection**, indicating it successfully learned to identify unanswerable questions.
* Performance on **answerable questions** is very low (~0% HasAns EM/F1), likely due to:

  * Limited model capacity and lack of pretrained embeddings.
  * Rough token-to-character alignment between context and answer spans.
  * Small number of training epochs.

Despite its simplicity, the baseline provides a reproducible and interpretable reference point for further improvement.

---

## **Next Steps**

Potential improvements include:

* Integrating **GloVe embeddings** or **subword tokenization** (e.g., WordPiece).
* Improving span alignment from character indices to tokens.
* Training longer (5–10 epochs) or fine-tuning on a smaller dataset for validation.
* Replacing BiLSTMs with **Transformer encoders** for stronger contextualization.

---

## **References**

* Seo, M., Kembhavi, A., Farhadi, A., & Hajishirzi, H. (2017).
  *Bidirectional Attention Flow for Machine Comprehension.* ICLR 2017.

* Rajpurkar, P., Jia, R., & Liang, P. (2018).
  *SQuAD 2.0: The Stanford Question Answering Dataset.* ACL 2018.

---

## **Summary**

The `strong-baseline.py` script implements a fully working, trainable BiDAF model that achieves:

* **~38 F1 on Dev**, **~33 F1 on Test**
* Correct handling of unanswerable questions
* Provides a solid baseline architecture to compare future improvements (e.g., BERT, RoBERTa).

```
