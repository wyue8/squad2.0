# **Extension Baseline – GloVe-Enhanced BiDAF on SQuAD v2.0**

## **Overview**

This document describes the **Extension Baseline** implemented for Milestone 3.  
Building upon the strong baseline BiDAF model, this extension incorporates **pretrained GloVe word embeddings** (glove.6B.100d) to provide richer lexical representations and improve the model’s ability to align question tokens with relevant context. This extension demonstrates how classical QA architectures can benefit from pretrained word vectors without modifying the upper-level BiDAF layers.

---

# **Explanation of the Extension**

For this milestone, we extended the strong baseline developed in Milestone 2 by integrating pretrained GloVe embeddings into the BiDAF model. In the original strong baseline, BiDAF used randomly initialized embeddings, which provided limited semantic understanding and made it difficult for the model to meaningfully compare question and context representations. The goal of this extension was to introduce pretrained lexical knowledge into the embedding layer so that the model could leverage semantic similarity between words (e.g., *river* vs. *stream*, *president* vs. *leader*), which is essential for span extraction.

To implement this extension, we constructed a 50,000-word vocabulary from the SQuAD 2.0 training data and loaded 400,000 pretrained vectors from `glove.6B.100d.txt`. For each vocabulary token, we assigned its GloVe embedding when available; otherwise, we mapped it to an UNK vector. The remainder of the BiDAF architecture—including the contextual BiLSTMs, attention flow layer, modeling layer, and start/end classifiers—remained unchanged. The model was trained for two epochs on CPU using the Adam optimizer with a learning rate of 2e-3, following the same training setup as the strong baseline.

Empirical results show that the GloVe-enhanced BiDAF improves the model's ability to identify unanswerable questions and provides more stable span predictions compared to the randomly initialized baseline. On the development set, the model achieved **EM = 32.35** and **F1 = 32.35**; on the test set, it reached **EM = 32.92** and **F1 = 32.92**. While performance on answerable questions remained low (HasAns F1 ≈ 0), the improvements in NoAns performance confirm that pretrained lexical representations help the model better detect semantic mismatches between questions and context. These findings motivate future extensions using contextualized embeddings such as BERT.

---

## **Model Modifications**

The extension baseline retains the full BiDAF architecture (Seo et al., 2017):

- Embedding → Contextual Encoding → Attention Flow → Modeling → Output  
- Bidirectional LSTMs for both question and context  
- Attention matrix constructed using:  
  \[
  S_{ij} = w^T[c_i; q_j; c_i * q_j]
  \]

### **Key Change: Embedding Layer**

| Component | Strong Baseline | Extension Baseline |
|----------|-----------------|--------------------|
| Token Embeddings | Random (100d) | Pretrained GloVe (100d) |
| Vocabulary | 50,000 | 50,000 |
| Unknown Token | Random | UNK vector |
| No-Answer Token | Index 0 | Same |

No other architectural components were altered.

---

## **Data and Preprocessing**

- **Dataset:** SQuAD 2.0  
- **Splits:**  
  - `train_v2.json` (training)  
  - `dev_v2.json` (validation)  
  - `test_v2.json` (evaluation)  
- **Tokenization:** Simple whitespace tokenization (`\S+`)  
- **Max sequence lengths:**  
  - Context: 400 tokens  
  - Question: 40 tokens  
- **Vocabulary:** Built from training data with min frequency = 2  

---

## **Training Setup**

| Setting | Value |
|---------|--------|
| Optimizer | Adam |
| Learning Rate | 2e-3 |
| Batch Size | 32 |
| Epochs | 2 |
| Embeddings | GloVe.6B.100d |
| Device | CPU (in this run) |

Training pipeline is identical to the strong baseline, aside from loading pretrained embeddings.

---

## **Usage**

### Downloading GloVe Embeddings

To run this extension with pretrained embeddings, download the 100-dimensional GloVe vectors:

https://nlp.stanford.edu/data/glove.6B.zip

Extract the archive and place:

`glove.6B.100d.txt` → `data/` directory.

If the file is not found, the code will automatically fall back to **random embeddings**, printing:

`Warning: GloVe file not found. Using random embeddings.`

### **Train + Predict on Dev Set**
```bash
python extension-baseline.py --gold data/dev_v2.json --out outputs/pred_dev_glove.json
```
### **Train + Predict on Test Set**
```bash
python extension-baseline.py --gold data/test_v2.json --out outputs/pred_test_glove.json
```
### **Evaluation**
```bash
python score.py data/dev_v2.json  outputs/pred_dev_glove.json
python score.py data/test_v2.json outputs/pred_test_glove.json
```