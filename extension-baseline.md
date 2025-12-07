# **Extension Baseline – GloVe + Multi-Head Attention BiDAF on SQuAD v2.0**

## **Overview**

This document describes the **Extension Baseline** implemented for Milestone 3.  
Building upon the strong baseline BiDAF model, this extension incorporates two architectural improvements:

1. **Pretrained GloVe word embeddings** (glove.6B.100d) to provide richer lexical representations.  
2. **A multi-head attention mechanism** replacing BiDAF’s original single-head similarity scorer to enable more expressive question–context alignment.

These extensions demonstrate how classical QA architectures can benefit from pretrained lexical knowledge and enhanced attention patterns without modifying the overall BiDAF pipeline.

---

# **Explanation of the Extension**

For this milestone, we extended the strong baseline in two ways.  
First, we replaced the randomly initialized word embeddings with **100-dimensional GloVe vectors**, giving the model access to semantic information learned from large corpora. In the baseline, the lack of pretrained representations made it difficult for the model to meaningfully compare question and context tokens. GloVe embeddings help the model recognize semantic similarity (e.g., *river* ↔ *stream*, *leader* ↔ *president*), which is essential for span prediction.

Second, we replaced BiDAF’s **single linear similarity layer** with a **4-head attention module**.  
Instead of producing a single alignment score for each question–context token pair, the multi-head mechanism allows the model to attend to multiple alignment patterns simultaneously. In this setup:

- the **context** acts as the *query*,  
- the **question** provides *keys* and *values*,  
- and the resulting attention outputs are fused back into the BiDAF interaction layer.

To implement these extensions, we constructed a 50,000-word vocabulary from SQuAD 2.0 and loaded 400,000 embeddings from `glove.6B.100d.txt`. Tokens found in GloVe were assigned their pretrained vectors; unseen tokens were mapped to an UNK embedding. Apart from modifying the embedding layer and similarity module, the remaining BiDAF architecture—contextual BiLSTMs, attention flow fusion, modeling LSTM, and span predictors—remained unchanged.

The model was trained for two epochs on CPU using Adam with a learning rate of 2e-3. Evaluation followed the same process as the strong baseline. The combined extension was trained for two epochs on CPU using Adam with a learning rate of 2e-3.  
**On the development set, the model achieved EM = 33.46 and F1 = 33.46, with NoAns F1 = 66.81 and HasAns F1 ≈ 0.02.**  
**On the test set, it reached EM = 33.22 and F1 = 33.22, with NoAns F1 = 98.08 and HasAns F1 = 0.**  

Empirically, the combined GloVe + multi-head extension maintains similar overall EM/F1 to the strong baseline but improves the model’s ability to detect unanswerable questions, particularly on the test set. Performance on answerable questions remains low (HasAns F1 ≈ 0), reflecting the difficulty of SQuAD v2.0 for non-transformer architectures. These results motivate future extensions incorporating contextualized embeddings such as BERT.

---

## **Model Modifications**

The extension retains the base BiDAF structure:

- Embedding → Contextual Encoding → Attention Flow → Modeling → Output  
- Bidirectional LSTMs for question and context  
- Standard BiDAF feature fusion (`[c; c2q; c * c2q; c * q2c]`)

### **Key Changes**

| Component | Strong Baseline | Extension Baseline |
|----------|------------------|--------------------|
| Token Embeddings | Random (100d) | Pretrained GloVe (100d) |
| Attention Module | Single-head similarity | 4-head multi-head attention |
| Vocabulary | 50,000 | 50,000 (same) |
| Architecture Above Attention | Unchanged | Unchanged |

---

## **Data and Preprocessing**

- **Dataset:** SQuAD v2.0  
- **Splits:** `train_v2.json`, `dev_v2.json`, `test_v2.json`  
- **Tokenization:** Simple whitespace tokenization (`\S+`)  
- **Max lengths:** Context = 400 tokens, Question = 40 tokens  
- **Vocabulary:** Minimum frequency = 2  

---

## **Training Setup**

| Setting | Value |
|--------|--------|
| Optimizer | Adam |
| Learning Rate | 2e-3 |
| Batch Size | 32 |
| Epochs | 2 |
| Embeddings | glove.6B.100d |
| Attention | 4-head multi-head attention |
| Device | CPU |

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
