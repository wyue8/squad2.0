# analyze_squad_stats.md

## Purpose
This script analyzes the dataset splits (train / dev / test) and reports key statistics, including:
- Total number of questions  
- Percentage of unanswerable questions  
- Average context length (in words)  
- Average answer length (in words)  

It helps characterize the data used in our experiments and provides quantitative information for the *Experimental Design* section of our report.

---

## Input
The script expects three JSON files in standard **SQuAD v2.0 format**, each containing:
```json
{
  "data": [
    {
      "title": "Some Article",
      "paragraphs": [
        {
          "context": "Full paragraph text ...",
          "qas": [
            {
              "id": "56ddde6b9a695914005b9628",
              "question": "In what country did the Normans originate?",
              "is_impossible": false,
              "answers": [
                {"text": "France", "answer_start": 30}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
##  Usage
```bash
python3 analyze_squad_stats.py --train data/train_v2.json --dev data/dev_v2.json --test data/test_v2.json
```
## Output
```json
 Train
  #Articles:          391
  #Paragraphs:        16767
  #Unique Questions:  114663
  %Unanswerable:      33.33%
  Avg context length: 119.0 words
  Avg answer length:  3.2 words (answerable only)
------------------------------------------------------------
 Dev
  #Articles:          35
  #Paragraphs:        1204
  #Unique Questions:  11873
  #Total Q rows:      26247 (includes duplicates)
  #Duplicate IDs:     14374
  %Unanswerable:      50.07%
  Avg context length: 128.8 words
  Avg answer length:  3.1 words (answerable only)
------------------------------------------------------------
 Test
  #Articles:          52
  #Paragraphs:        2115
  #Unique Questions:  15656
  %Unanswerable:      33.87%
  Avg context length: 125.7 words
  Avg answer length:  3.9 words (answerable only)
------------------------------------------------------------
 Summary Table (unique question IDs)
Split     #Q_unique       %Unans        AvgCtx(w)        AvgAns(w)          #DupIDs
Train        114663        33.33            119.0              3.2                0
Dev           11873        50.07            128.8              3.1            14374
Test          15656        33.87            125.7              3.9                0

```