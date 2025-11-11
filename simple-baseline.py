#!/usr/bin/env python3
"""
simple-baseline.py

A trivial SQuAD 2.0 baseline that predicts NO-ANSWER for every question.
This produces an empty string "" for each question ID, which can be scored
with the official SQuAD v2.0 evaluation script (EM/F1).

Usage:
  python3 simple-baseline.py --gold data/dev_v2.json --out outputs/pred_dev_simple.json
  python3 simple-baseline.py --gold data/test_v2.json --out outputs/pred_test_simple.json
"""

import argparse
import json
import os


def load_question_ids(gold_path):
    with open(gold_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    qids = []
    for article in js["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                qids.append(qa["id"])
    return qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to SQuAD v2.0 gold JSON (dev/test).")
    ap.add_argument("--out", required=True, help="Path to write predictions JSON (qid -> text).")
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    qids = load_question_ids(args.gold)
    preds = {qid: "" for qid in qids}  # empty string denotes NO-ANSWER

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(preds)} predictions to: {args.out}")
    print("Note: This baseline always predicts NO-ANSWER (\"\").")


if __name__ == "__main__":
    main()
