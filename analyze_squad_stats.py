#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_squad_stats.py

Analyze SQuAD v2.0 JSON splits and report:
- #Articles
- #Paragraphs
- #Unique Questions (by id)
- #Duplicate question ids (if any)
- % Unanswerable
- Avg context length (words)
- Avg answer length (words) over all gold spans from answerable questions
"""

import json
import argparse
from collections import Counter
import numpy as np


def tokenize_ws(s: str):
    return s.split()


def analyze_squad(path: str, name: str = "dataset"):
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)

    data = root.get("data", [])
    n_articles = len(data)

    n_paragraphs = 0
    qid_list = []
    qid_seen = set()
    qid_dups = Counter()

    context_lengths = []  # one entry per QA (so contexts are weighted by #QAs they support)
    n_unanswerable = 0

    # collect all gold answer lengths (in words) for answerable QAs
    all_answer_lengths = []

    for article in data:
        paragraphs = article.get("paragraphs", [])
        n_paragraphs += len(paragraphs)
        for para in paragraphs:
            context = para.get("context", "")
            ctx_len = len(tokenize_ws(context))
            qas = para.get("qas", [])
            for qa in qas:
                qid = qa.get("id")
                qid_list.append(qid)
                if qid in qid_seen:
                    qid_dups[qid] += 1
                else:
                    qid_seen.add(qid)

                context_lengths.append(ctx_len)
                is_imp = bool(qa.get("is_impossible", False))
                if is_imp:
                    n_unanswerable += 1
                else:
                    answers = qa.get("answers", []) or []
                    for ans in answers:
                        ans_text = ans.get("text", "") or ""
                        if ans_text.strip():
                            all_answer_lengths.append(len(tokenize_ws(ans_text)))

    n_q_unique = len(qid_seen)
    n_q_total = len(qid_list)  # includes duplicates if present
    n_dups = sum(qid_dups.values())

    unans_ratio = (n_unanswerable / n_q_unique * 100.0) if n_q_unique else 0.0
    avg_ctx_len = float(np.mean(context_lengths)) if context_lengths else 0.0
    avg_ans_len = float(np.mean(all_answer_lengths)) if all_answer_lengths else 0.0

    # Pretty print block
    print(f" {name}")
    print(f"  #Articles:          {n_articles}")
    print(f"  #Paragraphs:        {n_paragraphs}")
    print(f"  #Unique Questions:  {n_q_unique}")
    if n_dups > 0 or n_q_total != n_q_unique:
        print(f"  #Total Q rows:      {n_q_total} (includes duplicates)")
        print(f"  #Duplicate IDs:     {n_dups}")
    print(f"  %Unanswerable:      {unans_ratio:.2f}%")
    print(f"  Avg context length: {avg_ctx_len:.1f} words")
    print(f"  Avg answer length:  {avg_ans_len:.1f} words (answerable only)")
    print("-" * 60)

    return {
        "name": name,
        "articles": n_articles,
        "paragraphs": n_paragraphs,
        "q_unique": n_q_unique,
        "q_total": n_q_total,
        "q_dup": n_dups,
        "unans_ratio": unans_ratio,
        "avg_ctx": avg_ctx_len,
        "avg_ans": avg_ans_len,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze SQuAD v2.0 JSON splits with unique QID counting.")
    ap.add_argument("--train", required=True, help="Path to train JSON (SQuAD v2.0 format)")
    ap.add_argument("--dev", required=True, help="Path to dev JSON (SQuAD v2.0 format)")
    ap.add_argument("--test", required=True, help="Path to test JSON (SQuAD v2.0 format)")
    args = ap.parse_args()

    stats = []
    stats.append(analyze_squad(args.train, "Train"))
    stats.append(analyze_squad(args.dev, "Dev"))
    stats.append(analyze_squad(args.test, "Test"))

    # Summary table
    print(" Summary Table (unique question IDs)")
    header = "{:<8} {:>10} {:>12} {:>16} {:>16} {:>16}".format(
        "Split", "#Q_unique", "%Unans", "AvgCtx(w)", "AvgAns(w)", "#DupIDs"
    )
    print(header)
    for s in stats:
        print("{:<8} {:>10} {:>12.2f} {:>16.1f} {:>16.1f} {:>16}".format(
            s["name"], s["q_unique"], s["unans_ratio"], s["avg_ctx"], s["avg_ans"], s["q_dup"]
        ))


if __name__ == "__main__":
    main()
