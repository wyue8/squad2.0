#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import argparse
import collections

# === Column names matching CSV header ===
COL_ID = "id"
COL_TITLE = "title"
COL_CONTEXT = "context"
COL_QUESTION = "question"
COL_IS_IMPOSS = "is_impossible"
COL_ANS_TEXT = "answer_text"
COL_ANS_START = "answer_start"


def parse_bool(v):
    """Robust boolean parser: True/False/1/0/yes/no etc."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def to_int_or_none(x):
    """Convert '269', '269.0', or ' 269 ' to int; return None if invalid."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def split_multi(s):
    """Split multiple answers separated by '||' or ';' and strip spaces."""
    if s is None:
        return []
    s = str(s)
    if "||" in s:
        parts = [p.strip() for p in s.split("||") if p.strip()]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
    else:
        s = s.strip()
        parts = [s] if s else []
    return parts


def row_to_qa(row):
    """Convert one CSV row into a SQuAD v2 QA entry."""
    qid = row[COL_ID]
    question = row[COL_QUESTION]
    is_imp = parse_bool(row.get(COL_IS_IMPOSS, False))

    answers = []
    if not is_imp:
        texts = split_multi(row.get(COL_ANS_TEXT, ""))
        starts_raw = split_multi(row.get(COL_ANS_START, ""))

        if texts and starts_raw and len(texts) == len(starts_raw):
            for t, s in zip(texts, starts_raw):
                iv = to_int_or_none(s)
                if t and iv is not None:
                    answers.append({"text": t, "answer_start": iv})
        else:
            # Fallback: try single answer
            raw_text = str(row.get(COL_ANS_TEXT, "")).strip()
            iv = to_int_or_none(row.get(COL_ANS_START, ""))
            if raw_text and iv is not None:
                answers.append({"text": raw_text, "answer_start": iv})

    return {
        "id": qid,
        "question": question,
        "is_impossible": bool(is_imp),
        "answers": [] if is_imp else answers,
    }


def main():
    ap = argparse.ArgumentParser(description="Convert flat CSV to SQuAD v2.0 JSON format")
    ap.add_argument("--csv", required=True, help="Path to input CSV file")
    ap.add_argument("--out", required=True, help="Path to output SQuAD v2.0 JSON file")
    ap.add_argument("--default_title", default="SQuADv2", help="Default title if CSV lacks a title column")
    args = ap.parse_args()

    # Group QAs by title and context
    title_to_context_to_qas = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check for required columns
        required_cols = [
            COL_ID,
            COL_CONTEXT,
            COL_QUESTION,
            COL_IS_IMPOSS,
            COL_ANS_TEXT,
            COL_ANS_START,
        ]
        for rc in required_cols:
            if rc not in reader.fieldnames:
                raise ValueError(f"CSV is missing required column: '{rc}'. "
                                 f"Current columns: {reader.fieldnames}")

        for row in reader:
            title = (row.get(COL_TITLE) or args.default_title).strip() or args.default_title
            context = row[COL_CONTEXT]
            qa = row_to_qa(row)
            title_to_context_to_qas[title][context].append(qa)

    # Build final JSON structure
    data = []
    for title, ctx_map in title_to_context_to_qas.items():
        paragraphs = []
        for ctx, qas in ctx_map.items():
            paragraphs.append({"context": ctx, "qas": qas})
        data.append({"title": title, "paragraphs": paragraphs})

    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump({"data": data, "version": "v2.0"}, fo, ensure_ascii=False)
    print(f"Wrote SQuADv2 JSON to {args.out} (articles={len(data)})")


if __name__ == "__main__":
    main()
