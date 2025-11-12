#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strong-baseline.py
------------------
Implements a train→dev→test pipeline for a lightweight BiDAF model in PyTorch.

Usage:
  python strong-baseline.py --gold data/dev_v2.json  --out outputs/pred_dev_strong.json
  python strong-baseline.py --gold data/test_v2.json --out outputs/pred_test_strong.json
"""

import argparse, json, os, re, random
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -----------------------
# Utility / Setup
# -----------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
PAD, UNK, NOANS = "<pad>", "<unk>", "<no_answer>"

def simple_tokenize(text):
    return re.findall(r"\S+", text)

# -----------------------
# Data Processing
# -----------------------
def read_squad(path):
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)
    exs = []
    for art in root["data"]:
        for para in art["paragraphs"]:
            ctx = para["context"]
            ctx_toks = simple_tokenize(ctx)
            for qa in para["qas"]:
                qid = qa["id"]
                q_toks = simple_tokenize(qa["question"])
                if qa.get("is_impossible", False) or not qa["answers"]:
                    exs.append({"id": qid, "q": q_toks, "c": ctx_toks, "start": 0, "end": 0})
                else:
                    ans = qa["answers"][0]
                    ans_text = ans["text"]
                    ans_start = ans["answer_start"]
                    join_ctx = " ".join(ctx_toks)
                    try:
                        start_tok = join_ctx[:ans_start].count(" ") + 1
                    except Exception:
                        start_tok = 1
                    end_tok = min(start_tok + len(ans_text.split()), len(ctx_toks))
                    exs.append({"id": qid, "q": q_toks, "c": ctx_toks, "start": start_tok, "end": end_tok})
    return exs

def build_vocab(exs, min_freq=2, max_size=50000):
    cnt = Counter()
    for ex in exs:
        cnt.update([t.lower() for t in ex["q"] + ex["c"]])
    vocab = {PAD: 0, UNK: 1, NOANS: 2}
    for w, f in cnt.most_common():
        if f < min_freq or len(vocab) >= max_size:
            break
        vocab[w] = len(vocab)
    return vocab

def tokens_to_ids(toks, vocab):
    return [vocab.get(t.lower(), vocab[UNK]) for t in toks]

# -----------------------
# Dataset
# -----------------------
class SquadDataset(Dataset):
    def __init__(self, exs, vocab, max_ctx=400, max_q=40):
        self.data = []
        for ex in exs:
            q_ids = tokens_to_ids(ex["q"][:max_q], vocab)
            c_ids = [vocab[NOANS]] + tokens_to_ids(ex["c"][:max_ctx-1], vocab)
            start = min(ex["start"], len(c_ids)-1)
            end = min(ex["end"], len(c_ids)-1)
            self.data.append((ex["id"], q_ids, c_ids, start, end))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_fn(batch, pad_id=0):
    ids, qs, cs, starts, ends = zip(*batch)
    max_q = max(len(q) for q in qs)
    max_c = max(len(c) for c in cs)
    q_pad = torch.full((len(batch), max_q), pad_id, dtype=torch.long)
    c_pad = torch.full((len(batch), max_c), pad_id, dtype=torch.long)
    for i, (q, c) in enumerate(zip(qs, cs)):
        q_pad[i, :len(q)] = torch.tensor(q)
        c_pad[i, :len(c)] = torch.tensor(c)
    return ids, q_pad, c_pad, torch.tensor(starts), torch.tensor(ends)

# -----------------------
# BiDAF Model
# -----------------------
class BiDAF(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hid_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.enc_q = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.enc_c = nn.LSTM(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.mod = nn.LSTM(8*hid_dim, hid_dim, bidirectional=True, batch_first=True)
        self.sim = nn.Linear(6*hid_dim, 1)
        self.start = nn.Linear(2*hid_dim, 1)
        self.end = nn.Linear(2*hid_dim, 1)

    def forward(self, q, c):
        q_out, _ = self.enc_q(self.emb(q))
        c_out, _ = self.enc_c(self.emb(c))
        B, Lc, H2 = c_out.shape
        Lq = q_out.shape[1]
        c_exp = c_out.unsqueeze(2).expand(B, Lc, Lq, H2)
        q_exp = q_out.unsqueeze(1).expand(B, Lc, Lq, H2)
        S = self.sim(torch.cat([c_exp, q_exp, c_exp*q_exp], dim=-1)).squeeze(-1)
        a = F.softmax(S, dim=-1)
        c2q = torch.bmm(a, q_out)
        b = F.softmax(S.max(dim=-1)[0], dim=-1)
        q2c = torch.bmm(b.unsqueeze(1), c_out).repeat(1, Lc, 1)
        G = torch.cat([c_out, c2q, c_out*c2q, c_out*q2c], dim=-1)
        M, _ = self.mod(G)
        return self.start(M).squeeze(-1), self.end(M).squeeze(-1)

# -----------------------
# Training & Prediction
# -----------------------
def train_model(model, loader, device, epochs=2, lr=2e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train()
        tot = 0
        for _, q, c, s, e in loader:
            q, c, s, e = q.to(device), c.to(device), s.to(device), e.to(device)
            opt.zero_grad()
            s_log, e_log = model(q, c)
            loss = ce(s_log, s) + ce(e_log, e)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"[Train] Epoch {ep} Loss: {tot/len(loader):.4f}")

@torch.no_grad()
def predict(model, loader, device, out_path):
    model.eval(); model.to(device)
    preds = {}
    for ids, q, c, _, _ in loader:
        q, c = q.to(device), c.to(device)
        s_log, e_log = model(q, c)
        s_idx = s_log.argmax(dim=-1).cpu().tolist()
        e_idx = e_log.argmax(dim=-1).cpu().tolist()
        for i, qid in enumerate(ids):
            preds[qid] = "" if s_idx[i] == 0 or e_idx[i] == 0 or e_idx[i] < s_idx[i] else f"span_{s_idx[i]}_{e_idx[i]}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f" Saved {len(preds)} predictions → {out_path}")

# -----------------------
# Main Flow
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to dev/test JSON file for evaluation.")
    ap.add_argument("--out", required=True, help="Path to save predictions JSON.")
    args = ap.parse_args()

    train_path = "data/train_v2.json"
    dev_path   = "data/dev_v2.json"
    test_path  = "data/test_v2.json"

    print(" Loading data...")
    train_ex = read_squad(train_path)
    dev_ex   = read_squad(dev_path)
    test_ex  = read_squad(test_path)
    vocab = build_vocab(train_ex)
    print(f"Vocab size: {len(vocab)}")

    train_ds = SquadDataset(train_ex, vocab)
    dev_ds   = SquadDataset(dev_ex, vocab)
    test_ds  = SquadDataset(test_ex, vocab)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiDAF(vocab_size=len(vocab))

    print(" Training BiDAF on train_v2.json...")
    train_model(model, train_loader, device, epochs=2, lr=2e-3)

    print(" Evaluating on dev and test...")
    if "dev" in args.gold:
        predict(model, dev_loader, device, args.out)
    else:
        predict(model, test_loader, device, args.out)

    print(f"Done! Evaluate with: python -Xutf8 score.py {args.gold} {args.out}")

if __name__ == "__main__":
    main()
