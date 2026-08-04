import json
import os
import random
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import prepare
import train as T

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LEN = 256
BATCH = 16
NEG_PER_POS = 4
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def query_of(r):
    return " ".join(r.get("postText", [])) + " " + r.get("targetTitle", "")


def norm(s):
    return " ".join(s.lower().split())


def label_paragraphs(r):
    """(paragraph_text, 1/0) — positive iff it contains a gold spoiler string."""
    paras = [p for p in r.get("targetParagraphs", []) if p]
    golds = [norm(g) for g in r.get("spoiler", []) if g.strip()]
    out = []
    for p in paras:
        np_ = norm(p)
        out.append((p, 1 if any(g and g in np_ for g in golds) else 0))
    return out


def build_dataset(records, tokenizer):
    qs, ps, ys = [], [], []
    for r in records:
        q = query_of(r)
        labelled = label_paragraphs(r)
        pos = [p for p, y in labelled if y == 1]
        neg = [p for p, y in labelled if y == 0]
        if not pos:
            continue
        random.shuffle(neg)
        keep = neg[:NEG_PER_POS * len(pos)]
        for p in pos:
            qs.append(q); ps.append(p); ys.append(1.0)
        for p in keep:
            qs.append(q); ps.append(p); ys.append(0.0)
    enc = tokenizer(qs, ps, max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    print(f"  examples: {len(ys)} ({int(sum(ys))} pos / {len(ys)-int(sum(ys))} neg)")
    return TensorDataset(enc.input_ids, enc.attention_mask,
                         torch.tensor(ys, dtype=torch.float))


@torch.no_grad()
def evaluate(records, model, tokenizer, device):
    """Rank each record's paragraphs, emit top-1, report selection acc + METEOR."""
    model.eval()
    hits, mets = 0, []
    for r in records:
        paras = [p for p in r.get("targetParagraphs", []) if p]
        if not paras:
            continue
        q = query_of(r)
        enc = tokenizer([q] * len(paras), paras, max_length=MAX_LEN, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        scores = []
        for b in range(0, len(paras), 32):
            out = model(input_ids=enc["input_ids"][b:b + 32],
                        attention_mask=enc["attention_mask"][b:b + 32])
            scores += out.logits.squeeze(-1).tolist()
        best = paras[max(range(len(paras)), key=lambda i: scores[i])].strip()
        gold = " ".join(r["spoiler"]).strip()
        if norm(gold) in norm(best):
            hits += 1
        mets.append(prepare.meteor([best], [gold]))
    n = len(mets)
    return hits / n, sum(mets) / n, n


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Paragraph ranker {MODEL_NAME} | epochs={EPOCHS} lr={LR} device={device}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1).to(device)

    train = [json.loads(l) for l in open(os.path.join(DATA_DIR, "train.jsonl"))]
    val = prepare.load_split("val")
    val_passage = [r for r in val if r.get("tags", [""])[0] == "passage"]

    print("Building training set...", flush=True)
    ds = build_dataset(train, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1.0 - s / total))
    lossf = torch.nn.BCEWithLogitsLoss()

    # zero-shot baseline (ms-marco prior, before any fine-tuning)
    acc, met, n = evaluate(val_passage, model, tokenizer, device)
    print(f"Zero-shot ranker: sel_acc={acc:.3f} METEOR={met:.4f} (n={n})", flush=True)
    print("  baseline to beat: sel_acc=0.545 METEOR=0.4694 | oracle 0.955\n", flush=True)

    out_dir = os.path.join(DATA_DIR, "models", "para_ranker")
    os.makedirs(out_dir, exist_ok=True)
    best = -1.0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, (ids, att, y) in enumerate(loader):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device))
            loss = lossf(out.logits.squeeze(-1), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running += loss.item()
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} loss {running/200:.4f}",
                      flush=True)
                running = 0.0
        acc, met, n = evaluate(val_passage, model, tokenizer, device)
        print(f"Epoch {epoch+1}: sel_acc={acc:.3f} passage METEOR={met:.4f}", flush=True)
        if met > best:
            best = met
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  ^ new best — saved to {out_dir}", flush=True)

    print(f"\nDone. Best passage METEOR={best:.4f} (was 0.4694, oracle ~0.74)")


if __name__ == "__main__":
    main()
