import json
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import prepare  # objective metric (METEOR)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "google/flan-t5-base"
MAX_SRC = 384
MAX_TGT = 100
BATCH = 4
TOPK_PARA = 5
GEN_TYPES = ("passage", "multi")  # where generation is expected to beat extraction
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4  # T5 fine-tunes at higher lr
# Optional 3rd arg: init from an existing checkpoint (e.g. models/gen_ft) to
# CONTINUE training instead of restarting from the base model.
INIT_FROM = sys.argv[3] if len(sys.argv) > 3 else MODEL_NAME
SEED = 42

torch.manual_seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_types(split):
    p = os.path.join(DATA_DIR, "experiments", f"pred_types_{split}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def topk_paragraphs(post, paras, k=TOPK_PARA):
    """Keep the k paragraphs most similar to the post (TF-IDF cosine), original order."""
    if len(paras) <= k:
        return paras
    try:
        vec = TfidfVectorizer().fit([post] + paras)
        sims = cosine_similarity(vec.transform([post]), vec.transform(paras))[0]
        idx = sorted(range(len(paras)), key=lambda i: sims[i], reverse=True)[:k]
        return [paras[i] for i in sorted(idx)]
    except ValueError:
        return paras[:k]


def make_input(r, ptype):
    post = " ".join(r.get("postText", []))
    title = r.get("targetTitle", "")
    ctx = " ".join(topk_paragraphs(post, r.get("targetParagraphs", [])))
    return (f"spoiler type {ptype}: generate the spoiler for this clickbait. "
            f"post: {post} title: {title} context: {ctx}")


def gold_target(r):
    return " ".join(r.get("spoiler", []))


def build_tensors(records, types, tokenizer):
    inputs = [make_input(r, t) for r, t in zip(records, types)]
    targets = [gold_target(r) for r in records]
    enc = tokenizer(inputs, max_length=MAX_SRC, truncation=True,
                    padding="max_length", return_tensors="pt")
    lab = tokenizer(targets, max_length=MAX_TGT, truncation=True,
                    padding="max_length", return_tensors="pt")
    labels = lab.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return TensorDataset(enc.input_ids, enc.attention_mask, labels)


@torch.no_grad()
def generate(records, types, model, tokenizer, device):
    model.eval()
    preds = []
    for i in range(0, len(records), BATCH):
        batch, bt = records[i:i + BATCH], types[i:i + BATCH]
        inp = [make_input(r, t) for r, t in zip(batch, bt)]
        enc = tokenizer(inp, max_length=MAX_SRC, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=MAX_TGT, num_beams=4)
        preds += tokenizer.batch_decode(out, skip_special_tokens=True)
        if (i + BATCH) % 40 == 0:
            print(f"  gen {min(i + BATCH, len(records))}/{len(records)}", flush=True)
    return preds


def meteor_by_type(preds, golds, types):
    buckets = {}
    for p, g, t in zip(preds, golds, types):
        buckets.setdefault(t, ([], []))
        buckets[t][0].append(p)
        buckets[t][1].append(g)
    return {t: (prepare.meteor(ps, gs), len(ps)) for t, (ps, gs) in buckets.items()}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Fine-tune init={INIT_FROM} | epochs={EPOCHS} lr={LR} device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(INIT_FROM)
    model = AutoModelForSeq2SeqLM.from_pretrained(INIT_FROM).to(device)

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    train_types = [r.get("tags", ["phrase"])[0] for r in train]  # gold tags for training
    val_types = load_types("val") or [r.get("tags", ["phrase"])[0] for r in val]

    ds = build_tensors(train, train_types, tokenizer)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1.0 - s / total))

    # Evaluate only on the subset we will actually route to the generator.
    sub_idx = [i for i, t in enumerate(val_types) if t in GEN_TYPES]
    sub_val = [val[i] for i in sub_idx]
    sub_types = [val_types[i] for i in sub_idx]
    sub_gold = [gold_target(val[i]) for i in sub_idx]
    print(f"Val {'+'.join(GEN_TYPES)} subset: {len(sub_val)} records "
          f"(current extractive: passage 0.4102, multi 0.4260)", flush=True)

    out_dir = os.path.join(DATA_DIR, "models", "gen_ft")
    os.makedirs(out_dir, exist_ok=True)
    best_meteor = -1.0

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, (ids, att, lab) in enumerate(loader):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                        labels=lab.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running += out.loss.item()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} loss {running/100:.4f}", flush=True)
                running = 0.0

        print(f"Epoch {epoch+1} done — generating on val subset...", flush=True)
        preds = generate(sub_val, sub_types, model, tokenizer, device)
        overall = prepare.meteor(preds, sub_gold)
        bt = meteor_by_type(preds, sub_gold, sub_types)
        parts = "  ".join(f"{t}={m:.4f}(n={n})" for t, (m, n) in sorted(bt.items()))
        print(f"Epoch {epoch+1}: val {'+'.join(GEN_TYPES)} METEOR={overall:.4f}  [{parts}]", flush=True)

        if overall > best_meteor:
            best_meteor = overall
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            with open(os.path.join(DATA_DIR, "experiments", "val_preds_gen_ft.json"), "w") as f:
                json.dump({"idx": sub_idx, "preds": preds}, f)
            print(f"  ^ new best subset METEOR — saved epoch {epoch+1} to {out_dir}", flush=True)

    print(f"Done. Best val {'+'.join(GEN_TYPES)} METEOR={best_meteor:.4f} (model in {out_dir})")


if __name__ == "__main__":
    main()
