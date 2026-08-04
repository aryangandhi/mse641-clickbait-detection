import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sacrebleu

import prepare  # objective metric (METEOR) lives here — single source of truth

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# Base QA model. Override with env QA_MODEL_NAME to try a stronger extractor
# (e.g. deepset/roberta-large-squad2). Must be SQuAD-pretrained — a bare base
# model gets a random QA head (Pal 2024 / Hagen 2022 both init from SQuAD).
MODEL_NAME = os.environ.get("QA_MODEL_NAME", "deepset/roberta-base-squad2")
MAX_LEN = 384
STRIDE = 128
# roberta-base fits batch 8 on 16GB M4; larger models need less. Override w/ QA_BATCH.
BATCH = int(os.environ.get("QA_BATCH", "8"))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-5
MAX_ANS_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_context(r):
    parts = [r.get("targetTitle", "")] + r.get("targetParagraphs", [])
    return " ".join(p for p in parts if p)


def find_answers(r, context):
    """Return list of (answer_text, char_start) located in context."""
    out = []
    for spoiler in r.get("spoiler", []):
        s = spoiler.strip()
        if not s:
            continue
        idx = context.find(s)
        if idx < 0:
            # normalize whitespace and retry
            norm = " ".join(s.split())
            idx = context.find(norm)
            s = norm if idx >= 0 else s
        if idx >= 0:
            out.append((s, idx))
    return out


def make_train_features(records, tokenizer):
    questions, contexts, answers = [], [], []
    for r in records:
        ctx = build_context(r)
        q = " ".join(r.get("postText", []))
        for text, start in find_answers(r, ctx):
            questions.append(q)
            contexts.append(ctx)
            answers.append((start, start + len(text)))
    enc = tokenizer(
        questions, contexts,
        max_length=MAX_LEN, stride=STRIDE, truncation="only_second",
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length",
    )
    start_pos, end_pos = [], []
    for i, offsets in enumerate(enc["offset_mapping"]):
        sample_idx = enc["overflow_to_sample_mapping"][i]
        a_start, a_end = answers[sample_idx]
        seq_ids = enc.sequence_ids(i)
        cls_idx = enc["input_ids"][i].index(tokenizer.cls_token_id)
        # context token range
        ctx_tokens = [j for j, s in enumerate(seq_ids) if s == 1]
        if not ctx_tokens:
            start_pos.append(cls_idx); end_pos.append(cls_idx); continue
        c0, c1 = ctx_tokens[0], ctx_tokens[-1]
        if offsets[c0][0] > a_start or offsets[c1][1] < a_end:
            start_pos.append(cls_idx); end_pos.append(cls_idx); continue
        ts = c0
        while ts <= c1 and offsets[ts][0] <= a_start:
            ts += 1
        te = c1
        while te >= c0 and offsets[te][1] >= a_end:
            te -= 1
        start_pos.append(ts - 1)
        end_pos.append(te + 1)
    enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")

    # Downsample no-answer (CLS-labeled) windows to ~1:1 with answer windows —
    # inference always extracts a span, so null-heavy training just wastes steps.
    rng = np.random.RandomState(SEED)
    pos_idx = [i for i, (s, e) in enumerate(zip(start_pos, end_pos)) if not (s == e == 0)]
    neg_idx = [i for i, (s, e) in enumerate(zip(start_pos, end_pos)) if s == e == 0]
    keep_neg = rng.choice(len(neg_idx), size=min(len(pos_idx), len(neg_idx)), replace=False)
    keep = sorted(pos_idx + [neg_idx[i] for i in keep_neg])
    print(f"windows: {len(start_pos)} total, {len(pos_idx)} with answer, keeping {len(keep)}")

    return {
        "input_ids": torch.tensor([enc["input_ids"][i] for i in keep]),
        "attention_mask": torch.tensor([enc["attention_mask"][i] for i in keep]),
        "start_positions": torch.tensor([start_pos[i] for i in keep]),
        "end_positions": torch.tensor([end_pos[i] for i in keep]),
    }


@torch.no_grad()
def predict_spoilers(records, model, tokenizer, device, max_ans_len=MAX_ANS_LEN, batch_size=16):
    """Best-span prediction with sliding windows; returns list of strings."""
    model.eval()
    preds = []
    for r in records:
        ctx = build_context(r)
        q = " ".join(r.get("postText", []))
        enc = tokenizer(
            [q], [ctx],
            max_length=MAX_LEN, stride=STRIDE, truncation="only_second",
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding="max_length", return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping")
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        best_score, best_span = -1e9, (0, 0)
        for b in range(0, input_ids.shape[0], batch_size):
            out = model(input_ids=input_ids[b:b+batch_size],
                        attention_mask=attention[b:b+batch_size])
            for w in range(out.start_logits.shape[0]):
                widx = b + w
                seq_ids = enc.sequence_ids(widx)
                mask = torch.tensor(
                    [s == 1 for s in seq_ids], device=device
                )
                sl = out.start_logits[w].masked_fill(~mask, -1e9)
                el = out.end_logits[w].masked_fill(~mask, -1e9)
                top_s = torch.topk(sl, k=min(20, sl.shape[0])).indices
                top_e = torch.topk(el, k=min(20, el.shape[0])).indices
                for si in top_s.tolist():
                    for ei in top_e.tolist():
                        if ei < si or ei - si + 1 > max_ans_len:
                            continue
                        score = (sl[si] + el[ei]).item()
                        if score > best_score:
                            off = offsets[widx]
                            best_score = score
                            best_span = (off[si][0].item(), off[ei][1].item())
        s, e = best_span
        ans = ctx[s:e].strip()
        if not ans:
            ans = ctx.split(".")[0].strip()
        preds.append(ans)
        if len(preds) % 100 == 0:
            print(f"  predicted {len(preds)}/{len(records)}", flush=True)
    return preds


def bleu(preds, golds):
    return sacrebleu.corpus_bleu(preds, [golds]).score


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Fine-tune {MODEL_NAME} | epochs={EPOCHS} lr={LR} max_ans={MAX_ANS_LEN} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device).float()  # MPS dtype safety

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))

    feats = make_train_features(train, tokenizer)
    n = feats["input_ids"].shape[0]
    print(f"Train windows: {n}")
    ds = torch.utils.data.TensorDataset(
        feats["input_ids"], feats["attention_mask"],
        feats["start_positions"], feats["end_positions"],
    )
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1.0 - s / total))

    gold_joined = [" ".join(r["spoiler"]) for r in val]
    gold_first = [r["spoiler"][0] for r in val]

    out_dir = os.environ.get("QA_OUT_DIR", os.path.join(DATA_DIR, "models", "qa_ft"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "experiments"), exist_ok=True)
    preds_path = os.path.join(DATA_DIR, "experiments", "val_preds_qa_ft.json")
    best_meteor, best_epoch = -1.0, 0

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, (ids, att, sp, ep) in enumerate(loader):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                        start_positions=sp.to(device), end_positions=ep.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            running += out.loss.item()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} loss {running/100:.4f}", flush=True)
                running = 0.0

        print(f"Epoch {epoch+1} done — evaluating on val...", flush=True)
        preds = predict_spoilers(val, model, tokenizer, device)
        meteor_joined = prepare.meteor(preds, gold_joined)  # PRIMARY: competition metric
        b_joined = bleu(preds, gold_joined)
        b_first = bleu(preds, gold_first)
        em = sum(p.strip() == g.strip() for p, g in zip(preds, gold_joined)) / len(val)
        print(f"Epoch {epoch+1}: val METEOR(joined)={meteor_joined:.4f}  "
              f"[BLEU(joined)={b_joined:.2f} BLEU(first)={b_first:.2f} EM={em:.3f}]", flush=True)

        # Save the BEST epoch by val METEOR (our objective), not just the last.
        if meteor_joined > best_meteor:
            best_meteor, best_epoch = meteor_joined, epoch + 1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            with open(preds_path, "w") as f:
                json.dump(preds, f)
            print(f"  ^ new best val METEOR — saved epoch {epoch+1} to {out_dir}", flush=True)

    print(f"Done. Best epoch = {best_epoch} at val METEOR={best_meteor:.4f} (model in {out_dir})")


if __name__ == "__main__":
    main()
