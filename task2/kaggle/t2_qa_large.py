"""
Kaggle GPU notebook: deepset/roberta-large-squad2 QA fine-tune for Task 2
(spoiler generation). PAPER_DOSSIER.md Sec 6 names this "the one lever that
could improve extraction AND passage selection at once" -- the local attempt
is interrupted (task2/models/qa_ft_large/ is empty) because a roberta-large
fine-tune plus decoding sweep didn't fit the M4/16GB MPS budget.

Two-stage pipeline, mirroring the local repo's two-file split:
  1. Fine-tune (port of exp_finetune_qa.py): train on gold spoiler spans,
     select the best epoch by plain (non-type-conditioned) val METEOR --
     exactly what exp_finetune_qa.py does locally.
  2. Decode (port of train.py CONFIG + spans_for_record/passage_full_paragraph/
     predict): apply the current best type-conditioned decoding rule (phrase
     <=75/1 span, passage = full paragraph, multi <=30x5 spans) using this
     project's actual Task-1 predicted types, to get the real, comparable
     val METEOR and a test submission CSV.

prepare.meteor() and build_context() are embedded verbatim (byte-for-byte
copies of task2/prepare.py) so the reported number is directly comparable to
local tuning_runs.csv rows -- not a reimplementation.

Run via Kaggle MCP: save_notebook with
  competitionDataSources=["task-2-clickbait-detection-mse-641-s-26"],
  enableGpu=True, enableInternet=True, kernelType="script".
Outputs to /kaggle/working/: experiments/val_preds_qa_large.json,
experiments/qa_large_summary.json, submissions/qa_large.csv.
"""

import json
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "sacrebleu", "nltk"], check=True)
# Kaggle's preinstalled torch build only ships sm_70+ kernels, but the GPU
# scheduler sometimes hands out a P100 (sm_60, Pascal). Pin a stable CUDA 12.1
# wheel that still includes Pascal kernels so the script works on whichever
# accelerator (P100 or T4) actually gets assigned.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.4.1", "--index-url",
                 "https://download.pytorch.org/whl/cu121"], check=True)
# The preinstalled torchvision/torchaudio are compiled against the original
# (newer) torch and break at import time once torch is downgraded above
# (RuntimeError: operator torchvision::nms does not exist, surfacing as a
# transformers ModuleNotFoundError for the model class). We never use vision
# or audio code here, so just remove them instead of chasing a matching
# version triple.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                 "torchvision", "torchaudio"], check=True)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score as _meteor_sentence

DATA_DIR = "/kaggle/input/task-2-clickbait-detection-mse-641-s-26"
OUT_DIR = "/kaggle/working"
os.makedirs(os.path.join(OUT_DIR, "experiments"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "submissions"), exist_ok=True)

QA_MODEL_NAME = "deepset/roberta-large-squad2"
MAX_LEN = 384
STRIDE = 128
BATCH = 8
EPOCHS = 3
LR = 3e-5
MAX_ANS_LEN = 50  # default cap used during epoch-selection decoding only
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- verbatim from prepare.py: THE objective metric ---
_WORDNET_READY = False


def _ensure_wordnet():
    global _WORDNET_READY
    if _WORDNET_READY:
        return
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        _WORDNET_READY = True
        return
    except LookupError:
        pass
    import contextlib
    import io
    for res in ("wordnet", "omw-1.4"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    _WORDNET_READY = True


def meteor(preds, golds):
    _ensure_wordnet()
    if not preds:
        return 0.0
    total = sum(_meteor_sentence([g.lower().split()], p.lower().split())
                for p, g in zip(preds, golds))
    return total / len(preds)


def bleu(preds, golds):
    return sacrebleu.corpus_bleu(preds, [golds]).score


def build_context(r):
    parts = [r.get("targetTitle", "")] + r.get("targetParagraphs", [])
    return " ".join(p for p in parts if p)
# --- end verbatim from prepare.py ---


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# --- verbatim from exp_finetune_qa.py: fine-tuning ---
def find_answers(r, context):
    out = []
    for spoiler in r.get("spoiler", []):
        s = spoiler.strip()
        if not s:
            continue
        idx = context.find(s)
        if idx < 0:
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
    """Simple best-span decode (no type conditioning) -- used only to pick the
    best epoch during fine-tuning, matching exp_finetune_qa.py exactly."""
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
                mask = torch.tensor([s == 1 for s in seq_ids], device=device)
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
# --- end verbatim from exp_finetune_qa.py ---


# --- verbatim from train.py: type-conditioned decoding ---
CONFIG = {
    "max_len": MAX_LEN,
    "stride": STRIDE,
    "type_cfg": {"phrase": (75, 1), "passage": (60, 1), "multi": (30, 5)},
    "top_k_logits": 20,
    "passage_full_para": True,
}


@torch.no_grad()
def spans_for_record(r, model, tokenizer, device, max_ans_len, top_k=1,
                     return_offsets=False):
    ctx = build_context(r)
    q = " ".join(r.get("postText", []))
    enc = tokenizer([q], [ctx], max_length=CONFIG["max_len"], stride=CONFIG["stride"],
                    truncation="only_second", return_overflowing_tokens=True,
                    return_offsets_mapping=True, padding="max_length",
                    return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    candidates = []
    for b in range(0, input_ids.shape[0], 16):
        out = model(input_ids=input_ids[b:b+16], attention_mask=attention[b:b+16])
        for w in range(out.start_logits.shape[0]):
            widx = b + w
            seq_ids = enc.sequence_ids(widx)
            mask = torch.tensor([s == 1 for s in seq_ids], device=device)
            sl = out.start_logits[w].masked_fill(~mask, -1e9)
            el = out.end_logits[w].masked_fill(~mask, -1e9)
            k = min(CONFIG["top_k_logits"], sl.shape[0])
            for si in torch.topk(sl, k=k).indices.tolist():
                for ei in torch.topk(el, k=k).indices.tolist():
                    if ei < si or ei - si + 1 > max_ans_len:
                        continue
                    off = offsets[widx]
                    candidates.append(((sl[si] + el[ei]).item(),
                                       off[si][0].item(), off[ei][1].item()))
    candidates.sort(reverse=True)
    picked = []
    for score, s, e in candidates:
        if e <= s or any(not (e <= ps or s >= pe) for _, ps, pe in picked):
            continue
        picked.append((score, s, e))
        if len(picked) >= top_k:
            break
    if return_offsets:
        return [(s, e) for _, s, e in picked]
    texts = [t for t in (ctx[s:e].strip() for _, s, e in picked) if t]
    return texts or [ctx.split(".")[0].strip()]


def _paragraph_offsets(r):
    items = [(r.get("targetTitle", ""), False)]
    items += [(p, True) for p in r.get("targetParagraphs", [])]
    offs, pos = [], 0
    for text, is_para in items:
        if not text:
            continue
        start = pos
        end = pos + len(text)
        if is_para:
            offs.append((start, end, text))
        pos = end + 1
    return offs


def passage_full_paragraph(r, model, tokenizer, device, ctx):
    mal = CONFIG["type_cfg"].get("passage", (MAX_ANS_LEN, 1))[0]
    spans = spans_for_record(r, model, tokenizer, device, mal, top_k=1,
                             return_offsets=True)
    if not spans:
        return ctx.split(".")[0].strip()
    s, e = spans[0]
    mid = (s + e) / 2
    for ps, pe, text in _paragraph_offsets(r):
        if ps <= mid < pe:
            return text.strip()
    return ctx[s:e].strip() or ctx.split(".")[0].strip()


def predict_typed(records, types, model, tokenizer, device):
    preds = []
    for i, r in enumerate(records):
        t = types[i] if types else None
        if CONFIG.get("passage_full_para") and t == "passage":
            ctx = build_context(r)
            preds.append(passage_full_paragraph(r, model, tokenizer, device, ctx))
        else:
            mal, k = CONFIG["type_cfg"].get(t, (MAX_ANS_LEN, 1)) if t else (MAX_ANS_LEN, 1)
            preds.append(" ".join(spans_for_record(r, model, tokenizer, device, mal, k)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)}", flush=True)
    return preds


def meteor_by_type(preds, golds, types):
    buckets = {}
    for p, g, t in zip(preds, golds, types):
        buckets.setdefault(t, ([], []))
        buckets[t][0].append(p)
        buckets[t][1].append(g)
    parts = [f"{t}={meteor(ps, gs):.4f}(n={len(ps)})"
             for t, (ps, gs) in sorted(buckets.items())]
    print("  by type: " + "  ".join(parts))
# --- end verbatim from train.py ---


# Task-1 predicted types, embedded inline (task2/experiments/pred_types_{val,test}.json
# -- generated by the local Task-1 classifier; too small to warrant a dataset upload).
PRED_TYPES_VAL = json.loads(r'''["passage", "passage", "multi", "multi", "passage", "phrase", "phrase", "passage", "passage", "passage", "multi", "phrase", "multi", "phrase", "phrase", "multi", "passage", "phrase", "passage", "passage", "phrase", "passage", "passage", "phrase", "passage", "phrase", "phrase", "multi", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "multi", "passage", "phrase", "multi", "passage", "phrase", "multi", "passage", "phrase", "passage", "phrase", "passage", "multi", "multi", "multi", "passage", "passage", "passage", "phrase", "phrase", "multi", "passage", "phrase", "phrase", "passage", "phrase", "phrase", "multi", "passage", "phrase", "passage", "phrase", "passage", "phrase", "phrase", "phrase", "multi", "passage", "phrase", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "phrase", "phrase", "passage", "passage", "phrase", "multi", "passage", "multi", "passage", "passage", "passage", "multi", "multi", "passage", "multi", "phrase", "passage", "phrase", "passage", "phrase", "passage", "phrase", "passage", "passage", "phrase", "multi", "passage", "phrase", "phrase", "multi", "passage", "multi", "multi", "passage", "multi", "phrase", "passage", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "phrase", "multi", "multi", "multi", "passage", "multi", "multi", "phrase", "passage", "phrase", "multi", "passage", "passage", "phrase", "passage", "phrase", "phrase", "multi", "passage", "phrase", "phrase", "passage", "phrase", "phrase", "phrase", "passage", "phrase", "phrase", "phrase", "passage", "passage", "phrase", "passage", "multi", "phrase", "phrase", "phrase", "phrase", "phrase", "multi", "passage", "phrase", "passage", "phrase", "phrase", "multi", "passage", "phrase", "multi", "passage", "phrase", "phrase", "phrase", "phrase", "phrase", "multi", "passage", "passage", "multi", "multi", "passage", "passage", "phrase", "passage", "phrase", "passage", "multi", "phrase", "multi", "phrase", "phrase", "passage", "phrase", "passage", "passage", "multi", "multi", "phrase", "multi", "passage", "passage", "multi", "passage", "phrase", "passage", "passage", "multi", "multi", "passage", "multi", "phrase", "passage", "multi", "passage", "passage", "phrase", "phrase", "phrase", "phrase", "multi", "passage", "passage", "passage", "passage", "multi", "phrase", "phrase", "phrase", "passage", "multi", "multi", "multi", "passage", "phrase", "passage", "phrase", "phrase", "passage", "multi", "passage", "phrase", "passage", "passage", "phrase", "multi", "multi", "phrase", "multi", "passage", "multi", "passage", "phrase", "passage", "passage", "phrase", "passage", "phrase", "phrase", "multi", "multi", "phrase", "multi", "multi", "phrase", "passage", "multi", "phrase", "passage", "phrase", "phrase", "passage", "passage", "passage", "phrase", "passage", "phrase", "phrase", "multi", "passage", "phrase", "phrase", "passage", "passage", "multi", "phrase", "phrase", "passage", "phrase", "multi", "phrase", "passage", "passage", "passage", "passage", "multi", "phrase", "passage", "passage", "phrase", "passage", "passage", "phrase", "phrase", "multi", "passage", "passage", "phrase", "phrase", "phrase", "passage", "passage", "passage", "multi", "passage", "multi", "phrase", "passage", "passage", "passage", "phrase", "phrase", "phrase", "passage", "phrase", "multi", "phrase", "passage", "multi", "phrase", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "multi", "phrase", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "multi", "passage", "phrase", "phrase", "passage", "passage", "multi", "phrase", "multi", "phrase", "passage", "phrase", "multi", "multi", "passage", "phrase", "passage", "phrase", "passage", "passage", "phrase", "passage", "passage", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "passage", "phrase", "passage", "phrase", "phrase", "phrase", "phrase", "phrase", "phrase", "phrase", "passage", "passage", "passage", "passage", "multi", "phrase", "passage", "phrase", "passage", "multi", "phrase"]''')
PRED_TYPES_TEST = json.loads(r'''["phrase", "passage", "phrase", "phrase", "passage", "phrase", "multi", "passage", "phrase", "phrase", "passage", "passage", "phrase", "passage", "multi", "phrase", "passage", "passage", "multi", "phrase", "phrase", "passage", "passage", "phrase", "passage", "passage", "passage", "passage", "passage", "multi", "phrase", "passage", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "passage", "passage", "phrase", "passage", "multi", "passage", "passage", "passage", "multi", "passage", "multi", "passage", "phrase", "phrase", "multi", "passage", "phrase", "passage", "passage", "phrase", "phrase", "passage", "passage", "passage", "phrase", "phrase", "phrase", "phrase", "multi", "multi", "multi", "passage", "passage", "phrase", "multi", "phrase", "phrase", "phrase", "multi", "phrase", "passage", "passage", "passage", "phrase", "phrase", "multi", "phrase", "passage", "multi", "multi", "passage", "passage", "phrase", "passage", "phrase", "passage", "passage", "multi", "multi", "passage", "passage", "phrase", "passage", "phrase", "passage", "multi", "phrase", "passage", "passage", "passage", "passage", "phrase", "passage", "phrase", "passage", "passage", "phrase", "passage", "phrase", "phrase", "passage", "passage", "phrase", "passage", "passage", "phrase", "passage", "passage", "multi", "passage", "phrase", "phrase", "passage", "passage", "multi", "multi", "passage", "phrase", "phrase", "multi", "passage", "passage", "passage", "passage", "multi", "passage", "phrase", "passage", "phrase", "phrase", "multi", "passage", "passage", "multi", "phrase", "multi", "phrase", "phrase", "passage", "passage", "passage", "phrase", "passage", "passage", "phrase", "multi", "phrase", "phrase", "passage", "phrase", "phrase", "phrase", "phrase", "passage", "passage", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "phrase", "passage", "passage", "passage", "multi", "passage", "passage", "phrase", "phrase", "phrase", "passage", "passage", "multi", "passage", "phrase", "passage", "passage", "multi", "multi", "phrase", "phrase", "phrase", "phrase", "multi", "phrase", "phrase", "multi", "phrase", "passage", "passage", "multi", "passage", "phrase", "phrase", "phrase", "phrase", "phrase", "passage", "phrase", "passage", "phrase", "multi", "phrase", "phrase", "phrase", "passage", "phrase", "passage", "multi", "multi", "passage", "phrase", "passage", "multi", "passage", "passage", "phrase", "passage", "passage", "passage", "passage", "phrase", "passage", "phrase", "passage", "passage", "passage", "phrase", "phrase", "passage", "passage", "multi", "passage", "passage", "passage", "passage", "passage", "passage", "phrase", "phrase", "phrase", "phrase", "phrase", "passage", "phrase", "phrase", "passage", "passage", "passage", "phrase", "passage", "passage", "phrase", "passage", "passage", "multi", "passage", "passage", "passage", "multi", "multi", "passage", "passage", "passage", "phrase", "passage", "multi", "phrase", "multi", "passage", "multi", "passage", "phrase", "multi", "phrase", "phrase", "multi", "phrase", "multi", "multi", "phrase", "passage", "phrase", "multi", "phrase", "phrase", "phrase", "passage", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "phrase", "passage", "passage", "phrase", "phrase", "phrase", "multi", "multi", "passage", "phrase", "multi", "phrase", "phrase", "multi", "passage", "passage", "multi", "passage", "multi", "phrase", "phrase", "passage", "passage", "passage", "passage", "passage", "passage", "phrase", "passage", "phrase", "phrase", "passage", "phrase", "passage", "passage", "multi", "passage", "phrase", "phrase", "passage", "phrase", "phrase", "phrase", "passage", "passage", "passage", "phrase", "multi", "phrase", "passage", "phrase", "passage", "passage", "passage", "passage", "passage", "multi", "phrase", "phrase", "phrase", "phrase", "multi", "passage", "passage", "multi", "passage", "multi", "multi", "phrase", "phrase", "multi", "multi", "multi", "phrase", "multi", "multi", "passage", "passage", "phrase", "multi", "passage", "phrase"]''')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tune {QA_MODEL_NAME} | epochs={EPOCHS} lr={LR} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).to(device)

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

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
    # Real CUDA fp16 (unlike MPS, no correctness bug here -- see dossier Sec 2) for speed.
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    gold_joined = [" ".join(r["spoiler"]) for r in val]

    best_meteor, best_epoch, best_state = -1.0, 0, None

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, (ids, att, sp, ep) in enumerate(loader):
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                            start_positions=sp.to(device), end_positions=ep.to(device))
            scaler.scale(out.loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += out.loss.item()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(loader)} loss {running/100:.4f}", flush=True)
                running = 0.0

        print(f"Epoch {epoch+1} done -- evaluating on val (plain decode)...", flush=True)
        preds = predict_spoilers(val, model, tokenizer, device)
        meteor_joined = meteor(preds, gold_joined)
        print(f"Epoch {epoch+1}: val METEOR(joined, plain decode)={meteor_joined:.4f}", flush=True)

        if meteor_joined > best_meteor:
            best_meteor, best_epoch = meteor_joined, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ^ new best plain-decode val METEOR -- epoch {epoch+1}", flush=True)

    print(f"Fine-tune done. Best epoch = {best_epoch} at plain val METEOR={best_meteor:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    print("\nApplying type-conditioned decoding (current best local CONFIG) on val...", flush=True)
    val_preds = predict_typed(val, PRED_TYPES_VAL, model, tokenizer, device)
    val_meteor = meteor(val_preds, gold_joined)
    val_bleu = bleu(val_preds, gold_joined)
    print(f"val METEOR(joined)={val_meteor:.4f}", flush=True)
    meteor_by_type(val_preds, gold_joined, PRED_TYPES_VAL)
    print(f"val BLEU(joined)={val_bleu:.2f}", flush=True)

    print("\nApplying type-conditioned decoding on test...", flush=True)
    test_preds = predict_typed(test, PRED_TYPES_TEST, model, tokenizer, device)

    with open(os.path.join(OUT_DIR, "experiments", "val_preds_qa_large.json"), "w") as f:
        json.dump(val_preds, f)
    with open(os.path.join(OUT_DIR, "experiments", "qa_large_summary.json"), "w") as f:
        json.dump({"model": QA_MODEL_NAME, "epochs": EPOCHS, "lr": LR,
                    "best_epoch": best_epoch, "plain_val_meteor": best_meteor,
                    "typed_val_meteor": val_meteor, "typed_val_bleu": val_bleu}, f, indent=2)

    import csv
    with open(os.path.join(OUT_DIR, "submissions", "qa_large.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoiler"])
        w.writerows(zip([r.get("id") for r in test], test_preds))

    print(f"\nDone. typed val METEOR={val_meteor:.4f} (vs local best 0.4705)")


if __name__ == "__main__":
    main()
