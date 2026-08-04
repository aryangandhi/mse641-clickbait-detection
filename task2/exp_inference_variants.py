import json
import os
import sys
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sacrebleu

import exp_finetune_qa as E
import prepare  # objective metric (METEOR) — single source of truth

DATA_DIR = E.DATA_DIR
MODEL_DIR = os.path.join(DATA_DIR, "models", "qa_ft")


@torch.no_grad()
def spans_for_record(r, model, tokenizer, device, max_ans_len, top_k=1, batch_size=16):
    """Return top_k non-overlapping best spans (text, score)."""
    ctx = E.build_context(r)
    q = " ".join(r.get("postText", []))
    enc = tokenizer(
        [q], [ctx],
        max_length=E.MAX_LEN, stride=E.STRIDE, truncation="only_second",
        return_overflowing_tokens=True, return_offsets_mapping=True,
        padding="max_length", return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device)
    candidates = []  # (score, char_start, char_end)
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
                    off = offsets[widx]
                    candidates.append(((sl[si] + el[ei]).item(),
                                       off[si][0].item(), off[ei][1].item()))
    candidates.sort(reverse=True)
    picked = []
    for score, s, e in candidates:
        if e <= s:
            continue
        if any(not (e <= ps or s >= pe) for _, ps, pe in picked):
            continue  # overlaps an already-picked span
        picked.append((score, s, e))
        if len(picked) >= top_k:
            break
    texts = [ctx[s:e].strip() for _, s, e in picked]
    texts = [t for t in texts if t]
    if not texts:
        texts = [ctx.split(".")[0].strip()]
    return texts


def load_predicted_types(split):
    """Predicted spoiler types from task1 ensemble probs (transformer npz + linear)."""
    path = os.path.join(DATA_DIR, "experiments", f"pred_types_{split}.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "sweep_len"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    records = E.load_jsonl(os.path.join(DATA_DIR, f"{split}.jsonl"))
    if split == "val":
        gold = [" ".join(r["spoiler"]) for r in records]

    if variant == "sweep_len":
        for mal in [15, 30, 50, 80]:
            preds = [" ".join(spans_for_record(r, model, tokenizer, device, mal))
                     for r in records]
            print(f"max_ans_len={mal:<3} METEOR={prepare.meteor(preds, gold):.4f} "
                  f"[BLEU={E.bleu(preds, gold):.2f}]", flush=True)

    elif variant == "type_cond":
        types = load_predicted_types(split)
        if types is None:
            print("No predicted types file; using gold tags on val as oracle upper bound")
            types = [r["tags"][0] for r in records]
        cfg = {"phrase": (10, 1), "passage": (60, 1), "multi": (30, 3)}
        preds = []
        for r, t in zip(records, types):
            mal, k = cfg.get(t, (50, 1))
            texts = spans_for_record(r, model, tokenizer, device, mal, top_k=k)
            preds.append(" ".join(texts))
            if len(preds) % 100 == 0:
                print(f"  {len(preds)}/{len(records)}", flush=True)
        if split == "val":
            print(f"type_cond METEOR={prepare.meteor(preds, gold):.4f} "
                  f"[BLEU={E.bleu(preds, gold):.2f}]")
            json.dump(preds, open(os.path.join(DATA_DIR, "experiments", "val_preds_type_cond.json"), "w"))
        else:
            ids = [r.get("id") for r in records]
            out = os.path.join(DATA_DIR, "submissions", "qa_ft_type_cond.csv")
            pd.DataFrame({"id": ids, "spoiler": preds}).to_csv(out, index=False)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
