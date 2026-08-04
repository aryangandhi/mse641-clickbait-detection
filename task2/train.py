"""
Task 2 experiment file — the main decoding-experiment surface (edit this to iterate).

Strategy: extractive QA. The headline is the question, targetTitle + all
paragraphs the context; an extractive-QA model FINE-TUNED on this dataset's
gold spoiler spans (models/qa_ft, produced by exp_finetune_qa.py — see that
file for the current base model, MODEL_NAME) extracts the best span. This file
is the fast iteration surface: DECODING experiments (~6-8 min on MPS for 400
val samples).

Iterate here: max answer length, per-type length caps and top-k spans
(predicted types in experiments/pred_types_{val,test}.json), span scoring,
window handling. To change the fine-tune itself, edit exp_finetune_qa.py
and retrain — that is the slow loop, do it sparingly.

PRIMARY METRIC: METEOR. The Kaggle competition scores Task 2 with METEOR
(recall weighted higher than precision, plus stemming + WordNet synonymy),
NOT BLEU. So we SELECT/JUDGE runs on `val METEOR(joined)` and optimize for
it; BLEU is kept only as a secondary sanity number. Reference = the
space-joined gold spoiler (same target the submission is scored against).
Keep a change if val METEOR beats the best by ~>0.003 (the 400-sample val
split is noisy below that), else revert.

Contract: the last line must still contain `val BLEU(joined)=<float>` — the
fixed harness/grader parses it — but it is NOT the optimization objective.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import prepare

# ---------------- editable config ----------------
CONFIG = {
    "max_len": 384,
    "stride": 128,
    "max_ans_len": 50,          # used when type conditioning is off
    "type_cond": True,          # per-type (max_ans_len, top_k)
    # Round-2 METEOR sweep winners (exp_sweep_types.py): phrase 10->75 (+0.056),
    # multi top_k 3->5 (+0.104). Both were BLEU-era values; METEOR's recall
    # weighting rewards emitting more. passage stays fullpara (cap locates the span).
    "type_cfg": {"phrase": (75, 1), "passage": (60, 1), "multi": (30, 5)},
    "top_k_logits": 20,
    # Phase 1: for passage type, emit the FULL paragraph containing the model's
    # best span instead of a length-capped span. A passage spoiler IS a whole
    # paragraph, and METEOR weights recall > precision, so returning the whole
    # paragraph should beat a truncated span. Toggle to ablate.
    "passage_full_para": True,
}
# --------------------------------------------------


@torch.no_grad()
def spans_for_record(r, model, tokenizer, device, max_ans_len, top_k=1,
                     return_offsets=False):
    ctx = prepare.build_context(r)
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
    """Char [start, end) of each targetParagraph inside prepare.build_context(r).
    Mirrors build_context exactly: non-empty [title] + paragraphs joined by ' '."""
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
        pos = end + 1  # the single joining space
    return offs


def passage_full_paragraph(r, model, tokenizer, device, ctx):
    """Locate the model's best span, then return the WHOLE paragraph it falls in
    (high recall for METEOR). Falls back to the capped span if no paragraph maps."""
    mal = CONFIG["type_cfg"].get("passage", (CONFIG["max_ans_len"], 1))[0]
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


def predict(records, types, model, tokenizer, device):
    preds = []
    for i, r in enumerate(records):
        t = types[i] if (CONFIG["type_cond"] and types) else None
        if CONFIG.get("passage_full_para") and t == "passage":
            ctx = prepare.build_context(r)
            preds.append(passage_full_paragraph(r, model, tokenizer, device, ctx))
        else:
            mal, k = CONFIG["type_cfg"].get(t, (CONFIG["max_ans_len"], 1)) if t \
                else (CONFIG["max_ans_len"], 1)
            preds.append(" ".join(spans_for_record(r, model, tokenizer, device, mal, k)))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)}", flush=True)
    return preds


def load_types(split):
    p = os.path.join(prepare.DATA_DIR, "experiments", f"pred_types_{split}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def meteor_by_type(preds, golds, types):
    """Per-type METEOR breakdown for analysis (not logged) — lets an A/B change
    be attributed to the type bucket it actually moves."""
    if not types:
        return
    buckets = {}
    for p, g, t in zip(preds, golds, types):
        buckets.setdefault(t, ([], []))
        buckets[t][0].append(p)
        buckets[t][1].append(g)
    parts = [f"{t}={prepare.meteor(ps, gs):.4f}(n={len(ps)})"
             for t, (ps, gs) in sorted(buckets.items())]
    print("  by type: " + "  ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-submission", metavar="PATH")
    ap.add_argument("--passage-full-para", choices=["on", "off"],
                    help="override CONFIG['passage_full_para'] (Phase 1 A/B)")
    ap.add_argument("--gold-types", action="store_true",
                    help="DIAGNOSTIC ONLY: route with gold tags instead of Task 1 "
                         "predictions, to measure the oracle ceiling. Cannot be used "
                         "for a submission — the test split has no gold tags.")
    args = ap.parse_args()
    if args.passage_full_para:
        CONFIG["passage_full_para"] = (args.passage_full_para == "on")
    if args.gold_types and args.write_submission:
        raise SystemExit("--gold-types is a val diagnostic; refusing to write a "
                         "submission from oracle routing.")

    if not os.path.exists(prepare.QA_MODEL_DIR):
        raise SystemExit(
            f"Fine-tuned model missing at {prepare.QA_MODEL_DIR}. "
            "Run `python3 exp_finetune_qa.py` first (~60 min on MPS).")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(prepare.QA_MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(prepare.QA_MODEL_DIR).to(device)
    model.eval()

    val = prepare.load_split("val")
    if args.gold_types:
        val_types = [r["tags"][0] for r in val]
        print("*** ORACLE ROUTING: gold tags (diagnostic upper bound) ***")
    else:
        val_types = load_types("val")
    preds = predict(val, val_types, model, tokenizer, device)
    golds = prepare.gold_spoilers(val)
    meteor_val = prepare.meteor(preds, golds)  # PRIMARY: the competition metric
    bleu_val = prepare.bleu(preds, golds)      # secondary sanity number
    print(f"passage_full_para={CONFIG['passage_full_para']} gold_types={args.gold_types}")
    meteor_by_type(preds, golds, val_types)

    if args.gold_types:
        # Oracle diagnostic: not a real tuning run, so keep it out of the CSV
        # where it would sit incomparably beside prediction-routed rows.
        print("(oracle run — not logged to tuning_runs.csv)")
    else:
        cfg = {k: str(v) for k, v in CONFIG.items()}
        prepare.log_run(cfg, meteor_val, bleu_val)

    if args.write_submission:
        test = prepare.load_split("test")
        test_preds = predict(test, load_types("test"), model, tokenizer, device)
        prepare.write_submission([r.get("id") for r in test], test_preds,
                                 args.write_submission)

    # METEOR is the objective — judge runs on this line. BLEU is kept last
    # only because the fixed harness/grader parses `val BLEU(joined)=`.
    print(f"val METEOR(joined)={meteor_val:.4f}   <-- PRIMARY (competition metric)")
    print(f"val BLEU(joined)={bleu_val:.2f}")


if __name__ == "__main__":
    main()
