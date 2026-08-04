import sys

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import prepare
import train as T

# (mode, max_ans_len, top_k) candidates per type
# Round 2: round 1 found phrase(span,25) and multi(span,k=4) BEST but both sat at
# the grid edge while still improving monotonically, so push past the boundary.
# passage is settled (fullpara clearly beat every span variant) — keep one point
# so the weighted overall projection still works.
GRIDS = {
    "phrase": [("span", 25, 1), ("span", 35, 1), ("span", 50, 1), ("span", 75, 1),
               ("fullpara", 60, 1)],
    "passage": [("fullpara", 60, 1)],
    "multi": [("span", 30, 4), ("span", 30, 5), ("span", 30, 6), ("span", 30, 8),
              ("span", 50, 4)],
}


def predict_one(r, mode, mal, k, model, tok, device):
    ctx = prepare.build_context(r)
    if mode == "span":
        return " ".join(T.spans_for_record(r, model, tok, device, mal, k))
    spans = T.spans_for_record(r, model, tok, device, mal, k, return_offsets=True)
    if not spans:
        return ctx.split(".")[0].strip()
    paras = T._paragraph_offsets(r)
    out, seen = [], set()
    for s, e in spans:
        mid = (s + e) / 2
        hit = next((txt.strip() for ps, pe, txt in paras if ps <= mid < pe), None)
        if hit is None:
            hit = ctx[s:e].strip()
        if hit and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return " ".join(out) or ctx.split(".")[0].strip()


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(prepare.QA_MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(prepare.QA_MODEL_DIR).to(device)
    model.eval()

    val = prepare.load_split("val")
    types = T.load_types("val")
    golds = prepare.gold_spoilers(val)
    if not types:
        raise SystemExit("Need experiments/pred_types_val.json for the per-type sweep.")

    # Baseline for reference: current CONFIG behaviour per type.
    current = {"phrase": ("span", 25, 1), "passage": ("fullpara", 60, 1),
               "multi": ("span", 30, 4)}  # round-1 winners
    best_cfg, weighted = {}, 0.0
    n_total = len(val)

    for t, grid in GRIDS.items():
        if only and t != only:
            continue
        idx = [i for i, x in enumerate(types) if x == t]
        recs = [val[i] for i in idx]
        gold_t = [golds[i] for i in idx]
        print(f"\n=== {t}  (n={len(recs)}) ===", flush=True)
        results = []
        for (mode, mal, k) in grid:
            preds = [predict_one(r, mode, mal, k, model, tok, device) for r in recs]
            m = prepare.meteor(preds, gold_t)
            tag = " <- current" if current.get(t) == (mode, mal, k) else ""
            print(f"  {mode:8} max_ans={mal:<4} top_k={k}  METEOR={m:.4f}{tag}", flush=True)
            results.append((m, (mode, mal, k)))
        results.sort(reverse=True)
        bm, bcfg = results[0]
        best_cfg[t] = bcfg
        weighted += bm * len(recs)
        print(f"  BEST {t}: {bcfg} METEOR={bm:.4f}", flush=True)

    if not only:
        print("\n=== projected overall (best per type, weighted by n) ===")
        print(f"  overall val METEOR ~= {weighted / n_total:.4f}  (current best 0.4261)")
        print(f"  best config: {best_cfg}")


if __name__ == "__main__":
    main()
