import statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import prepare
import train as T


def norm(s):
    return " ".join(s.lower().split())


def main():
    val = prepare.load_split("val")
    recs = [r for r in val if r.get("tags", [""])[0] == "passage"]
    print(f"gold-passage val records: {len(recs)}\n")

    # ---------- Part A: data-only ----------
    ratios, para_lens, gold_lens, found = [], [], [], 0
    multi_para = 0
    for r in recs:
        ctx = prepare.build_context(r)
        gold = " ".join(r["spoiler"]).strip()
        i = ctx.find(gold)
        if i < 0:
            continue
        found += 1
        for ps, pe, txt in T._paragraph_offsets(r):
            if ps <= i < pe:
                ratios.append(len(gold) / max(1, len(txt)))
                para_lens.append(len(txt))
                gold_lens.append(len(gold))
                if i + len(gold) > pe:      # gold runs past this paragraph
                    multi_para += 1
                break

    print("=== A) Is a passage spoiler a whole paragraph? (no model) ===")
    print(f"  gold found verbatim in context: {found}/{len(recs)}")
    if ratios:
        print(f"  len(gold)/len(paragraph): mean={st.mean(ratios):.3f} "
              f"median={st.median(ratios):.3f}")
        print(f"  mean gold chars={st.mean(gold_lens):.0f}  "
              f"mean paragraph chars={st.mean(para_lens):.0f}")
        print(f"  gold spans BEYOND one paragraph: {multi_para}/{len(ratios)}")
        near1 = sum(1 for x in ratios if x >= 0.8)
        half = sum(1 for x in ratios if x < 0.5)
        print(f"  ratio>=0.8 (paragraph ~= gold): {near1}/{len(ratios)}")
        print(f"  ratio<0.5  (we emit >2x gold): {half}/{len(ratios)}")

    # ---------- Part B: live strategy ----------
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(prepare.QA_MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(prepare.QA_MODEL_DIR).to(device)
    model.eval()

    hit_m, miss_m, cover = [], [], []
    for n, r in enumerate(recs, 1):
        ctx = prepare.build_context(r)
        pred = T.passage_full_paragraph(r, model, tok, device, ctx)
        gold = " ".join(r["spoiler"]).strip()
        m = prepare.meteor([pred], [gold])
        if norm(gold) in norm(pred):
            hit_m.append(m)
        else:
            miss_m.append(m)
        cover.append(len(gold) / max(1, len(pred)))
        if n % 50 == 0:
            print(f"  ...{n}/{len(recs)}", flush=True)

    tot = len(hit_m) + len(miss_m)
    print("\n=== B) Live passage strategy ===")
    print(f"  SELECTION ok (emitted text contains gold): {len(hit_m)}/{tot} "
          f"= {len(hit_m)/tot:.3f}")
    print(f"  METEOR when selection ok  : {st.mean(hit_m):.4f}" if hit_m else "")
    print(f"  METEOR when selection bad : {st.mean(miss_m):.4f}" if miss_m else "")
    print(f"  overall passage METEOR    : {(sum(hit_m)+sum(miss_m))/tot:.4f}")
    print(f"  len(gold)/len(pred) coverage: mean={st.mean(cover):.3f} "
          f"median={st.median(cover):.3f}")
    print("\nRead: low SELECTION ok  -> fix paragraph choice (retrieval).")
    print("      high SELECTION ok but low METEOR/coverage -> fix EXTENT (windowing).")


if __name__ == "__main__":
    main()
