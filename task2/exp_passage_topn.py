
import statistics as st

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import prepare
import train as T


def norm(s):
    return " ".join(s.lower().split())


def top_paragraphs(r, model, tok, device, n_para, k_spans=8, mal=60):
    """Distinct paragraphs of the top spans, in article order, up to n_para."""
    ctx = prepare.build_context(r)
    spans = T.spans_for_record(r, model, tok, device, mal, top_k=k_spans,
                               return_offsets=True)
    paras = T._paragraph_offsets(r)
    chosen = []
    for s, e in spans:
        mid = (s + e) / 2
        for pi, (ps, pe, txt) in enumerate(paras):
            if ps <= mid < pe and pi not in chosen:
                chosen.append(pi)
                break
        if len(chosen) >= n_para:
            break
    if not chosen:
        return ctx.split(".")[0].strip()
    chosen.sort()
    return " ".join(paras[i][2].strip() for i in chosen)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(prepare.QA_MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(prepare.QA_MODEL_DIR).to(device)
    model.eval()

    val = prepare.load_split("val")
    recs = [r for r in val if r.get("tags", [""])[0] == "passage"]
    print(f"gold-passage val records: {len(recs)}\n")

    for n in (1, 2, 3):
        hits, mets = 0, []
        for r in recs:
            pred = top_paragraphs(r, model, tok, device, n)
            gold = " ".join(r["spoiler"]).strip()
            if norm(gold) in norm(pred):
                hits += 1
            mets.append(prepare.meteor([pred], [gold]))
        print(f"top-{n} paragraphs: containment={hits/len(recs):.3f} "
              f"METEOR={st.mean(mets):.4f}")
    print("\nbaseline (top-1 fullpara): containment 0.545, METEOR 0.4694")
    print("break-even for top-2 needs containment > ~0.70")


if __name__ == "__main__":
    main()
