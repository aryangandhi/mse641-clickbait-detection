import math
import statistics as st

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import prepare
import train as T

MAX_ANS = 60
TOPK_LOGITS = 20


@torch.no_grad()
def span_candidates(r, model, tokenizer, device):
    ctx = prepare.build_context(r)
    q = " ".join(r.get("postText", []))
    enc = tokenizer([q], [ctx], max_length=384, stride=128, truncation="only_second",
                    return_overflowing_tokens=True, return_offsets_mapping=True,
                    padding="max_length", return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    enc.pop("overflow_to_sample_mapping")
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    cands = []
    for b in range(0, ids.shape[0], 16):
        out = model(input_ids=ids[b:b + 16], attention_mask=att[b:b + 16])
        for w in range(out.start_logits.shape[0]):
            widx = b + w
            seq = enc.sequence_ids(widx)
            mask = torch.tensor([s == 1 for s in seq], device=device)
            sl = out.start_logits[w].masked_fill(~mask, -1e9)
            el = out.end_logits[w].masked_fill(~mask, -1e9)
            k = min(TOPK_LOGITS, sl.shape[0])
            for si in torch.topk(sl, k=k).indices.tolist():
                for ei in torch.topk(el, k=k).indices.tolist():
                    if ei < si or ei - si + 1 > MAX_ANS:
                        continue
                    off = offsets[widx]
                    s, e = off[si][0].item(), off[ei][1].item()
                    if e > s:
                        cands.append(((sl[si] + el[ei]).item(), s, e))
    return ctx, cands


def para_of(paras, pos):
    for i, (ps, pe, _) in enumerate(paras):
        if ps <= pos < pe:
            return i
    return None


def pick_span_top1(paras, cands, sims):
    if not cands:
        return None
    _, s, e = max(cands)
    return para_of(paras, (s + e) / 2)


def pick_span_mass(paras, cands, sims):
    if not cands:
        return None
    top = max(c[0] for c in cands)
    mass = [0.0] * len(paras)
    for sc, s, e in cands:
        i = para_of(paras, (s + e) / 2)
        if i is not None:
            mass[i] += math.exp(sc - top)
    return max(range(len(paras)), key=lambda i: mass[i]) if any(mass) else None


def pick_tfidf(paras, cands, sims):
    return max(range(len(paras)), key=lambda i: sims["post"][i]) if paras else None


def pick_tfidf_pt(paras, cands, sims):
    return max(range(len(paras)), key=lambda i: sims["pt"][i]) if paras else None


def _z(xs):
    if len(xs) < 2:
        return [0.0] * len(xs)
    m, sd = st.mean(xs), (st.pstdev(xs) or 1.0)
    return [(x - m) / sd for x in xs]


def pick_hybrid(paras, cands, sims):
    if not paras or not cands:
        return None
    top = max(c[0] for c in cands)
    mass = [0.0] * len(paras)
    for sc, s, e in cands:
        i = para_of(paras, (s + e) / 2)
        if i is not None:
            mass[i] += math.exp(sc - top)
    zm, zt = _z(mass), _z(list(sims["post"]))
    return max(range(len(paras)), key=lambda i: zm[i] + zt[i])


STRATEGIES = {
    "span_top1": pick_span_top1,
    "span_mass": pick_span_mass,
    "tfidf_post": pick_tfidf,
    "tfidf_pt": pick_tfidf_pt,
    "hybrid": pick_hybrid,
}


def norm(s):
    return " ".join(s.lower().split())


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(prepare.QA_MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(prepare.QA_MODEL_DIR).to(device)
    model.eval()

    val = prepare.load_split("val")
    recs = [r for r in val if r.get("tags", [""])[0] == "passage"]
    print(f"gold-passage val records: {len(recs)}\n", flush=True)

    hits = {k: 0 for k in STRATEGIES}
    mets = {k: [] for k in STRATEGIES}
    oracle_hits = 0

    for n, r in enumerate(recs, 1):
        ctx, cands = span_candidates(r, model, tok, device)
        paras = T._paragraph_offsets(r)
        if not paras:
            continue
        gold = " ".join(r["spoiler"]).strip()
        texts = [p[2] for p in paras]
        post = " ".join(r.get("postText", []))
        pt = post + " " + r.get("targetTitle", "")
        try:
            v = TfidfVectorizer().fit(texts + [post, pt])
            sims = {"post": cosine_similarity(v.transform([post]), v.transform(texts))[0],
                    "pt": cosine_similarity(v.transform([pt]), v.transform(texts))[0]}
        except ValueError:
            sims = {"post": [0.0] * len(texts), "pt": [0.0] * len(texts)}

        # oracle: is there ANY paragraph containing the gold?
        if any(norm(gold) in norm(t) for t in texts):
            oracle_hits += 1

        for name, fn in STRATEGIES.items():
            i = fn(paras, cands, sims)
            pred = texts[i].strip() if i is not None else ctx.split(".")[0].strip()
            if norm(gold) in norm(pred):
                hits[name] += 1
            mets[name].append(prepare.meteor([pred], [gold]))
        if n % 50 == 0:
            print(f"  ...{n}/{len(recs)}", flush=True)

    tot = len(mets["span_top1"])
    print(f"\n=== passage paragraph selection (n={tot}) ===")
    print(f"{'strategy':12} {'sel_acc':>8} {'METEOR':>8}")
    for name in STRATEGIES:
        tag = "  <- current" if name == "span_top1" else ""
        print(f"{name:12} {hits[name]/tot:8.3f} {st.mean(mets[name]):8.4f}{tag}")
    print(f"\noracle (gold paragraph exists & is selectable): {oracle_hits/tot:.3f}")


if __name__ == "__main__":
    main()
