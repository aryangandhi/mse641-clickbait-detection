"""
Task 1 experiment file — the main experiment surface (edit this to iterate).

Current config: word TF-IDF (uni+bi) over headline+title+description+first
3 paragraphs, + char_wb 3-5 gram TF-IDF over the headline, + 15 handcrafted
features, LogisticRegression C=1.0 balanced.  Val weighted F1: 0.6257.

With --blend-transformer it mixes in cached transformer probabilities
(experiments/probs_<model>.npz, w=0.60). Currently roberta-base (0.6949);
can switch to other models like microsoft/deberta-v3-small in CONFIG.

The cached probs come from exp_transformer.py (e.g. ~50 min for roberta-base
on MPS, ~30-40 min for deberta-v3-small). Re-run exp_transformer.py with a new
model to generate fresh cached probs:
  python3 exp_transformer.py microsoft/deberta-v3-small 6 2e-5 512

Then update CONFIG["transformer_model"] here and re-tune blend_w_transformer.

Budget: a plain run takes ~90 s. Iterate here: features, n-gram ranges,
C, class weights, blend weight, model choice. Print stays the contract:
the last line must contain `Val weighted F1: <float>`.
"""

import argparse
import os
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import prepare

# ---------------- editable config ----------------
CONFIG = {
    "word_ngrams": (1, 2),
    "word_max_features": 50_000,
    "char_ngrams": (3, 5),
    "char_max_features": 50_000,
    "C": 1.0,
    "class_weight": "balanced",
    "n_body_paragraphs": 0,
    "blend_w_transformer": 0.60,
    "transformer_model": "microsoft/deberta-v3-base",  # changed to deberta-v3-base (0.7004 best)
    # Exp A: pick the n_body_paragraphs most relevant to the post+title (lexical
    # overlap) instead of the first n. Billy-Batson (SemEval-2023 winner) showed
    # relevance-condensed articles beat raw first-N paragraphs.
    "relevance_paragraphs": False,
}
# --------------------------------------------------

# Minimal stopword set so overlap scoring keys on content words, not glue words.
_STOP = frozenset(
    "the a an and or but of to in on at for with as by from is are was were be been "
    "being this that these those it its it's he she they them his her their you your i "
    "we our us not no do does did has have had will would can could should about into "
    "over after before then than so if out up down more most some such which who what "
    "when where why how".split()
)
_TOK_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text):
    return [t for t in _TOK_RE.findall(text.lower()) if t not in _STOP and len(t) > 2]


def select_paragraphs(paras, query, k):
    """Return the k paragraphs most relevant to `query`, in original order.

    Relevance = sum over query terms of the term's frequency in the paragraph,
    length-normalized so long paragraphs don't win by sheer size. Falls back to
    first-k when the query is empty or there are <=k paragraphs."""
    if len(paras) <= k:
        return paras
    q = set(_content_tokens(query))
    if not q:
        return paras[:k]
    scored = []
    for idx, p in enumerate(paras):
        toks = _content_tokens(p)
        if not toks:
            scored.append((0.0, idx))
            continue
        overlap = sum(1 for t in toks if t in q)
        scored.append((overlap / (len(toks) ** 0.5), idx))
    top_idx = sorted(sorted(scored, reverse=True)[:k], key=lambda x: x[1])
    return [paras[i] for _, i in top_idx]

NUM_RE = re.compile(r"\b\d+\b")
WH_RE = re.compile(r"\b(what|why|how|when|where|who)\b", re.I)
THIS_RE = re.compile(r"\b(this|these)\b", re.I)
YOU_RE = re.compile(r"\byou\b", re.I)
LIST_RE = re.compile(
    r"\b(\d+)\s+(things|ways|reasons|times|facts|tips|photos|pictures|moments|signs|celebs|stars|people)\b",
    re.I)


def build_text(r):
    head = " ".join(r.get("postText", []))
    title = r.get("targetTitle", "") or ""
    paras = r.get("targetParagraphs", [])
    k = CONFIG["n_body_paragraphs"]
    if CONFIG["relevance_paragraphs"]:
        body_paras = select_paragraphs(paras, head + " " + title, k)
    else:
        body_paras = paras[:k]
    parts = [head, title, r.get("targetDescription", ""), " ".join(body_paras)]
    return " ".join(p for p in parts if p)


def headline(r):
    return " ".join(r.get("postText", []))


def hand_features(r):
    head = headline(r)
    title = r.get("targetTitle", "") or ""
    paras = r.get("targetParagraphs", [])
    n_words = sum(len(p.split()) for p in paras)
    kw = r.get("targetKeywords", "") or ""
    return [
        len(head.split()), len(NUM_RE.findall(head)),
        1.0 if NUM_RE.search(head) else 0.0,
        1.0 if LIST_RE.search(head) else 0.0,
        len(WH_RE.findall(head)), len(THIS_RE.findall(head)),
        len(YOU_RE.findall(head)),
        1.0 if head.strip().endswith("?") else 0.0,
        1.0 if ":" in title else 0.0,
        len(paras), np.log1p(n_words),
        np.mean([len(p.split()) for p in paras]) if paras else 0.0,
        len(title.split()), 1.0 if NUM_RE.search(title) else 0.0,
        len(kw.split(",")) if kw else 0.0,
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-transformer", action="store_true",
                    help="blend cached roberta-base probs (current best)")
    ap.add_argument("--write-submission", metavar="PATH",
                    help="also write a test-set submission CSV")
    args = ap.parse_args()

    train = prepare.load_split("train")
    val = prepare.load_split("val")
    test = prepare.load_split("test")
    y_tr, y_va = prepare.labels(train), prepare.labels(val)

    word_vec = TfidfVectorizer(ngram_range=CONFIG["word_ngrams"],
                               max_features=CONFIG["word_max_features"],
                               sublinear_tf=True, min_df=2)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=CONFIG["char_ngrams"],
                               max_features=CONFIG["char_max_features"],
                               sublinear_tf=True, min_df=2)
    scaler = StandardScaler()

    Xw = word_vec.fit_transform([build_text(r) for r in train])
    Xc = char_vec.fit_transform([headline(r) for r in train])
    F = scaler.fit_transform([hand_features(r) for r in train])

    def X(records):
        return sp.hstack([
            word_vec.transform([build_text(r) for r in records]),
            sp.csr_matrix(scaler.transform([hand_features(r) for r in records])),
            char_vec.transform([headline(r) for r in records]),
        ]).tocsr()

    X_tr = sp.hstack([Xw, sp.csr_matrix(F), Xc]).tocsr()
    clf = LogisticRegression(max_iter=3000, class_weight=CONFIG["class_weight"],
                             C=CONFIG["C"])
    clf.fit(X_tr, y_tr)

    order = [list(clf.classes_).index(l) for l in prepare.LABELS]
    val_probs = clf.predict_proba(X(val))[:, order]
    test_probs = clf.predict_proba(X(test))[:, order]

    used_blend = False
    transformer_tag = CONFIG["transformer_model"].replace("/", "_")
    npz_path = os.path.join(prepare.DATA_DIR, "experiments", f"probs_{transformer_tag}.npz")
    if args.blend_transformer and os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        w = CONFIG["blend_w_transformer"]
        val_probs = w * d["val_probs"] + (1 - w) * val_probs
        test_probs = w * d["test_probs"] + (1 - w) * test_probs
        used_blend = True

    val_pred = [prepare.LABELS[i] for i in val_probs.argmax(1)]
    f1 = prepare.weighted_f1(y_va, val_pred)

    prepare.log_run({**{k: str(v) for k, v in CONFIG.items()},
                     "blend_transformer": used_blend}, f1)

    if args.write_submission:
        ids = [r.get("id") for r in test]
        preds = [prepare.LABELS[i] for i in test_probs.argmax(1)]
        prepare.write_submission(ids, preds, args.write_submission)

    print(f"Val weighted F1: {f1:.4f}")


if __name__ == "__main__":
    main()
