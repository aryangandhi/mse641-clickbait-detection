
import json
import os
import re
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression


from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_text(r):
    parts = [
        " ".join(r.get("postText", [])),
        r.get("targetTitle", ""),
        r.get("targetDescription", ""),
        " ".join(r.get("targetParagraphs", [])[:3]),
    ]
    return " ".join(p for p in parts if p)


NUM_RE = re.compile(r"\b\d+\b")
WH_RE = re.compile(r"\b(what|why|how|when|where|who)\b", re.I)
THIS_RE = re.compile(r"\b(this|these)\b", re.I)
YOU_RE = re.compile(r"\byou\b", re.I)
LIST_RE = re.compile(r"\b(\d+)\s+(things|ways|reasons|times|facts|tips|photos|pictures|moments|signs|celebs|stars|people)\b", re.I)


def hand_features(r):
    head = " ".join(r.get("postText", []))
    title = r.get("targetTitle", "") or ""
    paras = r.get("targetParagraphs", [])
    n_words = sum(len(p.split()) for p in paras)
    kw = r.get("targetKeywords", "") or ""
    return [
        len(head.split()),                       # headline length
        len(NUM_RE.findall(head)),               # numbers in headline
        1.0 if NUM_RE.search(head) else 0.0,
        1.0 if LIST_RE.search(head) else 0.0,    # listicle pattern -> multi
        len(WH_RE.findall(head)),
        len(THIS_RE.findall(head)),
        len(YOU_RE.findall(head)),
        1.0 if head.strip().endswith("?") else 0.0,
        1.0 if ":" in title else 0.0,
        len(paras),                              # n paragraphs
        np.log1p(n_words),                       # article length
        np.mean([len(p.split()) for p in paras]) if paras else 0.0,
        len(title.split()),
        1.0 if NUM_RE.search(title) else 0.0,
        len(kw.split(",")) if kw else 0.0,
    ]


def prepare(records, has_labels=True):
    texts = [build_text(r) for r in records]
    feats = np.array([hand_features(r) for r in records])
    labels = [r["tags"][0] for r in records] if has_labels else None
    return texts, feats, labels


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "feats_lr"

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))

    tr_texts, tr_feats, tr_y = prepare(train)
    va_texts, va_feats, va_y = prepare(val)

    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                               sublinear_tf=True, min_df=2)
    Xw_tr = word_vec.fit_transform(tr_texts)
    Xw_va = word_vec.transform(va_texts)

    scaler = StandardScaler()
    F_tr = scaler.fit_transform(tr_feats)
    F_va = scaler.transform(va_feats)

    blocks_tr = [Xw_tr, sp.csr_matrix(F_tr)]
    blocks_va = [Xw_va, sp.csr_matrix(F_va)]

    if variant == "feats_chargrams":
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   max_features=50_000, sublinear_tf=True, min_df=2)
        heads_tr = [" ".join(r.get("postText", [])) for r in train]
        heads_va = [" ".join(r.get("postText", [])) for r in val]
        blocks_tr.append(char_vec.fit_transform(heads_tr))
        blocks_va.append(char_vec.transform(heads_va))

    X_tr = sp.hstack(blocks_tr).tocsr()
    X_va = sp.hstack(blocks_va).tocsr()

    if variant == "feats_svm":
        clf = LinearSVC(class_weight="balanced", C=0.5)
        clf.fit(X_tr, tr_y)
        preds = clf.predict(X_va)
        report(variant, va_y, preds)
    elif variant == "tune_c":
        for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
            clf.fit(X_tr, tr_y)
            preds = clf.predict(X_va)
            wf1 = f1_score(va_y, preds, average="weighted")
            print(f"C={C:<5} weighted F1={wf1:.4f}")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X_tr, tr_y)
        preds = clf.predict(X_va)
        report(variant, va_y, preds)


def report(name, y_true, y_pred):
    wf1 = f1_score(y_true, y_pred, average="weighted")
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{name}] Val weighted F1: {wf1:.4f} | macro F1: {mf1:.4f}")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
