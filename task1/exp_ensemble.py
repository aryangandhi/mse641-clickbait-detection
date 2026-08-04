

import json
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

import exp_features as E

DATA_DIR = E.DATA_DIR
LABELS = ["multi", "passage", "phrase"]  # must match exp_transformer.py order


def fit_linear(train, val, test):
    tr_texts, tr_feats, tr_y = E.prepare(train)
    va_texts, va_feats, va_y = E.prepare(val)
    te_texts, te_feats, _ = E.prepare(test, has_labels=False)

    wv = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                         sublinear_tf=True, min_df=2)
    Xw = wv.fit_transform(tr_texts)
    sc = StandardScaler().fit(tr_feats)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                         max_features=50_000, sublinear_tf=True, min_df=2)
    heads = lambda rs: [" ".join(r.get("postText", [])) for r in rs]
    Xc = cv.fit_transform(heads(train))

    def transform(texts, feats, recs):
        return sp.hstack([
            wv.transform(texts),
            sp.csr_matrix(sc.transform(feats)),
            cv.transform(heads(recs)),
        ]).tocsr()

    X_tr = sp.hstack([Xw, sp.csr_matrix(sc.transform(tr_feats)), Xc]).tocsr()
    X_va = transform(va_texts, va_feats, val)
    X_te = transform(te_texts, te_feats, test)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, tr_y)
    # align prob columns to LABELS order
    order = [list(clf.classes_).index(l) for l in LABELS]
    return (clf.predict_proba(X_va)[:, order], clf.predict_proba(X_te)[:, order],
            np.array([LABELS.index(y) for y in va_y]))


def main():
    probs_path = sys.argv[1]
    d = np.load(probs_path, allow_pickle=True)
    t_val, t_test, val_y_t = d["val_probs"], d["test_probs"], d["val_y"]
    test_ids = d["test_ids"]

    train = E.load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = E.load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test = E.load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    l_val, l_test, val_y = fit_linear(train, val, test)
    assert (val_y == val_y_t).all(), "label order mismatch between linear and transformer"

    f1_lin = f1_score(val_y, l_val.argmax(1), average="weighted")
    f1_tr = f1_score(val_y, t_val.argmax(1), average="weighted")
    print(f"linear val F1={f1_lin:.4f} | transformer val F1={f1_tr:.4f}")

    best_w, best_f1 = 0.0, -1
    for w in np.arange(0, 1.01, 0.05):
        blend = w * t_val + (1 - w) * l_val
        f1 = f1_score(val_y, blend.argmax(1), average="weighted")
        if f1 > best_f1:
            best_w, best_f1 = w, f1
    print(f"best blend: w_transformer={best_w:.2f} val weighted F1={best_f1:.4f}")
    blend_val = best_w * t_val + (1 - best_w) * l_val
    print(classification_report(val_y, blend_val.argmax(1), target_names=LABELS))

    os.makedirs(os.path.join(DATA_DIR, "submissions"), exist_ok=True)
    blend_test = best_w * t_test + (1 - best_w) * l_test
    pd.DataFrame({
        "id": test_ids,
        "spoilerType": [LABELS[i] for i in blend_test.argmax(1)],
    }).to_csv(os.path.join(DATA_DIR, "submissions", "ensemble.csv"), index=False)
    pd.DataFrame({
        "id": test_ids,
        "spoilerType": [LABELS[i] for i in l_test.argmax(1)],
    }).to_csv(os.path.join(DATA_DIR, "submissions", "linear_e003.csv"), index=False)
    print("Wrote submissions/ensemble.csv and submissions/linear_e003.csv")


if __name__ == "__main__":
    main()
