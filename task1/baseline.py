import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(__file__)
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_text(record):
    parts = [
        " ".join(record.get("postText", [])),
        record.get("targetTitle", ""),
        record.get("targetDescription", ""),
        " ".join(record.get("targetParagraphs", [])[:3]),
    ]
    return " ".join(p for p in parts if p)


def prepare(records, has_labels=True):
    texts = [build_text(r) for r in records]
    labels = [r["tags"][0] for r in records] if has_labels else None
    ids = [r.get("id") for r in records] if not has_labels else None
    return texts, labels, ids


def main():
    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_texts, train_labels, _ = prepare(train)
    val_texts,   val_labels,   _ = prepare(val)
    test_texts,  _,      test_ids = prepare(test, has_labels=False)

    # TF-IDF: unigrams + bigrams
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
    )
    clf.fit(X_train, train_labels)

    val_preds = clf.predict(X_val)
    val_f1_weighted = f1_score(val_labels, val_preds, average="weighted")
    val_f1_macro    = f1_score(val_labels, val_preds, average="macro")

    print(f"Val weighted F1 : {val_f1_weighted:.4f}")
    print(f"Val macro F1    : {val_f1_macro:.4f}")
    print()
    print(classification_report(val_labels, val_preds))

    # Generate test predictions and write submission
    test_preds = clf.predict(X_test)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSIONS_DIR, "baseline.csv")
    pd.DataFrame({"id": test_ids, "spoilerType": test_preds}).to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")


if __name__ == "__main__":
    main()
