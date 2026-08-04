

import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.dirname(__file__)
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    # Extract headlines
    train_headlines = [" ".join(r.get("postText", [])) for r in train]
    val_headlines   = [" ".join(r.get("postText", [])) for r in val]
    test_headlines  = [" ".join(r.get("postText", [])) for r in test]

    # Extract spoilers from train (we'll use these as retrieval targets)
    train_spoilers = [r["spoiler"][0] if r.get("spoiler") else "" for r in train]

    # TF-IDF: unigrams + bigrams on train headlines
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)
    X_train = vectorizer.fit_transform(train_headlines)
    X_test  = vectorizer.transform(test_headlines)

    # For each test sample, find most similar train sample
    similarities = cosine_similarity(X_test, X_train)  # shape: (n_test, n_train)
    nearest_indices = similarities.argmax(axis=1)

    test_preds = [train_spoilers[idx] for idx in nearest_indices]
    test_ids   = [r.get("id") for r in test]

    # Compute val accuracy (retrieval from train)
    X_val = vectorizer.transform(val_headlines)
    val_similarities = cosine_similarity(X_val, X_train)
    val_nearest_indices = val_similarities.argmax(axis=1)
    val_preds = [train_spoilers[idx] for idx in val_nearest_indices]
    val_spoilers = [r["spoiler"][0] if r.get("spoiler") else "" for r in val]

    # Exact match accuracy
    val_exact_match = sum(1 for pred, gold in zip(val_preds, val_spoilers) if pred == gold)
    val_acc = val_exact_match / len(val)

    print(f"Val exact match accuracy: {val_acc:.4f} ({val_exact_match}/{len(val)})")
    print(f"Average similarity to nearest neighbor (val): {val_similarities.max(axis=1).mean():.4f}")

    # Write submission
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSIONS_DIR, "baseline.csv")
    pd.DataFrame({"id": test_ids, "spoiler": test_preds}).to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")


if __name__ == "__main__":
    main()
