
import csv
import json
import os
from datetime import datetime

import nltk
import sacrebleu
from nltk.translate.meteor_score import meteor_score as _meteor_sentence

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(DATA_DIR, "experiments", "tuning_runs.csv")
QA_MODEL_DIR = os.path.join(DATA_DIR, "models", "qa_ft")


def load_split(name):
    assert name in ("train", "val", "test")
    with open(os.path.join(DATA_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def build_context(r):
    parts = [r.get("targetTitle", "")] + r.get("targetParagraphs", [])
    return " ".join(p for p in parts if p)


def gold_spoilers(records):
    return [" ".join(r["spoiler"]) for r in records]


_WORDNET_READY = False


def _ensure_wordnet():
    """METEOR's synonymy stage needs WordNet. If it is already available (the
    usual case) do nothing; otherwise try a one-time quiet download, swallowing
    any network/SSL errors so scoring still runs (WordNet ships with most nltk
    installs). Cached so we probe at most once per process."""
    global _WORDNET_READY
    if _WORDNET_READY:
        return
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        _WORDNET_READY = True
        return
    except LookupError:
        pass
    import contextlib
    import io
    for res in ("wordnet", "omw-1.4"):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
    _WORDNET_READY = True


def meteor(preds, golds):
    """THE objective metric: average sentence-level METEOR, matching the Kaggle
    competition (recall weighted higher than precision, with stemming + WordNet
    synonymy). Single reference = the space-joined gold spoiler (same string the
    submission is scored against). Returned on METEOR's native 0-1 scale."""
    _ensure_wordnet()
    if not preds:
        return 0.0
    total = sum(_meteor_sentence([g.lower().split()], p.lower().split())
                for p, g in zip(preds, golds))
    return total / len(preds)


def bleu(preds, golds):
    """Secondary sanity metric ONLY (sacrebleu corpus BLEU). Task 2 is scored on
    METEOR — see meteor(); do not select runs on this. Retained so train.py can
    still print the `val BLEU(joined)=` line the fixed harness/grader parses."""
    return sacrebleu.corpus_bleu(preds, [golds]).score


def write_submission(ids, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoiler"])
        w.writerows(zip(ids, preds))
    print(f"Submission written to {path}")


def _migrate_runs_csv(fields):
    """One-time: if an older tuning_runs.csv predates the val_meteor column,
    rewrite it with the new header (blank val_meteor for legacy BLEU-only rows)
    so the file stays a consistent table instead of going ragged."""
    if not os.path.exists(RUNS_CSV):
        return
    with open(RUNS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "val_meteor" in rows[0]:
        return
    with open(RUNS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def log_run(config: dict, val_meteor: float, val_bleu: float):
    """Append one tuning run to experiments/tuning_runs.csv (course-style log).
    METEOR is the objective (logged first); BLEU is a secondary sanity number."""
    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    fields = ["date"] + sorted(config.keys()) + ["val_meteor", "val_bleu"]
    _migrate_runs_csv(fields)
    exists = os.path.exists(RUNS_CSV)
    row = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           **config, "val_meteor": round(val_meteor, 4), "val_bleu": round(val_bleu, 2)}
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
