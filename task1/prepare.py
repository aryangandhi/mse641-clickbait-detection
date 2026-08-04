import csv
import json
import os
from datetime import datetime

from sklearn.metrics import f1_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(DATA_DIR, "experiments", "tuning_runs.csv")
LABELS = ["multi", "passage", "phrase"]


def load_split(name):
    assert name in ("train", "val", "test")
    with open(os.path.join(DATA_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def labels(records):
    return [r["tags"][0] for r in records]


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")


def write_submission(ids, preds, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "spoilerType"])
        w.writerows(zip(ids, preds))
    print(f"Submission written to {path}")


def log_run(config: dict, val_f1: float):
    """Append one tuning run to experiments/tuning_runs.csv (course-style log)."""
    os.makedirs(os.path.dirname(RUNS_CSV), exist_ok=True)
    exists = os.path.exists(RUNS_CSV)
    fields = ["date"] + sorted(config.keys()) + ["val_weighted_f1"]
    row = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"),
           **config, "val_weighted_f1": round(val_f1, 4)}
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
