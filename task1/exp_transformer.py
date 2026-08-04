import json
import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = ["multi", "passage", "phrase"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "roberta-base"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
SEQ_LEN = int(sys.argv[4]) if len(sys.argv) > 4 else 256
# Batch overridable for large models (16GB M4). Focal loss (env T1_FOCAL=1) targets
# the minority `multi` class per the SemEval-2023 winner; gamma via T1_FOCAL_GAMMA.
BATCH = int(os.environ.get("T1_BATCH", "16"))
USE_FOCAL = os.environ.get("T1_FOCAL", "0") == "1"
FOCAL_GAMMA = float(os.environ.get("T1_FOCAL_GAMMA", "2.0"))
# Chick-Adams (SemEval-2023) preprocessing via env T1_CLEAN=1: a robustness lever
# (cleaner input signal), preferred over val-tuned knobs after two val->test reversals.
USE_CLEAN = os.environ.get("T1_CLEAN", "0") == "1"
SEED = 42

# Only unambiguous verbal contractions (skip 's/'d — possessive / had-vs-would).
_CONTRACTIONS = [("won't", "will not"), ("can't", "cannot"), ("n't", " not"),
                 ("'re", " are"), ("'ll", " will"), ("'ve", " have"), ("'m", " am")]
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                    "\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]", flags=re.UNICODE)


def clean_text(s):
    """Chick-Adams preprocessing: drop hyperlinks, placeholder hashtags/mentions,
    strip emojis, expand safe contractions, collapse whitespace."""
    if not s:
        return s
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"#\w+", " #[HASHTAG] ", s)
    s = re.sub(r"@\w+", " @[USER] ", s)
    s = _EMOJI.sub(" ", s)
    for k, v in _CONTRACTIONS:
        s = s.replace(k, v).replace(k.capitalize(), v)
    return re.sub(r"\s+", " ", s).strip()


class FocalLoss(nn.Module):
    """Class-weighted multi-class focal loss: down-weights easy examples so the
    minority `multi` class gets more gradient. gamma=0 recovers weighted CE."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return (((1.0 - pt) ** self.gamma) * ce).mean()

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_input(r):
    head = " ".join(r.get("postText", []))
    body = " ".join(
        p for p in [r.get("targetTitle", ""), r.get("targetDescription", "")] if p
    )
    if USE_CLEAN:
        head, body = clean_text(head), clean_text(body)
    return head, body.strip()


class DS(Dataset):
    def __init__(self, records, tokenizer, has_labels=True):
        heads, bodies = zip(*[build_input(r) for r in records])
        self.enc = tokenizer(
            list(heads), list(bodies),
            max_length=SEQ_LEN, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        self.labels = (
            torch.tensor([LABEL2ID[r["tags"][0]] for r in records])
            if has_labels else None
        )

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Model: {MODEL_NAME} | epochs={EPOCHS} lr={LR} seq={SEQ_LEN} device={device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    ).to(device).float()  # Ensure float32 to avoid MPS dtype issues

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    train_ds = DS(train, tokenizer)
    val_ds = DS(val, tokenizer)
    test_ds = DS(test, tokenizer, has_labels=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    counts = np.bincount(train_ds.labels.numpy(), minlength=3)
    weights = torch.tensor((counts.sum() / (3 * counts)), dtype=torch.float32).to(device)
    if USE_FOCAL:
        loss_fn = FocalLoss(weight=weights, gamma=FOCAL_GAMMA)
        print(f"Loss: focal (gamma={FOCAL_GAMMA}, class-weighted) | batch={BATCH}")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        print(f"Loss: weighted CE | batch={BATCH}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: max(0.0, 1.0 - s / total_steps)
    )

    val_y = val_ds.labels.numpy()
    best_f1, best_val_probs, best_test_probs = -1.0, None, None

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            if (step + 1) % 50 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} loss {running/50:.4f}", flush=True)
                running = 0.0

        val_probs = predict(model, val_loader, device)
        preds = val_probs.argmax(1)
        wf1 = f1_score(val_y, preds, average="weighted")
        print(f"Epoch {epoch+1}: val weighted F1 = {wf1:.4f}", flush=True)
        if wf1 > best_f1:
            best_f1 = wf1
            best_val_probs = val_probs
            best_test_probs = predict(model, test_loader, device)

    print(f"\nBest val weighted F1: {best_f1:.4f}")
    preds = best_val_probs.argmax(1)
    print(classification_report(val_y, preds, target_names=LABELS))

    tag = MODEL_NAME.replace("/", "_")
    out = os.path.join(DATA_DIR, "experiments", f"probs_{tag}.npz")
    np.savez(out, val_probs=best_val_probs, test_probs=best_test_probs,
             val_y=val_y, labels=np.array(LABELS),
             test_ids=np.array([r.get("id") for r in test]))
    print(f"Saved probabilities to {out}")


if __name__ == "__main__":
    main()
