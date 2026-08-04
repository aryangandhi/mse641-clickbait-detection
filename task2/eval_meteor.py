import json
import pandas as pd
import sys
from nltk.translate.meteor_score import meteor_score
from sacrebleu import BLEU

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def compute_bleu(predictions, references):
    refs = [[r] for r in references]
    result = BLEU().corpus_score(predictions, list(zip(*refs)))
    return result.score

def compute_meteor(predictions, references):
    """Compute average METEOR score."""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

# Load val data
val = load_jsonl('val.jsonl')
val_gold = [r['spoiler'][0] if r.get('spoiler') else '' for r in val]

# Load existing predictions
pred_csv = pd.read_csv('submissions/extractive_qa.csv')
val_preds = pred_csv['spoiler'].tolist()[:len(val)]

# Compute both metrics
bleu = compute_bleu(val_preds, val_gold)
meteor = compute_meteor(val_preds, val_gold)
exact_match = sum(1 for p, g in zip(val_preds, val_gold) if p.strip() == g.strip())

print(f"\n{'='*60}")
print(f"TASK 2: METRIC RE-EVALUATION (using correct METEOR)")
print(f"{'='*60}\n")
print(f"Experiment 002: Extractive QA (roberta-base-squad2, zero-shot)")
print(f"\nMetrics on validation set (n={len(val)}):")
print(f"  BLEU              : {bleu:.2f}")
print(f"  METEOR (official) : {meteor:.4f}  ← Use this for Kaggle prediction")
print(f"  Exact match       : {exact_match}/{len(val)} ({exact_match/len(val)*100:.1f}%)")
print(f"\n{'='*60}\n")
