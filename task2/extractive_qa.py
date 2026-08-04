

import json
import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import sacrebleu
from nltk.translate.meteor_score import meteor_score

DATA_DIR = os.path.dirname(__file__)
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")

MODEL_NAME = "deepset/roberta-base-squad2"
MAX_CONTEXT_PARAGRAPHS = 5
MAX_SEQ_LEN = 512


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_context(record):
    paragraphs = record.get("targetParagraphs", [])[:MAX_CONTEXT_PARAGRAPHS]
    return " ".join(paragraphs)


def run_qa(model, tokenizer, question, context):
    if not context.strip():
        return context[:100]
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation="only_second",
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    start = outputs.start_logits.argmax().item()
    end   = outputs.end_logits.argmax().item() + 1
    if end <= start:
        end = start + 1
    answer_tokens = inputs["input_ids"][0][start:end]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    # Fallback: if empty, return first sentence of context
    if not answer:
        answer = context.split(".")[0].strip()
    return answer


def compute_bleu(predictions, references):
    refs = [[r] for r in references]
    result = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
    return result.score


def compute_meteor(predictions, references):
    """Compute average METEOR score across all predictions."""
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()

    train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    # --- Evaluate on val ---
    print("Evaluating on val set...")
    val_preds = []
    val_gold  = []
    for i, r in enumerate(val):
        question  = " ".join(r.get("postText", []))
        context   = build_context(r)
        pred      = run_qa(model, tokenizer, question, context)
        gold      = r["spoiler"][0] if r.get("spoiler") else ""
        val_preds.append(pred)
        val_gold.append(gold)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val)} done...")

    bleu = compute_bleu(val_preds, val_gold)
    meteor = compute_meteor(val_preds, val_gold)
    exact_match = sum(1 for p, g in zip(val_preds, val_gold) if p.strip() == g.strip())

    print(f"\nVal BLEU         : {bleu:.2f}")
    print(f"Val METEOR       : {meteor:.4f}  ← Official Kaggle metric")
    print(f"Val exact match  : {exact_match}/{len(val)} ({exact_match/len(val)*100:.1f}%)")

    # Sample predictions
    print("\nSample predictions (val):")
    for i in range(5):
        print(f"  Headline : {' '.join(val[i].get('postText', []))}")
        print(f"  Gold     : {val_gold[i]}")
        print(f"  Pred     : {val_preds[i]}")
        print()

    # --- Generate test predictions ---
    print("Generating test predictions...")
    test_preds = []
    for i, r in enumerate(test):
        question = " ".join(r.get("postText", []))
        context  = build_context(r)
        pred     = run_qa(model, tokenizer, question, context)
        test_preds.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test)} done...")

    test_ids = [r.get("id") for r in test]

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSIONS_DIR, "extractive_qa.csv")
    pd.DataFrame({"id": test_ids, "spoiler": test_preds}).to_csv(out_path, index=False)
    print(f"\nSubmission written to {out_path}")


if __name__ == "__main__":
    main()
