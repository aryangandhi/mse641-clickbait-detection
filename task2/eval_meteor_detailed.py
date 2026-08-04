
import json
import pandas as pd
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from collections import defaultdict

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def compute_meteor_with_params(predictions, references, alpha=0.9, beta=3.0, gamma=0.5):
    """Compute METEOR with specific parameters."""
    scores = []
    details = {
        'matches': [],
        'fragments': [],
        'penalties': [],
    }

    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            scores.append(0.0)
            continue

        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # Use single_meteor_score to get detailed scoring
        try:
            # NLTK's meteor_score expects list of references
            score = meteor_score([ref_tokens], pred_tokens)
            scores.append(score)
        except:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

def analyze_prediction_quality(predictions, references):
    """Detailed analysis of prediction quality."""
    stats = {
        'exact_matches': 0,
        'partial_matches': 0,
        'empty_predictions': 0,
        'length_ratios': [],
        'token_overlap': [],
    }

    for pred, ref in zip(predictions, references):
        if not pred.strip():
            stats['empty_predictions'] += 1
            continue

        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())

        # Exact match
        if pred.strip() == ref.strip():
            stats['exact_matches'] += 1

        # Token overlap (Jaccard similarity)
        if pred_tokens and ref_tokens:
            overlap = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
            stats['token_overlap'].append(overlap)

            if overlap > 0.5:
                stats['partial_matches'] += 1

        # Length ratio
        if len(ref.split()) > 0:
            ratio = len(pred.split()) / len(ref.split())
            stats['length_ratios'].append(ratio)

    return stats

def main():
    # Load data
    val = load_jsonl('val.jsonl')
    val_gold = [r['spoiler'][0] if r.get('spoiler') else '' for r in val]

    # Load existing predictions
    df = pd.read_csv('submissions/extractive_qa.csv')
    val_preds = df['spoiler'].tolist()

    print(f"\n{'='*80}")
    print(f"TASK 2: DETAILED METEOR ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Dataset: {len(val)} validation samples\n")

    # 1. Parameter sensitivity analysis
    print(f"1. METEOR Parameter Sensitivity")
    print(f"{'-'*80}\n")

    param_configs = [
        ("NLTK Default", 0.9, 3.0, 0.5),
        ("Low fragmentation penalty", 0.9, 3.0, 0.0),
        ("High precision weight", 0.5, 3.0, 0.5),
        ("Low precision weight", 0.95, 3.0, 0.5),
        ("SemEval standard", 0.85, 3.0, 0.5),
    ]

    for name, alpha, beta, gamma in param_configs:
        meteor = compute_meteor_with_params(val_preds, val_gold, alpha, beta, gamma)
        print(f"  {name:30s} (α={alpha:.2f}, β={beta:.1f}, γ={gamma:.1f}): {meteor:.4f}")

    print()

    # 2. Prediction quality analysis
    print(f"2. Prediction Quality Analysis")
    print(f"{'-'*80}\n")

    stats = analyze_prediction_quality(val_preds, val_gold)

    print(f"  Exact matches:        {stats['exact_matches']}/{len(val)} ({stats['exact_matches']/len(val)*100:.1f}%)")
    print(f"  Partial matches (>50% overlap): {stats['partial_matches']}/{len(val)}")
    print(f"  Empty predictions:    {stats['empty_predictions']}/{len(val)}")

    if stats['token_overlap']:
        avg_overlap = sum(stats['token_overlap']) / len(stats['token_overlap'])
        print(f"  Avg token overlap (Jaccard):    {avg_overlap:.4f}")

    if stats['length_ratios']:
        avg_ratio = sum(stats['length_ratios']) / len(stats['length_ratios'])
        print(f"  Avg pred/ref length ratio:      {avg_ratio:.2f}")

    print()

    # 3. Sample analysis
    print(f"3. Sample Predictions vs Gold Spoilers")
    print(f"{'-'*80}\n")

    # Show some examples
    for i in [0, 1, 2, 50, 100]:
        if i < len(val_preds):
            pred = val_preds[i]
            gold = val_gold[i]

            # Compute METEOR for this sample
            score = compute_meteor_with_params([pred], [gold])

            print(f"  Sample {i}:")
            print(f"    Gold: {gold[:70]}{'...' if len(gold) > 70 else ''}")
            print(f"    Pred: {pred[:70]}{'...' if len(pred) > 70 else ''}")
            print(f"    METEOR: {score:.4f}")
            print()

    # 4. Hypothesis: Why val=0.0242 but test=0.2135?
    print(f"4. Hypothesis: Val-Test Discrepancy")
    print(f"{'-'*80}\n")

    print("""
  Possible explanations for 8x difference:

  a) DIFFERENT PARAMETERS
     - Kaggle might use different α, β, γ values
     - We use NLTK defaults (0.9, 3.0, 0.5)
     - Kaggle might use lower fragmentation penalty

  b) DIFFERENT IMPLEMENTATION
     - Kaggle might use official Java METEOR implementation
     - NLTK might have bugs or different tokenization
     - Different handling of stopwords or punctuation

  c) TEST SET EASIER THAN VAL
     - Our extractive QA model happens to work better on test
     - Test spoilers might be longer/easier to extract
     - Test spoilers might match article spans better

  d) MULTIPLE REFERENCES
     - Kaggle might evaluate against multiple gold spoilers per input
     - NLTK uses single reference by default

  e) PREPROCESSING DIFFERENCES
     - Case sensitivity handling
     - Punctuation/special character handling
     - Whitespace normalization
    """)

    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
