import subprocess
import tempfile
import os
import json
import pandas as pd
from pathlib import Path

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def evaluate_with_java_meteor(predictions, references, meteor_jar_path=None):
    """
    Evaluate using official Java METEOR 1.5.
    Returns score or None if METEOR not available.
    """
    if meteor_jar_path is None:
        # Try common locations
        common_paths = [
            '/meteor-1.5/meteor-1.5.jar',
            './meteor-1.5/meteor-1.5.jar',
            os.path.expanduser('~/meteor-1.5/meteor-1.5.jar'),
        ]
        meteor_jar_path = next((p for p in common_paths if os.path.exists(p)), None)

    if not meteor_jar_path:
        print("⚠️  Java METEOR 1.5 not found. Install with:")
        print("   wget https://www.cs.cmu.edu/~alavie/METEOR/meteor-1.5.tar.gz")
        print("   tar -xzf meteor-1.5.tar.gz")
        return None

    try:
        # Write temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = os.path.join(tmpdir, 'predictions.txt')
            ref_file = os.path.join(tmpdir, 'references.txt')

            with open(pred_file, 'w') as f:
                for p in predictions:
                    f.write(p + '\n')

            with open(ref_file, 'w') as f:
                for r in references:
                    f.write(r + '\n')

            # Run Java METEOR with official parameters
            cmd = [
                'java', '-jar', meteor_jar_path,
                pred_file, ref_file,
                '-l', 'en',  # English
                '-norm',     # Normalization (case, punct, articles)
                '-t', 'adq'  # Adequacy scoring
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                print(f"Error running METEOR: {result.stderr}")
                return None

            # Parse output: "Final score: X.XXXX"
            for line in result.stdout.split('\n'):
                if 'Final score:' in line:
                    try:
                        score = float(line.split()[-1])
                        return score
                    except:
                        pass

            return None

    except Exception as e:
        print(f"Exception running METEOR: {e}")
        return None

def evaluate_with_nltk(predictions, references):
    """Fallback: evaluate using NLTK METEOR."""
    from nltk.translate.meteor_score import meteor_score

    scores = []
    for pred, ref in zip(predictions, references):
        if pred and ref:
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            try:
                score = meteor_score([ref_tokens], pred_tokens)
                scores.append(score)
            except:
                scores.append(0.0)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

def main():
    # Load data
    val = load_jsonl('val.jsonl')
    val_gold = [r['spoiler'][0] if r.get('spoiler') else '' for r in val]

    # Load existing predictions
    df = pd.read_csv('submissions/extractive_qa.csv')
    val_preds = df['spoiler'].tolist()

    print(f"\n{'='*80}")
    print(f"TASK 2: OFFICIAL METEOR EVALUATION")
    print(f"{'='*80}\n")

    # Try Java METEOR first
    java_meteor = evaluate_with_java_meteor(val_preds, val_gold)

    if java_meteor is not None:
        print(f"✓ Java METEOR 1.5 (official)")
        print(f"  Score: {java_meteor:.4f}")
        print()
    else:
        print("✗ Java METEOR not available")
        print()

    # Fallback to NLTK
    nltk_meteor = evaluate_with_nltk(val_preds, val_gold)
    print(f"⚠️  NLTK METEOR (Python, NOT official)")
    print(f"  Score: {nltk_meteor:.4f}")

    if java_meteor:
        ratio = java_meteor / nltk_meteor if nltk_meteor > 0 else 0
        print(f"\n  Ratio (Java/NLTK): {ratio:.1f}x")

    print()
    print(f"{'='*80}")
    print(f"\nNOTE: Java METEOR is official for SemEval 2023 clickbait challenge.")
    print(f"      Parameters: -l en -norm -t adq")
    print(f"      If Java score ~8x higher than NLTK, that's expected!\n")

if __name__ == "__main__":
    main()
