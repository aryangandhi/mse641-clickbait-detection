# MSE 641 Project Setup Guide

## Environment

This project requires Python 3.14+ with PyTorch MPS support for M-series Macs.

### Option 1: Use System Python (Recommended)
The project is configured to use the system `python3` (3.14, pre-installed on your M4 MacBook with torch/transformers/sklearn).

```bash
# Just run directly, no venv needed
cd /Users/aryan/Documents/School/Waterloo/MSE\ 641/Project
python3 train.py  # in task1 or task2
```

### Option 2: Set Up Virtual Environment (For Reproducibility)
If you want an isolated environment with pinned versions:

```bash
# Create virtual environment
cd /Users/aryan/Documents/School/Waterloo/MSE\ 641/Project
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Now run experiments
cd task1
python3 train.py
```

### Important Notes

- **MPS GPU Support:** PyTorch must be compiled with MPS support. The system python3 already has this.
- **DeBERTa Models:** DeBERTa-v3-base and DeBERTa-v3-large require additional tokenizer dependencies:
  ```bash
  pip install sentencepiece tiktoken
  ```
- **Float32 Safety:** When using DeBERTa on MPS, models are explicitly cast to float32 to avoid dtype mismatches.

## Key Dependencies

- **torch 2.1+**: Deep learning (MPS accelerated)
- **transformers 4.35+**: HuggingFace models (BERT, RoBERTa, DeBERTa)
- **scikit-learn 1.3+**: ML models and metrics
- **nltk 3.8+**: METEOR — the official Task 2 metric (needs the WordNet corpus, auto-downloaded on first run)
- **sacrebleu 2.3+**: BLEU — secondary sanity metric for Task 2 only
- **pandas 2.1+**: Data manipulation

## Task-Specific Commands

### Task 1: Spoiler Type Classification
```bash
cd task1
python3 train.py                              # Run with current config
python3 train.py --blend-transformer          # Blend linear + transformer
python3 train.py --write-submission path.csv  # Write Kaggle submission
python3 exp_transformer.py model epochs lr seq_len  # Fine-tune transformer
```

### Task 2: Spoiler Generation
```bash
cd task2
python3 train.py                              # Run with current config
python3 train.py --write-submission path.csv  # Write Kaggle submission
python3 exp_finetune_qa.py epochs lr max_ans_len  # Fine-tune QA model (~2-10h)

# With caffeinate (keeps laptop awake)
caffeinate -dimsu python3 exp_finetune_qa.py 4 3e-5 50
```

## Experiment Workflow

Each experiment:
1. Edit only `train.py` for fast iterations (~1-10 min)
2. Record results in `experiments/log.md` with a short narrative + config
3. Metrics are auto-logged to `experiments/tuning_runs.csv`
4. Keep a change only if it beats the best by >0.005 F1 / METEOR; otherwise revert with a note

For slow-loop changes (model fine-tuning), edit the `exp_*.py` scripts and record results.

## Troubleshooting

### OOM (Out of Memory) Errors
- Reduce `BATCH` size in exp_finetune_qa.py
- Use DeBERTa-base instead of -large

### MPS Dtype Errors
- Models are automatically cast to float32
- If still failing, check transformers version compatibility

### Missing HuggingFace Models
- First run downloads models (requires internet)
- Models cached in `~/.cache/huggingface/`

## Data

- Train: 3,200 samples
- Val: 400 samples  
- Test: 400 samples (no ground truth)
- Format: JSONL (JSON Lines)
