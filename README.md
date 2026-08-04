# MSE 641 — Clickbait Detection Challenge



Source Code: [https://github.com/aryangandhi/mse641-clickbait-detection](https://github.com/aryangandhi/mse641-clickbait-detection)



Two-stage clickbait spoiling on the Webis Clickbait Spoiling Corpus 2022:

- **Task 1** — spoiler *type* classification (`phrase` / `passage` / `multi`), metric **weighted F1**.
- **Task 2** — spoiler *generation*, metric **METEOR**. Task 1's predicted types route Task 2's decoder.

## Results (Kaggle test, team "Aryan Gandhi", username "arystatistics")


| Task | Best score         | System                                                           |
| ---- | ------------------ | ---------------------------------------------------------------- |
| 1    | **0.75852** wF1    | roberta-large + weighted-CE + post-focused input + preprocessing |
| 2    | **0.44731** METEOR | roberta-base-squad2 fine-tuned QA + type-conditioned decoding    |




## Documentation (start here)

- `task1/experiments/log.md`, `task2/experiments/log.md` — per-experiment lab notes.
- `task*/experiments/tuning_runs.csv` — auto-logged run metrics.
- `SETUP.md` — environment details.



## Repository layout

```
task1/  spoiler type classification    task2/  spoiler generation
  prepare.py     fixed harness: data, metric, logging
  train.py       main experiment surface (T1: linear+blend; T2: QA decoder)
  exp_*.py       slow-loop trainers / sweeps / diagnostics
  baseline.py    milestone baseline
  experiments/   log.md, tuning_runs.csv, pred_types_*.json, qualitative_examples.txt
  *.jsonl        train / val / test data
Research Papers/ reference PDFs (Hagen 2022, Pal 2024, SemEval writeups)
```

**Not tracked in git** (see `.gitignore`, regenerate with the commands below):
`.venv/`, `task*/models/` (checkpoints), `*.npz`, `*.log`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```



## Reproduce the best models

Everything is deterministic (seed 42).

**Task 1 (val wF1 0.7490 → test 0.75852):**

```bash
cd task1
T1_BATCH=8 T1_FOCAL=0 T1_CLEAN=1 python exp_transformer.py roberta-large 5 1e-5 256
# writes experiments/probs_roberta-large.npz; argmax test_probs -> id,spoilerType CSV
```

**Task 2 (val METEOR 0.4705 → test 0.44731):**

```bash
cd task2
python exp_finetune_qa.py 3 3e-5 50        # builds models/qa_ft (best epoch by METEOR)
python train.py --write-submission submissions/out.csv   # type-conditioned decoding
```



## Best submissions (for the grading form)

- **Task 1:** score 0.75852, `task1/submissions/roberta_large_ce_clean.csv`, submitted 2026-07-25.
- **Task 2:** score 0.44731, `task2/submissions/sweep_tuned.csv`, submitted 2026-07-19.

