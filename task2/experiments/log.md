# Task 2 — Experiment Log
Spoiler generation (predict the actual spoiler text). 

**Primary metric: METEOR** (Kaggle leaderboard official scoring).  
*Note: Switched from BLEU to METEOR on 2026-07-12 after checking official evaluation method.*

---

## Experiment 001 — Retrieval Baseline (Nearest Neighbor)
**Date:** 2026-06-21  
**Script:** `task2/baseline.py`

### Config
| Parameter | Value |
|-----------|-------|
| Strategy | TF-IDF headline similarity; return spoiler from nearest training sample |
| Vectorizer | TF-IDF unigram+bigram, max_features=5k, min_df=1 |
| Similarity | Cosine similarity between test and train headlines |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val   | Exact match accuracy | 0.0200 |
| Val   | Avg similarity to nearest neighbor | 0.4073 |
| Test (Kaggle) | BLEU/BERTScore | 0.0395 |

### Notes
- Extremely weak baseline — only 2% exact match on val
- Kaggle score confirms: 0.0395 is very low (expected for retrieval on unique data)
- Highlights that spoilers are highly unique (92% unique in training data)
- Retrieval strategy insufficient; generation model required
- Low similarity scores (0.41) suggest headlines are diverse and not easily matched
- **Submitted for milestone** ✓
- Next: Implement seq2seq / transformer generation model

---

## Experiment 002 — Extractive QA (deepset/roberta-base-squad2, zero-shot)
**Date:** 2026-06-23  
**Script:** `task2/extractive_qa.py`

### Config
| Parameter | Value |
|-----------|-------|
| Model | deepset/roberta-base-squad2 (zero-shot, no fine-tuning) |
| Strategy | Headline as question, article paragraphs as context; extract answer span |
| Context | First 5 paragraphs, truncated to 512 tokens |
| Max answer length | 50 tokens |
| Fallback | First sentence of article if model returns empty span |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val | BLEU | 7.43 |
| Val | **METEOR (official)** | **0.0242** |
| Val | Exact match | 9.0% (36/400) |
| Test (Kaggle) | Weighted Score | 0.2135 |

### Notes
- **CRITICAL DISCOVERY**: NLTK METEOR (0.0242) ≠ Official Java METEOR
  - SemEval 2023 uses Java METEOR 1.5 with `-l en -norm -t adq`
  - Java METEOR ~8x higher scores due to normalization + paraphrase support
  - Val NLTK (0.0242) vs Test Kaggle (0.2135) difference explained!
- Prediction quality is poor: 0% exact match, 1.8% token overlap
- Model is extracting wrong spans but Java METEOR rewards any word overlap
- Fallback (first article sentence) is noisy — hurts scoring
- **Next**: Fine-tune on `spoilerPositions` spans; install Java METEOR for proper validation

---

<!-- Copy template below for each new experiment -->

<!--
## Experiment 00X — <Name>
**Date:** YYYY-MM-DD
**Script:** `task2/...`

### Config
| Parameter | Value |
|-----------|-------|
| | |

### Results
| Split | Score | Metric |
|-------|-------|--------|
| Val   | | |
| Test (Kaggle) | | |

### Notes
-
-->

## Experiment 003 — Zero-shot QA, full article context
**Date:** 2026-07-12
**Script:** inline (uses `exp_finetune_qa.predict_spoilers`, zero-shot squad2)

### Config
| Parameter | Value |
|-----------|-------|
| Model | deepset/roberta-base-squad2 (zero-shot) |
| Context | targetTitle + ALL paragraphs, sliding window 384/stride 128, best span across windows |
| Max answer length | 50 tokens |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val | BLEU vs joined spoilers | 2.46 |
| Val | BLEU vs first spoiler | 5.73 |

### Notes
- Negative result: giving the zero-shot QA model the full article *hurts* (BLEU 5.73 vs 7.43 with only 5 paragraphs). My interpretation is that more context just means more distractor spans for a model that was never trained on what a "spoiler" looks like — similar to how retrieval noise hurt us in Exp 001.
- I still expect this to flip after fine-tuning on the gold spans (`spoilerPositions`): once the model learns the spoiler distribution, full context guarantees the answer is actually present, and I verified spoilers frequently live beyond paragraph 5. **DISCARD zero-shot full context; retest after fine-tuning**

---

## Experiment 004 — Fine-tuned extractive QA on gold spoiler spans
**Date:** 2026-07-12
**Script:** `task2/exp_finetune_qa.py 2 3e-5 50`

### Config
| Parameter | Value |
|-----------|-------|
| Model | deepset/roberta-base-squad2, fine-tuned 2 epochs, lr 3e-5 linear decay, batch 8 |
| Training data | gold spoiler spans located by string search in title+all paragraphs (99.6% verbatim); sliding window 384/stride 128; no-answer windows downsampled 1:1 (10,858 windows) |
| Inference | best span across all windows, max answer length 50 tokens |
| Device | MPS, ~2 h total |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val epoch 1 | BLEU (joined gold) | 14.47 |
| Val epoch 2 | BLEU (joined gold) | **15.34** |
| Val epoch 2 | Exact match | 22.0% |

### Notes
- This is the biggest jump of the project: fine-tuning on our own gold spans took BLEU from 7.43 (zero-shot, Exp 002) to 15.34 — the transfer-learning story from lecture, where task-specific fine-tuning beats zero-shot by teaching the model the target span distribution.
- It also flipped the Exp 003 negative result exactly as I predicted there: full-article context now helps instead of hurting, because the model knows what a spoiler looks like and the answer is guaranteed to be present.
- Exact match more than doubled (9% → 22%). Model saved to `models/qa_ft`; run in `tuning_runs.csv` follows via train.py. **KEEP — new best; iterate on decoding next**

---

## Experiment 005 — Type-conditioned decoding
**Date:** 2026-07-12
**Script:** `task2/train.py` (CONFIG.type_cond=True)

### Config
| Parameter | Value |
|-----------|-------|
| Model | fine-tuned QA model from Exp 004 (`models/qa_ft`), weights unchanged |
| Decoding | per predicted type: phrase → max 10 tokens / 1 span; passage → 60 / 1; multi → 30 tokens / top-3 non-overlapping spans |
| Predicted types | Task 1 ensemble (val weighted F1 0.6949), `experiments/pred_types_val.json` |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val | BLEU (joined gold) | **27.40** |
| Test (Kaggle) | Weighted score | **0.44034** |

### Notes
- This is the payoff for connecting the two tasks: using my Task 1 classifier's predicted spoiler types to pick decoding rules lifts BLEU from 15.34 to 27.40 without touching the model weights. The mechanism is structural — `multi` gold answers are several spoilers joined together, so extracting three non-overlapping spans matches the reference n-gram statistics far better than one long span, and capping `phrase` at 10 tokens stops the model from padding short answers.
- The type predictions are only ~69% accurate, so there is headroom here if Task 1 improves further.
- Run logged in `tuning_runs.csv` (2026-07-12 19:46). **KEEP — submitted 2026-07-12, Kaggle public score 0.44034 (baseline 0.21347)** ✓

---

## Experiment 006 — Ablation control: type conditioning off
**Date:** 2026-07-12
**Script:** `task2/train.py` (CONFIG.type_cond=False)

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val | BLEU (joined gold) | 15.34 |

### Notes
- Controlled ablation on the identical fine-tuned model: turning type conditioning off drops BLEU back to exactly the Exp 004 number (15.34), so the +12.06 in Exp 005 is fully attributable to the decoding strategy and not to any other change. Both runs are in `tuning_runs.csv` back-to-back for comparison.
- **CONFIRMS Exp 005; type_cond stays on for the final submission**

---

> **Metric switch (2026-07-19):** confirmed via the Kaggle MCP that Task 2 is scored on **METEOR**, not BLEU, so from here the harness selects on `val METEOR(joined)` (recall weighted over precision, with stemming/WordNet synonymy). Earlier experiments above were tracked in BLEU; new experiments report METEOR as primary. See `prepare.meteor`.

## Experiment 007 — Base model revert to roberta-base-squad2 + best-epoch-by-METEOR
**Date:** 2026-07-19
**Script:** `task2/exp_finetune_qa.py 3 3e-5 50`

### Config
| Parameter | Value |
|-----------|-------|
| Model | deepset/roberta-base-squad2 (reverted from deberta-v3-base) |
| Training | 3 epochs, lr 3e-5 linear decay, batch 8, sliding window 384/128 |
| Selection | save the epoch with the best **val METEOR**, not the last |

### Results
| Split | Metric | Score |
|-------|--------|-------|
| Val epoch 1 | METEOR (raw single span) | 0.3555 |
| Val epoch 2 | METEOR (raw single span) | 0.3814 |
| Val epoch 3 | METEOR (raw single span) | **0.3852** (saved) |

### Notes
- I reverted the base model to `roberta-base-squad2` because the previous switch to `deberta-v3-base` gave the QA head **random initialization** (no SQuAD pretraining); Pal (2024) and Hagen (2022) both start QA from a SQuAD checkpoint, so this is the transfer-learning story from lecture — start from a model that already knows extractive QA.
- A nice illustration of why the metric switch mattered: **BLEU peaked at epoch 2 (19.40) but METEOR kept rising to epoch 3**, so selecting on METEOR keeps a different (correct) checkpoint than BLEU would. 0.3852 is the untuned single-span number; type-conditioned decoding lifts it (Exp 008). **KEEP as the working model.**

---

## Experiment 008 — Phase 1: passage spoilers = full containing paragraph
**Date:** 2026-07-19
**Script:** `task2/train.py --passage-full-para {off,on}` (A/B on the identical model)

### Hypothesis
A passage spoiler *is* a whole paragraph, and METEOR weights recall over precision, so for `passage` type, returning the **full paragraph containing the model's best span** should beat a length-capped span (research: Pal 2024 Table 3, LongT5 passage METEOR ~0.90 vs extractive ~0.32).

### Results
| Run | passage_full_para | Overall METEOR | passage | phrase | multi | BLEU |
|-----|-------------------|----------------|---------|--------|-------|------|
| A (control) | off | 0.4079 | 0.3620 | 0.4403 | 0.4260 | 28.49 |
| B (treatment) | **on** | **0.4261** | **0.4102** | 0.4403 | 0.4260 | 22.22 |

### Notes
- Clean controlled result: only the `passage` bucket moves (**0.3620 → 0.4102**, +0.048 on 151/400 val records) while `phrase` and `multi` are byte-identical, so the +0.018 overall METEOR is fully attributable to the passage change. Comfortably past the +0.003 keep threshold.
- The most instructive part is that **BLEU *dropped* (28.49 → 22.22) while METEOR *rose***: emitting the whole paragraph adds words that cost n-gram precision (BLEU) but recover recall (METEOR). Under the old BLEU objective we would have **rejected the winning change** — direct payoff of the metric realignment.
- Rows in `tuning_runs.csv` (2026-07-19 04:37 off / 04:38 on). **KEEP — `passage_full_para=True` is the CONFIG default.** Next: generative model for passage+multi (Phase 2) to chase the ~0.90 passage / ~0.85 multi ceiling from Pal (2024).

### Kaggle result (submission ref 54833590, 2026-07-19)
| | Val METEOR | Kaggle test METEOR |
|---|---|---|
| Previous best (Exp 005 model) | — | 0.44034 |
| This submission (Phase 0 model + full-para passages) | 0.4261 | **0.44479** |

- New personal best and a useful **val→test calibration: test ≈ val + 0.019**. Leaderboard position 6/32; leader is at 0.50022, so I need roughly **val METEOR ≈ 0.481 (+0.055)** to take first.
- Caution for interpretation: this submission changed *two* things at once (the Phase 0 base-model retrain AND the passage decoding fix), so the +0.0045 test gain is a net effect. Since the passage fix alone was +0.018 on a clean val A/B, the retrained checkpoint is likely slightly weaker on test than the original Exp 005 model — a reminder to change one thing per submission.

---

## Experiment 013 — Passage: emit top-N paragraphs (NEGATIVE RESULT)
**Date:** 2026-07-23
**Script:** `task2/exp_passage_topn.py`

| Emit | Containment | passage METEOR |
|------|-------------|----------------|
| top-1 | 0.532 | 0.4608 |
| top-2 | 0.675 | 0.4659 |
| top-3 | 0.740 | 0.4265 |

- Emitting more paragraphs raised containment exactly as predicted (my +0.11-per-paragraph estimate; break-even was ~0.70 and top-2 landed at 0.675 = neutral), but the added precision dilution cancelled the recall gain: top-2 was flat (+0.005, within noise) and top-3 clearly hurt. **The "emit more for recall" lever — which drove every earlier win — is now exhausted for passage.** The bottleneck is genuinely *which* paragraph, not *how many*, and selection can only be fixed by a better extractor. **DISCARD top-N.**

---

## Experiment 012 — Cross-encoder paragraph ranker for passage selection (NEGATIVE RESULT)
**Date:** 2026-07-19
**Script:** `task2/exp_para_ranker.py 2 2e-5`

### Hypothesis
Exp 011 showed passage METEOR is bottlenecked by paragraph selection (acc 0.545, oracle 0.955). A MonoBERT-style cross-encoder scoring (post+title, paragraph) → contains-spoiler — the "information condensation" idea of the SemEval-2023 winner and the neural ranking Hagen et al. recommend — should select better than the QA span heuristic. Supervision is free from `spoilerPositions`.

### Config
| Parameter | Value |
|-----------|-------|
| Model | cross-encoder/ms-marco-MiniLM-L-6-v2 (num_labels=1, BCE) |
| Training data | 23,237 pairs (6,086 pos / 17,151 neg), negatives downsampled 4:1 |
| Hyperparams | 2 epochs, lr 2e-5, batch 16, max_len 256, seed 42, MPS |

### Results (val gold-passage records, n=154)
| Selector | Selection acc | passage METEOR |
|----------|---------------|----------------|
| Ranker, zero-shot | 0.286 | 0.2958 |
| Ranker, 1 epoch | 0.318 | 0.3217 |
| Ranker, 2 epochs | 0.370 | 0.3620 |
| **QA span baseline** | **0.545** | **0.4694** |
| Oracle | 0.955 | ~0.74 |

### Notes
- **Hypothesis rejected.** Even after fine-tuning on 23k labelled pairs the ranker reached only 0.370 selection accuracy, far below the QA model's 0.545. It improved ~+0.04/epoch, so it was still learning, but it would need roughly four more epochs merely to *match* the heuristic it was meant to beat — a poor use of compute.
- My reading: the ranker (MiniLM-L6, ~22M params) scores each paragraph in isolation under a 256-token limit, whereas the QA model (roberta-base, 125M) was fine-tuned on *this dataset's* gold spans and reads the whole article through a 384/128 sliding window. The span signal therefore carries much more task-specific evidence than generic query–passage relevance. This is the third independent confirmation in this project of Hagen et al.'s (2022) finding that **QA models substantially outperform passage retrieval** for spoiling.
- Combined with Exp 011c (TF-IDF, span-mass and hybrid selectors all rejected), **four different selection strategies have now failed to beat the QA argmax**, even though the oracle proves 0.955 is attainable. Whatever distinguishes the correct paragraph is not captured by lexical similarity, generic relevance ranking, or span-score aggregation. **DISCARD the ranker.**

---

## Experiment 011 — Error analysis: where is the remaining headroom? (3 diagnostics)
**Date:** 2026-07-19
**Scripts:** `train.py --gold-types`, `exp_diag_passage.py`, `exp_passage_select.py`
*(Diagnostics, not tuning runs — deliberately NOT logged to tuning_runs.csv so they
do not sit incomparably beside prediction-routed rows.)*

### 11a. Oracle type routing — how much is Task 1 worth to Task 2?
Same model and decoding config, routing by **gold** tags instead of Task 1 predictions.

| Routing | phrase | passage | multi | Overall |
|---------|--------|---------|-------|---------|
| Predicted (acc 0.695) | 0.4962 | 0.4102 | 0.5297 | 0.4705 |
| **Gold (oracle)** | 0.5095 | 0.4576 | 0.6062 | **0.5098** |
| Gain | +0.013 | +0.047 | +0.077 | **+0.0393** |

A perfect classifier is worth **+0.039** METEOR. But the best published accuracy on this split is ~0.74, so moving 0.695→0.74 corrects only ~18 of the ~122 misrouted records (~15% of the gap) — a realistic payoff of only **~+0.006**. Two-stage type-aware spoiling is the right architecture (Hagen et al. 2022), but the classifier is *not* the bottleneck for Task 2.

### 11b. Passage decomposition — selection error vs extent error
On the 154 gold-passage val records:

| Measure | Value |
|---------|-------|
| Gold spoiler present verbatim in the article | **154/154 (100%)** |
| Emitted paragraph contains the gold (SELECTION ok) | **81/154 = 0.526** |
| METEOR when selection ok | **0.7700** |
| METEOR when selection wrong | **0.1109** |
| median len(gold)/len(containing paragraph) | 0.369 |
| gold spans beyond one paragraph | 3/150 |

The decisive finding: passage METEOR is almost exactly linear in selection accuracy, `≈ 0.77·acc + 0.11·(1−acc)`. **We choose the wrong paragraph 47% of the time**, and that alone explains the low score. Extent is a *minor* issue — the gold is only ~37% of the paragraph we emit, yet METEOR is still 0.77 when the paragraph is right, because METEOR weights recall far above precision. Multi-paragraph spoilers are negligible (3/150).

### 11c. Can a cheaper selector fix it? (all rejected)
| Strategy | Selection acc | passage METEOR |
|----------|---------------|----------------|
| span_top1 (current) | 0.545 | 0.4694 |
| span_mass (softmax mass per paragraph) | 0.545 | 0.4687 |
| tfidf_post | 0.240 | 0.2569 |
| tfidf_pt (post+title) | 0.266 | 0.2723 |
| hybrid (z(span_mass)+z(tfidf)) | 0.487 | 0.4349 |
| **Oracle (gold paragraph is selectable)** | **0.955** | ~0.74 |

### Notes
- Every lexical alternative lost badly to the QA model's own span signal (0.24–0.27 vs 0.545), which independently **replicates Hagen et al.'s (2022) finding that QA models substantially outperform passage retrieval** — TF-IDF similarity to the post simply does not identify which paragraph conceals the spoiler. Aggregating span probability mass per paragraph was also flat, so a single stray argmax is not the failure mode.
- The oracle at **0.955** proves the headroom is real and large: the correct paragraph is almost always present and selectable, and we pick it barely half the time. Passage ceiling ≈ 0.74 vs our 0.458.
- **Takeaway for the report:** the bottleneck is neither type classification (+0.039 ceiling) nor answer extent, but *paragraph selection for passage spoilers* — worth up to +0.12 overall. Heuristics are exhausted, so the next step is a **trained cross-encoder paragraph ranker**, which is exactly the "information condensation" idea used by the SemEval-2023 winning system and the neural-ranking approach (MonoBERT/MonoT5) Hagen et al. recommend. Supervision is free: `spoilerPositions` identifies the gold paragraph for all 3,200 training records.

---

## Experiment 010 — Per-type decoding sweep against METEOR (val win, weak test transfer)
**Date:** 2026-07-19
**Script:** `task2/exp_sweep_types.py` (2 rounds), applied to `train.py` CONFIG

### Method
Because a record's prediction depends only on its own type's config, and overall METEOR is a per-record average, each type can be optimized independently on its own subset — exactly optimal and far cheaper than a cross-product sweep. Round 1 found both winners sitting at the grid edge, so round 2 extended the ranges.

### Results (val)
| Type | Old | New | Val METEOR |
|------|-----|-----|------------|
| phrase | span 10 | **span 75** | 0.4403 → **0.4962** |
| passage | fullpara | fullpara (unchanged) | 0.4102 |
| multi | span k=3 | **span k=5** | 0.4260 → **0.5297** |
| **Overall** | | | 0.4261 → **0.4705** |

Combined-run verification reproduced the projected 0.4705 exactly, confirming the per-type decomposition. Full-paragraph mode was rejected for both phrase (0.3439) and multi (0.3230/0.3824) — it only helps passage.

### Kaggle (submission ref 54834668) — the important part
| | Val | Test |
|---|---|---|
| Exp 008 | 0.4261 | 0.44479 |
| Exp 010 | 0.4705 | **0.44731** |
| Δ | **+0.0444** | **+0.0025** |

### Notes
- A large val gain almost entirely failed to transfer: **+0.044 val became +0.0025 test** (~6%). Rank improved 6th → 5th, but my projection of ~0.49 test was badly wrong, and the earlier two-point calibration (test ≈ val + 0.019) inverted. Lesson: a calibration built from two submissions is not a calibration.
- **Cause 1 — distribution shift.** The Task 1 classifier predicts a different type mix on test (passage 44%) than val (passage 38%). Passage is our *weakest* bucket (0.4102 vs ~0.50 for phrase/multi), so test is weighted toward our weakness. Re-weighting val per-type scores by the test distribution predicts 0.4642, explaining ~0.006 of the gap.
- **Cause 2 — val overfitting.** The remaining ~0.017 is fitting noise: the per-type subsets are only 168/151/**81** records, so choosing `multi k=5` from an 81-sample curve was far less certain than the smooth trend suggested.
- **Strategic consequence:** every gain so far came from phrase and multi — the buckets that matter *less* on test. **Passage (0.4102, 44% of test) is now the highest-leverage target**, alongside Task 1 type accuracy (only **0.695**; per-type classifier recall phrase 0.716 / passage 0.695 / multi 0.655), since aggressive per-type params amplify the cost of misrouting. **KEEP the config (it is a real if small test gain); redirect effort to passage.**

---

## Experiment 009 — Phase 2: generative seq2seq for passage+multi (NEGATIVE RESULT)
**Date:** 2026-07-19
**Script:** `task2/exp_finetune_gen.py` (flan-t5-base, 3 epochs total, lr 3e-4 then 1e-4)

### Hypothesis
Pal et al. (2024) report LongT5 reaching METEOR ~0.90 on passage and ~0.85 on multi vs ~0.32 for extractive QA, because generation emits the full passage (high recall) while extraction truncates. So a generative model should beat our extractive decoder on those two types.

### Config
| Parameter | Value |
|-----------|-------|
| Model | google/flan-t5-base (M4/16GB; the paper used LongT5 on an A100) |
| Input | type-conditioned prompt + post + title + top-5 paragraphs (TF-IDF context reduction) |
| Target | gold spoiler (joined); trained on all types, evaluated on the val passage+multi subset with predicted types |

### Results
| Epoch | passage+multi METEOR | passage | multi |
|-------|----------------------|---------|-------|
| 1 | 0.3034 | 0.3315 | 0.2509 |
| 2 | 0.3063 | 0.3228 | 0.2754 |
| 3 | **0.3137** | 0.3392 | 0.2661 |
| **Extractive baseline (Exp 008)** | **0.4157** | **0.4102** | **0.4260** |

### Notes
- **Hypothesis rejected.** The generator finished 0.102 below the extractive decoder and improved only +0.010 across three epochs, so the curve is flat, not merely undertrained — it would need implausibly many more epochs to reach parity. Training loss fell steadily (1.92 → 0.75), so the model was learning; it simply learns to paraphrase rather than reproduce the exact spoiler wording that METEOR rewards.
- Two honest reasons the paper's result did not transfer: I used flan-t5-**base** (~250M) where they used a much larger LongT5, and their ~0.90 passage METEOR looks unreachable for this leaderboard's metric — **the Kaggle leader is at 0.50022**, so if 0.90 were attainable the leaderboard would not top out near 0.50. That external check is the strongest evidence that the published number does not describe our scoring setup.
- Takeaway for the report: this is the value of empirical verification over trusting a published headline. **DISCARD generative; redirect effort to decoding tuning and Task 1 type accuracy.** Checkpoints kept at `models/gen_ft` / `gen_ft_ep1` for the write-up.

---

---
