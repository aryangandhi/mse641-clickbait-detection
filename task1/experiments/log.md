# Task 1 — Experiment Log
Spoiler type classification (phrase / passage / multi). Primary metric: **weighted F1** (Kaggle leaderboard).

---

## Experiment 001 — TF-IDF + Logistic Regression (Baseline)
**Date:** 2026-06-21  
**Script:** `task1/baseline.py`

### Config
| Parameter | Value |
|-----------|-------|
| Features | postText + targetTitle + targetDescription + targetParagraphs[:3] |
| Vectorizer | TF-IDF, unigram+bigram, max_features=50k, sublinear_tf=True, min_df=2 |
| Model | LogisticRegression, C=1.0, class_weight=balanced, max_iter=1000 |

### Results
| Split | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| Val   | 0.5270      | 0.5170   |
| Test (Kaggle) | 0.5404 | — |

### Per-class Val F1
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| phrase | 0.53 | 0.53 | 0.53 |
| passage | 0.52 | 0.60 | 0.56 |
| multi | 0.56 | 0.39 | 0.46 |

### Notes
- `multi` class is weakest — underrepresented (17.5%) and harder to distinguish from passage
- Test F1 (0.5404) is slightly better than val (0.5270) — suggests model generalizes reasonably
- **Submitted for milestone** ✓
- Next: try including more paragraphs and feature engineering (spoiler count heuristic)

---

<!-- Copy template below for each new experiment -->

<!--
## Experiment 00X — <Name>
**Date:** YYYY-MM-DD
**Script:** `task1/...`

### Config
| Parameter | Value |
|-----------|-------|
| | |

### Results
| Split | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| Val   | | |
| Test (Kaggle) | | |

### Notes
-
-->

## Experiment 002 — TF-IDF + Handcrafted Features + LR
**Date:** 2026-07-12
**Script:** `task1/exp_features.py feats_lr`

### Config
| Parameter | Value |
|-----------|-------|
| Features | Baseline TF-IDF + 15 handcrafted feats (headline numbers, listicle regex, wh-words, this/these, you, ?, title colon, n_paragraphs, log article words, mean para len, title len/nums, keyword count) |
| Model | LogisticRegression, C=1.0, class_weight=balanced |

### Results
| Split | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| Val   | 0.5480      | 0.5420   |

### Notes
- From my EDA, numbers in the headline are a strong `multi` signal (34% of multi headlines contain one vs ~6% for the other classes), so I added count/regex features on top of the baseline TF-IDF — the same feature-engineering-before-learned-representations progression we followed in the assignments. Multi F1 improved 0.46 → 0.51, consistent with the hypothesis.
- Overall +0.021 over the baseline val weighted F1 (0.5270). **KEEP**

---

## Experiment 003 — + Headline char n-grams (3-5)
**Date:** 2026-07-12
**Script:** `task1/exp_features.py feats_chargrams`

### Config
| Parameter | Value |
|-----------|-------|
| Features | Exp 002 + char_wb 3-5 gram TF-IDF of headline (50k feats) |
| Model | LogisticRegression, C=1.0, class_weight=balanced |

### Results
| Split | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| Val   | **0.6257**  | 0.6145   |

### Notes
- I tried character n-grams on the headline because clickbait style lives in surface patterns (punctuation, casing, word shapes) that word-level unigrams/bigrams miss — the subword-features intuition from lecture. The jump was bigger than I expected: +0.078 over Exp 002, and every class improved (multi 0.56, passage 0.62, phrase 0.67).
- I swept regularization strength the way we swept L2 in A4: C=1.0 was best (0.3→0.585, 2→0.612, 5→0.621, 10→0.604). **KEEP — new best linear model**

---

## Experiment 004 — LinearSVC comparison
**Date:** 2026-07-12
**Script:** `task1/exp_features.py feats_svm`

### Config
| Parameter | Value |
|-----------|-------|
| Features | Exp 002 features (no char-grams) |
| Model | LinearSVC, C=0.5, class_weight=balanced |

### Results
| Split | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| Val   | 0.5726      | 0.5666   |

### Notes
- The linear SVM beats LR on the same features (0.5726 vs 0.5480) but still loses to LR with char-grams (0.6257), so the feature representation mattered more than the classifier here.
- LinearSVC also gives no calibrated probabilities, which I want for ensembling later. **DISCARD in favor of Exp 003**

---

## Experiment 005 — Char n-gram ablations
**Date:** 2026-07-12
**Script:** inline (variants of `exp_features.py`)

### Results
| Variant | Val Weighted F1 |
|---------|-----------------|
| headline char 3-5 grams (=Exp 003) | 0.6257 |
| headline+title char grams | 0.6102 |
| headline char 2-6 grams | 0.6258 |
| char grams only, no word TF-IDF | 0.5966 |

### Notes
- The ablation shows the signal is specifically in the headline's writing style — adding the article title's char-grams actually diluted it (0.6102), which makes sense since titles are written by journalists, not the clickbait poster.
- Word TF-IDF still contributes about +0.03 over char-grams alone, so both views of the text are complementary.
- The 2-6 range ties 3-5, so I kept 3-5 (smaller feature space). **Config frozen: Exp 003 = best linear**

---

## Experiment 006 — roberta-base fine-tune
**Date:** 2026-07-12
**Script:** `task1/exp_transformer.py roberta-base 4 2e-5 256`

### Config
| Parameter | Value |
|-----------|-------|
| Model | roberta-base, 4 epochs, lr 2e-5 linear decay, batch 16, seq 256 |
| Input | headline </s> title + description + first 4 paragraphs |
| Loss | CE with balanced class weights |
| Device | MPS (Apple GPU), ~50 min |

### Results
| Split | Weighted F1 |
|-------|-------------|
| Val epoch 1 | 0.4288 |
| Val epoch 2 | 0.6617 |
| Val epoch 3 | 0.6834 |
| Val epoch 4 | **0.6851** |

### Notes
- Fine-tuning a pretrained transformer (the contextual-embeddings step up from the static Word2Vec embeddings of A3/A4) beats my best linear model by +0.06. Epoch 1 looked broken (0.4288) but it was just warm-up — loss only started dropping mid-epoch, a good reminder not to early-stop too aggressively.
- Val F1 was still climbing at epoch 4, so more epochs (or a larger model) is a cheap follow-up. Val/test probabilities saved to `experiments/probs_roberta-base.npz` for ensembling. **KEEP**

---

## Experiment 007 — Ensemble: linear (Exp 003) + transformer (Exp 006)
**Date:** 2026-07-12
**Script:** `task1/exp_ensemble.py experiments/probs_roberta-base.npz`

### Config
| Parameter | Value |
|-----------|-------|
| Blend | w * transformer_probs + (1-w) * linear_probs, w swept 0..1 step 0.05 |
| Best w | 0.60 (transformer) |

### Results
| Split | Weighted F1 |
|-------|-------------|
| Val   | **0.6949** |
| Test (Kaggle) | **0.70573** |

### Notes
- Blending gives +0.010 over the transformer alone, confirming the two models make complementary errors (the linear model sees character-level style; the transformer sees meaning).
- Submissions written: `submissions/ensemble.csv` (best) and `submissions/linear_e003.csv`.
- One caveat I want to note honestly: the blend weight was tuned on the same val split I used for model selection, so there is mild overfitting risk — but every w in 0.5-0.7 beat both single models, so the conclusion is robust. This mirrors the val-vs-test gap lesson from A4. **KEEP — submitted 2026-07-12, Kaggle public score 0.70573 (baseline 0.5404)** ✓

---

## Experiment 008 — Relevance-filtered paragraphs (information condensation)
**Date:** 2026-07-18
**Script:** `task1/train.py` (CONFIG `relevance_paragraphs`); rows in tuning_runs.csv at 2026-07-18 15:13

### Config
| Parameter | Value |
|-----------|-------|
| Change | Replace `targetParagraphs[:3]` with the 3 paragraphs whose content-word overlap with post+title is highest (length-normalized), sorted back into reading order |
| Motivation | Billy-Batson (SemEval-2023 winner) condensed the article to its most-relevant paragraphs before classifying; Chick Adams found raw document content dilutes the post signal |

### Results
| Variant | Val Weighted F1 |
|---------|-----------------|
| Relevance OFF, linear (Exp 003 baseline) | 0.6257 |
| Relevance ON, linear | 0.6105 |
| Relevance OFF + blend (best) | 0.6949 |
| Relevance ON + blend | 0.6924 |

### Notes
- My cheap lexical-overlap proxy for "relevance" actually hurt the linear model (−0.015) and the blend (−0.0025). The winner's condensation used a *contrastively-trained* RoBERTa-large ranker; a bag-of-words overlap score is too crude and, worse, it tends to pick paragraphs that echo the post's own words — redundant with features the TF-IDF already extracts from the post, rather than adding new evidence.
- This also fits the Chick Adams finding differently than I expected: their point was that document content is weak *relative to post+title*, and condensation only mattered because their transformer had a 200-token cap. My linear bag-of-words already ingests the whole post+title+description, so shuffling which 3 paragraphs get appended is marginal — and my selector made it worse. **REVERT — keep `relevance_paragraphs: False`.**
- Takeaway for the next step: the real lever from these papers is *post-focus* (post+title >> document by +14-21% F1 in Chick Adams), which I should test directly by varying `n_body_paragraphs` down toward 0-1, and ultimately a DeBERTa fine-tune. Negative result worth keeping in the report as evidence that naive condensation ≠ the winner's learned condensation.

---

## Experiment 009 — Post-focus: sweep n_body_paragraphs
**Date:** 2026-07-18
**Script:** `task1/train.py` (CONFIG `n_body_paragraphs`); rows in tuning_runs.csv at 2026-07-18

### Config
| Parameter | Value |
|-----------|-------|
| Change | Number of article body paragraphs appended to the text (post + title + description are always kept) |
| Motivation | Chick Adams (SemEval-2023): post+title features beat post+document features by +14-21% F1 — so *less* document content may help |

### Results
| n_body_paragraphs | Linear F1 | Blend F1 (w=0.60) |
|-------------------|-----------|-------------------|
| 0 (no paragraphs) | **0.6304** | 0.6948 |
| 1 | 0.6228 | — |
| 2 | 0.6201 | — |
| 3 (previous default) | 0.6257 | 0.6949 |
| 5 | 0.6049 | — |

### Notes
- Dropping the article paragraphs entirely (post + title + description only) gave the best *linear* model, 0.6304 vs 0.6257 at n=3 (+0.0047). This directly confirms the Chick Adams post-focus finding on our own data: the raw body paragraphs were diluting the bag-of-words signal, and more paragraphs (n=5) was strictly worse. The clickbait style lives in the post and title, exactly the surface-pattern intuition from Exp 003's char-grams.
- But with the RoBERTa blend the metric is flat (0.6948 vs 0.6949) — the transformer already encodes the post-focus signal, so the linear model's gain is redundant in the ensemble. **This is the important lesson: cheap linear/feature tweaks have plateaued because the cached roberta-base is the ceiling of the blend.** To move the leaderboard I need a stronger transformer (deberta-v3), not more feature engineering.
- Decision: **KEEP n_body_paragraphs=0** — it is the best standalone linear model, ties the blend, is simpler (no paragraph handling), and is the better-motivated foundation for when the transformer is upgraded. Blend weight held at 0.60 (best, ties). Next: DeBERTa-v3-base fine-tune (Exp D in the plan).

---

## Experiment 011 — DeBERTa-v3-base fine-tune + blend sweep
**Date:** 2026-07-18
**Script:** `exp_transformer.py microsoft/deberta-v3-base 6 1e-5 256` (post-focused); `train.py --blend-transformer` with blend_w sweep; rows in tuning_runs.csv at 2026-07-18

### Config
| Parameter | Value |
|-----------|-------|
| Transformer | DeBERTa-v3-base, 6 epochs, lr 1e-5, seq 256 |
| Input | Post + title + description (no article paragraphs — post-focused per Exp 009) |
| Blend sweep | w in {0.50, 0.55, 0.60, 0.65, 0.70} with linear model |

### Results
| Config | Val Weighted F1 |
|--------|-----------------|
| **DeBERTa alone (epoch 6 best)** | **0.7004** ✓ **NEW BEST** |
| DeBERTa + blend w=0.50 | 0.6926 |
| DeBERTa + blend w=0.55 | 0.6975 |
| DeBERTa + blend w=0.60 | 0.6977 |
| DeBERTa + blend w=0.65 | 0.6924 |
| DeBERTa + blend w=0.70 | 0.6924 |
| Previous best (RoBERTa + blend, Exp 007) | 0.6949 |

### Epoch Progression (DeBERTa)
| Epoch | Val F1 | Δ |
|-------|--------|-----|
| 1 | 0.5621 | — |
| 2 | 0.6537 | +0.0916 |
| 3 | 0.6790 | +0.0253 |
| 4 | 0.6853 | +0.0063 |
| 5 | 0.6922 | +0.0069 |
| 6 | 0.7004 | +0.0082 |

### Notes
- DeBERTa-v3-base fine-tuned on post-focused input (post + title + description, **no article paragraphs**) reached **0.7004 F1** — beats previous RoBERTa+blend best (0.6949) by **+0.0055**, exceeding the >0.005 threshold for keeping. **KEEP this model.**
- Epoch progression shows steady improvement with diminishing returns (typical learning curve). Model kept improving through epoch 6, suggesting more epochs might squeeze out marginal gains, but ROI drops sharply.
- **Surprise finding**: blending DeBERTa with the linear model **hurts** (0.6977 < 0.7004). This is the inverse of RoBERTa's behavior (where blending helped +0.0098). Interpretation: DeBERTa's contextual embeddings are now rich enough to encode all the signal the linear bag-of-words model provides. The character-grams and handcrafted features don't add new information; they're just noise.
- **Decision: KEEP DeBERTa alone (0.7004), discard blend.** For submission, use DeBERTa-v3-base standalone probabilities. This is significant: once a transformer is strong enough, ensemble-with-linear-baseline stops helping. The synergy between linear and RoBERTa came from complementary errors; DeBERTa alone fixes both.
- **Next step:** Error analysis on multi class (still ~0.46 F1, lowest of the three) or try DeBERTa-v3-large, focal loss, or longer fine-tuning for marginal gains toward 0.71-0.72 (path to 0.78 would require multi-specific architecture, per paper findings).

---

## Experiment 012 — DeBERTa-v3-base KAGGLE SUBMISSION (Standalone)
**Date:** 2026-07-18
**Script:** Generated from `exp_transformer.py` probabilities; no blending

### Submission Details
| Metric | Value |
|--------|-------|
| Val weighted F1 | 0.7004 (from Exp 011, best epoch) |
| **Test weighted F1 (Kaggle)** | **0.72938** ✓ **NEW BEST** |
| vs. Previous best (Exp 007, RoBERTa+blend) | +0.02365 |
| vs. Baseline (Exp 001) | +0.18898 |
| Submission file | `submissions/deberta_standalone.csv` |
| Status | COMPLETE |

### Test Set Predictions
| Class | Count | Pred % |
|-------|-------|--------|
| phrase | 152 | 38.0% |
| passage | 200 | 50.0% |
| multi | 48 | 12.0% |

### Notes
- **DeBERTa-v3-base test F1 = 0.72938**, exceeding Chick Adams' post-features baseline (0.7259 val on their setup). Our val→test generalization (0.7004→0.72938) is robust despite modest gap.
- **Post-focused input design confirmed**: post + title + description only, no article paragraphs. This aligns with Chick Adams' +14-21% F1 finding that post+title >> post+document.
- **Blending would have hurt** (0.6977 < 0.7004), so standalone DeBERTa is the right submission choice. Demonstrates that once a transformer is strong enough, the linear-model ensemble stops helping.
- **Test class distribution (38% phrase, 50% passage, 12% multi) is close to training balance**, suggesting the model generalizes without significant class-balance artifacts.
- **+0.02365 absolute improvement** over previous best (RoBERTa+blend), equivalent to +3.4% relative F1 gain. Architecture upgrade (RoBERTa→DeBERTa) is the primary lever, not ensemble tricks.

### Comparison to SemEval Leaders
| System | Test/Val F1 |
|--------|------------|
| **Our submission (DeBERTa-v3-base)** | **0.72938** ✓ |
| Chick Adams DeBERTa-large (post) | 0.7259 val |
| Chick Adams DeBERTa-large (combined) | 0.7415 val |
| Billy-Batson (SemEval winner) | 0.7414 val |
| Kaggle leader | 0.7866 test |

Our test score (0.72938) on DeBERTa-base already rivals the SemEval post-features approaches when accounting for larger model variants. Gap to leader: ~0.057 (~7.2%).

### Next steps to push toward 0.73-0.74 range
1. **DeBERTa-v3-large** (+0.5-1% typically over base) → ~0.73-0.74 expected
2. **Focal loss + fine-grained class weighting** on the multi bottleneck
3. **Ensemble with different random seeds** (small gains, ~0.1-0.3%)
4. Multi-class-specific architecture (requires error analysis, diminishing ROI)

**SUBMITTED TO KAGGLE 2026-07-18 21:32 — TEST SCORE 0.72938** ✓

---

## Experiment 013 — roberta-large + focal loss (NEW BEST val)
**Date:** 2026-07-24
**Script:** `T1_BATCH=8 T1_FOCAL=1 T1_FOCAL_GAMMA=2.0 exp_transformer.py roberta-large 5 1e-5 256`

### Motivation
Push past DeBERTa-v3-base (0.7004) with (a) a larger model and (b) **focal loss**
(γ=2, class-weighted) to attack the minority `multi` class — the SemEval-2023
winner's trick. **DeBERTa-v3-large was ruled out first**: on the M4/MPS it runs at
~54 s/step (~6 h/epoch) because its disentangled attention has no efficient MPS
kernel, so roberta-large (standard attention, MPS-friendly) is the feasible larger model.

### Config
| Parameter | Value |
|-----------|-------|
| Model | roberta-large (355M), post-focused input (post + title + description) |
| Loss | class-weighted **focal**, γ=2.0 |
| Train | 5 epochs, lr 1e-5, batch 8, seq 256, seed 42, MPS |

### Results
| Epoch | Val wF1 |
|-------|---------|
| 1 | 0.5968 |
| 2 | 0.7016 |
| **3** | **0.7156** (saved) |
| 4 | 0.6893 |
| 5 | 0.6962 |

Per-class (best epoch): multi P0.61 R0.70 **F1 0.65** · passage F1 0.72 · phrase F1 0.75.

### Notes
- **New best val weighted F1 0.7156, +0.0152 over DeBERTa-v3-base (0.7004)** — clears
  the +0.005 keep threshold. Focal loss did its job: `multi` F1 rose to **0.65** (recall
  0.70) from the ~0.46–0.51 it sat at with weighted CE in earlier experiments; since
  `multi` was the class dragging the weighted score, fixing it is what moved the metric.
- Clean peak at epoch 3 with mild overfitting after (0.716→0.689→0.696), correctly kept
  by best-epoch selection. Test-prob class distribution 81 multi / 148 passage / 171
  phrase — more `multi` than the deberta submission (48), consistent with focal raising
  minority recall.
- Two changes at once (arch roberta-large + focal) vs the deberta-base baseline, so
  attribution is not clean; a weighted-CE roberta-large ablation is the follow-up before
  claiming focal's isolated effect in the report.

### Kaggle result (submission ref 54952608, 2026-07-24) — val win did NOT transfer
| | Val wF1 | Kaggle test |
|---|---|---|
| DeBERTa-v3-base (Exp 012) | 0.7004 | **0.72938** |
| roberta-large + focal (this) | **0.7156** | 0.70250 |
| Δ | +0.0152 | **−0.0269** |

- **REVERT.** Val rose but test *fell* — a full reversal, the second val→test failure in the
  project (cf. Task 2 Exp 010). Mechanism: focal loss over-predicted `multi` (81 test preds
  vs deberta's 48). On the 400-sample val this raised `multi` recall (what focal targets), but
  on test the extra `multi` false positives cost precision and dragged weighted F1 down. **The
  DeBERTa-v3-base 0.72938 submission remains our Task 1 best; Kaggle keeps best-per-team so
  rank (16/33) is unaffected.** A roberta-large + weighted-CE ablation is running to separate
  "roberta-large is worse on test" from "focal over-prediction is the culprit."
- Report takeaway: with a 400-record val split, aggressive minority-class optimisation
  overfits the proxy. This is strong evidence for the paper's discussion of metric reliability.

---

## Experiment 014 — roberta-large + weighted-CE (attribution ablation → new best val)
**Date:** 2026-07-24
**Script:** `T1_BATCH=8 T1_FOCAL=0 exp_transformer.py roberta-large 5 1e-5 256`

### Purpose
Ablate Exp 013: same roberta-large, **weighted CE instead of focal**, to separate
"roberta-large is weak on test" from "focal over-predicted multi".

### Results
| Epoch | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|
| Val wF1 | 0.7211 | **0.7418** | 0.7097 | 0.7387 | 0.7393 |

Best val **0.7418**. Per-class: multi P0.68 R0.64 **F1 0.66** · passage F1 0.77 · phrase F1 0.76.
Test-pred class dist 63 multi / 194 passage / 143 phrase.

### Notes
- **Focal was the culprit, not roberta-large.** Plain weighted-CE roberta-large reaches
  **0.7418 val — +0.041 over deberta-base (0.7004) and +0.026 over the focal variant (0.7156)** —
  while focal's minority over-prediction (81 test multi) is gone here (63, near deberta's 48).
  Focal traded test-precision for val-recall on `multi`; standard CE does not.
- Val is **stable across epochs** (4–5 also ~0.739), so 0.7418 is a genuine level, not a one-epoch
  spike — unlike focal. Balanced class distribution + safe loss = the two properties the failed
  focal submission lacked, so this is a legitimate test candidate. Submission
  `roberta_large_ce.csv` written. **KEEP — best val model so far.**

### Kaggle result (submission ref 54958802, 2026-07-24) — TRANSFERRED, new best
| | Val wF1 | Kaggle test |
|---|---|---|
| DeBERTa-v3-base (prev best) | 0.7004 | 0.72938 |
| roberta-large + focal (Exp 013) | 0.7156 | 0.70250 (regressed) |
| **roberta-large + weighted-CE (this)** | **0.7418** | **0.73187** |

- **NEW BEST, and it transferred** (+0.0025 test over deberta-base) — unlike focal. This validates
  the diagnosis: focal's minority over-prediction was the failure; balanced weighted-CE is safe.
  Val over-estimated (0.7418 → 0.73187, ~−0.01) but the direction held.
- Rank still 16/33: the gain wasn't enough to pass the 0.7387 cluster (need ~+0.007 more).
  **roberta-large + weighted-CE is now the Task 1 best and the foundation for further work.**

---

## Experiment 015 — roberta-large + CE + Chick-Adams preprocessing (NEW BEST val)
**Date:** 2026-07-24
**Script:** `T1_BATCH=8 T1_FOCAL=0 T1_CLEAN=1 exp_transformer.py roberta-large 5 1e-5 256`

### Config
Exp 014 + input cleaning (`clean_text`): strip hyperlinks, `#tag`→`#[HASHTAG]`,
`@user`→`@[USER]`, remove emojis, expand safe contractions. A *robustness* lever
(cleaner signal), chosen over val-tuned knobs after two val→test reversals.

### Results
| Epoch | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|
| Val wF1 | 0.6790 | 0.7230 | 0.7394 | 0.7433 | **0.7490** |

Best val **0.7490** (+0.0072 over Exp 014's 0.7418). Per-class: multi P0.72 R0.64 **F1 0.68** ·
passage F1 0.77 · phrase F1 0.77. Test-pred dist 63 multi / 183 passage / 154 phrase (balanced).

### Notes
- **New best val, and still climbing at epoch 5** — the cleaned-input model learned slower
  (epoch 2 was only 0.7230, *below* the uncleaned 0.7418) but overtook by epoch 5. A caution
  against judging a run from an early epoch. Class distribution stays balanced (63 multi, same
  as the transferring Exp 014), so it carries the same good-transfer signal.
- Preprocessing helped here despite the worry that stripping hashtags removes style — likely
  because removing URLs/emojis/mention-noise sharpens the post+title signal the model keys on.
- Submission `roberta_large_ce_clean.csv` written (balanced). was
  still climbing so a longer run (7–8 ep) may gain more.

### Kaggle result (submission ref 54963664, 2026-07-25) — BIG WIN, best of project
| | Val wF1 | Kaggle test |
|---|---|---|
| roberta-large + CE (Exp 014) | 0.7418 | 0.73187 |
| **+ Chick-Adams preprocessing (this)** | 0.7490 | **0.75852** |
| Δ | +0.0072 | **+0.0267** |

- **Test 0.75852 — rank 16→11** (passed tang 0.7418, Henry 0.7468, Emma 0.75836). Preprocessing
  gave a far larger test gain (+0.027) than val gain (+0.007), and **test OVERSHOT val** (0.7490→
  0.75852) — the only run this session where the val→test gap ran in our favour. Interpretation:
  clean input generalises better to the (differently-noisy) test posts than the 400-val suggested,
  so a robustness lever transfers where val-tuned knobs didn't. Strong evidence for the paper that
  *input cleaning > metric-tuning* on small noisy val splits. **KEEP — Task 1 best, 0.75852.**

---
