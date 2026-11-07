# Viva prep

## 1. The story in one minute (learn this by heart)

Hybrid search mixes BM25 (keyword) and dense (meaning) scores with one fixed weight, α.
We asked: can a router pick the best retriever **per query** and beat that fixed mix?

- **No.** The router never beat a well-tuned hybrid, on any dataset.
- **Why:** which retriever "wins" a query mostly depends on **which relevant document** you look at, not on the
  query itself. So nothing about the query can predict the winner well.
- **So the fix is to decide per document, not per query.** A small model that mixes BM25 and dense per document
  beat fixed-α hybrid when all 6 new datasets are pooled (+0.003). It was ahead on every one of the 6, but clearly
  ahead (confidence interval above zero) only on NQ and HotpotQA. Small, but in the right direction everywhere.
- **Exception:** multi-hop questions (HotpotQA), where every relevant document is needed. There the query does
  predict the winner better, but even there routing did not beat hybrid.

## 2. Every hypothesis in one line

| | What it says | Result |
| :--- | :--- | :--- |
| E1 | A router trained on fake (synthetic) queries fails on real ones | Accepted |
| E2 | No group of real queries is better served by BM25 or dense alone | Accepted, repeated on new data and with e5-base |
| E3 / E3′ | A router can skip dense search for 20% of queries at little cost | Not met / inconclusive (both predicted) |
| E4 | The winner depends on the document, not the query | Inconclusive: true everywhere except HotpotQA (wordpress too small to measure) |
| E4-R | Same as E4, but tested per dataset on new data (Climate-FEVER) | **Accepted**: 0.507, basically a coin flip |
| E5 | A small special group of queries exists where routing wins | Rejected (predicted) |
| E6 | Mixing per document beats fixed-α hybrid | **Accepted** when pooled; ahead on all 6, clearly ahead on 2 |
| E7 | A free one-line rule skips as well as the learned router | Inconclusive: true on 5 of 6, not HotpotQA |
| E8 | The E4 idea fails on multi-hop data, as on HotpotQA | Inconclusive: multi-hop is clearly more stable than DBpedia (+0.09), but only 2Wiki (0.82) acts like HotpotQA; MuSiQue (0.60) doesn't. Both side predictions failed |

Naming trap: the first pre-registration file calls the skip-dense test "E4". In the report it is E3 / E3′, and
"E4" is the split-half test. Say this if an examiner opens the pre-registration files.

## 3. Numbers to remember

- E1: synthetic-trained router vs hybrid: **−0.021** nDCG@10.
- E2: router vs hybrid ≈ **0.000**; best feature AUC′ ≈ **0.58–0.60** (0.5 is a coin flip).
- E4: split-half AUC′ **0.52** without HotpotQA, **0.84** on HotpotQA. Pooled it was **0.7498**, just under the
  0.75 "reject" line, so "inconclusive" by a hair. 73% of exploration queries (37% of fresh ones) have only one
  relevant document, and the split-half test can't use those.
- E6: **+0.0029** vs hybrid; **+0.0011** vs hybrid tuned on each dataset's own labels (there NQ is worse, −0.004);
  extra time ≈ **+5 ms** per query (22.7 → 27.9 ms, about +23%).
- Changing the dense model (e5-small → e5-base): only **1.8%** of winners flip to the other retriever; **44%** become
  "no clear winner". The e5-small score gap still predicts the e5-base winner very well (AUC′ 0.95). So the encoder
  mostly changes *whether* there is a clear winner, not *which* one.
- E7: the H1 rule saves about **19%** latency at a 20% skip rate; the learned router is slower than plain hybrid
  unless it skips more than ~**35%** of queries (post hoc numbers without the laptop-sleep measurement).
- Predictions: **9 of 13** correct. The 4 misses (E4, E7, E8, learning curve) all involve multi-hop data.

## 4. Terms you must be able to explain simply

- **nDCG@10:** how good the top 10 results are; relevant documents near the top count more.
- **α:** how much weight the dense score gets in the hybrid (0 = only BM25, 1 = only dense).
- **AUC′:** how well a score separates two groups. 0.5 = no better than chance, 1.0 = perfect. We take the better of
  the two directions, which is why there's a prime.
- **95% CI (bootstrap):** resample the queries many times; the range where the true difference most likely lies.
  If it doesn't include 0, the difference is unlikely to be due to chance alone.
- **Pre-registration:** we wrote down the test, the thresholds and the prediction, and hashed the file, before
  downloading the test data. Honest caveat: the files were committed to git together with the results, so git alone
  can't prove the order; the hashes prove only that nothing was changed afterwards.
- **Split-half test (E4):** split a query's relevant documents into two halves at random. If the winner were a property
  of the query, both halves would agree. They mostly don't.
- **Holm correction:** when you test 4 things, one can look good by luck; Holm makes the bar stricter.
- **Held-out / fresh data:** data never used while designing the method; only used once for the final test.

## 5. Questions you'll likely get (with short answers)

- **"Your method didn't work. What's the contribution?"** We show *why* per-query routing can't work, that explanation
  points to a fix that does work (per-document mixing), and we find where the explanation breaks (multi-hop).
- **"+0.003 is tiny. Does it matter?"** It's small. Pooled over 6 new datasets it is clearly above zero, and it was
  ahead on every one of them with a model that never saw them; individually only 2 are clearly above zero. We don't
  claim a big jump; we claim it's real when pooled and that it follows from the explanation.
- **"Isn't per-document mixing just learning-to-rank?"** Yes, the technique is known. The new part is the evidence for
  *why* it beats routing.
- **"Why is HotpotQA different?"** Its questions need two documents together. A retriever that understands the
  question finds both, so the winner really does depend on the question.
- **"Did you tune things on the test data?"** No. All choices were fixed on old data, written down and hashed first;
  each test set was used once. One disclosure: a team member had looked at DBpedia (E8's control set) in an earlier
  exploratory check, so that one set was not completely unseen.
- **"E4 was inconclusive, so why do you trust the mechanism?"** The original test pooled six datasets, and one
  multi-hop dataset (HotpotQA, 46% of the queries) dominated it. That was a design mistake. We fixed the design,
  pre-registered it again and tested it on new data (Climate-FEVER): AUC′ 0.507, a coin flip. Both results are
  reported; we didn't change the original verdict. Caveat: Climate-FEVER's labels are noisy, which could also push it
  towards a coin flip.
- **"Why did some predictions fail?"** We report them as failures. All four misses (E4, E7, E8 and the learning curve)
  involve multi-hop data (HotpotQA; MuSiQue), and that led to E8. E8 was partly supported: multi-hop data is more
  query-dependent on average, but MuSiQue (built to block shortcuts) behaves like ordinary data, and both of E8's side
  predictions failed. So the exception holds for some multi-hop datasets, not all.
- **"What does it cost?"** Per-document mixing adds about 5 ms per query (about 23% slower than plain hybrid). The H1
  skip rule makes search ~19% faster for a small quality loss.
- **"Why not just use more features or more data?"** We tried: query embeddings (Exp 2c) and cross-encoder scores
  (Exp 2e) didn't help. More data made the router better at *predicting* the winner but not better at *results*.
  (We had predicted the predicting part would level off; it didn't, which is one of our four misses.)

- **"How were the thresholds decided?"**
  - 0.005 = smallest gain worth caring about; −0.01 and the 20% skip rate came from the Phase 1 plan (§4.1).
  - AUC′: 0.5 is chance, under 0.65 means no useful signal, 0.70+ means a real one (common rule of thumb).
  - κ: the standard scale (under 0.2 "slight", 0.4 "moderate", 0.6 "substantial").
  - All of them were written down and hashed **before** the test data was downloaded. Some were chosen after looking
    at the old (exploration) data, which is allowed: old data designs the test, new data runs it.
  - If they say "arbitrary?": "Somewhat, like any cut-off. That's why we report the real numbers with confidence
    intervals too."

## 6. Weak spots: admit them, don't hide them

- The positive result is small (+0.003), and it costs about 5 ms extra per query.
- E4 was inconclusive overall because of HotpotQA, and E8 only partly explained why.
- One family of dense models (e5) and one mixing method (min-max) were used. e5 was trained on NQ data, so NQ was not
  new to the dense model.
- The split-half test only works on queries with 2+ relevant documents (a minority in the forums).
- The laptop went to sleep during one CPU timing run (one query recorded at 57 minutes). The pre-registered cost table
  averaged over it; we recomputed it post hoc without that query (`tables_cpu_clean_posthoc.md`). No verdict changed.

## 7. Before the viva

- Be able to explain sections 1 and 4 **without notes**.
- Know which script does what: `run_experiment2_heldout_routing.py` (router), `analyze_label_reliability.py` (E4),
  `run_learned_fusion.py` (E6), `run_cost_frontier.py` (cost), `compare_boundary.py` (E8).
- Know where the pre-registrations are: `results/preregistration/`.
