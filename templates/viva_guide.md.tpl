<!-- GENERATED from templates/viva_guide.md.tpl and saved results (config fingerprint {{config_sha}}). -->
# Viva / presentation guide - {{student_name}} ({{roll}})

## 30-second pitch
At Belle II, B-pair events (signal-like for B decays, spherical) must be separated from continuum e+e- -> qqbar events (usually jet-like). Event-shape variables (R2, thrust, CLEO cones, KSFW moments) measure that shape difference. I trained classifiers on {{n_feat}} such variables to separate **signal (type +1)** from **continuum (type 0 / -1)**, using the code meanings given by the instructor. Only {{n_continuum}} continuum events exist ({{n_signal}} signal), so I used stratified splits, training-only class weighting, an explicit operating point and exact confidence intervals.

## Key points
* **Mapping:** type +1 signal; 0 or -1 continuum; 2-5 other decays (instructor, relayed by me). I excluded 2-5 as **my analysis decision**. File has {{n_minus1}} type -1 rows.
* **Why not accuracy?** {{sig_frac}} of rows are signal: predict-all-signal gets {{maj_acc}} accuracy and rejects no continuum.
* **Operating point:** max continuum rejection subject to signal efficiency >= {{floor}} on validation ({{floor}} is my choice); threshold frozen before test.
* **Result:** {{verdict_sentence}} {{separation_sentence}}
* **Uncertainty:** only {{test_con}} test continuum events -> exact (Clopper-Pearson) intervals, stratified bootstrap for AUC, split repeats.

## Likely questions and honest answers
| Question | Answer |
|---|---|
| Where did the labels come from? | The instructor's clarification, relayed by me. I could not verify it from the data. |
| Did the instructor say to exclude codes 2-5? | No. Excluding them is my analysis decision because they are neither signal nor continuum. |
| Why are there no type -1 events? | The CSV has none ({{cont_codes_present}} only). I did not create any. |
| Why not report accuracy? | Because of the {{sig_frac}} signal fraction; I report signal efficiency, continuum rejection, balanced accuracy and macro-F1. |
| Efficiency vs rejection? | Efficiency = fraction of true signal kept; rejection = fraction of true continuum rejected; acceptance = 1 - rejection. Different denominators ({{test_sig}} signal, {{test_con}} continuum in the test set). |
| Is the {{floor}} efficiency a requirement? | No, an illustrative analysis choice. |
| Why is the test result not exactly {{floor}} efficiency? | The threshold was frozen on validation data; the test set gives what it gives. |
| ROC-AUC at different thresholds? | Unchanged: it is computed from the continuous scores of the same rows. |
| How many models? | Four learned classifiers (logistic regression, decision tree, random forest, gradient boosting) plus two baselines (majority; single-feature cut using one feature, `{{single_feature}}`, from {{n_candidates}} candidates). |
| Did class weighting help? | Compared on validation data per model (Table F in the report); weights use training labels only; validation/test are never reweighted or resampled. |
| How good is the separation? | {{separation_sentence}} |
| Is that physical? | Continuum is normally more jet-like than signal, but the type-0 events look signal-like in these features; I flag it for the instructor rather than changing the mapping. |
| How did you avoid leakage? | Excluded `type`, index and identifiers; stratified duplicate-aware split before anything learned; preprocessing/weights on training only; model, hyper-parameters and threshold on validation; test scored once; an isolation check with scrambled test data gave identical choices. |
| Feature reduction? | {{n_feat}} -> {{n_corr}}: {{corr_sentence}}. {{n_feat}} -> {{k_sel}}: {{reduced_sentence}}. Validation is noisy; importance is associational. |
| Is the weak result a bug? | I checked: labels and positive-class column aligned; metrics recomputed with independent sklearn functions; shuffled-label control AUC {{sa_neg_range}}; the same code reaches validation AUC {{sa_pos_logreg}}/{{sa_pos_boost}} on a learnable task; pooled CV on train+val ({{sa_cv_n_con}} continuum events) gives {{sa_cv_logreg}} (logistic). No bug found. |
| Has the test set influenced development? | {{sa_disclosure}} |
| Can event-level leakage be excluded? | No: there is no physical event identifier. Only exact duplicates were checked. |
| What about the old `type==1` vs rest results? | A different task (negative class contained other decays); kept only as a separate historical note, never compared. |

## Limitations to state yourself
Instructor-provided, unverifiable mapping; very few continuum events; wide intervals; no event ID; simulated data; bounded tuning; associational importance.
