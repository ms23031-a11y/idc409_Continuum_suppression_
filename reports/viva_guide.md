<!-- GENERATED from templates/viva_guide.md.tpl and saved results (config fingerprint 4d55652eba8e68384e84f9b45973034903f7c4fb1abdab57660c062c084d99c4). -->
# Viva / presentation guide - Daman Kumar (MS23031)

## 30-second pitch
At Belle II, B-pair events (signal-like for B decays, spherical) must be separated from continuum e+e- -> qqbar events (usually jet-like). Event-shape variables (R2, thrust, CLEO cones, KSFW moments) measure that shape difference. I trained classifiers on 59 such variables to separate **signal (type +1)** from **continuum (type 0 / -1)**, using the code meanings given by the instructor. Only 462 continuum events exist (35,303 signal), so I used stratified splits, training-only class weighting, an explicit operating point and exact confidence intervals.

## Key points
* **Mapping:** type +1 signal; 0 or -1 continuum; 2-5 other decays (instructor, relayed by me). I excluded 2-5 as **my analysis decision**. File has 0 type -1 rows.
* **Why not accuracy?** 98.7% of rows are signal: predict-all-signal gets 0.987 accuracy and rejects no continuum.
* **Operating point:** max continuum rejection subject to signal efficiency >= 90% on validation (90% is my choice); threshold frozen before test.
* **Result:** On the instructor-provided mapping, at the validation-chosen operating point (signal efficiency floor 90%) the validation-selected model keeps 93.3% of true signal and rejects 5.8% of true continuum on the test set (only 69 continuum events; 95% exact interval 1.6%, 14.2%). For comparison, a score with no discrimination would reject about 6.7% at this efficiency. The validation-selected model (Random forest) has test ROC-AUC 0.505 (close to chance); its 95% interval 0.434-0.573 includes 0.5 (chance). The highest test AUC of any model is 0.567 (Single-feature cut, interval 0.504-0.630, which excludes 0.5 only barely; it was not the validation choice). Models whose AUC interval includes 0.5: Logistic regression, Decision tree, Random forest, Gradient boosting.
* **Uncertainty:** only 69 test continuum events -> exact (Clopper-Pearson) intervals, stratified bootstrap for AUC, split repeats.

## Likely questions and honest answers
| Question | Answer |
|---|---|
| Where did the labels come from? | The instructor's clarification, relayed by me. I could not verify it from the data. |
| Did the instructor say to exclude codes 2-5? | No. Excluding them is my analysis decision because they are neither signal nor continuum. |
| Why are there no type -1 events? | The CSV has none (type 0: 462 only). I did not create any. |
| Why not report accuracy? | Because of the 98.7% signal fraction; I report signal efficiency, continuum rejection, balanced accuracy and macro-F1. |
| Efficiency vs rejection? | Efficiency = fraction of true signal kept; rejection = fraction of true continuum rejected; acceptance = 1 - rejection. Different denominators (5,297 signal, 69 continuum in the test set). |
| Is the 90% efficiency a requirement? | No, an illustrative analysis choice. |
| Why is the test result not exactly 90% efficiency? | The threshold was frozen on validation data; the test set gives what it gives. |
| ROC-AUC at different thresholds? | Unchanged: it is computed from the continuous scores of the same rows. |
| How many models? | Four learned classifiers (logistic regression, decision tree, random forest, gradient boosting) plus two baselines (majority; single-feature cut using one feature, `thrustOm`, from 59 candidates). |
| Did class weighting help? | Compared on validation data per model (Table F in the report); weights use training labels only; validation/test are never reweighted or resampled. |
| How good is the separation? | The validation-selected model (Random forest) has test ROC-AUC 0.505 (close to chance); its 95% interval 0.434-0.573 includes 0.5 (chance). The highest test AUC of any model is 0.567 (Single-feature cut, interval 0.504-0.630, which excludes 0.5 only barely; it was not the validation choice). Models whose AUC interval includes 0.5: Logistic regression, Decision tree, Random forest, Gradient boosting. |
| Is that physical? | Continuum is normally more jet-like than signal, but the type-0 events look signal-like in these features; I flag it for the instructor rather than changing the mapping. |
| How did you avoid leakage? | Excluded `type`, index and identifiers; stratified duplicate-aware split before anything learned; preprocessing/weights on training only; model, hyper-parameters and threshold on validation; test scored once; an isolation check with scrambled test data gave identical choices. |
| Feature reduction? | 59 -> 47: logistic regression -0.0030 (CI includes 0); decision tree -0.0084 (CI includes 0); random forest -0.0072 (CI includes 0); gradient boosting -0.0094 (CI includes 0). 59 -> 3: logistic regression +0.0360 (CI includes 0); decision tree +0.0444 (CI includes 0); random forest +0.0428 (CI includes 0); gradient boosting +0.0063 (CI includes 0). Validation is noisy; importance is associational. |
| What about the old `type==1` vs rest results? | A different task (negative class contained other decays); kept only as a separate historical note, never compared. |

## Limitations to state yourself
Instructor-provided, unverifiable mapping; very few continuum events; wide intervals; no event ID; simulated data; bounded tuning; associational importance.
