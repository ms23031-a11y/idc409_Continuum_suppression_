# Historical comparison - a DIFFERENT classification task (superseded)

**Do not compare these numbers with the signal-versus-continuum results.** Before the instructor's clarification, an earlier analysis classified `type == 1` against **all** other codes (0, 2, 3, 4, 5). Its negative class therefore contained other decays (codes 2-5, 34,841 rows) as well as the 462 continuum rows, the two classes had 35,303 rows each, and the scores say nothing specific about performance against continuum. The current task keeps only codes 1, 0, -1 (35,303 signal vs 462 continuum). These historical results are not used in any current table, figure or conclusion.

Full-feature test results of that earlier task (`artifacts/historical_type1_vs_rest/results_full_superseded.csv`):

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Majority class | 0.500 | 0.500 | 0.667 |
| Single-feature cut | 0.897 | 0.818 | 0.829 |
| Logistic regression | 0.915 | 0.844 | 0.851 |
| Decision tree | 0.908 | 0.834 | 0.842 |
| Random forest | 0.926 | 0.850 | 0.857 |
| Gradient boosting (HistGB) | 0.944 | 0.871 | 0.879 |
