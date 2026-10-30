<!-- GENERATED from templates/requirements_checklist.md.tpl (config fingerprint {{config_sha}}). -->
# Requirements checklist (IDC409 PDF pages 1-2, Project 1 + instructor clarification) - {{student_name}} ({{roll}})

| Requirement | Status | Where |
|---|---|---|
| NumPy/pandas to store data | Done | `src/data.py`, `src/prepare.py` |
| Database read / write / modify | Done (SQLite: raw table, SQL UPDATE of role/in_primary/target/split/n_rows, `primary_events` view, read back, verified) | `src/database.py`, notebook section 2 |
| EDA: distributions, class balance, correlations, signal/background separation | Done | `artifacts/figures/01-07` |
| Understand and reduce features | Done: feature dictionary; correlation filter + permutation importance | `reports/feature_dictionary.md`, figures 16-17 |
| >= 3 models in one comparison table | Done: four learned classifiers + two baselines | report Tables A-F, `artifacts/metrics/results_tables.md` |
| Metrics: accuracy, precision, recall, F1, ROC-AUC (+ balanced accuracy, macro-F1, per-class, efficiency, rejection, acceptance) | Done at the operating point and at 0.5, with exact / bootstrap intervals | report Tables A-D |
| Confusion matrix and ROC curve (+ continuum-positive ROC/PR) | Done | figures 10, 11 |
| Reduced-feature model; full vs reduced comparison | Done | figure 17, Table E |
| Clean executed notebook | Done (generated and executed by `src.docs`) | `notebooks/continuum_suppression.ipynb` |
| Baseline (simple cut or majority) | Done | models |
| Signal vs continuum labelling | Instructor-provided via the student (type +1 signal; 0/-1 continuum; 2-5 other decays, excluded by analysis decision); not independently verifiable | report section 1 |
| GitHub repository with README and instructions | README complete; publication status is given in the delivery message | `README.md` |
