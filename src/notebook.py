"""Builds and executes notebooks/continuum_suppression.ipynb from SAVED results (markdown filled from src.narrative)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from .paths import ROOT


SETUP_CELL = """# --- Setup: locate the project root reliably (safe to re-run any number of times) ---------------------------------
# Works whether Jupyter was started in the project root, in notebooks/, or anywhere inside the project tree.
# If the notebook is opened outside the project, set the environment variable CONTINUUM_ROOT to the project folder.
import os, sys, json
from pathlib import Path

def find_project_root():
    env = os.environ.get("CONTINUUM_ROOT")
    starts = ([Path(env).expanduser()] if env else []) + [Path.cwd(), *Path.cwd().parents]
    for base in starts:
        for cand in [base] + sorted(p for p in base.glob("*") if p.is_dir())[:50]:     # the folder itself, or a direct child
            if all((cand / m).exists() for m in ("configs/experiment.json", "src/prepare.py", "src/narrative.py", "templates/report.md.tpl")):   # files unique to this project
                return cand.resolve()
    raise RuntimeError("Project root not found (looked for configs/experiment.json, src/prepare.py and templates/ starting from "
                       f"{Path.cwd()}). Start Jupyter inside the project folder or set CONTINUUM_ROOT=/path/to/continuum_suppression.")

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))     # no duplicate entries on re-run
os.chdir(PROJECT_ROOT)                                                           # absolute path -> idempotent

import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
from src import data, database, predict
from src.paths import ART, FIG, MET, MOD, LOG
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 40)
print("project root:", PROJECT_ROOT)"""


def build(F):
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
    md(f"""# Signal vs continuum classification on Belle II features ({F['course']}, Project 1)
**Student:** {F['student_name']} ({F['roll']}) - {F['institution']}

**Question.** Can Belle II event-shape features separate **signal** (`type = +1`) from **continuum background** (`type = 0` or `-1`), and how much do classifier choice and feature reduction matter?

> **Label mapping.** The code meanings come from the instructor's clarification relayed by the student: type +1 = signal; type 0 or -1 = continuum; codes 2-5 = other decays. This cannot be verified from the data files. Codes 2-5 are neither category, so I excluded them: an *analysis decision*, not an instructor instruction. The CSV has {F['n_signal']} signal, {F['n_continuum']} continuum ({F['cont_codes_present']}) and **{F['n_minus1']} type -1 rows** - none were invented. The sample is {F['sig_frac']} signal, so accuracy alone is misleading.

Notebook map: 1 data & audit - 2 SQLite - 3 EDA - 4 splits - 5 models - 6 evaluation - 7 reproducing from the saved bundle - 8 feature reduction - 9 pipeline checks and robustness - 10 conclusions.

**How to run this notebook.** From the project folder: `pip install -r requirements.txt`, then `jupyter lab` (or `jupyter nbconvert --to notebook --execute --inplace notebooks/continuum_suppression.ipynb`). Use a kernel of the same Python environment. The first code cell finds the project root automatically (start Jupyter anywhere inside the project, or set `CONTINUUM_ROOT`) and can be re-run safely. The notebook reads the results saved by `python -m src.run_all` (already included); the original CSV must be at `data/raw/data_hep.csv` only if you regenerate them.
*Generated and executed by `python -m src.docs` from saved results (config fingerprint `{F['config_sha'][:16]}`).*""")
    code(SETUP_CELL)
    code(f"""cfg = data.load_config(); show = lambda n, w=900: display(Image(str(FIG / n), width=w))
tlog = json.load(open(LOG / 'training_log.json'))
# stale-output guard: this notebook must describe results produced under the CURRENT configuration
assert tlog['config_sha256'] == data.config_fingerprint(cfg), 'STALE RESULTS: re-run python -m src.run_all'
print('student:', cfg['student']['name'], cfg['student']['roll'], '| seed', cfg['seed'], '| config fingerprint', data.config_fingerprint(cfg)[:16])
print(json.dumps(cfg['label_mapping'], indent=1)); print(json.dumps(cfg['operating_point'], indent=1))""")
    md("## 1. Data loading, selection and audit\nThe original CSV is never modified. Provenance: supplied as the course's Belle II sample; how it was generated was not independently verified.")
    code("""raw = data.load_raw(cfg); audit = json.load(open(MET / 'data_audit.json')); m = cfg['label_mapping']
print('shape', raw.shape, '| SHA-256', audit['csv_sha256']); print('type counts', audit['type_counts'])
selected = raw.loc[raw['type'].isin([1, 0, -1])].copy(); y = (selected['type'] == 1).astype(int)
print('selected', len(selected), '| signal', int(y.sum()), '| continuum', int((1 - y).sum()), '| type -1 rows in file:', int((raw['type'] == -1).sum()), '| signal fraction', round(float(y.mean()), 4))
print('excluded (analysis decision):', audit['experiment']['excluded_by_code'], '| total', audit['experiment']['n_excluded'])
print('majority (predict-all-signal) accuracy:', round(audit['experiment']['majority_accuracy'], 4))
for k in ['missing_total','inf_total','constant_features','duplicate_rows_excl_index','duplicate_feature_vectors','feature_vectors_with_conflicting_type','common_sentinel_hits']: print(f'{k:42s}', audit[k])
print('\\nunnamed index-like column:'); print(json.dumps(audit['index_column'], indent=1))""")
    md(f"""**Findings.** {F['n_rows_total']} rows, {F['n_feat']} numeric features. No missing/infinite values, constant columns, duplicate feature vectors or conflicting duplicates. The unnamed first column repeats {F['idx_repeats']} values, is unique only within each type code and its range depends on the code; that pattern resembles a running index from separate files, but its origin is undocumented, so I do not assert it. It is not a reliable event identifier and could encode the label, so it is **excluded from all predictors** (with `type` and every identifier). Exact zeros occur only in CLEO-cone variables and are kept.""")
    md("## 2. SQLite database: write, read, modify, verify")
    code("""con = database.connect(cfg)
print(pd.read_sql("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')", con))
print(pd.read_sql("SELECT * FROM type_mapping", con))
print(pd.read_sql("SELECT split, role, COUNT(*) n FROM events GROUP BY split, role ORDER BY split, role", con))
print('primary_events view rows:', con.execute("SELECT COUNT(*) FROM primary_events").fetchone()[0], '| distinct codes:', [r[0] for r in con.execute("SELECT DISTINCT type FROM primary_events ORDER BY type")])
chk = database.verify(con, raw, cfg); print({k: v for k, v in chk.items() if isinstance(v, bool)})
con.execute("CREATE TEMP TABLE scratch AS SELECT row_id, type, target FROM events WHERE in_primary = 1")     # MODIFY a scratch copy, project DB untouched
con.execute("UPDATE scratch SET target = 1 - target WHERE type = 0")
print('scratch UPDATE changed', con.execute("SELECT COUNT(*) FROM scratch s JOIN events e USING(row_id) WHERE s.target <> e.target").fetchone()[0], 'rows (= number of type-0 rows:', int((raw.type==0).sum()), ')')
con.close()""")
    md("`events_raw` keeps every original value (verified identical to the CSV). `type_mapping`, `events.role`, `in_primary`, `target`, `split` and `n_rows` were **modified with SQL `UPDATE`**, read back and checked against pandas; the `primary_events` view is exactly the modelling subset.")
    md("## 3. Exploratory data analysis (training rows for decisions)")
    code("show('01_label_balance.png', 950)")
    md(f"**Shows:** all six original codes and the primary experiment ({F['n_signal']} signal vs {F['n_continuum']} continuum, log scale). **Cannot conclude:** the class sizes say nothing about physics; they show why accuracy is not informative (predict-all-signal scores {F['maj_acc']}).")
    code("show('02_univariate_separation.png', 700)")
    md(f"**Shows:** single-feature separation on training rows: the best feature (`{F['raw_top_feat']}`) reaches an AUC of only {F['raw_top_sep']}. **Cannot conclude:** AUC ignores correlation between features and says nothing about causal importance.")
    code("show('03_event_shape_distributions.png', 1100)")
    md("**Shows:** the signal and continuum distributions overlap almost completely. **Note:** continuum is normally expected to be more jet-like (higher R2/thrust, cosTBTO near 1) than B-pair signal; the type-0 events show the signal-like pattern. This is flagged for the instructor, not 'fixed' by changing the mapping.")
    code("show('04_distributions_by_type_code.png', 1000)")
    md("**Shows:** each original code (grey = other decays 2-5, excluded; shown for context only).")
    code("show('05_correlation_heatmap.png', 800); print(pd.read_csv(MET/'highly_correlated_pairs_train.csv').head(10))")
    md(f"**Shows:** strong redundancy: KSFW moments from B primary daughters and from final-state particles correlate up to 1.0; {F['n_high_pairs']} feature pairs have |r| >= 0.9. This motivates the correlation filter. Correlation does not mean the variables are physically identical.")
    code("show('06_cleo_cone_profiles.png', 900); show('07_data_quality.png', 1000)")
    md("## 4. Train / validation / test split")
    code("""sp = pd.read_csv(MET/'split_assignments.csv')
print(pd.crosstab([sp.in_primary, sp.role], sp.split, margins=True)); g = sp[sp.in_primary == 1]
print(pd.crosstab(g.split, g.role, margins=True)); print('class proportion by split:', g.groupby('split').target.mean().round(4).to_dict())
print('primary rows:', len(g), '| groups spanning >1 split:', int((g.groupby('group').split.nunique() > 1).sum()), '| excluded rows in train/val/test:', int(((sp.in_primary == 0) & sp.split.isin(['train','val','test'])).sum()))""")
    md(f"""Stratified 70/15/15 (seed 42) on the signal/continuum class, so every split keeps the class proportions: train {F['train_sig']} / {F['train_con']}, validation {F['val_sig']} / {F['val_con']}, test {F['test_sig']} / {F['test_con']} (signal / continuum). No physical event identifier exists; rows are grouped by a rounded feature-vector hash so duplicates could never cross splits (none exist). **Limitation:** distinct rows from one physical event cannot be detected. Scalers and class weights are fitted on training rows only; validation and test data are never resampled or weighted; the test split is scored once.""")
    md("## 5. Models and tuning")
    code("""rows=[]
for k, v in tlog['models'].items():
    if k.endswith('|full'): rows.append({'model': k.split('|')[0], 'features used': v['n_features_used'], 'val AUC': round(v['best_val_auc'],4), 'best params': v['best_params'], 'configs tried': len(v['search']), 'tuning s': v['tuning_seconds']})
display(pd.DataFrame(rows)); print('Best by VALIDATION AUC:', tlog['best_by_validation_auc']); print('single-feature cut uses:', tlog['models']['single_cut|full']['single_cut_feature'], '(best of', tlog['models']['single_cut|full']['n_features'], 'candidates)')
display(pd.read_csv(MET/'class_weighting_comparison_validation.csv'))""")
    md("**Four learned classifiers** (logistic regression, decision tree, random forest, gradient boosting) and **two baselines** (majority class; single-feature cut using one feature). Training-only class weighting is part of each learned model's search and is compared on validation ROC-AUC (second table). Selection metric: validation ROC-AUC (threshold-free, never test).")
    md("## 6. Evaluation on the held-out test set")
    code("""R = pd.read_csv(MET/'results_full.csv'); n_sig, n_con = int(R.n_signal_test.iloc[0]), int(R.n_continuum_test.iloc[0]); print('test denominators: signal', n_sig, '| continuum', n_con)
display(Markdown('**Table A - required metrics at the validation-selected operating point**')); display(pd.read_csv(MET/'table_A_required_metrics_operating_point.csv'))
display(Markdown('**Table B - signal efficiency and continuum rejection (exact 95% CI)**')); display(pd.read_csv(MET/'table_B_efficiency_rejection_operating_point.csv'))
display(Markdown('**Table C - fixed threshold 0.5 (ROC-AUC identical: same scores, same rows)**')); display(pd.read_csv(MET/'table_C_threshold_0.5.csv'))
display(Markdown('**Table D - per-class precision / recall / F1**')); display(pd.read_csv(MET/'table_D_per_class_operating_point.csv'))""")
    md(f"The majority baseline predicts every row signal: accuracy {F['maj_acc']} but balanced accuracy {F['maj_bal']} and 0% continuum rejection. The operating point maximises *validation* continuum rejection subject to signal efficiency >= {F['floor']} (an analysis choice); it was frozen before the test set was scored and the test values are what they are. Exact (Clopper-Pearson) intervals use the real denominators; AUC intervals use a stratified bootstrap.")
    code("display(pd.read_csv(MET/'paired_auc_vs_best.csv').round(4)); show('10_roc_pr.png', 1000); show('14_model_comparison.png', 1000)")
    code("show('11_confusion_matrices.png', 1150)")
    md("Counts and row-normalised shading are shown for the fixed threshold 0.5 (top) and the validation-selected operating point (bottom). Rows are the TRUE classes, so each row sums to the true class size.")
    code("show('12_score_distribution.png', 1000); show('13_threshold_analysis.png', 700)")
    code("""b = tlog['best_by_validation_auc']; bm = R.set_index('key').loc[b]
display(Markdown(f"**Selected model: `{b}`** - threshold {bm.threshold:.3f}. Test: signal efficiency {bm.op_signal_efficiency:.1%} [{bm.op_eff_lo:.1%}, {bm.op_eff_hi:.1%}] (n={n_sig}); continuum rejection {bm.op_continuum_rejection:.1%} [{bm.op_rej_lo:.1%}, {bm.op_rej_hi:.1%}] (n={n_con}); continuum acceptance {bm.op_continuum_acceptance:.1%}; ROC-AUC {bm.test_auc:.3f} [{bm.auc_ci_lo:.3f}, {bm.auc_ci_hi:.3f}]. Validation at the same threshold: efficiency {bm.val_op_signal_efficiency:.1%}, rejection {bm.val_op_continuum_rejection:.1%}."))
show('15_other_decays_diagnostic.png', 600); display(pd.read_csv(MET/'other_decays_diagnostic.csv').round(3))""")
    md("The last table is **out of scope context**: the final model applied to the excluded other-decay codes 2-5 (never in training, validation or test). It is not a performance measure.")
    md("## 7. Reproducing predictions from the saved bundle")
    code("""bundle = predict.load_bundle(MOD/'best_model.joblib'); print(bundle['model_name'], len(bundle['features']), 'features | frozen operating threshold', round(bundle['threshold'],4), '| mapping:', bundle['mapping']['status'])
ev = database.read_events(database.connect(cfg)); P = pd.read_csv(ART/'predictions/val_test_scores.csv', float_precision='round_trip'); te = P[P.split=='test']
out = predict.predict_df(ev.set_index('row_id').loc[te.row_id], bundle)
print('max |recomputed - saved score| =', float(np.abs(out.signal_score.to_numpy() - te[b].to_numpy()).max()))
for name, bad in (('missing column', raw.drop(columns=['R2']).head(3)), ('infinite value', raw.head(3).assign(thrust=np.inf)), ('non-numeric value', raw.head(3).astype({'thrust': object}).assign(thrust='abc'))):
    try: predict.predict_df(bad, bundle); print(name, '-> NOT rejected (unexpected)')
    except ValueError as e: print(name, '-> ValueError:', str(e)[:110])""")
    md("## 8. Feature reduction")
    code("""fr = tlog['feature_reduction']; print(fr['n_full'], 'features ->', fr['n_after_corr_filter'], 'after |Spearman| <', fr['corr_threshold'], 'filter ->', fr['k_selected'], 'by permutation importance'); print('criterion:', fr['criterion']); print('reduced set:', fr['reduced_features'])
show('16_feature_importance_reduction.png', 1100)""")
    code("display(pd.read_csv(MET/'full_vs_reduced_table.csv')); show('17_full_vs_reduced.png', 1000)")
    md(f"""**Method:** (1) drop one of each highly correlated pair (|Spearman| >= {F['corr_thr']}, keep the more separating feature, training rows); (2) rank by permutation importance of a class-weighted gradient-boosting model on *validation* data; (3) {F['crit']}. All four learned models were then re-tuned and re-evaluated with the identical protocol.
**Result:** correlation-filtered set ({F['n_corr']} features), AUC change vs full: {F['corr_sentence']}. Reduced set ({F['k_sel']} features): {F['reduced_sentence']}. With so few continuum events the validation AUC used for these choices is noisy, so differences should not be over-read; importance is associational.""")
    md("## 9. Pipeline checks (is the weak result a bug?) and robustness")
    code("""sa = json.load(open(MET / 'sanity_audit.json'))
print('label alignment / positive class:', json.dumps(sa['label_alignment'], indent=1))
print('metrics recomputed with independent sklearn functions:', sa['metrics_recomputed_with_sklearn'])
print('negative control (shuffled training labels), validation AUC:', json.dumps(sa['negative_control']))
print('positive control (learnable task, same code path), validation AUC:', sa['positive_control']['validation_auc'], '|', sa['positive_control']['task'])
print('pooled train+validation 5-fold CV (test set NOT used):'); print(json.dumps(sa['pooled_train_val_5fold_cv'], indent=1))
print('verdict:', sa['verdict']); print(); print(sa['development_disclosure'])""")
    md(f"**Reading the checks.** Shuffled training labels give validation AUC {F['sa_neg_range']} (no information); the same pipeline reaches {F['sa_pos_logreg']} / {F['sa_pos_boost']} on a learnable task, so it can find separation when it exists; metrics recompute exactly with independent library functions; pooled train+validation CV ({F['sa_cv_n_con']} continuum events, no test data) gives out-of-fold AUC {F['sa_cv_logreg']} (logistic) and {F['sa_cv_boost']} (boosting). No implementation problem was found, and **no physical separation is claimed**. Event-level leakage cannot be ruled out because no physical event identifier exists. {F['sa_disclosure']}")
    md("### Split variability")
    code("display(pd.read_csv(MET/'seed_repeats_summary.csv')); display(pd.read_csv(MET/'seed_repeats.csv').pivot_table(index='model', columns='seed', values='continuum_rej').round(3))")
    md(f"Re-splits with further seeds and *frozen* hyper-parameters (operating threshold re-selected on each split's validation set). The spread of continuum rejection across splits shows how much a single {F['test_con']}-event test set can move; the best model's AUC over {F['n_splits']} splits is {F['seed_best_auc']} and its continuum rejection {F['seed_best_rej']}.")
    md("## 10. Conclusions and limitations")
    md(f"""* **Task:** signal (type +1) vs continuum (type 0/-1) with instructor-provided code meanings (relayed by the student; unverifiable from the data). Codes 2-5 excluded by analysis decision. Only type 0 continuum exists ({F['n_continuum']} rows); no type -1 rows.
* {F['separation_sentence']}
* {F['verdict_sentence']}
* {F['paired_sentence']}
* Feature reduction: correlation-filtered set - {F['corr_sentence']}; reduced set - {F['reduced_sentence']}.
{('* ' + F['flag_block']) if F['flag_block'] else ''}
* Pipeline checks found no implementation problem (section 9); "signal-like" means only 'score above the threshold', not a known signal label. {F['sa_disclosure']}
* **Not established:** the mapping itself; absence of shared physical events across splits; any causal reading of importance; performance on real data. With {F['test_con']} test continuum events all estimates are imprecise.""")
    from .notebook_comments import apply as add_plain_language
    C = add_plain_language(C, nbf)
    nb = nbf.v4.new_notebook(); nb["cells"] = C; nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def build_and_execute(F):
    nb = build(F); path = ROOT / "notebooks/continuum_suppression.ipynb"
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    errs = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    assert not errs, "notebook has error cells"
    nbf.write(nb, path); return len(nb.cells)
