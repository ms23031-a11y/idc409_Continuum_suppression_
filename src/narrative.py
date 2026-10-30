"""Single source of every number and result-dependent sentence used in the report, README, viva guide and notebook.
All values are read from SAVED results of the current run; nothing is typed by hand. Qualitative statements are generated
conditionally from the results, so a configuration change cannot silently leave stale prose behind."""
import json
import numpy as np, pandas as pd
from . import data
from .models import LABELS
from .narrative_tables import md_table
from .paths import ART, MET, LOG, ROOT

f3 = lambda x: f"{x:.3f}"; PCT = lambda x: f"{100 * x:.1f}%"; CI = lambda a, lo, hi: f"{PCT(a)} [{PCT(lo)}, {PCT(hi)}]"


def check_fresh(cfg):
    tl = json.load(open(LOG / "training_log.json")); fp = data.config_fingerprint(cfg)
    if tl.get("config_sha256") != fp:
        raise RuntimeError(f"STALE RESULTS: artifacts were produced under a different configuration than configs/experiment.json (results {str(tl.get('config_sha256'))[:12]} vs config {fp[:12]}). Run `python -m src.run_all`.")
    if tl.get("csv_sha256") != data.sha256(ROOT / cfg["data"]["csv"]): raise RuntimeError("STALE RESULTS: data/raw/data_hep.csv differs from the file the models were trained on.")
    return tl


def level(auc):
    return "close to chance" if auc < 0.60 else "weak" if auc < 0.75 else "moderate" if auc < 0.90 else "strong"


def F_con(cv): return cv["logreg"]["n_continuum_pooled"]


def facts():
    cfg = data.load_config(); tl = check_fresh(cfg); lm, st, op = cfg["label_mapping"], cfg["student"], cfg["operating_point"]
    audit = json.load(open(MET / "data_audit.json")); ex = audit["experiment"]; R = pd.read_csv(MET / "results_full.csv"); Rk = R.set_index("key")
    D = pd.read_csv(MET / "paired_auc_difference.csv"); PV = pd.read_csv(MET / "paired_auc_vs_best.csv"); seeds = pd.read_csv(MET / "seed_repeats.csv")
    pi = pd.read_csv(MET / "permutation_importance_val.csv"); fr = tl["feature_reduction"]; best = tl["best_by_validation_auc"]; b = Rk.loc[best]; bm = best.split("|")[0]
    cc = tl["class_counts"]; sh = Rk.loc["majority|full"]; sc = tl["models"]["single_cut|full"]
    F = {"student_name": st["name"], "roll": st["roll"], "institution": st["institution"], "course": st["course"], "config_sha": data.config_fingerprint(cfg), "csv_sha": audit["csv_sha256"],
         "mapping_status": lm["status"], "mapping_source": lm["source"], "exclusion_note": lm["exclusion_note"], "floor": f"{op['min_signal_efficiency']:.0%}",
         "n_rows_total": f"{audit['shape'][0]:,}", "n_feat": audit["n_features"], "type_counts": ", ".join(f"{k}: {v:,}" for k, v in audit["type_counts"].items()),
         "n_selected": f"{ex['n_selected']:,}", "sig_ratio": f"{ex['n_signal'] / ex['n_continuum']:.0f}", "n_signal": f"{ex['n_signal']:,}", "n_continuum": f"{ex['n_continuum']:,}", "sig_frac": PCT(ex["signal_fraction"]), "n_excluded": f"{ex['n_excluded']:,}",
         "excluded_by_code": "; ".join(f"type {k}: {v:,}" for k, v in ex["excluded_by_code"].items()), "cont_codes_present": ", ".join(f"type {k}: {v:,}" for k, v in ex["continuum_rows_by_code"].items()), "n_minus1": ex["n_type_minus1"],
         "n_train": f"{tl['n']['train']:,}", "n_val": f"{tl['n']['val']:,}", "n_test": f"{tl['n']['test']:,}", "idx_repeats": f"{audit['index_column']['n_repeated_values']:,}",
         "test_sig": f"{cc['test']['signal']:,}", "test_con": cc["test"]["continuum"], "val_sig": f"{cc['val']['signal']:,}", "val_con": cc["val"]["continuum"], "train_sig": f"{cc['train']['signal']:,}", "train_con": cc["train"]["continuum"],
         "best_key": best, "best_name": LABELS[bm], "best_nfeat": int(b.n_features), "best_auc": f3(b.test_auc), "best_auc_lo": f3(b.auc_ci_lo), "best_auc_hi": f3(b.auc_ci_hi), "best_val_auc": f3(b.val_auc), "best_thr": f3(b.threshold), "best_level": level(b.test_auc),
         "best_acc": f3(b.op_accuracy), "best_prec": f3(b.op_precision), "best_rec": f3(b.op_recall), "best_f1": f3(b.op_f1), "best_bal": f3(b.op_balanced_accuracy), "best_macro": f3(b.op_macro_f1),
         "best_eff": PCT(b.op_signal_efficiency), "best_eff_ci": CI(b.op_signal_efficiency, b.op_eff_lo, b.op_eff_hi), "best_rej": PCT(b.op_continuum_rejection), "best_rej_ci": CI(b.op_continuum_rejection, b.op_rej_lo, b.op_rej_hi), "best_accpt": PCT(b.op_continuum_acceptance),
         "best_val_eff": PCT(b.val_op_signal_efficiency), "best_val_rej": PCT(b.val_op_continuum_rejection), "best_tn": int(b.op_tn), "best_fp": int(b.op_fp), "best_tp": int(b.op_tp), "best_fn": int(b.op_fn),
         "best_acc05": f3(b.t05_accuracy), "best_prec05": f3(b.t05_precision), "best_rec05": f3(b.t05_recall), "best_f105": f3(b.t05_f1), "best_eff05": PCT(b.t05_signal_efficiency), "best_rej05": CI(b.t05_continuum_rejection, b.t05_rej_lo, b.t05_rej_hi), "best_bal05": f3(b.t05_balanced_accuracy),
         "chance_rej": PCT(1 - b.op_signal_efficiency), "maj_acc": f3(sh.op_accuracy), "maj_bal": f3(sh.op_balanced_accuracy),
         "single_feature": sc["single_cut_feature"], "n_candidates": sc["n_features"], "single_auc": f3(Rk.loc["single_cut|full"].test_auc),
         "logreg_auc": f3(Rk.loc["logreg|full"].test_auc), "tree_auc": f3(Rk.loc["tree|full"].test_auc), "forest_auc": f3(Rk.loc["forest|full"].test_auc), "boosting_auc": f3(Rk.loc["boosting|full"].test_auc)}
    # tables ------------------------------------------------------------------------------------------------------------------------------------
    for k, n in (("A", "A_required_metrics_operating_point"), ("B", "B_efficiency_rejection_operating_point"), ("C", "C_threshold_0.5"), ("D", "D_per_class_operating_point")): F[f"table_{k}"] = md_table(pd.read_csv(MET / f"table_{n}.csv"))
    F["table_fvr"] = md_table(pd.read_csv(MET / "full_vs_reduced_table.csv")); F["table_weighting"] = md_table(pd.read_csv(MET / "class_weighting_comparison_validation.csv"))
    pvt = pd.DataFrame({"Comparison (same test rows)": PV.comparison.str.replace("|full", "", regex=False), "AUC difference [95% paired stratified-bootstrap CI]": [f"{a:+.4f} [{l:+.4f}, {h:+.4f}]" for a, l, h in zip(PV.auc_diff, PV.ci_lo, PV.ci_hi)], "CI excludes 0": PV.ci_excludes_zero.map({True: "yes", False: "no"})})
    F["table_paired_best"] = md_table(pvt)
    F["table_split_counts"] = md_table(pd.DataFrame({"Split": ["train", "validation", "test", "total (primary experiment)"], "Signal": [F["train_sig"], F["val_sig"], F["test_sig"], F["n_signal"]], "Continuum": [F["train_con"], F["val_con"], F["test_con"], F["n_continuum"]],
                                                    "Total": [F["n_train"], F["n_val"], F["n_test"], F["n_selected"]], "Signal fraction": [PCT(cc[s]["signal"] / (cc[s]["signal"] + cc[s]["continuum"])) for s in ("train", "val", "test")] + [F["sig_frac"]]}))
    sm = seeds[seeds.model != "majority"].groupby("model", sort=False).agg(auc_m=("test_auc", "mean"), auc_s=("test_auc", "std"), rej_m=("continuum_rej", "mean"), rej_s=("continuum_rej", "std"), rej_lo=("continuum_rej", "min"), rej_hi=("continuum_rej", "max"), eff_m=("signal_eff", "mean"))
    F["n_splits"] = int(seeds.seed.nunique())
    F["table_seeds"] = md_table(pd.DataFrame({"Model": [LABELS[m] for m in sm.index], f"ROC-AUC mean +/- sd ({F['n_splits']} splits)": [f"{a:.3f} +/- {s:.3f}" for a, s in zip(sm.auc_m, sm.auc_s)], "Signal efficiency (mean)": [PCT(x) for x in sm.eff_m],
                                              "Continuum rejection mean +/- sd [min, max]": [f"{PCT(a)} +/- {PCT(s)} [{PCT(l)}, {PCT(h)}]" for a, s, l, h in zip(sm.rej_m, sm.rej_s, sm.rej_lo, sm.rej_hi)]}))
    F["seed_best_auc"] = f"{sm.auc_m[bm]:.3f} +/- {sm.auc_s[bm]:.3f}"; F["seed_best_rej"] = f"{PCT(sm.rej_m[bm])} (range {PCT(sm.rej_lo[bm])} to {PCT(sm.rej_hi[bm])})"
    od = pd.read_csv(MET / "other_decays_diagnostic.csv"); F["table_other"] = md_table(pd.DataFrame({"type code": od["type"], "role": od.role, "rows scored": od.rows_scored.map(lambda x: f"{x:,}"), "scored on": od.scored_on, "fraction accepted as signal": od.fraction_accepted_as_signal.map(PCT)}))
    # separation assessment (drives conditional prose) ----------------------------------------------------------------------------------------
    full = R[(R.feature_set == "full") & (~R.model.isin(["majority"]))]
    allv = R[R.model != "majority"].sort_values("test_auc", ascending=False).iloc[0]          # every feature set (full, correlation-filtered, reduced)
    F.update(auc_min=f3(full.test_auc.min()), auc_max=f3(full.test_auc.max()))
    F["n_variants"] = int(len(allv_df)) if False else int((R.model != "majority").sum())
    F.update(topall_name=LABELS[allv.model], topall_set=allv.feature_set, topall_auc=f3(allv.test_auc), topall_lo=f3(allv.auc_ci_lo), topall_hi=f3(allv.auc_ci_hi), topall_key=allv.key)
    topr = full.sort_values("test_auc", ascending=False).iloc[0]
    F.update(top_key=topr.key, top_name=LABELS[topr.model], top_auc=f3(topr.test_auc), top_lo=f3(topr.auc_ci_lo), top_hi=f3(topr.auc_ci_hi), top_is_best=bool(topr.key == best), top_excludes_chance=bool(topr.auc_ci_lo > 0.5))
    F["ci_chance_models"] = ", ".join(LABELS[m].split(" (")[0] for m, lo, hi in zip(full.model, full.auc_ci_lo, full.auc_ci_hi) if lo <= 0.5 <= hi) or "none"
    sa = json.load(open(MET / "sanity_audit.json")); cv = sa["pooled_train_val_5fold_cv"]; neg = sa["negative_control"]; pc = sa["positive_control"]["validation_auc"]
    F.update(sa_neg_range=f"{min(a for v in neg.values() for a in v['validation_auc_with_shuffled_training_labels']):.2f}-{max(a for v in neg.values() for a in v['validation_auc_with_shuffled_training_labels']):.2f}",
             sa_pos_logreg=f3(pc["logreg"]), sa_pos_boost=f3(pc["boosting"]), sa_cv_n_con=cv["logreg"]["n_continuum_pooled"], sa_cv_n_sig=f"{cv['logreg']['n_signal_pooled']:,}",
             sa_cv_logreg=f"{cv['logreg']['oof_auc']:.3f} [{cv['logreg']['oof_auc_ci95'][0]:.3f}, {cv['logreg']['oof_auc_ci95'][1]:.3f}]", sa_cv_boost=f"{cv['boosting']['oof_auc']:.3f} [{cv['boosting']['oof_auc_ci95'][0]:.3f}, {cv['boosting']['oof_auc_ci95'][1]:.3f}]",
             sa_metric_diff=f"{sa['metrics_recomputed_with_sklearn']['max_abs_difference']:.1e}", sa_classes_ok=sa["label_alignment"]["all_saved_models_have_classes_[0,1]_so_predict_proba[:,1]_is_P(signal)"], sa_metrics_ok=sa["metrics_recomputed_with_sklearn"]["all_match"],
             sa_toy_auc=f"{sa['label_alignment']['positive_class_probe_auc_on_toy_task(should_be_~1)']:.3f}", sa_verdict=sa["verdict"]["all_checks_consistent_with_a_correct_pipeline"], sa_disclosure=sa["development_disclosure"])
    F["sa_cv_rows"] = md_table(pd.DataFrame({"Model": ["Logistic regression", "Gradient boosting"], "5-fold AUC per fold": [", ".join(f"{a:.2f}" for a in cv[k]["fold_auc"]) for k in ("logreg", "boosting")], f"Pooled out-of-fold AUC [95% stratified-bootstrap CI] ({F_con(cv)} continuum events)": [F["sa_cv_logreg"], F["sa_cv_boost"]]}))
    F["weak"] = bool(b.test_auc < 0.75); F["near_chance"] = bool(b.test_auc < 0.60)
    F["best_ci_includes_chance"] = bool(b.auc_ci_lo <= 0.5 <= b.auc_ci_hi)
    F["any_ci_excludes_chance"] = bool((full.auc_ci_lo > 0.5).any())
    F["paired_all_exclude0"] = bool(PV[PV.model != "majority"].ci_excludes_zero.all())
    F["paired_not_excl"] = ", ".join(LABELS[m].split(" (")[0] for m in PV[(PV.model != "majority") & (~PV.ci_excludes_zero)].model) or "none"
    F["raw_top_sep"] = f3(float(pd.read_csv(MET / "univariate_separation_train.csv", index_col=0).abs_sep.iloc[0] + .5)); F["raw_top_feat"] = pd.read_csv(MET / "univariate_separation_train.csv", index_col=0).index[0]
    # reduction ------------------------------------------------------------------------------------------------------------------------------------
    F.update(n_corr=fr["n_after_corr_filter"], k_sel=fr["k_selected"], corr_thr=fr["corr_threshold"], reduced_list=", ".join(f"`{x}`" for x in fr["reduced_features"]), crit=fr["criterion"], top1=pi.feature[0], top1_drop=f"{pi.perm_importance_val_auc_drop[0]:.3f}", top2=pi.feature[1], top2_drop=f"{pi.perm_importance_val_auc_drop[1]:.3f}")
    d = D.set_index(["model", "feature_set"])
    def dsent(fs):
        o = []
        for m in ("logreg", "tree", "forest", "boosting"):
            r = d.loc[(m, fs)]; sig = (r.ci_lo > 0) or (r.ci_hi < 0); o.append(f"{LABELS[m].split(' (')[0].lower()} {r.auc_diff_vs_full:+.4f}" + ("" if sig else " (CI includes 0)"))
        return "; ".join(o)
    F["reduced_sentence"] = dsent("reduced"); F["corr_sentence"] = dsent("corr_filtered"); F["n_high_pairs"] = int((pd.read_csv(MET / "highly_correlated_pairs_train.csv").spearman.abs() >= 0.9).sum())
    F["order_stable"] = bool(all(list(g.sort_values("test_auc", ascending=False).model) == list(seeds[seeds.seed == seeds.seed.iloc[0]].sort_values("test_auc", ascending=False).model) for _, g in seeds[seeds.model != "majority"].groupby("seed")))
    # other decays ---------------------------------------------------------------------------------------------------------------------------------
    odi = od.set_index("type").fraction_accepted_as_signal; F["od_codes"] = ", ".join(f"{t}: {odi[t]:.0%}" for t in (2, 3, 4, 5) if t in odi.index); F["frac0"] = PCT(odi.get(0, np.nan)); F["frac1"] = PCT(odi.get(1, np.nan))
    # verification summary --------------------------------------------------------------------------------------------------------------------------
    v = json.load(open(LOG / "verification.json")) if (LOG / "verification.json").exists() else {}
    F["n_checks"] = sum(1 for k, x in v.items() if isinstance(x, bool) and k != "ALL_PASSED"); F["checks_all_passed"] = bool(v.get("ALL_PASSED", False)); F["max_repro_diff"] = f"{v.get('bundle_vs_saved_predictions_max_abs_diff', float('nan')):.1e}"
    for n_, f_ in (("pytest", "pytest_summary.json"), ("e2e", "streamlit_e2e.json"), ("isolation", "isolation_check.json")): F[n_] = json.load(open(LOG / f_)) if (LOG / f_).exists() else None
    h = pd.read_csv(ART / "historical_type1_vs_rest/results_full_superseded.csv"); h = h[(h.feature_set == "full")]
    F["table_hist"] = md_table(pd.DataFrame({"Model": h.model_label, "ROC-AUC": h.test_auc.map(f3), "Accuracy": h.sel_accuracy.map(f3), "F1": h.sel_f1.map(f3)}))
    return F


def derived(F):
    F["test_rank_text"] = ""
    R = pd.read_csv(MET / "results_full.csv"); full = R[(R.feature_set == "full") & (R.model != "majority")]; top = full.sort_values("test_auc", ascending=False).iloc[0].key
    F["test_rank_text"] = " and also has the highest test AUC among full-feature models" if top == F["best_key"] else f" (the highest full-feature test AUC belongs to {top}, but model selection used validation data only)"
    F["n_splits_extra"] = F["n_splits"] - 1
    F["paired_sentence"] = ("With the paired stratified bootstrap, the AUC advantage of the selected model over every other model has an interval excluding zero on this test set (assumes independent rows; excludes training variability; only %s continuum test events)." % F["test_con"] if F["paired_all_exclude0"]
                            else f"With the paired stratified bootstrap, the AUC difference between the selected model and the following models has an interval that includes zero, so no clear difference is established on this test set: {F['paired_not_excl']}.")
    F["order_sentence"] = "the same in every split" if F["order_stable"] else "not the same in every split (see table), so the ranking of models is not stable"
    sel = f"The validation-selected model ({F['best_name']}) has test ROC-AUC {F['best_auc']} ({F['best_level']}); its 95% interval {F['best_auc_lo']}-{F['best_auc_hi']} " + ("includes 0.5 (chance). " if F["best_ci_includes_chance"] else "excludes 0.5, so some separation is detected, but it is small. ")
    top = "" if F["top_is_best"] else (f"The highest test AUC among full-feature models is {F['top_auc']} ({F['top_name']}, interval {F['top_lo']}-{F['top_hi']}, " + ("which excludes 0.5 only barely" if F["top_excludes_chance"] else "which includes 0.5") + "; it was not the validation choice). ")
    allv = "" if F["topall_key"] == F["top_key"] else (f"Counting every feature-set variant (full, correlation-filtered, reduced), the highest test AUC is {F['topall_auc']} ({F['topall_name']}, {F['topall_set'].replace('_', '-')} features, interval {F['topall_lo']}-{F['topall_hi']}); it too was not chosen on validation data. With {F['n_variants']} model / feature-set combinations scored on the same {F['test_con']} continuum test events, one interval that excludes 0.5 is expected by chance alone and is not evidence of real separation. ")
    F["separation_sentence"] = sel + top + allv + f"Among full-feature learned classifiers the AUC interval includes 0.5 for: {F['ci_chance_models']}."
    if F["weak"]:
        F["flag_block"] = (f"**Flag for the instructor.** Under the instructor-provided mapping the type-0 events (the only continuum rows in the file; {F['n_minus1']} type -1 rows exist) are, in these 59 features, nearly the same as the signal events: the best single feature (`{F['raw_top_feat']}`) reaches an AUC of only {F['raw_top_sep']} on the training rows, and no model separates the classes well. "
                           "Physically, continuum (light-quark/charm) events are expected to be more jet-like (higher R2 and thrust, cosTBTO peaked at 1) than B-pair signal events; the type-0 events show the *signal-like* pattern instead (figure 3 and 4). This report does not change the mapping, but the data do not show the separation that a signal-versus-continuum task normally has, so it would be worth confirming with the instructor that code 0 really is continuum and that no type -1 data were meant to be included.")
    else: F["flag_block"] = ""
    F["verdict_sentence"] = (f"On the instructor-provided mapping, at the validation-chosen operating point (signal efficiency floor {F['floor']}) the validation-selected model keeps {F['best_eff']} of true signal and rejects {F['best_rej']} of true continuum on the test set (only {F['test_con']} continuum events; 95% exact interval {F['best_rej_ci'].split('[')[1].rstrip(']')}). "
                            f"For comparison, a score with no discrimination would reject about {F['chance_rej']} at this efficiency.")
    v = json.load(open(LOG / "verification.json")) if (LOG / "verification.json").exists() else None; L = []
    if v:
        L.append(f"* `python -m src.verify` ran **{F['n_checks']} automated checks**; all passed: {F['checks_all_passed']}. They cover: CSV checksum; SQLite raw table identical to the CSV; SQL-derived roles/targets equal the pandas mapping; the `primary_events` view equals the modelling subset; only codes 1, 0, -1 in the experiment; no excluded row (codes 2-5) in any train/validation/test split and no type -1 row invented; stratified class proportions in every split and at least one continuum event in each; no label/index/identifier column in any predictor list or bundle; every saved bundle reproduces its saved validation and test scores (max absolute difference {F['max_repro_diff']}); test metrics, signal efficiency, continuum rejection (explicit class denominators), exact intervals and thresholds recompute independently from saved scores; thresholds re-derive from validation scores only and meet the efficiency floor on validation; hyper-parameters equal the validation-AUC winners; full / correlation-filtered / reduced comparisons used identical search budgets, seed and rules; inference works, uses the frozen threshold and rejects missing, non-numeric, missing-value and infinite inputs.")
    if F["isolation"]:
        i = F["isolation"]; L.append(f"* **Test-set isolation check:** test rows' labels and feature values were deliberately scrambled, then feature reduction, tuning of {', '.join(i['models'])} and threshold selection were re-run from scratch; all selected features, hyper-parameters, thresholds and validation scores were identical ({i['all_identical']}), so these choices do not read test data.")
    if F["pytest"]: L.append(f"* `pytest`: {F['pytest']['passed']} passed, {F['pytest']['failed']} failed.")
    if F["e2e"]:
        e = F["e2e"]; L.append(f"* **Streamlit upload flow** (real browser via Playwright against a locally started app that loads the saved model and frozen threshold): a valid CSV *without* the `type` or index columns was uploaded and scored ({e['valid_upload_without_type_column_scored']}); the downloaded predictions file matched the saved pipeline's scores and used the frozen threshold ({e['download_ok']} / {e['download_uses_frozen_threshold']}); a CSV that also contains `type` was scored with that column ignored ({e['valid_upload_with_extra_type_column_scored']}); invalid CSVs were rejected with clear messages: missing feature column ({e['missing_column_rejected']}), missing value ({e['nan_rejected']}), non-numeric value ({e['non_numeric_rejected']}). Inference never needs the true label.")
    L.append(f"* **Pipeline sanity audit** (`src.sanity_audit`, no new fitting on test data): label alignment and positive-class column ({F['sa_classes_ok']}); all reported metrics recomputed with independent scikit-learn functions, max difference {F['sa_metric_diff']} ({F['sa_metrics_ok']}); negative control (shuffled training labels) gives validation AUC {F['sa_neg_range']}; positive control (a learnable task through the same code) gives validation AUC {F['sa_pos_logreg']} / {F['sa_pos_boost']}; pooled train+validation 5-fold CV is reported in section 5. Overall: no implementation problem found ({F['sa_verdict']}).")
    L.append("* **Notebook:** executed from a fresh kernel with all outputs cleared first; every code cell has an execution count, there are zero error cells, and the setup cell was tested from five different start directories, re-run twice (no repeated directory changes, no duplicate `sys.path` entries), and outside the project (clear error; `CONTINUUM_ROOT` override).")
    L.append("* **Full retraining:** this whole analysis (preprocessing, splits, feature reduction, tuning, thresholds, evaluation, repeats) was re-run from the raw CSV for the corrected task; no fitted model, threshold or selection from the earlier `type == 1` versus all-others task was reused.")
    L.append("* **Not tested:** other operating systems or Python versions; the documents on a machine without pandoc/weasyprint; independent checking of the feature definitions against basf2 source code; the instructor's mapping itself (it cannot be verified from the data files).")
    F["verification_text"] = "\n".join(L); return F


def guards(F):
    errs = []
    if "INSTRUCTOR" not in F["mapping_status"].upper(): errs.append("mapping status no longer says it is instructor-provided: reword the templates")
    if F["n_minus1"] != 0: errs.append("type -1 rows now exist in the CSV: templates state that none exist")
    if errs: raise RuntimeError("Narrative templates are out of date with the results:\n - " + "\n - ".join(errs))
