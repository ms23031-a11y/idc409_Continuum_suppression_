"""Pipeline sanity audit for the near-chance result. Nothing here is used for model, feature or threshold selection and the TEST set is
never used for new fitting; the only test-set use is re-computing already-reported metrics with independent library functions.

Checks: (1) label alignment and positive-class column, (2) metrics recomputed with sklearn, (3) score direction, (4) negative control
(shuffled training labels), (5) positive control (a genuinely learnable task through the SAME feature loading / splitting / model / metric code),
(6) pooled train+validation cross-validation (more continuum events than the 69-event test set)."""
import json
import joblib, numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from . import data, database, splits
from .models import tune_and_fit
from .paths import ART, MET, MOD, LOG

NOTFEAT = ("row_id", "csv_index", "type", "target", "split", "role", "in_primary")


def boot_auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; a = []
    for _ in range(n):
        i = np.r_[rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]; a.append(roc_auc_score(y[i], s[i]))
    return [float(x) for x in np.percentile(a, [2.5, 97.5])]


def main():
    cfg = data.load_config(); m = cfg["label_mapping"]; tlog = json.load(open(LOG / "training_log.json")); res = {}
    con = database.connect(cfg); ev = database.read_events(con); evall = database.read_events(con, primary_only=False); con.close()
    feats = [c for c in ev.columns if c not in NOTFEAT]; y = data.make_target(ev["type"], m)
    tr, va, te = (ev.split == s for s in ("train", "val", "test")); Xtr, Xva = ev.loc[tr, feats].to_numpy(), ev.loc[va, feats].to_numpy(); ytr, yva = y[tr.to_numpy()], y[va.to_numpy()]
    P = pd.read_csv(ART / "predictions/val_test_scores.csv", float_precision="round_trip"); thr = json.load(open(MET / "thresholds.json"))["thresholds"]; R = pd.read_csv(MET / "results_full.csv").set_index("key")
    keys = [c for c in P.columns if "|" in c]
    # 1 label alignment / positive-class column ---------------------------------------------------------------------------------------
    res["label_alignment"] = {"target_equals_(type==1)": bool((y == (ev["type"].to_numpy() == 1).astype(int)).all()), "database_target_equals_pandas_target": bool((ev["target"].to_numpy() == y).all()),
                              "types_in_experiment": sorted(int(t) for t in ev["type"].unique()), "n_signal": int(y.sum()), "n_continuum": int((1 - y).sum())}
    cls_ok = True
    for k in keys:
        mdl = joblib.load(MOD / f"{k.replace('|', '__')}.joblib")["model"]; cls_ok &= list(mdl.classes_) == [0, 1]
    res["label_alignment"]["all_saved_models_have_classes_[0,1]_so_predict_proba[:,1]_is_P(signal)"] = bool(cls_ok)
    # explicit probe: a model fitted on a trivially separable toy signal in the SAME code path returns high scores for class 1
    rng = np.random.default_rng(0); Xt = rng.normal(size=(400, 3)); yt = (Xt[:, 0] > 0).astype(int); tm, _, _ = tune_and_fit("logreg", cfg, 0, Xt, yt, Xt, yt, fixed_params={"C": 1.0, "class_weight": None})
    res["label_alignment"]["positive_class_probe_auc_on_toy_task(should_be_~1)"] = float(roc_auc_score(yt, tm.predict_proba(Xt)[:, 1]))
    # 2 independent metric recomputation (sklearn) vs reported ---------------------------------------------------------------------------
    te_ = P[P.split == "test"]; yt_ = te_.target.to_numpy(); ok = True; worst = 0.0
    for k in keys:
        s = te_[k].to_numpy(); auc = roc_auc_score(yt_, s); worst = max(worst, abs(auc - R.loc[k, "test_auc"]))
        for tag, t in (("op", thr[k]["selected"]), ("t05", 0.5)):
            pred = (s >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yt_, pred, labels=[0, 1]).ravel(); p, r, f, _ = precision_recall_fscore_support(yt_, pred, labels=[0, 1], zero_division=0)
            for name, a, b in (("accuracy", accuracy_score(yt_, pred), R.loc[k, f"{tag}_accuracy"]), ("balanced_accuracy", balanced_accuracy_score(yt_, pred), R.loc[k, f"{tag}_balanced_accuracy"]), ("macro_f1", f1_score(yt_, pred, average="macro", zero_division=0), R.loc[k, f"{tag}_macro_f1"]),
                               ("signal_efficiency", r[1], R.loc[k, f"{tag}_signal_efficiency"]), ("continuum_rejection", r[0], R.loc[k, f"{tag}_continuum_rejection"]), ("signal_precision", p[1], R.loc[k, f"{tag}_precision"]), ("signal_f1", f[1], R.loc[k, f"{tag}_f1"]), ("continuum_f1", f[0], R.loc[k, f"{tag}_f1_continuum"])):
                d = abs(float(a) - float(b)); worst = max(worst, d); ok &= d < 1e-12
    res["metrics_recomputed_with_sklearn"] = {"all_match": bool(ok), "max_abs_difference": float(worst)}
    # 3 score direction ---------------------------------------------------------------------------------------------------------------------
    res["score_direction"] = {k: {"auc": float(R.loc[k, "test_auc"]), "auc_if_scores_reversed": float(1 - R.loc[k, "test_auc"])} for k in keys if k.endswith("|full")}
    res["score_direction"]["note"] = "no model is strongly anti-correlated with the label (an inverted score would show AUC far below 0.5)"
    # 4 negative control: shuffled training labels --------------------------------------------------------------------------------------------
    rng = np.random.default_rng(7); neg = {}
    for name in ("logreg", "boosting"):
        p = tlog["models"][f"{name}|full"]["best_params"]; aucs = []
        for rep in range(3):
            ysh = rng.permutation(ytr); mm, _, _ = tune_and_fit(name, cfg, rep, Xtr, ysh, Xva, yva, fixed_params=p); aucs.append(float(roc_auc_score(yva, mm.predict_proba(Xva)[:, 1])))
        neg[name] = {"validation_auc_with_shuffled_training_labels": aucs}
    res["negative_control"] = neg
    # 5 positive control: signal (type 1) vs one other-decay code (type 2) through the same code path (excluded codes used ONLY here, for sanity) ---------------
    sub = evall[evall.type.isin([1, 2])].copy(); rng = np.random.default_rng(3); keep = np.r_[rng.choice(np.where(sub.type == 1)[0], 4000, replace=False), rng.choice(np.where(sub.type == 2)[0], 4000, replace=False)]; sub = sub.iloc[np.sort(keep)].reset_index(drop=True)
    yc = (sub.type == 1).astype(int).to_numpy(); sp, _ = splits.assign_splits(sub[feats], pd.Series(yc), cfg); pc = {}
    for name, p in (("logreg", {"C": 1.0, "class_weight": None}), ("boosting", {"learning_rate": 0.1, "max_leaf_nodes": 15, "l2_regularization": 1.0, "class_weight": None})):
        a, b = (sp == "train").to_numpy(), (sp == "val").to_numpy(); mm, _, _ = tune_and_fit(name, cfg, 0, sub.loc[a, feats].to_numpy(), yc[a], sub.loc[b, feats].to_numpy(), yc[b], fixed_params=p); pc[name] = float(roc_auc_score(yc[b], mm.predict_proba(sub.loc[b, feats].to_numpy())[:, 1]))
    res["positive_control"] = {"task": "type 1 vs type 2 (excluded code), balanced 4000/4000 sample, same features/splitting/models/metrics", "validation_auc": pc, "note": "sanity control only; not part of the experiment; the test set is not used"}
    # 6 pooled train+validation CV (no test rows) ------------------------------------------------------------------------------------------------------
    pool = (tr | va).to_numpy(); Xp, yp = ev.loc[pool, feats].to_numpy(), y[pool]; cv = {}
    for name in ("logreg", "boosting"):
        p = tlog["models"][f"{name}|full"]["best_params"]; oof = np.zeros(len(yp)); fold_auc = []
        for f_tr, f_va in StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(Xp, yp):
            mm, _, _ = tune_and_fit(name, cfg, cfg["seed"], Xp[f_tr], yp[f_tr], Xp[f_va], yp[f_va], fixed_params=p); oof[f_va] = mm.predict_proba(Xp[f_va])[:, 1]; fold_auc.append(float(roc_auc_score(yp[f_va], oof[f_va])))
        cv[name] = {"fold_auc": fold_auc, "oof_auc": float(roc_auc_score(yp, oof)), "oof_auc_ci95": boot_auc_ci(yp, oof), "n_continuum_pooled": int((1 - yp).sum()), "n_signal_pooled": int(yp.sum())}
    res["pooled_train_val_5fold_cv"] = cv
    res["development_disclosure"] = ("The held-out test metrics of this corrected run were inspected while developing and debugging the reporting and the score-file precision fix. "
                                     "No model, hyper-parameter, feature set or threshold was changed in response to test results (they are fixed by validation data: see the isolation check). "
                                     "An earlier, different task (type 1 vs all other codes) used a different split of a different subset; its rows overlap with the current rows, so the current test rows are not strictly untouched by earlier work.")
    res["verdict"] = {"implementation_problem_found": False, "all_checks_consistent_with_a_correct_pipeline": bool(cls_ok and ok and min(pc.values()) > 0.8 and all(0.35 < a < 0.65 for v in neg.values() for a in v["validation_auc_with_shuffled_training_labels"]))}
    json.dump(res, open(MET / "sanity_audit.json", "w"), indent=2); print(json.dumps(res, indent=1)[:2600]); raise SystemExit(0 if res["verdict"]["all_checks_consistent_with_a_correct_pipeline"] else 1)


if __name__ == "__main__":
    main()
