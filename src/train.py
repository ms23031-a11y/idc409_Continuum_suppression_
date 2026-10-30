"""Step 3: train all models on TRAIN rows, tune on VALIDATION, freeze threshold, save predictions + bundles.
Test rows are only *scored* here; no choice reads test metrics."""
import argparse, time, json, sys
import joblib, numpy as np, pandas as pd, sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from . import data, database, splits as splits_mod
from .metrics import operating_threshold, op_metrics
from .models import MODEL_ORDER, tune_and_fit
from .paths import ART, ROOT


def corr_filter(Xtr, feats, sep, thr):
    C = pd.DataFrame(Xtr, columns=feats).corr(method="spearman").abs()
    kept = []
    for f in sep.sort_values(ascending=False).index:     # keep the more separating feature of a correlated pair
        if all(C.loc[f, k] < thr for k in kept): kept.append(f)
    return [f for f in feats if f in kept]


def default_hgb(seed):
    # training-only class weighting (the sample is ~98.7% signal); validation/test data are never reweighted
    return HistGradientBoostingClassifier(max_iter=200, early_stopping=False, class_weight="balanced", random_state=seed)


def feature_reduction(cfg, Xtr, ytr, Xva, yva, feats, seed, out):
    """Validation-based, training-fitted reduction: correlation filter -> permutation-importance ranking -> smallest k."""
    sep = pd.Series({f: abs(roc_auc_score(ytr, Xtr[f]) - .5) for f in feats})
    kept = corr_filter(Xtr, feats, sep, cfg["corr_threshold"])
    base_full = roc_auc_score(yva, default_hgb(seed).fit(Xtr[feats], ytr).predict_proba(Xva[feats])[:, 1])
    m = default_hgb(seed).fit(Xtr[kept], ytr)
    auc_kept = roc_auc_score(yva, m.predict_proba(Xva[kept])[:, 1])
    pi = permutation_importance(m, Xva[kept], yva, scoring="roc_auc", n_repeats=cfg["feature_reduction"]["perm_repeats"],
                                random_state=seed, n_jobs=1)
    rank = pd.DataFrame({"feature": kept, "perm_importance_val_auc_drop": pi.importances_mean, "std": pi.importances_std}
                        ).sort_values("perm_importance_val_auc_drop", ascending=False).reset_index(drop=True)
    curve = []
    ks = sorted({k for k in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, len(kept)] if k <= len(kept)})
    for k in ks:
        fs = rank.feature[:k].tolist()
        curve.append({"k": k, "val_auc": float(roc_auc_score(yva, default_hgb(seed).fit(Xtr[fs], ytr).predict_proba(Xva[fs])[:, 1]))})
    need = cfg["feature_reduction"]["min_val_auc_fraction"] * base_full
    k_sel = next((c["k"] for c in curve if c["val_auc"] >= need), len(kept))
    reduced = rank.feature[:k_sel].tolist()
    info = {"n_full": len(feats), "n_after_corr_filter": len(kept), "corr_threshold": cfg["corr_threshold"],
            "dropped_by_corr_filter": [f for f in feats if f not in kept], "val_auc_default_hgb_full": float(base_full),
            "val_auc_default_hgb_corr": float(auc_kept), "k_curve": curve, "criterion": f"smallest k with val AUC >= {cfg['feature_reduction']['min_val_auc_fraction']} x full-feature val AUC",
            "k_selected": k_sel, "reduced_features": reduced}
    rank.to_csv(out / "metrics/permutation_importance_val.csv", index=False)
    return kept, reduced, info


def run(cfg, mapping_name=None, outdir=None, lite=False, ev=None, models=None, quiet=False):
    seed = cfg["seed"]; np.random.seed(seed)
    mapping = cfg["label_mapping"]
    out = ART if outdir is None else outdir
    for d in ("metrics", "models", "predictions", "logs"): (out / d).mkdir(parents=True, exist_ok=True)
    if ev is None:
        con = database.connect(cfg); ev = database.read_events(con); con.close()
    feats = [c for c in ev.columns if c not in ("row_id", "csv_index", "type", "target", "split", "role", "in_primary")]
    assert not ({"type", "csv_index", "target", "split", "role", "in_primary", "row_id"} & set(feats)) and len(feats) == 59
    assert set(ev["type"].unique()) <= set(mapping["signal_codes"] + mapping["continuum_codes"]), "excluded codes reached the primary experiment"
    y_all = data.make_target(ev["type"], mapping)
    assert (y_all == ev["target"].to_numpy()).all()          # SQL-derived target agrees with the config mapping
    tr, va, te = (ev.split == s for s in ("train", "val", "test"))
    Xtr, Xva, Xte = ev.loc[tr, feats], ev.loc[va, feats], ev.loc[te, feats]
    ytr, yva, yte = y_all[tr.to_numpy()], y_all[va.to_numpy()], y_all[te.to_numpy()]
    log = {"config_sha256": data.config_fingerprint(cfg), "csv_sha256": data.sha256(ROOT / cfg["data"]["csv"]), "seed": seed, "mapping": mapping, "sklearn": sklearn.__version__, "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
           "class_counts": {n: {"signal": int((y == 1).sum()), "continuum": int((y == 0).sum())} for n, y in (("train", ytr), ("val", yva), ("test", yte))}}

    fsets = {"full": feats}
    if not lite:
        t0 = time.time(); kept, reduced, info = feature_reduction(cfg, Xtr, ytr, Xva, yva, feats, seed, out)
        info["seconds"] = round(time.time() - t0, 1); log["feature_reduction"] = info
        fsets.update({"corr_filtered": kept, "reduced": reduced})
        print(f"reduction: {len(feats)} -> corr {len(kept)} -> reduced {len(reduced)}", flush=True)

    preds, thr, tlog, bundles = {}, {}, {}, {}
    for fs_name, fs in fsets.items():
        for name in (models or MODEL_ORDER):
            key = f"{name}|{fs_name}"
            if name == "majority" and fs_name != "full": continue
            t0 = time.time()
            model, params, tl = tune_and_fit(name, cfg, seed, Xtr[fs].to_numpy(), ytr, Xva[fs].to_numpy(), yva)
            sv, st = model.predict_proba(Xva[fs].to_numpy())[:, 1], model.predict_proba(Xte[fs].to_numpy())[:, 1]
            t_sel, vm = operating_threshold(yva, sv, cfg["operating_point"]["min_signal_efficiency"])      # VALIDATION data only
            preds[key] = (sv, st); thr[key] = {"selected": t_sel, **vm, "fixed": 0.5}
            tl["n_features"] = len(fs)                       # candidate features offered to the model
            tl["n_features_used"] = 0 if name == "majority" else (1 if name == "single_cut" else len(fs))
            if name == "single_cut": tl["single_cut_feature"] = fs[model.j_]; tl["single_cut_value"] = model.cut_; tl["single_cut_sign"] = model.sign_
            tlog[key] = tl
            bundles[key] = {"model": model, "features": fs, "threshold": t_sel, "model_name": name, "feature_set": fs_name, "threshold_fixed": 0.5, "operating_point": cfg["operating_point"],
                            "mapping": mapping, "seed": seed, "sklearn_version": sklearn.__version__,
                            "csv_sha256": data.sha256(ROOT / cfg["data"]["csv"]), "params": params}
            print(f"{key:28s} val AUC {tl['best_val_auc']:.4f}  thr {t_sel:.3f}  tune {tl['tuning_seconds']}s  ({time.time()-t0:.0f}s)", flush=True)

    # save everything
    rows = []
    for split_name, idx, ysub, col in (("val", va, yva, 0), ("test", te, yte, 1)):
        d = ev.loc[idx, ["row_id", "type"]].assign(split=split_name, target=ysub)
        for k, v in preds.items(): d[k] = v[col]
        rows.append(d)
    pd.concat(rows).to_csv(out / "predictions/val_test_scores.csv", index=False, float_format="%.17g")   # exact float64 round trip (ties at thresholds matter)
    json.dump({"thresholds": thr}, open(out / "metrics/thresholds.json", "w"), indent=2)
    log["models"] = tlog; log["feature_sets"] = {k: v for k, v in fsets.items()}
    json.dump(log, open(out / "logs/training_log.json", "w"), indent=2, default=str)
    for k, b in bundles.items(): joblib.dump(b, out / "models" / f"{k.replace('|', '__')}.joblib", compress=3)
    # best bundle chosen by VALIDATION AUC among tuned (non-baseline) full-feature models
    cand = {k: tlog[k]["best_val_auc"] for k in tlog if k.endswith("|full") and not k.startswith(("majority", "single"))}
    best = max(cand, key=cand.get); joblib.dump(bundles[best], out / "models/best_model.joblib", compress=3)
    log["best_by_validation_auc"] = best
    json.dump(log, open(out / "logs/training_log.json", "w"), indent=2, default=str)
    print("best (validation):", best, cand[best])
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--outdir"); ap.add_argument("--lite", action="store_true")
    a = ap.parse_args(); cfg = data.load_config()
    t = time.time(); run(cfg, None, (ROOT / a.outdir) if a.outdir else None, a.lite); print(f"total {time.time()-t:.0f}s")
