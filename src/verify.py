"""Verification checks on the finished project. Writes artifacts/logs/verification.json; exit code 1 on any failure."""
import json, sys
import joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from . import data, database, predict
from .metrics import clopper_pearson, op_metrics, operating_threshold
from .paths import ART, ROOT, MOD, MET, LOG

LEAK = {"type", "target", "split", "row_id", "csv_index", "role", "in_primary"}


def main():
    cfg = data.load_config(); m = cfg["label_mapping"]; floor = cfg["operating_point"]["min_signal_efficiency"]; raw = data.load_raw(cfg); res = {}
    audit = json.load(open(MET / "data_audit.json")); tlog = json.load(open(LOG / "training_log.json"))
    res["csv_sha256_matches_audit"] = data.sha256(ROOT / cfg["data"]["csv"]) == audit["csv_sha256"]
    res["results_match_current_config"] = tlog.get("config_sha256") == data.config_fingerprint(cfg)
    con = database.connect(cfg); db = database.verify(con, raw, cfg); evall = database.read_events(con, primary_only=False); ev = database.read_events(con); con.close()
    res["database_checks"] = db; res["database_all_ok"] = db["all_ok"]
    # ---- the primary experiment contains exactly the intended codes ----------------------------------
    sp = pd.read_csv(MET / "split_assignments.csv"); pm = data.primary_mask(raw["type"], m)
    res["only_intended_codes_in_experiment"] = bool(set(ev["type"].unique()) <= {1, 0, -1} and set(sp.loc[sp.split != "excluded", "type"].unique()) <= {1, 0, -1})
    res["selection_counts_match_csv"] = int(pm.sum()) == len(ev) == int((raw["type"].isin([1, 0, -1])).sum()) == int((sp.split != "excluded").sum())
    res["no_type_minus1_rows_invented"] = int((raw["type"] == -1).sum()) == int((evall["type"] == -1).sum()) == int((sp["type"] == -1).sum())
    res["no_excluded_row_in_any_split"] = bool(sp.loc[sp.in_primary == 0, "split"].eq("excluded").all() and sp.loc[sp.in_primary == 1, "split"].isin(["train", "val", "test"]).all())
    res["excluded_row_counts_recorded"] = bool(sum(audit["experiment"]["excluded_by_code"].values()) == int((~pm).sum()))
    res["split_csv_matches_db"] = bool((sp.set_index("row_id").loc[evall.row_id, "split"].to_numpy() == evall["split"].to_numpy()).all())
    g = sp[sp.in_primary == 1]; res["no_group_crosses_splits"] = bool((g.groupby("group").split.nunique() == 1).all()); res["row_ids_preserved_and_unique"] = bool(sp.row_id.is_unique and (sp.row_id.to_numpy() == np.arange(len(raw))).all())
    frac = g.groupby("split").target.mean(); res["class_proportion_max_deviation_across_splits"] = float((frac - g.target.mean()).abs().max()); res["splits_stratified_by_class"] = res["class_proportion_max_deviation_across_splits"] < 0.002
    res["every_split_has_continuum"] = bool((g[g.target == 0].groupby("split").size() > 0).all() and g[g.target == 0].split.nunique() == 3)
    feats = [c for c in evall.columns if c not in LEAK]
    res["leakage_columns_not_in_predictors"] = not (LEAK & set(feats)) and len(feats) == 59 and cfg["data"]["index_column"] not in feats
    # ---- bundles, scores, metrics -----------------------------------------------------------------------
    P = pd.read_csv(ART / "predictions/val_test_scores.csv", float_precision="round_trip"); keys = [c for c in P.columns if "|" in c]; thr = json.load(open(MET / "thresholds.json"))["thresholds"]; R = pd.read_csv(MET / "results_full.csv").set_index("key")
    res["predictions_only_val_test_primary_rows"] = bool(set(P.row_id) <= set(ev.row_id) and P.split.isin(["val", "test"]).all() and set(P.row_id).isdisjoint(set(sp[sp.split.isin(["train", "excluded"])].row_id)))
    evi = ev.set_index("row_id"); repro, met_ok, thr_ok, eff_ok, nl = {}, True, True, True, True
    for k in keys:
        b = joblib.load(MOD / f"{k.replace('|', '__')}.joblib"); nl &= not (LEAK & set(b["features"]))
        for sname in ("val", "test"):
            sub = P[P.split == sname]; repro[f"{k}|{sname}"] = float(np.abs(b["model"].predict_proba(evi.loc[sub.row_id, b["features"]].to_numpy())[:, 1] - sub[k].to_numpy()).max())
        te, va = P[P.split == "test"], P[P.split == "val"]; y, s = te.target.to_numpy(), te[k].to_numpy(); t = thr[k]["selected"]
        # independent re-computation with explicit class masks (denominators)
        sig, con = y == 1, y == 0; eff = float((s[sig] >= t).mean()); rej = float((s[con] < t).mean()); bal = (eff + rej) / 2
        acc = float((((s >= t).astype(int)) == y).mean()); lo, hi = clopper_pearson(int((s[con] < t).sum()), int(con.sum()))
        eff_ok &= abs(eff - R.loc[k, "op_signal_efficiency"]) < 1e-12 and abs(rej - R.loc[k, "op_continuum_rejection"]) < 1e-12 and R.loc[k, "n_signal_test"] == sig.sum() and R.loc[k, "n_continuum_test"] == con.sum() and abs(lo - R.loc[k, "op_rej_lo"]) < 1e-9
        met_ok &= abs(roc_auc_score(y, s) - R.loc[k, "test_auc"]) < 1e-12 and abs(acc - R.loc[k, "op_accuracy"]) < 1e-12 and abs(bal - R.loc[k, "op_balanced_accuracy"]) < 1e-12 and abs((1 - rej) - R.loc[k, "op_continuum_acceptance"]) < 1e-12
        t2, _ = operating_threshold(va.target.to_numpy(), va[k].to_numpy(), floor); thr_ok &= abs(t2 - t) < 1e-12
        vm = op_metrics(va.target.to_numpy(), va[k].to_numpy(), t); thr_ok &= vm["signal_efficiency"] >= floor - 1e-9
    res["bundle_vs_saved_predictions_max_abs_diff"] = max(repro.values()); res["saved_models_reproduce_saved_predictions"] = max(repro.values()) < 1e-9
    res["recomputed_test_metrics_match_reported"] = bool(met_ok); res["signal_efficiency_and_continuum_rejection_use_correct_class_denominators"] = bool(eff_ok)
    res["operating_thresholds_reproduced_from_validation_only_and_meet_efficiency_floor"] = bool(thr_ok); res["bundles_have_no_leaky_columns"] = bool(nl)
    cand = {k: tlog["models"][k]["best_val_auc"] for k in tlog["models"] if k.endswith("|full") and not k.startswith(("majority", "single"))}
    res["best_model_chosen_by_validation_auc"] = max(cand, key=cand.get) == tlog["best_by_validation_auc"]
    hp_ok = True
    for k, mm in tlog["models"].items():
        top = max(mm["search"], key=lambda e: e["val_auc"]); hp_ok &= top["params"] == mm["best_params"] and abs(top["val_auc"] - mm["best_val_auc"]) < 1e-12
    res["hyperparameters_are_validation_winners"] = bool(hp_ok)
    proto = True
    for n in ("logreg", "tree", "forest", "boosting"):
        ms = [tlog["models"][f"{n}|{fs}"] for fs in ("full", "corr_filtered", "reduced")]
        proto &= len({len(x["search"]) for x in ms}) == 1 and all([e["params"] for e in ms[0]["search"]] == [e["params"] for e in x["search"]] for x in ms)
        proto &= len({joblib.load(MOD / f"{n}__{fs}.joblib")["seed"] for fs in ("full", "corr_filtered", "reduced")}) == 1
    res["full_vs_reduced_follow_same_protocol"] = bool(proto)
    res["scored_row_counts_match_training_log"] = bool(tlog["n"]["val"] == int((P.split == "val").sum()) and tlog["n"]["test"] == int((P.split == "test").sum()) and tlog["class_counts"]["test"]["continuum"] == int(((P.split == "test") & (P.target == 0)).sum()))
    # ---- inference -------------------------------------------------------------------------------------------
    best = predict.load_bundle(); samp = raw[raw["type"].isin([1, 0])].sample(200, random_state=0); out = predict.predict_df(samp, best)
    res["inference_works"] = bool(len(out) == 200 and out.signal_score.between(0, 1).all() and set(out.predicted_class) <= {"signal", "continuum"})
    ref = P[P.split == "test"].set_index("row_id")[tlog["best_by_validation_auc"]]; te_ids = ev[ev.split == "test"].row_id.to_numpy()[:300]
    o2 = predict.predict_df(raw.loc[te_ids], best); res["inference_matches_saved_test_scores"] = bool(np.allclose(o2.signal_score.to_numpy(), ref.loc[te_ids].to_numpy()))
    res["inference_uses_frozen_operating_threshold"] = bool(abs(best["threshold"] - thr[tlog["best_by_validation_auc"]]["selected"]) < 1e-12 and (o2.predicted_class.eq("signal").to_numpy() == (ref.loc[te_ids].to_numpy() >= best["threshold"])).all())
    try: predict.predict_df(samp.drop(columns=["R2"]), best); res["missing_column_rejected"] = False
    except ValueError as e: res["missing_column_rejected"] = "R2" in str(e)
    for nm, val in (("inf_rejected", np.inf), ("non_numeric_rejected", "abc"), ("nan_rejected", np.nan)):
        b2 = samp.copy().astype({"thrust": object}); b2.iloc[0, b2.columns.get_loc("thrust")] = val
        try: predict.predict_df(b2, best); res[nm] = False
        except ValueError: res[nm] = True
    flags = {k: bool(v) for k, v in res.items() if isinstance(v, (bool, np.bool_))}
    res["ALL_PASSED"] = all(flags.values()); (LOG / "verification.json").write_text(json.dumps(res, indent=2, default=str))
    for k, v in flags.items(): print(("PASS " if v else "FAIL ") + k)
    print("max |bundle - saved score| =", res["bundle_vs_saved_predictions_max_abs_diff"], "| class-proportion max deviation across splits =", round(res["class_proportion_max_deviation_across_splits"], 5))
    sys.exit(0 if res["ALL_PASSED"] else 1)


if __name__ == "__main__":
    main()
