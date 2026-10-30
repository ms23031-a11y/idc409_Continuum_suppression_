"""Test-set isolation check: scramble the TEST rows (labels and feature values), re-run feature reduction, hyper-parameter
tuning (logistic regression, tree, boosting) and threshold selection from scratch, and confirm every choice is identical
to the saved run. If any step read test data, the choices would change."""
import json, tempfile
from pathlib import Path
import numpy as np
from . import data, database, train
from .paths import LOG

MODELS = ["single_cut", "logreg", "tree", "boosting"]


def main():
    cfg = data.load_config(); mapping = cfg["label_mapping"]
    con = database.connect(cfg); ev = database.read_events(con); con.close()
    feats = [c for c in ev.columns if c not in ("row_id", "csv_index", "type", "target", "split", "role", "in_primary")]
    rng = np.random.default_rng(123); te = ev.split == "test"
    ev2 = ev.copy()
    ev2.loc[te, "type"] = rng.permutation(ev.loc[te, "type"].to_numpy())      # test labels shuffled among test rows
    ev2["target"] = data.make_target(ev2["type"], mapping)
    ev2.loc[te, feats] = rng.normal(size=(int(te.sum()), len(feats))) * 1e3          # garbage features on test rows
    assert (ev2.loc[~te, feats].to_numpy() == ev.loc[~te, feats].to_numpy()).all()
    with tempfile.TemporaryDirectory() as d:
        log2 = train.run(cfg, None, Path(d), lite=False, ev=ev2, models=MODELS, quiet=True)
        import pandas as pd
        P0 = pd.read_csv(Path(LOG).parent / "predictions/val_test_scores.csv", float_precision="round_trip"); P2 = pd.read_csv(Path(d) / "predictions/val_test_scores.csv", float_precision="round_trip")
        thr2 = json.load(open(Path(d) / "metrics/thresholds.json"))["thresholds"]
    log0 = json.load(open(LOG / "training_log.json")); thr0 = json.load(open(LOG.parent / "metrics/thresholds.json"))["thresholds"]
    res = {"models": MODELS, "reduced_features_identical": log0["feature_reduction"]["reduced_features"] == log2["feature_reduction"]["reduced_features"],
           "corr_features_identical": log0["feature_sets"]["corr_filtered"] == log2["feature_sets"]["corr_filtered"]}
    same_h, same_t, same_v = True, True, True
    for k in log2["models"]:
        same_h &= log0["models"][k]["best_params"] == log2["models"][k]["best_params"]
        same_t &= abs(thr0[k]["selected"] - thr2[k]["selected"]) < 1e-12
        a = P0[P0.split == "val"].set_index("row_id")[k]; b = P2[P2.split == "val"].set_index("row_id")[k]
        same_v &= bool(np.allclose(a.loc[b.index], b, atol=1e-12))
    res.update(hyperparameters_identical=bool(same_h), thresholds_identical=bool(same_t), validation_scores_identical=bool(same_v))
    res["all_identical"] = bool(all(v for k, v in res.items() if isinstance(v, bool)))
    json.dump(res, open(LOG / "isolation_check.json", "w"), indent=2); print(res)
    raise SystemExit(0 if res["all_identical"] else 1)


if __name__ == "__main__":
    main()
