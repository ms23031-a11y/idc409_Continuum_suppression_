"""Split-sensitivity: re-split with other seeds, refit with FROZEN hyper-parameters (full features), re-select the operating threshold on that split's validation set.
With only ~460 continuum events the test-set continuum count is ~70, so split-to-split spread is a necessary companion to the exact intervals."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from . import data, database, splits
from .metrics import op_metrics, operating_threshold
from .models import tune_and_fit, MODEL_ORDER, LABELS
from .paths import ART


def main():
    cfg = data.load_config(); tlog = json.load(open(ART / "logs/training_log.json")); floor = cfg["operating_point"]["min_signal_efficiency"]
    con = database.connect(cfg); ev = database.read_events(con); con.close()
    feats = [c for c in ev.columns if c not in ("row_id", "csv_index", "type", "target", "split", "role", "in_primary")]; y = data.make_target(ev["type"], cfg["label_mapping"]); rows = []
    main = pd.read_csv(ART / "metrics/results_full.csv"); main = main[main.feature_set == "full"]
    for _, r in main.iterrows(): rows.append({"seed": cfg["seed"], "model": r.model, "test_auc": r.test_auc, "signal_eff": r.op_signal_efficiency, "continuum_rej": r.op_continuum_rejection, "balanced_acc": r.op_balanced_accuracy, "n_continuum_test": int(r.n_continuum_test)})
    for sd in cfg["seed_repeats"]:
        sp, _ = splits.assign_splits(ev[feats], pd.Series(y, index=ev.index), cfg, seed=sd); tr, va, te = (sp == s for s in ("train", "val", "test"))
        for name in MODEL_ORDER:
            if name == "majority": continue
            m, _, _ = tune_and_fit(name, cfg, sd, ev.loc[tr, feats].to_numpy(), y[tr], ev.loc[va, feats].to_numpy(), y[va], fixed_params=tlog["models"][f"{name}|full"]["best_params"])
            sv, st = m.predict_proba(ev.loc[va, feats].to_numpy())[:, 1], m.predict_proba(ev.loc[te, feats].to_numpy())[:, 1]
            t, _ = operating_threshold(y[va], sv, floor); mm = op_metrics(y[te], st, t)
            rows.append({"seed": sd, "model": name, "test_auc": roc_auc_score(y[te], st), "signal_eff": mm["signal_efficiency"], "continuum_rej": mm["continuum_rejection"], "balanced_acc": mm["balanced_accuracy"], "n_continuum_test": mm["n_continuum"]})
        print("seed", sd, "done", flush=True)
    R = pd.DataFrame(rows); R.to_csv(ART / "metrics/seed_repeats.csv", index=False)
    S = R.groupby("model", sort=False).agg(auc_mean=("test_auc", "mean"), auc_sd=("test_auc", "std"), eff_mean=("signal_eff", "mean"), rej_mean=("continuum_rej", "mean"), rej_sd=("continuum_rej", "std"), rej_min=("continuum_rej", "min"), rej_max=("continuum_rej", "max"), n_splits=("seed", "nunique")).reindex([m for m in MODEL_ORDER if m != "majority"]).round(4)
    S.to_csv(ART / "metrics/seed_repeats_summary.csv"); print(S)


if __name__ == "__main__":
    main()
