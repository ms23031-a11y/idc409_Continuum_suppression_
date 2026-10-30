"""Out-of-scope diagnostic: how the final model scores rows of the EXCLUDED other-decay codes (2-5), which belong to no split and
never influenced training, thresholds or selection. Shown for context only; it is not a performance measure for any labelled task."""
import json
import numpy as np, pandas as pd
from . import data, database, predict
from .paths import ART, FIG, MET
from .plotstyle import *


def main():
    cfg = data.load_config(); m = cfg["label_mapping"]; b = predict.load_bundle(); con = database.connect(cfg); allv = database.read_events(con, primary_only=False); con.close()
    sc = b["model"].predict_proba(allv[b["features"]].to_numpy())[:, 1]; allv["score"] = sc; allv["accepted_as_signal"] = sc >= b["threshold"]
    rows = []
    for t in sorted(allv.type.unique()):
        sub = allv[(allv.type == t) & ((allv.split == "test") if t in m["signal_codes"] + m["continuum_codes"] else (allv.split == "excluded"))]
        rows.append({"type": int(t), "role": "signal" if t in m["signal_codes"] else "continuum" if t in m["continuum_codes"] else "other_decay (excluded, out of scope)", "rows_scored": len(sub), "scored_on": "held-out test rows" if t in m["signal_codes"] + m["continuum_codes"] else "all rows of the excluded code",
                     "fraction_accepted_as_signal": float(sub.accepted_as_signal.mean()), "mean_score": float(sub.score.mean())})
    D = pd.DataFrame(rows); D.to_csv(MET / "other_decays_diagnostic.csv", index=False)
    cols = [POS if r == "signal" else NEG if r == "continuum" else "#9a9a9a" for r in D.role.str.split(" ").str[0]]
    fig, ax = plt.subplots(figsize=(7.8, 4)); ax.bar(D["type"].astype(str), D.fraction_accepted_as_signal, color=cols)
    for i, (v, n) in enumerate(zip(D.fraction_accepted_as_signal, D.rows_scored)): ax.text(i, v + .02, f"{v:.0%}\n(n={n:,})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.2); ax.set_xlabel("original `type` code (blue signal, orange continuum, grey other decays)"); ax.set_ylabel("fraction accepted as signal"); ax.set_title(f"Final model at its operating point ({b['model_name']}); codes 0/1: test rows, codes 2-5: out of scope", fontsize=9.5)
    fig.savefig(FIG / "15_other_decays_diagnostic.png"); plt.close(fig); print(D.round(3).to_string())


if __name__ == "__main__":
    main()
