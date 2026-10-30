"""Step 4: metrics, exact / stratified-bootstrap uncertainty, tables and figures - computed from SAVED scores + frozen thresholds.
Class 1 = signal, class 0 = continuum. Operating point = validation-chosen threshold (max continuum rejection s.t. signal efficiency >= floor)."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from . import data
from .metrics import clopper_pearson, op_metrics
from .models import LABELS, MODEL_ORDER
from .narrative_tables import md_table
from .paths import ART
from .plotstyle import *

MC = {"majority": "#8c8c8c", "single_cut": "#999933", "logreg": "#0072B2", "tree": "#CC79A7", "forest": "#009E73", "boosting": "#D55E00"}
f3 = lambda x: f"{x:.3f}"
PCT = lambda x: f"{100 * x:.1f}%"


def evaluate(out=ART, n_boot=1000, seed=42):
    cfg = data.load_config(); floor = cfg["operating_point"]["min_signal_efficiency"]
    P = pd.read_csv(out / "predictions/val_test_scores.csv", float_precision="round_trip"); thr = json.load(open(out / "metrics/thresholds.json"))["thresholds"]; tlog = json.load(open(out / "logs/training_log.json"))
    keys = [c for c in P.columns if "|" in c]; va, te = P[P.split == "val"], P[P.split == "test"]
    yv, yt = va.target.to_numpy(), te.target.to_numpy(); isig, icon = np.where(yt == 1)[0], np.where(yt == 0)[0]
    rng = np.random.default_rng(seed)
    bs = np.concatenate([rng.choice(isig, (n_boot, len(isig))), rng.choice(icon, (n_boot, len(icon)))], axis=1)       # stratified resamples
    rows, bootauc = [], {}
    for k in keys:
        n, f = k.split("|"); s = te[k].to_numpy(); t = thr[k]["selected"]
        r = {"key": k, "model": n, "model_label": LABELS[n], "feature_set": f, "n_features": tlog["models"][k]["n_features_used"], "n_candidates": tlog["models"][k]["n_features"],
             "single_cut_feature": tlog["models"][k].get("single_cut_feature", ""), "threshold": t, "val_auc": roc_auc_score(yv, va[k]), "test_auc": roc_auc_score(yt, s),
             "test_ap_signal": average_precision_score(yt, s), "test_ap_continuum": average_precision_score(1 - yt, -s),     # continuum-positive: reversed score
             "tuning_s": tlog["models"][k]["tuning_seconds"], "final_fit_s": tlog["models"][k]["final_fit_seconds"], "best_params": json.dumps(tlog["models"][k]["best_params"]),
             "n_signal_test": int(len(isig)), "n_continuum_test": int(len(icon))}
        for tag, tt in (("op", t), ("t05", .5)):
            m = op_metrics(yt, s, tt)
            for mk, v in m.items(): r[f"{tag}_{mk}"] = v
            r[f"{tag}_eff_lo"], r[f"{tag}_eff_hi"] = clopper_pearson(m["tp"], m["tp"] + m["fn"]); r[f"{tag}_rej_lo"], r[f"{tag}_rej_hi"] = clopper_pearson(m["tn"], m["tn"] + m["fp"])
        for mk, v in op_metrics(yv, va[k].to_numpy(), t).items(): r[f"val_op_{mk}"] = v
        aucs, bal = np.empty(n_boot), np.empty(n_boot)
        for i, b in enumerate(bs):
            yb, sb = yt[b], s[b]; aucs[i] = roc_auc_score(yb, sb); pr = sb >= t; bal[i] = (pr[yb == 1].mean() + (~pr[yb == 0]).mean()) / 2
        bootauc[k] = aucs; r["auc_ci_lo"], r["auc_ci_hi"] = np.percentile(aucs, [2.5, 97.5]); r["op_bal_ci_lo"], r["op_bal_ci_hi"] = np.percentile(bal, [2.5, 97.5])
        rows.append(r)
    R = pd.DataFrame(rows)
    diffs = []
    for n in MODEL_ORDER:
        for f2 in ("corr_filtered", "reduced"):
            if f"{n}|{f2}" in bootauc and f"{n}|full" in bootauc:
                d = bootauc[f"{n}|{f2}"] - bootauc[f"{n}|full"]
                diffs.append({"model": n, "feature_set": f2, "auc_diff_vs_full": float(d.mean()), "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))})
    D = pd.DataFrame(diffs); best = tlog["best_by_validation_auc"]; pv = []
    for n in MODEL_ORDER:
        k = f"{n}|full"
        if k in bootauc and k != best:
            d = bootauc[best] - bootauc[k]; lo, hi = np.percentile(d, [2.5, 97.5])
            pv.append({"comparison": f"{best} minus {k}", "model": n, "auc_diff": float(roc_auc_score(yt, te[best]) - roc_auc_score(yt, te[k])), "ci_lo": float(lo), "ci_hi": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    PV = pd.DataFrame(pv)
    R.to_csv(out / "metrics/results_full.csv", index=False); D.to_csv(out / "metrics/paired_auc_difference.csv", index=False); PV.to_csv(out / "metrics/paired_auc_vs_best.csv", index=False)
    return R, D, PV, P, thr, tlog


def feat_used(r):
    return "0" if r.model == "majority" else (f"1 ({r.single_cut_feature}; {r.n_candidates} examined)" if r.model == "single_cut" else str(r.n_features))


def tables(R, D, PV, tlog, out):
    full = R[R.feature_set == "full"].copy(); ns, nc = int(full.n_signal_test.iloc[0]), int(full.n_continuum_test.iloc[0])
    ci = lambda a, lo, hi: f"{PCT(a)} [{PCT(lo)}, {PCT(hi)}]"
    T1 = pd.DataFrame({"Model": full.model_label, "Accuracy": full.op_accuracy.map(f3), "Precision (signal)": full.op_precision.map(f3), "Recall (signal)": full.op_recall.map(f3), "F1 (signal)": full.op_f1.map(f3),
                       "ROC-AUC [95% CI]": [f"{a:.3f} [{l:.3f}, {h:.3f}]" for a, l, h in zip(full.test_auc, full.auc_ci_lo, full.auc_ci_hi)], "Features used": [feat_used(r) for r in full.itertuples()],
                       "Threshold": full.threshold.map(f3), "Final fit (s)": full.final_fit_s.round(1)})
    T2 = pd.DataFrame({"Model": full.model_label, f"Signal efficiency [95% exact CI], n={ns:,}": [ci(a, l, h) for a, l, h in zip(full.op_signal_efficiency, full.op_eff_lo, full.op_eff_hi)],
                       f"Continuum rejection [95% exact CI], n={nc}": [ci(a, l, h) for a, l, h in zip(full.op_continuum_rejection, full.op_rej_lo, full.op_rej_hi)],
                       "Continuum acceptance": full.op_continuum_acceptance.map(PCT), "Balanced accuracy": full.op_balanced_accuracy.map(f3), "Macro-F1": full.op_macro_f1.map(f3)})
    T3 = pd.DataFrame({"Model": full.model_label, "Accuracy": full.t05_accuracy.map(f3), "Precision (signal)": full.t05_precision.map(f3), "Recall (signal)": full.t05_recall.map(f3), "F1 (signal)": full.t05_f1.map(f3),
                       "ROC-AUC": full.test_auc.map(f3), "Signal efficiency": full.t05_signal_efficiency.map(PCT), f"Continuum rejection (n={nc})": full.t05_continuum_rejection.map(PCT), "Balanced accuracy": full.t05_balanced_accuracy.map(f3), "Macro-F1": full.t05_macro_f1.map(f3)})
    T4 = pd.DataFrame({"Model": full.model_label, "Signal P": full.op_precision_signal.map(f3), "Signal R": full.op_recall_signal.map(f3), "Signal F1": full.op_f1_signal.map(f3),
                       "Continuum P": full.op_precision_continuum.map(f3), "Continuum R": full.op_recall_continuum.map(f3), "Continuum F1": full.op_f1_continuum.map(f3)})
    parts = [f"## Table A - required metrics at the validation-selected operating point (test: {ns:,} signal, {nc} continuum)\n", md_table(T1),
             "\nPrecision/recall/F1 are for the signal class. ROC-AUC uses the continuous scores of the same test rows, so it does not depend on the threshold.\n",
             f"\n## Table B - signal efficiency and continuum rejection at the operating point\n", md_table(T2),
             "\n## Table C - the same models at the fixed threshold 0.5\n", md_table(T3), "\n## Table D - per-class precision, recall, F1 at the operating point\n", md_table(T4)]
    for nme, T in (("A_required_metrics_operating_point", T1), ("B_efficiency_rejection_operating_point", T2), ("C_threshold_0.5", T3), ("D_per_class_operating_point", T4)): T.to_csv(out / f"metrics/table_{nme}.csv", index=False)
    if (R.feature_set != "full").any():
        fr = R[R.model.isin(["logreg", "tree", "forest", "boosting"])].merge(D, on=["model", "feature_set"], how="left")
        fr["o"] = fr.model.map({m: i for i, m in enumerate(MODEL_ORDER)}); fr["fo"] = fr.feature_set.map({"full": 0, "corr_filtered": 1, "reduced": 2}); fr = fr.sort_values(["o", "fo"])
        FZ = pd.DataFrame({"Model": fr.model_label, "Feature set": fr.feature_set, "Features": fr.n_features, "ROC-AUC": fr.test_auc.map(f3),
                           "AUC diff vs full [95% paired CI]": [("-" if pd.isna(a) else f"{a:+.4f} [{l:+.4f}, {h:+.4f}]") for a, l, h in zip(fr.auc_diff_vs_full, fr.ci_lo, fr.ci_hi)],
                           "Signal eff.": fr.op_signal_efficiency.map(PCT), f"Continuum rej. (n={nc})": fr.op_continuum_rejection.map(PCT), "Balanced acc.": fr.op_balanced_accuracy.map(f3)})
        FZ.to_csv(out / "metrics/full_vs_reduced_table.csv", index=False); parts += ["\n## Table E - full vs reduced feature sets (same protocol)\n", md_table(FZ)]
    cw = []
    for n in ("logreg", "tree", "forest", "boosting"):
        for f in ("full",):
            s = tlog["models"][f"{n}|{f}"]["search"]; g = lambda w: max([e["val_auc"] for e in s if (e["params"].get("class_weight") in w)] or [np.nan])
            unw, wt = g([None]), g(["balanced", "balanced_subsample"]); cw.append({"Model": LABELS[n], "best validation AUC, no weighting": f3(unw), "best validation AUC, training-only class weighting": f3(wt), "weighting chosen": "yes" if tlog["models"][f"{n}|{f}"]["best_params"].get("class_weight") else "no"})
    CW = pd.DataFrame(cw); CW.to_csv(out / "metrics/class_weighting_comparison_validation.csv", index=False); parts += ["\n## Table F - training-only class weighting compared on validation data (full features)\n", md_table(CW)]
    (out / "metrics/results_tables.md").write_text("\n".join(parts)); return T1, T2


def figures(R, D, PV, P, thr, tlog, out, cfg):
    fd = out / "figures"; fd.mkdir(parents=True, exist_ok=True); te, va = P[P.split == "test"], P[P.split == "val"]; yt, yv = te.target.to_numpy(), va.target.to_numpy()
    ns, nc = int((yt == 1).sum()), int((yt == 0).sum()); main = [m for m in MODEL_ORDER if f"{m}|full" in P.columns]; best = tlog["best_by_validation_auc"]; bm = best.split("|")[0]
    # 10: ROC / PR, signal-positive and continuum-positive (reversed score)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.5))
    for m in main:
        s = te[f"{m}|full"].to_numpy(); lab = f"{LABELS[m].split(' (')[0]} (AUC {roc_auc_score(yt, s):.3f})"; kw = dict(color=MC[m], lw=2, ls="--" if m == "majority" else "-")
        fpr, tpr, _ = roc_curve(yt, s); ax[0, 0].plot(fpr, tpr, label=lab, **kw)
        fpr2, tpr2, _ = roc_curve(1 - yt, -s); ax[0, 1].plot(fpr2, tpr2, **kw)
        if m == "majority": continue          # constant scores give a degenerate PR line; its prevalence-level AP is in the tables
        p, r, _ = precision_recall_curve(yt, s); ax[1, 0].plot(r, p, label=f"{LABELS[m].split(' (')[0]} (AP {average_precision_score(yt, s):.3f})", **kw)
        p2, r2, _ = precision_recall_curve(1 - yt, -s); ax[1, 1].plot(r2, p2, **kw)
    for m, (a, key) in ((bm, (ax[0, 0], "op")),):
        pass
    bo = R.set_index("key").loc[best]
    ax[0, 0].plot(bo.op_continuum_acceptance, bo.op_signal_efficiency, "o", color=INK, ms=7, label="selected operating point (best model)"); ax[0, 1].plot(1 - bo.op_signal_efficiency, bo.op_continuum_rejection, "o", color=INK, ms=7)
    ax[0, 0].set_xlabel("continuum acceptance (fraction of continuum kept as signal)"); ax[0, 0].set_ylabel("signal efficiency"); ax[0, 0].set_title(f"ROC, signal = positive class (test: {ns:,} signal, {nc} continuum)", fontsize=9.5); ax[0, 0].legend(fontsize=7.5, loc="lower right")
    ax[0, 1].set_xlabel("fraction of signal wrongly tagged as continuum (1 - signal efficiency)"); ax[0, 1].set_ylabel("continuum rejection"); ax[0, 1].set_title("ROC, continuum = positive class (reversed score)", fontsize=9.5)
    ax[1, 0].set_xlabel("recall (signal efficiency)"); ax[1, 0].set_ylabel("precision (signal purity)"); ax[1, 0].set_title("Precision-recall, signal = positive class", fontsize=9.5); ax[1, 0].set_ylim(.9, 1.003); ax[1, 0].legend(fontsize=7.5, loc="lower left")
    ax[1, 1].set_xlabel("recall (continuum rejection)"); ax[1, 1].set_ylabel("precision (continuum purity)"); ax[1, 1].set_title("Precision-recall, continuum = positive class (reversed score)", fontsize=9.5); ax[1, 1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(fd / "10_roc_pr.png"); plt.close(fig)
    # 11: confusion matrices with counts
    ms = [m for m in ("logreg", "tree", "forest", "boosting") if m in main]; fig, axs = plt.subplots(2, len(ms), figsize=(3.9 * len(ms), 7.4))
    for j, m in enumerate(ms):
        for i, (nm, t) in enumerate((("fixed threshold", .5), ("operating point", thr[f"{m}|full"]["selected"]))):
            mm = op_metrics(yt, te[f"{m}|full"].to_numpy(), t); cm = np.array([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]]); a = axs[i, j]; a.grid(False); a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for u in range(2):
                for v in range(2): a.text(v, u, f"{cm[u, v]:,}\n({cm[u, v] / cm[u].sum():.1%} of true row)", ha="center", va="center", fontsize=8.5, color="white" if cm[u, v] / cm[u].sum() > .5 else "black")
            a.set_xticks([0, 1], ["predicted\ncontinuum", "predicted\nsignal"]); a.set_yticks([0, 1], [f"true continuum\n(n={nc})", f"true signal\n(n={ns:,})"]); a.set_title(f"{LABELS[m].split(' (')[0]}\n{nm} = {t:.3f}", fontsize=9)
    fig.suptitle("Confusion matrices, test set (shading = fraction of each TRUE class; counts shown)", y=1.0, fontsize=10); fig.tight_layout(); fig.savefig(fd / "11_confusion_matrices.png"); plt.close(fig)
    # 12: score distribution of best model
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); bins = np.linspace(0, 1, 41)
    for a, d, nm in ((ax[0], va, "validation"), (ax[1], te, "test")):
        for t, c, l in ((0, NEG, "continuum"), (1, POS, "signal")):
            x = d.loc[d.target == t, best]; a.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=.35, color=c); a.hist(x, bins=bins, density=True, histtype="step", color=c, lw=1.6, label=f"{l} (n={len(x):,})")
        a.axvline(thr[best]["selected"], color=INK, ls="--", lw=1.2, label=f"operating point {thr[best]['selected']:.3f}"); a.axvline(.5, color=MUTED, ls=":", lw=1.2, label="0.5")
        a.set_xlabel(f"{LABELS[bm].split(' (')[0]} signal score"); a.set_ylabel("density (each class normalised)"); a.set_title(f"Score distribution - {nm}"); a.legend(fontsize=8)
    fig.savefig(fd / "12_score_distribution.png"); plt.close(fig)
    # 13: threshold analysis on validation
    ts = np.unique(np.quantile(va[best], np.linspace(0, 1, 400))); sw = pd.DataFrame([{"t": t, **op_metrics(yv, va[best].to_numpy(), t)} for t in ts]); sw.to_csv(out / "metrics/threshold_sweep_validation_best_model.csv", index=False)
    swt = pd.DataFrame([{"t": t, **op_metrics(yt, te[best].to_numpy(), t)} for t in ts]); swt.to_csv(out / "metrics/threshold_sweep_test_best_model.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); ax.plot(sw.t, sw.signal_efficiency, color=POS, lw=2, label="signal efficiency"); ax.plot(sw.t, sw.continuum_rejection, color=NEG, lw=2, label=f"continuum rejection (validation n={int((yv == 0).sum())})")
    ax.axhline(cfg["operating_point"]["min_signal_efficiency"], color=MUTED, ls=":", label=f"efficiency floor {cfg['operating_point']['min_signal_efficiency']:.0%} (analysis choice)"); ax.axvline(thr[best]["selected"], color=INK, ls="--", lw=1)
    ax.set_xlabel("score threshold (predict signal if score >= threshold)"); ax.set_ylabel("VALIDATION metric"); ax.set_title(f"Threshold analysis on validation data - {LABELS[bm].split(' (')[0]}"); ax.legend(fontsize=8, loc="center left"); ax.set_ylim(0, 1.02)
    fig.savefig(fd / "13_threshold_analysis.png"); plt.close(fig)
    # 14: model comparison
    full = R[R.feature_set == "full"]; x = np.arange(len(full)); lab = [LABELS[m].replace(" (HistGB)", "\n(HistGB)").replace(" ", "\n", 1) for m in full.model]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].bar(x, full.test_auc, .6, color=POS, yerr=[full.test_auc - full.auc_ci_lo, full.auc_ci_hi - full.test_auc], capsize=3, ecolor=INK); ax[0].set_xticks(x, lab, fontsize=8); ax[0].set_ylim(.4, 1); ax[0].set_title(f"ROC-AUC, test (95% stratified-bootstrap CI; {nc} continuum events)", fontsize=9.5)
    ax[1].bar(x, full.op_continuum_rejection, .6, color=NEG, yerr=[full.op_continuum_rejection - full.op_rej_lo, full.op_rej_hi - full.op_continuum_rejection], capsize=3, ecolor=INK)
    for i, (e) in enumerate(full.op_signal_efficiency): ax[1].text(i, 1.01, f"eff {e:.0%}", ha="center", fontsize=8)
    ax[1].set_xticks(x, lab, fontsize=8); ax[1].set_ylim(0, 1.1); ax[1].set_title("Continuum rejection at the operating point, test (95% exact CI)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(fd / "14_model_comparison.png"); plt.close(fig)
    fr = tlog.get("feature_reduction")
    if fr:
        pi = pd.read_csv(out / "metrics/permutation_importance_val.csv"); top = pi.head(20)[::-1]; red = set(fr["reduced_features"])
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
        ax[0].barh([f.replace("KSFWVariables__", "KSFW ").replace("CleoConeCS__", "CLEO ").replace("__bc", "") for f in top.feature], top.perm_importance_val_auc_drop, xerr=top["std"], color=[POS if f in red else "#b8b8b8" for f in top.feature], ecolor=INK)
        ax[0].set_xlabel("drop in validation ROC-AUC when the feature is permuted (mean +/- sd, 5 repeats)"); ax[0].set_title(f"Permutation importance, validation data (blue = in the {fr['k_selected']}-feature reduced set)", fontsize=10)
        k = pd.DataFrame(fr["k_curve"]); ax[1].plot(k.k, k.val_auc, marker="o", color=POS); ax[1].axhline(fr["val_auc_default_hgb_full"], color=NEG, ls="--", label=f"all {fr['n_full']} features"); ax[1].axhline(cfg["feature_reduction"]["min_val_auc_fraction"] * fr["val_auc_default_hgb_full"], color=MUTED, ls=":", label=f"{cfg['feature_reduction']['min_val_auc_fraction']:.1%} of full")
        ax[1].axvline(fr["k_selected"], color=INK, lw=.8); ax[1].set_xlabel("number of top-ranked features kept"); ax[1].set_ylabel("validation ROC-AUC (default weighted HistGB)"); ax[1].set_title("Choosing the reduced-set size (validation; noisy: few continuum events)", fontsize=10); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(fd / "16_feature_importance_reduction.png"); plt.close(fig)
        fz = R[R.model.isin(["logreg", "tree", "forest", "boosting"])]; fig, ax = plt.subplots(1, 2, figsize=(13, 4.4)); w = .26
        for j, (fs, c) in enumerate((("full", POS), ("corr_filtered", "#56B4E9"), ("reduced", NEG))):
            sub = fz[fz.feature_set == fs].set_index("model").reindex(["logreg", "tree", "forest", "boosting"]); lab4 = [LABELS[m].split(" (")[0] for m in sub.index]
            ax[0].bar(np.arange(4) + (j - 1) * w, sub.test_auc, w, color=c, label=f"{fs} ({int(sub.n_features.iloc[0])} features)", yerr=[sub.test_auc - sub.auc_ci_lo, sub.auc_ci_hi - sub.test_auc], capsize=2, ecolor=INK)
            ax[1].bar(np.arange(4) + (j - 1) * w, sub.op_continuum_rejection, w, color=c, yerr=[sub.op_continuum_rejection - sub.op_rej_lo, sub.op_rej_hi - sub.op_continuum_rejection], capsize=2, ecolor=INK)
        for a_, ttl, yl in ((ax[0], "Test ROC-AUC (95% stratified bootstrap CI)", "ROC-AUC"), (ax[1], f"Continuum rejection at the operating point (95% exact CI, n={nc})", "continuum rejection")):
            a_.set_xticks(range(4), lab4, fontsize=8); a_.set_title(ttl, fontsize=9.5); a_.set_ylabel(yl)
        ax[0].set_ylim(.4, 1); ax[1].set_ylim(0, 1); ax[0].legend(fontsize=8, loc="upper left"); fig.suptitle("Full vs reduced feature sets - same protocol", y=1.0); fig.tight_layout(); fig.savefig(fd / "17_full_vs_reduced.png"); plt.close(fig)


def main():
    cfg = data.load_config(); R, D, PV, P, thr, tlog = evaluate(n_boot=cfg["bootstrap"]["n"], seed=cfg["seed"])
    T1, T2 = tables(R, D, PV, tlog, ART); figures(R, D, PV, P, thr, tlog, ART, cfg); print(T1.to_string()); print(T2.to_string()); print(PV.round(4).to_string())


if __name__ == "__main__":
    main()
