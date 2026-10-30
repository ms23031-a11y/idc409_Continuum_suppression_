"""Step 2: exploratory figures for the signal-vs-continuum experiment. Decisions use TRAINING rows only; rows of the
excluded other-decay codes (2-5) are shown only as grey context (they belong to no split)."""
import json
import numpy as np, pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from . import data, database
from .paths import FIG, MET
from .plotstyle import *

GREY = "#9a9a9a"
KEY = ["R2", "thrustBm", "thrustOm", "cosTBTO", "cosTBz", "thrust", "thrustAxisCosTheta", "foxWolframR1", "foxWolframR2", "KSFWVariables__boet__bc", "KSFWVariables__bomm2__bc", "CleoConeCS__bo1__bc"]
NICE = lambda f: f.replace("KSFWVariables__", "KSFW ").replace("CleoConeCS__", "CLEO ").replace("__bc", "")
NOTFEAT = ("row_id", "csv_index", "type", "target", "split", "role", "in_primary")


def load(cfg):
    con = database.connect(cfg); allv = database.read_events(con, primary_only=False); con.close()
    feats = [c for c in allv.columns if c not in NOTFEAT]
    return allv, allv[allv.split == "train"].reset_index(drop=True), feats


def main():
    cfg = data.load_config(); m = cfg["label_mapping"]; allv, tr, feats = load(cfg); rep = json.load(open(MET / "data_audit.json")); ex = rep["experiment"]
    ctr, str_ = int((tr.target == 0).sum()), int((tr.target == 1).sum())
    # 1 label distribution and selection ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    c = rep["type_counts"]; codes = list(c.keys()); role = {k: ("signal" if int(k) in m["signal_codes"] else "continuum" if int(k) in m["continuum_codes"] else "other") for k in codes}
    col = {"signal": POS, "continuum": NEG, "other": GREY}
    ax[0].bar(codes, [c[k] for k in codes], color=[col[role[k]] for k in codes])
    for k in codes: ax[0].text(k, c[k], f"{c[k]:,}", ha="center", va="bottom", fontsize=8.5)
    ax[0].set_xlabel("original `type` code (blue = signal, orange = continuum, grey = other decays, excluded)"); ax[0].set_ylabel("rows"); ax[0].set_title("All 70,606 rows by original code (no type -1 rows exist)", fontsize=10)
    ax[1].bar(["continuum", "signal"], [ex["n_continuum"], ex["n_signal"]], color=[NEG, POS])
    ax[1].set_yscale("log"); ax[1].set_ylim(100, 1e6)
    for i, v in enumerate([ex["n_continuum"], ex["n_signal"]]): ax[1].text(i, v * 1.2, f"{v:,}\n({v / ex['n_selected']:.1%})", ha="center", fontsize=9)
    ax[1].set_title(f"Primary experiment (codes 1, 0, -1): {ex['n_selected']:,} rows\n(log scale)", fontsize=10)
    fig.suptitle("Class balance: the primary sample is ~98.7% signal, so a predict-all-signal rule is 98.7% accurate and rejects no continuum", y=1.03, fontsize=10, color=MUTED)
    fig.savefig(FIG / "01_label_balance.png"); plt.close(fig)
    # 2 univariate separation (training rows of the experiment) --------------------------------------
    auc = pd.Series({f: roc_auc_score(tr.target, tr[f]) for f in feats}); sep = (auc - .5).abs().sort_values(ascending=False)
    auc.rename("train_auc").to_frame().assign(abs_sep=sep.reindex(auc.index), family=[data.feature_family(f) for f in auc.index]).sort_values("abs_sep", ascending=False).to_csv(MET / "univariate_separation_train.csv")
    top = sep.head(20)[::-1]; fig, ax = plt.subplots(figsize=(8.5, 6)); ax.barh([NICE(f) for f in top.index], top.values + .5, color=POS); ax.set_xlim(.5, max(.6, float(top.max()) + .55))
    ax.set_xlabel(f"single-feature ROC-AUC, max(AUC, 1-AUC) (0.5 = no separation) - training rows: {str_:,} signal, {ctr} continuum"); ax.set_title("Top 20 single-feature separators: signal vs continuum"); fig.savefig(FIG / "02_univariate_separation.png"); plt.close(fig)
    # 3 distributions signal vs continuum -------------------------------------------------------------
    fig, axs = plt.subplots(3, 4, figsize=(13, 8.5))
    for a, f in zip(axs.ravel(), KEY):
        lo, hi = np.percentile(tr[f], [0.5, 99.5]); bins = np.linspace(lo, hi, 40)
        for t, colr, nm in ((0, NEG, f"continuum (n={ctr})"), (1, POS, f"signal (n={str_:,})")): a.hist(tr.loc[tr.target == t, f].clip(lo, hi), bins=bins, density=True, histtype="step", lw=1.8, color=colr, label=nm)
        a.set_title(NICE(f), fontsize=10); a.set_yticks([])
    axs[0, 0].legend(fontsize=8); fig.suptitle("Event-shape distributions, signal vs continuum (training rows, each class density-normalised; values clipped to the 0.5-99.5 percentile range, so edge spikes are clipping artefacts)", y=1.0, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "03_event_shape_distributions.png"); plt.close(fig)
    # 4 per original code, including excluded codes as grey context ----------------------------------
    sel = ["R2", "thrust", "cosTBTO", "foxWolframR2", "KSFWVariables__boet__bc", "CleoConeCS__bo1__bc"]; cs = sorted(allv.type.unique())
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.8))
    for a, f in zip(axs.ravel(), sel):
        g = [allv.loc[(allv.type == t) & ((allv.split == "train") | (allv.split == "excluded")), f].to_numpy() for t in cs]
        bp = a.boxplot(g, tick_labels=[str(t) for t in cs], patch_artist=True, showfliers=False, widths=.6, medianprops=dict(color=INK))
        for p, t in zip(bp["boxes"], cs): p.set_facecolor(col[role[str(t)]]); p.set_alpha(.75)
        a.set_title(NICE(f), fontsize=10); a.set_xlabel("original `type` code")
    fig.suptitle("Each original code (blue = signal, orange = continuum; grey codes 2-5 = other decays, excluded from the experiment and shown only for context)\nBoxes = IQR, whiskers 1.5 IQR, outliers hidden; codes 1 and 0 use training rows", y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "04_distributions_by_type_code.png"); plt.close(fig)
    # 5 correlation ------------------------------------------------------------------------------------
    C = tr[feats].corr(method="spearman").to_numpy(); D = 1 - np.abs(C); np.fill_diagonal(D, 0); D = (D + D.T) / 2
    order = hierarchy.leaves_list(hierarchy.linkage(squareform(D, checks=False), method="average")); names = [feats[i] for i in order]; Co = C[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(10, 8.6)); im = ax.imshow(Co, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    lab = [NICE(n).replace("KSFW ", "K:").replace("CLEO ", "C:") for n in names]; ax.set_xticks(range(len(names)), lab, rotation=90, fontsize=5.5); ax.set_yticks(range(len(names)), lab, fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=.7, label="Spearman correlation"); ax.set_title("Feature correlation (training rows of the experiment, hierarchically ordered)"); fig.savefig(FIG / "05_correlation_heatmap.png"); plt.close(fig)
    iu = np.triu_indices(len(feats), 1); pairs = pd.DataFrame({"a": np.array(feats)[iu[0]], "b": np.array(feats)[iu[1]], "spearman": tr[feats].corr(method="spearman").to_numpy()[iu]})
    pairs[pairs.spearman.abs() >= .9].sort_values("spearman", key=abs, ascending=False).to_csv(MET / "highly_correlated_pairs_train.csv", index=False)
    # 6 CLEO cones: signal vs continuum ---------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, suf in enumerate(["", "__cmROE"]):
        for t, colr, nm in ((1, POS, "signal"), (0, NEG, "continuum")):
            mu = [tr.loc[tr.target == t, f"CleoConeCS__bo{k}{suf}__bc"].mean() for k in range(1, 10)]; ax[j].plot(range(1, 10), mu, marker="o", ms=4, color=colr, label=nm)
        ax[j].set_xlabel("cone index"); ax[j].set_ylabel("mean cone value (as stored)"); ax[j].set_title("CLEO cones" + (" (ROE only, cmROE)" if suf else " (all particles)"))
    ax[0].legend(fontsize=8); fig.savefig(FIG / "06_cleo_cone_profiles.png"); plt.close(fig)
    # 7 data quality ------------------------------------------------------------------------------------------
    zf = (tr[feats] == 0).mean(); zf = zf[zf > 0].sort_values()[-15:]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].barh([NICE(n) for n in zf.index], zf.values * 100, color=MUTED); ax[0].set_xlabel("% of training rows exactly 0.0"); ax[0].set_title("Exact zeros (all in CLEO-cone features) - kept, physically meaningful")
    ax[1].axis("off"); ax[1].grid(False)
    txt = (f"Missing values: {rep['missing_total']}\nInfinite values: {rep['inf_total']}\nConstant features: {len(rep['constant_features'])}\nDuplicate feature vectors: {rep['duplicate_feature_vectors']}\n"
           f"Conflicting-label vectors: {rep['feature_vectors_with_conflicting_type']}\nSentinel values (-999, 9999, -1 ...): {sum(rep['common_sentinel_hits'].values())}\nIndex-like column repeated values: {rep['index_column']['n_repeated_values']:,}\nType -1 rows in file: {ex['n_type_minus1']}")
    ax[1].text(0, .95, "Data-quality summary (full file)", fontweight="bold", va="top"); ax[1].text(0, .85, txt, va="top", family="monospace", fontsize=9.5); fig.savefig(FIG / "07_data_quality.png"); plt.close(fig)
    print("eda figures written")


if __name__ == "__main__":
    main()
