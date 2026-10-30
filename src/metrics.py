"""Metrics for the imbalanced signal-vs-continuum task. Class 1 = signal, class 0 = continuum.
signal efficiency   = TP / (TP + FN)  = fraction of TRUE SIGNAL retained (denominator: true signal rows)
continuum rejection = TN / (TN + FP)  = fraction of TRUE CONTINUUM rejected (denominator: true continuum rows)
continuum acceptance= FP / (TN + FP)  = 1 - continuum rejection (continuum wrongly kept as signal)"""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def clopper_pearson(k, n, level=0.95):
    """Exact binomial confidence interval for k successes out of n (appropriate for small n such as ~70 continuum events)."""
    if n == 0: return (np.nan, np.nan)
    a = (1 - level) / 2
    return (0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def confusion(y, s, t):
    pred = np.asarray(s) >= t; y = np.asarray(y).astype(bool)
    return int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & ~y).sum()), int((~pred & y).sum())   # tp, fp, tn, fn


def op_metrics(y, s, t):
    tp, fp, tn, fn = confusion(y, s, t); n = tp + fp + tn + fn; sd = lambda a, b: a / b if b else float("nan")
    eff, rej = sd(tp, tp + fn), sd(tn, tn + fp)
    p_sig, p_con = sd(tp, tp + fp), sd(tn, tn + fn)
    z = lambda x: 0.0 if np.isnan(x) else x                      # undefined precision (nothing predicted in that class) counts as 0
    f1 = lambda p, r: 0.0 if (z(p) + z(r)) == 0 else 2 * z(p) * z(r) / (z(p) + z(r))
    f_sig, f_con = f1(p_sig, eff), f1(p_con, rej); p_sig, p_con = z(p_sig), z(p_con)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_signal": tp + fn, "n_continuum": tn + fp,
            "accuracy": (tp + tn) / n, "precision": p_sig, "recall": eff, "f1": f_sig,
            "signal_efficiency": eff, "continuum_rejection": rej, "continuum_acceptance": 1 - rej,
            "balanced_accuracy": (eff + rej) / 2, "macro_f1": (f_sig + f_con) / 2,
            "precision_signal": p_sig, "recall_signal": eff, "f1_signal": f_sig, "precision_continuum": p_con, "recall_continuum": rej, "f1_continuum": f_con}


def operating_threshold(y_val, s_val, min_eff):
    """Largest threshold with validation signal efficiency >= min_eff (continuum rejection is non-decreasing in the threshold,
    so this maximises validation continuum rejection subject to the efficiency floor). Uses VALIDATION data only."""
    y = np.asarray(y_val).astype(bool); ss = np.sort(np.asarray(s_val)[y]); n = len(ss)
    cands = np.unique(ss)
    eff = 1 - np.searchsorted(ss, cands, side="left") / n               # fraction of signal with score >= candidate
    ok = np.where(eff >= min_eff - 1e-12)[0]
    t = float(cands[ok.max()])
    m = op_metrics(y_val, s_val, t); return t, {"val_signal_efficiency": m["signal_efficiency"], "val_continuum_rejection": m["continuum_rejection"]}


def strat_boot_auc(y, scores, idx_sig, idx_con):
    """AUC on one stratified resample (signal and continuum resampled separately)."""
    i = np.concatenate([idx_sig, idx_con]); return roc_auc_score(np.asarray(y)[i], np.asarray(scores)[i])
