"""Model definitions, a single-feature-cut baseline, tuning on VALIDATION data, threshold selection."""
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_ORDER = ["majority", "single_cut", "logreg", "tree", "forest", "boosting"]
LABELS = {"majority": "Majority class", "single_cut": "Single-feature cut", "logreg": "Logistic regression",
          "tree": "Decision tree", "forest": "Random forest", "boosting": "Gradient boosting (HistGB)"}


class SingleFeatureCut(BaseEstimator, ClassifierMixin):
    """Cut on the ONE training feature with the highest single-feature AUC; the cut value maximises TRAIN balanced accuracy
    (plain accuracy would just predict 'signal' for everything in this ~98.7% signal sample).
    predict_proba is a monotone logistic of the signed distance to the cut, so 0.5 == the cut."""
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])]); self.n_examined_ = X.shape[1]
        self.j_ = int(np.argmax(np.abs(aucs - .5))); self.sign_ = 1.0 if aucs[self.j_] >= .5 else -1.0
        v = X[:, self.j_]; cands = np.quantile(v, np.linspace(.01, .99, 197)); bal = []
        for c in cands:
            p = (self.sign_ * (v - c)) > 0; bal.append((p[y == 1].mean() + (~p[y == 0]).mean()) / 2)
        self.cut_ = float(cands[int(np.argmax(bal))]); self.scale_ = float(v.std()) or 1.0
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        z = self.sign_ * (np.asarray(X, float)[:, self.j_] - self.cut_) / self.scale_
        p = 1 / (1 + np.exp(-4 * z)); return np.c_[1 - p, p]

    def predict(self, X): return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def candidates(name, cfg, seed):
    t = cfg["tuning"]
    if name == "majority":
        return [dict()], lambda p: DummyClassifier(strategy="most_frequent")
    if name == "single_cut":
        return [dict()], lambda p: SingleFeatureCut()
    if name == "logreg":
        return list(ParameterGrid(t["logreg"])), lambda p: Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=seed, **p))])
    if name == "tree":
        return list(ParameterGrid(t["tree"])), lambda p: DecisionTreeClassifier(random_state=seed, **p)
    if name == "forest":
        f = t["forest"]; g = {k: f[k] for k in ("max_depth", "min_samples_leaf", "max_features", "class_weight")}
        return list(ParameterSampler(g, f["n_iter"], random_state=seed)), lambda p: RandomForestClassifier(
            n_estimators=f["n_estimators"], n_jobs=-1, random_state=seed, **p)
    if name == "boosting":
        b = t["boosting"]; g = {k: b[k] for k in ("learning_rate", "max_leaf_nodes", "l2_regularization", "class_weight")}
        return list(ParameterSampler(g, b["n_iter"], random_state=seed)), lambda p: HistGradientBoostingClassifier(
            max_iter=b["max_iter"], early_stopping=False, random_state=seed, **p)
    raise KeyError(name)


def tune_and_fit(name, cfg, seed, Xtr, ytr, Xva, yva, fixed_params=None):
    """Pick hyper-parameters by validation ROC-AUC (never test), then report the final train-only fit.
    Returns (fitted model, params, log dict)."""
    cands, make = candidates(name, cfg, seed)
    if fixed_params is not None:
        cands = [fixed_params]
    log, t_tune = [], time.time()
    best, best_auc = None, -1
    for p in cands:
        t0 = time.time(); m = make(p).fit(Xtr, ytr)
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1]) if len(cands) > 1 or name != "majority" else .5
        log.append({"params": p, "val_auc": float(auc), "fit_seconds": round(time.time() - t0, 2)})
        if auc > best_auc: best, best_auc = p, auc
    t0 = time.time(); model = make(best).fit(Xtr, ytr); fit_s = time.time() - t0
    return model, best, {"search": log, "tuning_seconds": round(time.time() - t_tune, 2), "final_fit_seconds": round(fit_s, 2),
                         "best_params": best, "best_val_auc": float(best_auc)}
