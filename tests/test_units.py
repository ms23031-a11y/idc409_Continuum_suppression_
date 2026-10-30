"""Unit tests on tiny SYNTHETIC frames - they test code logic only; no reported result uses synthetic data."""
import numpy as np, pandas as pd, pytest
from src import data, predict, splits
from src.models import SingleFeatureCut

M = {"status": "test mapping", "signal_codes": [1], "continuum_codes": [0, -1], "other_decay_codes": [2, 3], "positive_name": "signal", "negative_name": "continuum"}


def test_make_target_and_excluded_codes():
    assert list(data.make_target([0, 1, -1, 1], M)) == [0, 1, 0, 1]
    with pytest.raises(ValueError): data.make_target([0, 5], M)
    assert list(data.primary_mask([1, 0, -1, 2, 3, 5], M)) == [True, True, True, False, False, False]
    assert list(data.role_of([1, 0, -1, 2], M)) == ["signal", "continuum", "continuum", "other_decay"]


def test_exact_interval_and_operating_threshold_denominators():
    from src.metrics import clopper_pearson, operating_threshold, op_metrics
    lo, hi = clopper_pearson(70, 70); assert hi == 1.0 and 0.94 < lo < 0.96
    lo, hi = clopper_pearson(0, 70); assert lo == 0.0 and 0.04 < hi < 0.06
    rng = np.random.default_rng(0); y = np.r_[np.ones(900, int), np.zeros(100, int)]; s = np.r_[rng.normal(.7, .15, 900), rng.normal(.3, .15, 100)]
    t, v = operating_threshold(y, s, 0.9); m = op_metrics(y, s, t)
    assert m["signal_efficiency"] >= 0.9 and m["n_signal"] == 900 and m["n_continuum"] == 100 and abs(m["continuum_rejection"] + m["continuum_acceptance"] - 1) < 1e-12
    assert operating_threshold(y, s, 0.9)[0] >= operating_threshold(y, s, 0.95)[0]


def test_metrics_finite_when_a_class_is_never_predicted():
    from src.metrics import op_metrics
    y = np.r_[np.ones(50, int), np.zeros(5, int)]; s = np.ones(55)                       # predicts signal for everything
    m = op_metrics(y, s, 0.5)
    assert m["continuum_rejection"] == 0 and m["signal_efficiency"] == 1 and np.isfinite(m["macro_f1"]) and np.isfinite(m["f1_continuum"]) and m["precision_continuum"] == 0


def test_single_cut_uses_one_feature_and_balanced_cut():
    rng = np.random.default_rng(0); y = (rng.random(3000) < 0.98).astype(int); X = np.c_[rng.normal(size=3000), -3 * y + rng.normal(size=3000)]
    m = SingleFeatureCut().fit(X, y); assert m.j_ == 1 and m.sign_ == -1 and m.n_examined_ == 2
    p = m.predict(X); assert (p[y == 0] == 0).mean() > .7 and (p[y == 1] == 1).mean() > .7        # balanced cut, not predict-all-majority


def test_duplicates_stay_in_same_split():
    rng = np.random.default_rng(1); X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd")); X = pd.concat([X, X.iloc[:100]], ignore_index=True)
    strat = np.r_[rng.integers(0, 3, 400), np.zeros(100, int)]; strat[400:] = strat[:100]
    cfg = {"seed": 1, "split": {"val_frac": .15, "test_frac": .15, "n_folds": 20, "round_decimals": 6}}
    sp, g = splits.assign_splits(X, strat, cfg); assert (pd.DataFrame({"s": sp, "g": g}).groupby("g").s.nunique() == 1).all()


def test_predict_validation_errors():
    class Dummy:
        def predict_proba(self, X): return np.c_[1 - X[:, 0], X[:, 0]]
    b = {"features": ["a", "b"], "model": Dummy(), "threshold": .5, "mapping": M}
    with pytest.raises(ValueError, match="missing"): predict.predict_df(pd.DataFrame({"a": [.1]}), b)
    with pytest.raises(ValueError, match="Non-numeric"): predict.predict_df(pd.DataFrame({"a": [np.nan], "b": [1.]}), b)
    with pytest.raises(ValueError, match="Non-numeric"): predict.predict_df(pd.DataFrame({"a": [np.inf], "b": [1.]}), b)
    with pytest.raises(ValueError, match="Non-numeric"): predict.predict_df(pd.DataFrame({"a": ["x"], "b": [1.]}), b)
    out = predict.predict_df(pd.DataFrame({"a": [.9, .1], "b": [0, 0], "extra": [1, 2]}), b)
    assert list(out.predicted_class) == ["signal", "continuum"]
