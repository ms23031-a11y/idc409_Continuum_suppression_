"""Inference with a saved bundle (preprocessing + model + feature list + frozen threshold)."""
import argparse, sys
import joblib, numpy as np, pandas as pd
from .paths import MOD


def load_bundle(path=None):
    return joblib.load(path or MOD / "best_model.joblib")


def class_names(bundle):
    m = bundle["mapping"]
    return m["negative_name"], m["positive_name"]          # ("continuum", "signal")


def validate(df, bundle):
    feats = bundle["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required feature column(s): {missing}")
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(X.to_numpy(dtype=float))
    if bad.any():
        cols = list(X.columns[bad.any(axis=0)])
        raise ValueError(f"Non-numeric, missing or infinite values in column(s): {cols}")
    return X


def predict_df(df, bundle):
    X = validate(df, bundle)
    score = bundle["model"].predict_proba(X.to_numpy())[:, 1]
    neg, pos = class_names(bundle)
    lab = np.where(score >= bundle["threshold"], pos, neg)
    return pd.DataFrame({"signal_score": score, "predicted_class": lab, "threshold_used": bundle["threshold"],
                         "label_mapping": bundle["mapping"]["status"]}, index=df.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a CSV of Belle II continuum-suppression features")
    ap.add_argument("--input", required=True); ap.add_argument("--model"); ap.add_argument("--output")
    a = ap.parse_args()
    try:
        out = predict_df(pd.read_csv(a.input), load_bundle(a.model))
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    (out.to_csv(a.output, index=False) if a.output else print(out.head(20).to_string()))
