"""Data loading, config, labelling and the audit (NumPy / pandas)."""
import hashlib, json
import numpy as np
import pandas as pd
from .paths import ROOT, CFG_PATH, MET


def load_config(path=CFG_PATH):
    return json.loads(open(path).read())


def config_fingerprint(cfg):
    """SHA-256 of the experiment configuration (student block excluded): ties every output to the settings that produced it."""
    return hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k != "student"}, sort_keys=True).encode()).hexdigest()


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_raw(cfg):
    """Original CSV, untouched."""
    return pd.read_csv(ROOT / cfg["data"]["csv"])


def feature_columns(df, cfg):
    """All numeric columns except the index-like column and the label code."""
    drop = {cfg["data"]["index_column"], cfg["data"]["label_column"]}
    return [c for c in df.columns if c not in drop]


def role_of(type_codes, mapping):
    """'signal' / 'continuum' / 'other_decay' for each row (instructor-provided mapping)."""
    t = np.asarray(type_codes)
    return np.where(np.isin(t, mapping["signal_codes"]), "signal", np.where(np.isin(t, mapping["continuum_codes"]), "continuum", "other_decay"))


def primary_mask(type_codes, mapping):
    """Rows used in the primary signal-versus-continuum experiment: type in {1, 0, -1}."""
    return np.isin(np.asarray(type_codes), list(mapping["signal_codes"]) + list(mapping["continuum_codes"]))


def make_target(type_codes, mapping):
    """1 = signal, 0 = continuum. Defined only for primary rows; raises on any other code."""
    t = np.asarray(type_codes)
    if not primary_mask(t, mapping).all():
        raise ValueError(f"codes outside the primary experiment: {sorted(set(t[~primary_mask(t, mapping)]))}")
    return np.isin(t, mapping["signal_codes"]).astype(int)


def feature_family(name):
    if name.startswith("CleoConeCS"):
        return "CLEO cones (cmROE)" if "cmROE" in name else "CLEO cones"
    if name.startswith("KSFWVariables"):
        if "__bohso" in name or "__bohoo" in name:
            return "KSFW moments (cmFS1)" if "cmFS1" in name else "KSFW moments"
        return "KSFW (et, mm2)"
    if name.startswith("foxWolfram") or name == "R2":
        return "Fox-Wolfram"
    if "thrust" in name.lower() or name.startswith("cosTB"):
        return "Thrust / angular"
    return "other"


def audit(df, cfg):
    idx, lab = cfg["data"]["index_column"], cfg["data"]["label_column"]
    feats = feature_columns(df, cfg)
    X = df[feats]
    num = df.select_dtypes("number")
    rep = {
        "shape": list(df.shape), "n_features": len(feats),
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "type_counts": {int(k): int(v) for k, v in df[lab].value_counts().sort_index().items()},
        "missing_total": int(df.isna().sum().sum()), "inf_total": int(np.isinf(num.to_numpy()).sum()),
        "constant_features": [c for c in feats if X[c].nunique() <= 1],
        "duplicate_rows_excl_index": int(df.drop(columns=idx).duplicated().sum()),
        "duplicate_feature_vectors": int(X.duplicated().sum()),
    }
    g = X.assign(_t=df[lab].values).groupby(feats, sort=False)["_t"].nunique()
    rep["feature_vectors_with_conflicting_type"] = int((g > 1).sum())
    # partial-vector duplicates: same (R2, thrust, thrustBm, thrustOm) could hint at same physical event
    key = ["R2", "thrust", "thrustBm", "thrustOm"]
    rep["duplicate_event_shape_keys"] = int(df.duplicated(subset=key).sum())
    # index column diagnostics
    i = df[idx]
    rep["index_column"] = {
        "unique": bool(i.is_unique), "n_repeated_values": int(i.duplicated().sum()),
        "equals_row_number": bool((i.to_numpy() == np.arange(len(df))).all()),
        "range_by_type": {int(k): [int(v.min()), int(v.max())] for k, v in df.groupby(lab)[idx]},
        "unique_within_each_type": bool(df.groupby(lab)[idx].apply(lambda s: s.is_unique).all()),
        "verdict": "Observed: repeats across the file, is unique within each type code, and its range depends on the code (type 1 runs exactly 0..N-1). This resembles a running index from separate per-category files, but its origin is NOT documented. It is not a reliable event ID and could encode the label, so it is EXCLUDED from predictors.",
    }
    zeros = (X == 0).mean()
    rep["exact_zero_fraction_top"] = zeros[zeros > 0].sort_values(ascending=False).head(10).round(4).to_dict()
    common_sentinels = [-999, -999.0, -9999, 9999, 999, -1]
    rep["common_sentinel_hits"] = {str(s): int((X == s).sum().sum()) for s in common_sentinels}
    rep["feature_ranges"] = {c: [float(X[c].min()), float(X[c].max())] for c in feats}
    rep["leakage_candidates"] = [idx, lab]
    m = cfg["label_mapping"]; t = df[lab]; pm = primary_mask(t, m)
    ns, nc = int(t.isin(m["signal_codes"]).sum()), int(t.isin(m["continuum_codes"]).sum())
    rep["experiment"] = {"primary_codes_requested": sorted(m["signal_codes"] + m["continuum_codes"]), "n_selected": int(pm.sum()), "n_signal": ns, "n_continuum": nc,
                         "continuum_rows_by_code": {int(k): int(v) for k, v in t[t.isin(m["continuum_codes"])].value_counts().items()},
                         "n_type_minus1": int((t == -1).sum()), "signal_fraction": ns / max(ns + nc, 1), "majority_accuracy": max(ns, nc) / max(ns + nc, 1),
                         "n_excluded": int((~pm).sum()), "excluded_by_code": {int(k): int(v) for k, v in t[~pm].value_counts().sort_index().items()}}
    return rep


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
