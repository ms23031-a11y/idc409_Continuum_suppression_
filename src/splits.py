"""Leakage-aware train/val/test assignment for the PRIMARY experiment rows only (excluded rows never enter any split).

No reliable event identifier exists (the unnamed CSV index is not one). Rows are therefore grouped by a rounded hash of the
full feature vector so any exact/near-exact duplicate observations stay together, and the split is stratified on the binary
class (signal / continuum) so every split keeps the class proportions - essential with only a few hundred continuum events.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def group_ids(X, decimals=6):
    r = np.round(X.to_numpy(dtype="float64"), decimals)
    return pd.factorize(pd.Series([hashlib.md5(row.tobytes()).hexdigest() for row in r]))[0]


def assign_splits(X, y, cfg, seed=None):
    s = cfg["split"]; seed = cfg["seed"] if seed is None else seed; n_f = s["n_folds"]
    n_val, n_test = round(n_f * s["val_frac"]), round(n_f * s["test_frac"])
    groups = group_ids(X, s["round_decimals"]); fold = np.empty(len(X), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=n_f, shuffle=True, random_state=seed).split(X, y, groups)): fold[te] = k
    out = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))
    return pd.Series(out, index=X.index, name="split"), groups
