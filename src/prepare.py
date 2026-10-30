"""Step 1: audit -> SQLite write -> mapping UPDATE -> splits (primary rows only) -> SQL UPDATE -> read back & verify."""
import time
import numpy as np, pandas as pd
from . import data, database, splits
from .paths import ROOT, MET


def main():
    cfg = data.load_config(); csv = ROOT / cfg["data"]["csv"]; sha = data.sha256(csv); raw = data.load_raw(cfg)
    feats = data.feature_columns(raw, cfg); m = cfg["label_mapping"]
    rep = data.audit(raw, cfg); rep["csv_sha256"] = sha; rep["csv_bytes"] = csv.stat().st_size
    rep["feature_families"] = pd.Series([data.feature_family(c) for c in feats]).value_counts().to_dict()
    exp = rep["experiment"]
    if exp["n_selected"] != exp["n_signal"] + exp["n_continuum"]: raise RuntimeError("selection counts inconsistent")
    con = database.build(cfg, raw, sha); database.apply_mapping(con, cfg)
    ev = database.read_events(con)                                   # primary rows only (view)
    y = data.make_target(ev["type"], m); X = ev[feats]
    split, groups = splits.assign_splits(X, pd.Series(y, index=X.index), cfg)
    database.write_splits(con, pd.Series(split.to_numpy(), index=ev["row_id"]))
    rep["database_check"] = database.verify(con, raw, cfg)
    d = pd.DataFrame({"split": split.to_numpy(), "type": ev["type"].to_numpy(), "target": y, "group": groups})
    rep["split_counts"] = d.split.value_counts().to_dict()
    rep["split_class_counts"] = {s: {"signal": int(((d.split == s) & (d.target == 1)).sum()), "continuum": int(((d.split == s) & (d.target == 0)).sum())} for s in ("train", "val", "test")}
    rep["groups_crossing_splits"] = int((d.groupby("group").split.nunique() > 1).sum()); rep["n_groups"] = int(d.group.nunique())
    data.save_json(rep, MET / "data_audit.json")
    allr = raw[[cfg["data"]["index_column"], cfg["data"]["label_column"]]].rename(columns={cfg["data"]["index_column"]: "csv_index", cfg["data"]["label_column"]: "type"}); allr.insert(0, "row_id", np.arange(len(raw)))
    allr["in_primary"] = data.primary_mask(raw["type"], m).astype(int); allr["role"] = data.role_of(raw["type"], m)
    man = allr.merge(pd.DataFrame({"row_id": ev["row_id"], "split": split.to_numpy(), "group": groups, "target": y}), on="row_id", how="left")
    man["split"] = man["split"].fillna("excluded"); man.to_csv(MET / "split_assignments.csv", index=False)
    con.close()
    print(f"selected {exp['n_selected']} (signal {exp['n_signal']}, continuum {exp['n_continuum']}, type -1: {exp['n_type_minus1']}), excluded {exp['n_excluded']}; db ok={rep['database_check']['all_ok']}; splits={rep['split_class_counts']}; crossing={rep['groups_crossing_splits']}")


if __name__ == "__main__":
    t = time.time(); main(); print(f"{time.time()-t:.1f}s")
