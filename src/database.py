"""SQLite layer: write raw data, read it back, modify it with SQL UPDATE, verify consistency.

Tables / views
  events_raw      original CSV rows (never modified)
  events          working copy of ALL rows + columns filled by SQL UPDATE: role, in_primary, target, split
  type_mapping    code -> role (instructor-provided), in_primary_experiment flag, n_rows (filled by SQL UPDATE), source
  primary_events  VIEW: exactly the rows of the signal-vs-continuum experiment (in_primary = 1)
  meta            provenance: CSV sha256, row counts, mapping status
"""
import sqlite3
import numpy as np
import pandas as pd
from . import data
from .paths import ROOT


def connect(cfg):
    p = ROOT / cfg["data"]["sqlite"]; p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


def build(cfg, raw, checksum):
    """WRITE: create all tables from the untouched CSV."""
    p = ROOT / cfg["data"]["sqlite"]
    if p.exists(): p.unlink()
    con = connect(cfg)
    r = raw.rename(columns={cfg["data"]["index_column"]: "csv_index"}); r.insert(0, "row_id", np.arange(len(r)))
    r.to_sql("events_raw", con, index=False)
    r.assign(role=None, in_primary=pd.array([None] * len(r), dtype="Int64"), target=pd.array([None] * len(r), dtype="Int64"), split=None).to_sql("events", con, index=False)
    con.execute("CREATE UNIQUE INDEX ix_raw ON events_raw(row_id)"); con.execute("CREATE UNIQUE INDEX ix_ev ON events(row_id)")
    m = cfg["label_mapping"]; rows = []
    for role, codes in (("signal", m["signal_codes"]), ("continuum", m["continuum_codes"]), ("other_decay", m["other_decay_codes"])):
        for c in codes: rows.append({"code": int(c), "role": None, "in_primary_experiment": None, "n_rows": None, "source": None})
    con.execute("CREATE TABLE type_mapping (code INTEGER PRIMARY KEY, role TEXT, in_primary_experiment INTEGER, n_rows INTEGER, source TEXT)")
    con.executemany("INSERT INTO type_mapping (code) VALUES (?)", [(r["code"],) for r in rows])
    pd.DataFrame({"key": ["csv_sha256", "n_rows", "mapping_status", "mapping_source"], "value": [checksum, str(len(raw)), m["status"], m["source"]]}).to_sql("meta", con, index=False)
    con.execute("CREATE VIEW primary_events AS SELECT * FROM events WHERE in_primary = 1")
    con.commit(); return con


def apply_mapping(con, cfg):
    """MODIFY: UPDATE type_mapping from the config, then derive events.role / in_primary / target / n_rows by SQL."""
    m = cfg["label_mapping"]
    for role, codes, prim in (("signal", m["signal_codes"], 1), ("continuum", m["continuum_codes"], 1), ("other_decay", m["other_decay_codes"], 0)):
        for c in codes:
            con.execute("UPDATE type_mapping SET role=?, in_primary_experiment=?, source=? WHERE code=?", (role, prim, "instructor-provided via student" + ("" if prim else "; excluded by analysis decision"), int(c)))
    con.execute("UPDATE events SET role = (SELECT role FROM type_mapping t WHERE t.code = events.type), in_primary = (SELECT in_primary_experiment FROM type_mapping t WHERE t.code = events.type)")
    con.execute("UPDATE events SET target = CASE WHEN in_primary = 1 AND role = 'signal' THEN 1 WHEN in_primary = 1 AND role = 'continuum' THEN 0 ELSE NULL END")
    con.execute("UPDATE type_mapping SET n_rows = (SELECT COUNT(*) FROM events e WHERE e.type = type_mapping.code)")
    con.commit()


def write_splits(con, split_series):
    """MODIFY: UPDATE events.split for primary rows from a Series indexed by row_id; excluded rows are marked 'excluded'."""
    con.executemany("UPDATE events SET split=? WHERE row_id=?", [(v, int(k)) for k, v in split_series.items()])
    con.execute("UPDATE events SET split='excluded' WHERE in_primary = 0"); con.commit()


def read_events(con, primary_only=True):
    """READ: the primary experiment rows (default) or all rows, ordered by row_id."""
    return pd.read_sql(f"SELECT * FROM {'primary_events' if primary_only else 'events'} ORDER BY row_id", con)


def verify(con, raw, cfg):
    """Consistency checks between SQLite, the CSV and the SQL UPDATEs."""
    lab, idx, m = cfg["data"]["label_column"], cfg["data"]["index_column"], cfg["label_mapping"]
    feats = [c for c in raw.columns if c not in (idx, lab)]
    rr = pd.read_sql("SELECT * FROM events_raw ORDER BY row_id", con); ev = read_events(con, primary_only=False); pe = read_events(con)
    pm = data.primary_mask(raw[lab], m)
    out = {
        "raw_table_rowcount_matches_csv": len(rr) == len(raw),
        "raw_table_values_identical_to_csv": bool(np.array_equal(rr[feats].to_numpy(), raw[feats].to_numpy())),
        "raw_type_and_index_identical": bool((rr["type"].to_numpy() == raw[lab].to_numpy()).all() and (rr["csv_index"].to_numpy() == raw[idx].to_numpy()).all()),
        "events_features_equal_raw": bool(np.array_equal(ev[feats].to_numpy(), rr[feats].to_numpy())),
        "primary_view_rows_equal_pandas_subset": bool(len(pe) == int(pm.sum()) and (pe["row_id"].to_numpy() == np.where(pm)[0]).all()),
        "primary_view_only_intended_codes": bool(set(pe["type"].unique()) <= set(m["signal_codes"] + m["continuum_codes"])),
        "target_matches_pandas_mapping_on_primary": bool((pe["target"].to_numpy() == data.make_target(raw.loc[pm, lab], m)).all()),
        "target_null_for_excluded_rows": bool(ev.loc[~pm.astype(bool), "target"].isna().all()),
        "primary_rows_have_target_and_split": bool(pe["target"].notna().all() and pe["split"].isin(["train", "val", "test"]).all()),
        "excluded_rows_marked_excluded": bool((ev.loc[~pm.astype(bool), "split"] == "excluded").all()),
        "type_mapping_counts_match_csv": bool(all(r[1] == int((raw[lab] == r[0]).sum()) for r in con.execute("SELECT code, n_rows FROM type_mapping"))),
        "no_type_minus1_rows_created": int((ev["type"] == -1).sum()) == int((raw[lab] == -1).sum()),
        "sql_count_by_split": {r[0]: r[1] for r in con.execute("SELECT split, COUNT(*) FROM events GROUP BY split")},
        "sql_count_by_role": {r[0]: r[1] for r in con.execute("SELECT role, COUNT(*) FROM events GROUP BY role")},
        "mapping_rows": con.execute("SELECT code, role, in_primary_experiment, n_rows FROM type_mapping ORDER BY code").fetchall(),
    }
    out["all_ok"] = all(v for k, v in out.items() if isinstance(v, bool)); return out
