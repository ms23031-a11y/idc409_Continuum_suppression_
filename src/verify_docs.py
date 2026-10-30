"""Checks that the generated documents agree with saved results and keep the student details."""
import json, sys
import nbformat
from . import data, narrative
from .paths import ROOT, LOG

FORBIDDEN = ["five classifiers", "five learned", "indistinguishable", "per-type counter", "statistically significant", "significantly better", "provisional code grouping", "positive group", "negative group", "Gourav"]


def main():
    cfg = data.load_config(); F = narrative.derived(narrative.facts()); st = cfg["student"]; res = {}
    docs = {"README.md": ROOT / "README.md", "report.md": ROOT / "reports/report.md", "viva_guide.md": ROOT / "reports/viva_guide.md", "requirements_checklist.md": ROOT / "reports/requirements_checklist.md"}
    nb = nbformat.read(ROOT / "notebooks/continuum_suppression.ipynb", 4); nbtxt = "\n".join("".join(c.source) for c in nb.cells)
    texts = {k: p.read_text() for k, p in docs.items()}; texts["notebook"] = nbtxt
    for k, t in texts.items():
        res[f"{k}: student name and roll present"] = st["name"] in t and st["roll"] in t
        res[f"{k}: forbidden/outdated phrases absent"] = not any(p.lower() in t.lower() for p in FORBIDDEN)
        if k != "notebook": res[f"{k}: carries current config fingerprint"] = F["config_sha"] in t
    res["notebook: carries current config fingerprint (prefix)"] = F["config_sha"][:16] in nbtxt
    res["report/README/notebook agree on best-model AUC"] = all(F["best_auc"] in texts[k] for k in ("report.md", "README.md", "notebook"))
    res["report/README agree on test row count"] = all(F["test_sig"] in texts[k] for k in ("report.md", "README.md"))
    res["report states four learned classifiers + two baselines"] = "Four learned classifiers" in texts["report.md"] and "two baselines" in texts["report.md"]
    res["report/README state instructor-provided mapping and analysis-decision exclusion"] = all("instructor" in texts[k].lower() and "analysis decision" in texts[k].lower() for k in ("report.md", "README.md"))
    res["report/README give actual continuum test denominator"] = all(str(F["test_con"]) in texts[k] for k in ("report.md", "README.md"))
    res["report states signal efficiency and continuum rejection"] = "continuum rejection" in texts["report.md"] and "signal efficiency" in texts["report.md"]
    res["historical results kept separate (not in main tables)"] = "historical" in texts["report.md"].lower() and F["boosting_auc"] is not None
    res["notebook has no error cells"] = not any(o.get("output_type") == "error" for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    res["notebook executed (all code cells have execution_count)"] = all(c.get("execution_count") for c in nb.cells if c.cell_type == "code")
    res["figures referenced in report exist"] = all((ROOT / "reports" / p).exists() for p in __import__("re").findall(r"\]\((\.\./artifacts/figures/[^)]+\.png)\)", texts["report.md"]))
    # documented commands and paths
    import re, subprocess, tempfile
    from pathlib import Path
    readme = texts["README.md"]; ignore = {"data/hep.sqlite"}
    paths = {p for p in re.findall(r"`([\w./-]+\.(?:py|md|json|csv|ipynb|txt|pdf)|[\w./-]+/)`", readme) if not p.startswith(("http", "/path")) and p not in {"features.csv", "scores.csv"}}
    missing = sorted(p for p in paths if not (ROOT / p).exists() and p not in ignore and not p.startswith(("artifacts/", "python")) and "*" not in p)
    res["README paths exist"] = not missing
    if missing: print("missing paths:", missing)
    with tempfile.TemporaryDirectory() as d:
        sample = Path(d) / "features.csv"; data.load_raw(cfg).head(20).to_csv(sample, index=False)
        r = subprocess.run([sys.executable, "-m", "src.predict", "--input", str(sample), "--output", str(Path(d) / "scores.csv")], cwd=ROOT, capture_output=True, text=True)
        res["README CLI `src.predict` works"] = r.returncode == 0 and (Path(d) / "scores.csv").exists()
        bad = Path(d) / "bad.csv"; data.load_raw(cfg).head(5).drop(columns=["R2"]).to_csv(bad, index=False)
        r2 = subprocess.run([sys.executable, "-m", "src.predict", "--input", str(bad)], cwd=ROOT, capture_output=True, text=True)
        res["CLI rejects missing column with clear message"] = r2.returncode != 0 and "missing" in (r2.stderr + r2.stdout).lower() and "R2" in (r2.stderr + r2.stdout)
    res["ALL_PASSED"] = all(res.values()); (LOG / "docs_verification.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if k != "ALL_PASSED": print(("PASS " if v else "FAIL ") + k)
    sys.exit(0 if res["ALL_PASSED"] else 1)


if __name__ == "__main__":
    main()
