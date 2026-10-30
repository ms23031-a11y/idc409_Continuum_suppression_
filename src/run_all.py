"""One command to regenerate EVERYTHING from the raw CSV:  python -m src.run_all
   python -m src.run_all --docs-only   re-renders README tables, report, viva guide, notebook (and PDF) from saved results.
Stages: audit/SQLite/splits -> EDA -> training -> evaluation -> split repeats -> other-decays diagnostic -> feature dictionary
        -> verification -> test-isolation check -> pytest -> Streamlit browser test -> documents -> document verification."""
import json, re, subprocess, sys
from .paths import LOG, ROOT

ANALYSIS = ["src.prepare", "src.eda", "src.train", "src.evaluate", "src.repeats", "src.other_decays", "src.feature_dictionary"]
CHECKS = ["src.verify", "src.isolation"]


def sh(args, **kw):
    print(f"\n=== {' '.join(args)}", flush=True); return subprocess.run([sys.executable, *args], cwd=ROOT, **kw)


def main():
    docs_only = "--docs-only" in sys.argv
    if not docs_only:
        for s in ANALYSIS + CHECKS:
            if sh(["-m", s]).returncode: sys.exit(f"step failed: {s}")
    r = sh(["-m", "pytest", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True); print(r.stdout[-600:])
    m = lambda w: int((re.search(rf"(\d+) {w}", r.stdout) or [0, 0])[1])
    (LOG / "pytest_summary.json").write_text(json.dumps({"passed": m("passed"), "failed": m("failed"), "returncode": r.returncode}))
    if r.returncode: sys.exit("pytest failed")
    if not docs_only and sh(["tests/e2e_streamlit.py"]).returncode: sys.exit("Streamlit browser test failed")
    for s in ["src.docs", "src.verify_docs"]:
        if sh(["-m", s]).returncode: sys.exit(f"step failed: {s}")
    print("\nALL STAGES PASSED")


if __name__ == "__main__":
    main()
