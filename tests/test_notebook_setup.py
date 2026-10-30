"""The notebook's setup cell must find the project root from any start directory and be safe to re-run."""
import subprocess, sys, tempfile
from pathlib import Path
import pytest
from src.notebook import SETUP_CELL
from src.paths import ROOT


def run(code, cwd, env_extra=None):
    import os
    env = {**os.environ, **(env_extra or {})}; env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-I", "-c", code], cwd=cwd, capture_output=True, text=True, env=env)


PROBE = SETUP_CELL + "\nimport os, sys\nprint('CWD', os.getcwd()); print('COUNT', sys.path.count(str(PROJECT_ROOT)))\n"


@pytest.mark.parametrize("start", [ROOT, ROOT / "notebooks", ROOT / "tests", ROOT / "artifacts" / "figures", ROOT.parent])
def test_root_found_from_any_start_directory_and_cwd_is_stable(start):
    # sys.path has no project entry under -I, so the cell itself must add it; run the cell twice in one process (re-run safety)
    r = run(PROBE + "\n" + SETUP_CELL + "\nprint('CWD2', os.getcwd()); print('COUNT2', sys.path.count(str(PROJECT_ROOT)))", start)
    assert r.returncode == 0, r.stderr[-800:]
    out = dict(l.split(" ", 1) for l in r.stdout.splitlines() if l.split(" ", 1)[0] in ("CWD", "COUNT", "CWD2", "COUNT2"))
    assert Path(out["CWD"]).resolve() == ROOT.resolve() and out["CWD"] == out["CWD2"]          # no repeated directory moves
    assert out["COUNT"] == "1" and out["COUNT2"] == "1"                                           # no duplicate sys.path entries


def test_clear_error_outside_project_and_env_override():
    with tempfile.TemporaryDirectory() as d:
        r = run("import os\nos.chdir(r'%s')\n" % d + SETUP_CELL, d)
        # /tmp has no project below or above it
        assert r.returncode != 0 and "Project root not found" in r.stderr and "CONTINUUM_ROOT" in r.stderr
        r2 = run(SETUP_CELL + "\nprint('OK', PROJECT_ROOT)", d, {"CONTINUUM_ROOT": str(ROOT)})
        assert r2.returncode == 0 and "OK" in r2.stdout
