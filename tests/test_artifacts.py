"""Integration tests on the real pipeline outputs (run after `python -m src.run_all`)."""
import json, subprocess, sys
import pytest
from src.paths import ROOT, LOG

pytestmark = pytest.mark.skipif(not (LOG / "training_log.json").exists(), reason="pipeline outputs not generated yet")


def test_verification_script_passes():
    r = subprocess.run([sys.executable, "-m", "src.verify"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.load(open(LOG / "verification.json"))["ALL_PASSED"]
