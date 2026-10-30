import pytest
from src.paths import MOD, ROOT
pytestmark = pytest.mark.skipif(not (MOD / "best_model.joblib").exists(), reason="no trained bundle")


def test_app_loads_and_shows_saved_heldout_performance():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("held-out test performance" in w.value.lower() for w in at.warning)
    assert any("threshold" in c.value for c in at.caption)
