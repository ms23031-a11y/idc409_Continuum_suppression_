import json, pytest
from src import data, narrative
from src.paths import LOG
pytestmark = pytest.mark.skipif(not (LOG / "training_log.json").exists(), reason="no results")


def test_fresh_results_pass_and_changed_mapping_is_flagged():
    cfg = data.load_config(); narrative.check_fresh(cfg)                      # current results match current config
    cfg2 = json.loads(json.dumps(cfg)); cfg2["label_mapping"]["continuum_codes"] = [0]
    with pytest.raises(RuntimeError, match="STALE RESULTS"): narrative.check_fresh(cfg2)


def test_student_details_do_not_change_fingerprint():
    cfg = data.load_config(); cfg2 = json.loads(json.dumps(cfg)); cfg2["student"]["name"] = "X"
    assert data.config_fingerprint(cfg) == data.config_fingerprint(cfg2)
