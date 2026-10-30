from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG, MET, MOD, PRED, LOG = (ART / n for n in ("figures", "metrics", "models", "predictions", "logs"))
CFG_PATH = ROOT / "configs" / "experiment.json"
