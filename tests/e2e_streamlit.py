"""End-to-end test of the Streamlit upload flow in a real browser (Playwright/Chromium) against a locally started app.
Run:  python tests/e2e_streamlit.py     (writes artifacts/logs/streamlit_e2e.json; exit 1 on failure)"""
import json, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import data, predict
from playwright.sync_api import sync_playwright

PORT = 8765


def main():
    cfg = data.load_config(); idx, lab = cfg["data"]["index_column"], cfg["data"]["label_column"]
    raw = data.load_raw(cfg).sample(200, random_state=1).reset_index(drop=True); feats = [c for c in raw.columns if c not in (idx, lab)]
    tmp = Path(tempfile.mkdtemp()); good, withtype, miss, nan, txt = (tmp / n for n in ("good_features_only.csv", "with_type.csv", "missing.csv", "nan.csv", "text.csv"))
    raw[feats].to_csv(good, index=False)                    # NO type column, NO index column: inference must not need the true label
    raw.to_csv(withtype, index=False)                       # extra columns (type, index) must be ignored, not used
    raw[feats].drop(columns=["R2"]).to_csv(miss, index=False)
    raw[feats].assign(thrust=np.where(raw.index == 3, np.nan, raw.thrust)).to_csv(nan, index=False)
    raw[feats].astype({"thrust": object}).assign(thrust=lambda d: d.thrust.where(d.index != 5, "abc")).to_csv(txt, index=False)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true", "--server.port", str(PORT), "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = {}
    try:
        for _ in range(60):
            try: urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1); break
            except Exception: time.sleep(1)
        with sync_playwright() as p:
            br = p.chromium.launch(); page = br.new_page(accept_downloads=True)
            def upload(path, wait_text):
                page.goto(f"http://localhost:{PORT}"); page.wait_for_selector("input[type=file]", state="attached", timeout=30000)
                page.set_input_files("input[type=file]", str(path)); page.wait_for_selector(f"text={wait_text}", timeout=30000)
                return page.inner_text("body")
            body = upload(good, "Download predictions"); res["valid_upload_without_type_column_scored"] = "Scored 200 rows" in body
            res["app_states_mapping_and_heldout_performance"] = ("instructor" in body.lower()) and ("held-out test performance" in body.lower())
            with page.expect_download() as dl: page.get_by_role("button", name="Download predictions").click()
            out = pd.read_csv(dl.value.path()); bundle = predict.load_bundle(); ref = predict.predict_df(raw, bundle)
            res["download_ok"] = bool(len(out) == 200 and {"signal_score", "predicted_class"} <= set(out.columns) and "type" not in out.columns and np.allclose(out.signal_score.to_numpy(), ref.signal_score.to_numpy()))
            res["download_uses_frozen_threshold"] = bool((out.predicted_class.eq("signal").to_numpy() == (ref.signal_score.to_numpy() >= bundle["threshold"])).all())
            body = upload(withtype, "Download predictions"); res["valid_upload_with_extra_type_column_scored"] = "Scored 200 rows" in body
            body = upload(miss, "Input is missing"); res["missing_column_rejected"] = "R2" in body and "missing" in body.lower() and "Download predictions" not in body
            body = upload(nan, "Non-numeric"); res["nan_rejected"] = "thrust" in body and "Download predictions" not in body
            body = upload(txt, "Non-numeric"); res["non_numeric_rejected"] = "thrust" in body and "Download predictions" not in body
            br.close()
    finally:
        proc.terminate()
    res["all_ok"] = all(res.values()); (ROOT / "artifacts/logs/streamlit_e2e.json").write_text(json.dumps(res, indent=2)); print(res); sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
