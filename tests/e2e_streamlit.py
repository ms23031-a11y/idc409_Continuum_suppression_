"""End-to-end test of the Streamlit upload flow in a real browser (Playwright/Chromium) against a locally started app.
Run:  python tests/e2e_streamlit.py     (writes artifacts/logs/streamlit_e2e.json; exit 1 on failure)"""
import json, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src import data
from playwright.sync_api import sync_playwright

PORT = 8765


def main():
    cfg = data.load_config(); raw = data.load_raw(cfg).sample(200, random_state=1).reset_index(drop=True)
    tmp = Path(tempfile.mkdtemp()); good, miss, nan, txt = (tmp / n for n in ("good.csv", "missing.csv", "nan.csv", "text.csv"))
    raw.to_csv(good, index=False); raw.drop(columns=["R2"]).to_csv(miss, index=False)
    raw.assign(thrust=np.where(raw.index == 3, np.nan, raw.thrust)).to_csv(nan, index=False)
    raw.astype({"thrust": object}).assign(thrust=lambda d: d.thrust.where(d.index != 5, "abc")).to_csv(txt, index=False)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true", "--server.port", str(PORT), "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = {}
    try:
        for _ in range(60):
            try: urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1); break
            except Exception: time.sleep(1)
        with sync_playwright() as p:
            br = p.chromium.launch(); page = br.new_page()
            def upload(path, wait_text):
                page.goto(f"http://localhost:{PORT}"); page.wait_for_selector("input[type=file]", state="attached", timeout=30000)
                page.set_input_files("input[type=file]", str(path)); page.wait_for_selector(f"text={wait_text}", timeout=30000)
                return page.inner_text("body")
            body = upload(good, "Download predictions"); res["valid_upload_ok"] = "Scored 200 rows" in body
            res["valid_upload_label_text"] = ("instructor" in body.lower()) and ("held-out test performance" in body.lower())
            body = upload(miss, "Input is missing"); res["missing_column_rejected"] = "R2" in body and "missing" in body.lower() and "Download predictions" not in body
            if not res["missing_column_rejected"]: print("DEBUG body:", repr(body[:600]))
            body = upload(nan, "Non-numeric"); res["nan_rejected"] = "thrust" in body and "Download predictions" not in body
            body = upload(txt, "Non-numeric"); res["non_numeric_rejected"] = "thrust" in body and "Download predictions" not in body
            br.close()
    finally:
        proc.terminate()
    res["all_ok"] = all(v for v in res.values()); (ROOT / "artifacts/logs/streamlit_e2e.json").write_text(json.dumps(res, indent=2)); print(res); sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
