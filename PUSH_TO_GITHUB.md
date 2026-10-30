# Publishing to GitHub

Status of this repository is stated in the delivery message. The files below are ready to push; nothing here claims a successful upload.

1. Create an empty repository on github.com (public or private; do not add a README there).
2. In this folder (already a git repository with one commit):
```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main        # authenticate in your own terminal/browser; never paste tokens into chat
git log -1 --format=%H         # commit identifier to quote
```
Tracked size is about 30 MB in 110 files (largest: random-forest bundles, ~5 MB each). The CSV (80 MB) and the generated SQLite database are **git-ignored**: permission to redistribute the dataset is not established. Readers place their own copy at `data/raw/data_hep.csv` and run `python -m src.run_all` (see README). If you are allowed to share the CSV, delete the `data/raw/*.csv` line from `.gitignore`.
