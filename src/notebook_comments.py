"""Plain-language explanations for the notebook: a glossary, a short 'in simple words' note per section, and comments inside code cells.
Applied by src.notebook.build() so they survive every regeneration. Words only - no result numbers, so nothing here can go stale."""

GLOSSARY = """## Words used in this notebook (in simple words)

| Word | What it means here |
|---|---|
| **Signal** | The events we want to keep (type +1). |
| **Continuum** | Background events we want to throw away (type 0 or -1). |
| **Other decays** | Types 2-5. They are neither signal nor continuum, so I left them out of the main experiment (my own choice, not the instructor's order). |
| **Feature** | One measured number describing an event. There are 59 of them. |
| **Training / validation / test** | Three separate piles of events. *Training*: the model learns from it. *Validation*: we use it to pick settings. *Test*: kept locked away and used only once, for the final score. |
| **Model (classifier)** | A program that learns to guess "signal" or "continuum" from the 59 numbers. |
| **Score and threshold** | The model gives every event a score between 0 and 1 (higher = looks more like signal). The threshold is the cut: above it we say "signal", below it "continuum". |
| **Signal efficiency** | Of all the *true signal* events, the fraction we kept. |
| **Continuum rejection** | Of all the *true continuum* events, the fraction we threw away. This is the number we really want to be high. |
| **Continuum acceptance** | The opposite: the fraction of true continuum we wrongly kept (= 1 - rejection). |
| **Accuracy** | The fraction of all guesses that are right. It is misleading here because almost all events are signal, so "always say signal" already looks great. |
| **Balanced accuracy** | The average of signal efficiency and continuum rejection, so both groups count equally. |
| **ROC-AUC** | How well the scores put signal above continuum. 0.5 = no better than a coin flip, 1.0 = perfect. |
| **Confidence interval** | A range showing how unsure a number is. With very few continuum events the ranges are wide. |
| **Class weight** | Tells the model to treat each rare continuum event as more important while learning. Used on the training pile only. |
| **Operating point** | The one threshold I picked on the validation pile: throw away as much continuum as possible while keeping at least the chosen share of signal. |
"""

SECTION_NOTES = {
    "## 1. Data loading": "**In simple words:** we open the original data file (we never change it), count how many events of each type there are, keep only signal and continuum, and run basic health checks (missing numbers, duplicates, odd values).",
    "## 2. SQLite database": "**In simple words:** a database is like a very organised set of tables on disk. We stored the raw data in it, wrote the labels and the train/validation/test pile into it with commands, read them back, and checked that everything matches the original file.",
    "## 3. Exploratory data analysis": "**In simple words:** before any machine learning, we simply *look* at the data. The question for every picture is: can I already see a difference between signal and continuum? Only the training pile is used to make decisions.",
    "## 4. Train / validation / test split": "**In simple words:** we cut the events into three piles so we can test honestly. The model learns from the training pile, we choose settings on the validation pile, and the test pile is used just once at the end, like a final exam.",
    "## 5. Models and tuning": "**In simple words:** we try two very simple baselines and four real models. Each model has a few settings (for example how deep a tree may grow). We try a limited number of settings and keep the best one **on the validation pile**, never on the test pile.",
    "## 6. Evaluation on the held-out test set": "**In simple words:** this is the final exam. We score the test pile once and report how many signal events were kept and how many continuum events were thrown away. Because the test pile has only a few dozen continuum events, every number comes with a range showing how uncertain it is.",
    "## 7. Reproducing predictions": "**In simple words:** a result only counts if it can be repeated. We load the saved model from disk and check it gives exactly the same scores again. We also check that obviously bad input (a missing column, infinity, text) is refused with a clear message.",
    "## 8. Feature reduction": "**In simple words:** 59 numbers may be more than needed, and many are almost copies of each other. We try to keep fewer numbers and see whether the models still work about as well.",
    "## 9. Pipeline checks": "**In simple words:** the results are close to a coin flip, so the first question is whether I made a mistake. These checks try to catch mistakes: shuffled labels should give a coin flip, an easy task should give a high score, and the numbers should be recomputable in a second way.",
    "## 10. Conclusions": "**In simple words:** below is the honest summary. The models cannot separate signal from continuum with these features, and I do not claim otherwise.",
}

# (text that identifies the cell, comment lines to put on top, [(text of a line, short comment to add at its end), ...])
CODE = [
 ("# --- Setup: locate the project root",
  ["# WHAT THIS CELL DOES (simple words):", "# The notebook must know where the project folder is, so it can find our code (src), the data and the saved results.",
   "# It looks in the current folder and the folders above it until it finds the project. You can run this cell as many times as you like.",
   "# If Jupyter was started somewhere else, set the CONTINUUM_ROOT variable to the project folder first."],
  [("PROJECT_ROOT = find_project_root()", "# find the project folder"), ("sys.path.insert(0, str(PROJECT_ROOT))", "# lets Python import our own code from src/"),
   ("os.chdir(PROJECT_ROOT)", "# always work from the project folder (so re-running is safe)"), ("import numpy as np, pandas as pd", "# standard tools for numbers and tables"),
   ("from src import data, database, predict", "# our own helper code"), ('print("project root:"', "# show where we are")]),
 ("cfg = data.load_config(); show = lambda",
  ["# Load the settings file and the training log.", "# Then make sure the saved results were made with these same settings. If someone changed the settings but did not re-run the training,",
   "# the 'assert' line stops the notebook so we never show old numbers by mistake."],
  [("cfg = data.load_config()", "# read configs/experiment.json (seed, which code means what, split sizes, ...)"), ("tlog = json.load", "# the log written during training"),
   ("assert tlog", "# stop with a message if results and settings do not match"), ("print(json.dumps(cfg['label_mapping']", "# show which codes are signal, continuum and other decays")]),
 ("raw = data.load_raw(cfg); audit",
  ["# Load the ORIGINAL data file (we never change it) and count the events.", "# 'selected' keeps only type 1, 0 and -1. 'y' is 1 for signal and 0 for continuum.",
   "# Note: the file has no type -1 events at all. We did not make any up."],
  [("raw = data.load_raw(cfg)", "# the original table, untouched"), ("selected = raw.loc[", "# keep only signal and continuum rows"), ("y = (selected['type'] == 1)", "# 1 = signal, 0 = continuum"),
   ("print('majority", "# what a lazy 'always say signal' rule would score"), ("for k in ['missing_total'", "# data health checks: missing numbers, duplicates, ..."),
   ("unnamed index-like column", "# the unnamed first column is not a real event ID, so it is NOT used as a model input")]),
 ("con = database.connect(cfg)",
  ["# Look inside the SQLite database. It stores the raw table and a working table.", "# The labels (target) and the train/validation/test pile (split) were written into it with SQL UPDATE commands.",
   "# Here we READ them back and check that they match what pandas says. Then we show a small MODIFY example on a throw-away copy."],
  [("con = database.connect(cfg)", "# open the database file"), ("FROM type_mapping", "# the table that says which code means what"), ("GROUP BY split, role", "# how many events are in each pile"),
   ("primary_events", "# the 'view' = exactly the rows used for modelling"), ("database.verify(con", "# many automatic checks: database equals CSV, labels are right, ..."),
   ("CREATE TEMP TABLE scratch", "# MODIFY demo: we change a throw-away copy, never the real table"), ("UPDATE scratch", "# flip the label of the type-0 rows in the copy"),
   ("print('scratch UPDATE", "# how many rows changed (should equal the number of type-0 rows)")]),
 ("show('01_label_balance.png'", ["# Picture 1: how many events of each type, and how unbalanced signal vs continuum is."], []),
 ("show('02_univariate_separation.png'", ["# Picture 2: which single number separates signal from continuum best (0.5 = useless)."], []),
 ("show('03_event_shape_distributions.png'", ["# Picture 3: for 12 key numbers, signal (blue) and continuum (orange). If the two curves sit on top of each other, that number cannot tell them apart."], []),
 ("show('04_distributions_by_type_code.png'", ["# Picture 4: the same kind of numbers for every original type code. Grey = other decays (not used in the experiment)."], []),
 ("show('05_correlation_heatmap.png'", ["# Picture 5: which numbers are almost copies of each other (dark red/blue = strongly related). Then a table of the most similar pairs."], []),
 ("show('06_cleo_cone_profiles.png'", ["# Pictures 6 and 7: the average CLEO-cone numbers for signal and continuum, and a summary of data-quality checks."], []),
 ("sp = pd.read_csv(MET/'split_assignments.csv')",
  ["# Show how the events were divided into train / validation / test and check the rules were followed:", "#  - every pile has about the same share of continuum (this is called 'stratified')",
   "#  - no event of the excluded types 2-5 is in any pile", "#  - identical-looking rows (duplicates) would stay in the same pile (there are none)"],
  [("sp = pd.read_csv", "# the list saying which pile every row is in"), ("pd.crosstab([sp.in_primary", "# counts of rows by type/role and by pile"), ("g = sp[sp.in_primary == 1]", "# keep only rows used in the experiment"),
   ("print(pd.crosstab(g.split", "# signal / continuum counts per pile"), ("groups spanning", "# both should be 0: no group of identical rows split across piles, no excluded row in any pile")]),
 ("rows=[]\nfor k, v in tlog['models']",
  ["# Show what each model chose during tuning on the VALIDATION pile (never the test pile).", "# 'features used' = how many of the 59 numbers the model really uses (the single-cut baseline uses only 1).",
   "# 'configs tried' = how many settings were compared. The second table compares learning with and without class weights."],
  [("rows=[]", "# we collect one line per model"), ("for k, v in tlog['models'].items()", "# go through every trained model"), ("if k.endswith('|full')", "# only the models that use all 59 numbers"),
   ("print('Best by VALIDATION AUC", "# the model picked on the validation pile"), ("print('single-feature cut uses", "# the one number the simple cut looks at")]),
 ("R = pd.read_csv(MET/'results_full.csv'); n_sig",
  ["# The main result tables on the TEST pile.", "# Table A: accuracy, precision, recall, F1 and ROC-AUC (the course requirement).", "# Table B: signal efficiency and continuum rejection (what physicists care about), with 95% ranges.",
   "# Table C: the same models using the plain cut 0.5.   Table D: precision, recall and F1 for each class separately."],
  [("n_sig, n_con =", "# how many true signal / continuum events are in the test pile")]),
 ("display(pd.read_csv(MET/'paired_auc_vs_best.csv')",
  ["# Table of AUC differences between the chosen model and the others. If the range includes 0, the difference could be just luck.", "# Then the ROC / precision-recall curves and a bar chart comparing the models."], []),
 ("show('11_confusion_matrices.png'", ["# Confusion matrices: for each model, how many true signal / continuum events were called signal or continuum. Top row = plain cut 0.5, bottom row = our chosen cut."], []),
 ("show('12_score_distribution.png'", ["# Left: the scores the chosen model gave to signal and continuum events. Right: the validation curves used to pick the threshold."], []),
 ("b = tlog['best_by_validation_auc']; bm",
  ["# One-line summary of the chosen model, plus a side check on the other-decay types 2-5 (they were never used for training or testing)."],
  [("b = tlog[", "# name of the model chosen on the validation pile"), ("bm = R.set_index", "# its row in the results table")]),
 ("bundle = predict.load_bundle(",
  ["# Prove that the saved model gives the same answers again.", "# We load the saved model file, score the test events, and compare with the scores saved earlier (difference should be about 0).",
   "# Then we try three kinds of bad input (missing column, infinity, text) and check the code refuses them with a clear message."],
  [("bundle = predict.load_bundle", "# load the saved model, its 59 input names and its frozen threshold"), ("ev = database.read_events", "# read the modelling rows from the database"),
   ("out = predict.predict_df(", "# score them again with the saved model"), ("print('max |recomputed", "# should be about 0 (only tiny rounding)"), ("for name, bad in", "# three kinds of bad input"),
   ("try: predict.predict_df(bad", "# the code must refuse it")]),
 ("fr = tlog['feature_reduction']",
  ["# Feature reduction: how many numbers were kept at each step, and which ones survived.", "# Step 1 drops numbers that are almost copies of another. Step 2 ranks the rest by how much the model needs them (on validation data).",
   "# Step 3 keeps the smallest set that works about as well."], []),
 ("display(pd.read_csv(MET/'full_vs_reduced_table.csv'))", ["# Compare the models using all 59 numbers, the set left after step 1, and the short list - same rules each time."], []),
 ("sa = json.load(open(MET / 'sanity_audit.json'))",
  ["# Is the weak result caused by a bug? These checks do NOT use the test pile for any new learning:", "#  1) labels and the 'signal probability' column are correct", "#  2) every reported number can be recomputed in a second, independent way",
   "#  3) shuffled labels should give about 0.5 (nothing to learn)", "#  4) an easy task should give a high score (so the pipeline can find a difference when there is one)",
   "#  5) cross-validation on train+validation, which has more continuum events than the test pile alone."], []),
 ("display(pd.read_csv(MET/'seed_repeats_summary.csv'))", ["# How much the results change when the data is split in other ways (5 different splits). Big changes = a small test pile gives noisy answers."], []),
]


def annotate_source(src, header, inline):
    lines = src.split("\n")
    out = []
    for ln in lines:
        if "#" not in ln:
            for key, note in inline:
                if key in ln:
                    ln = ln + "  " + note; break
        out.append(ln)
    return "\n".join(list(header) + [""] + out) if header else "\n".join(out)


def apply(cells, nbf):
    """cells: list of nbformat cells (modified copy returned)."""
    res = []
    for c in cells:
        if c.cell_type == "code":
            for marker, header, inline in CODE:
                if c.source.lstrip().startswith(marker) or (marker.startswith("rows=[]") and c.source.startswith("rows=[]")):
                    c = nbf.v4.new_code_cell(annotate_source(c.source, header, inline)); break
        res.append(c)
        if c.cell_type == "markdown":
            for key, note in SECTION_NOTES.items():
                if c.source.startswith(key): res.append(nbf.v4.new_markdown_cell(note)); break
    res.insert(1, nbf.v4.new_markdown_cell(GLOSSARY))        # right after the title cell
    return res
