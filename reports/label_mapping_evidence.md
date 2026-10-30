# Evidence search for the meaning of `type` codes 0-5

**Status: UNRESOLVED.** No source linking this dataset's numeric codes to physical categories was found. The classification in this project is therefore a *code-group classification* (provisional grouping `type == 1` versus all other codes), **not** a verified continuum-suppression result. No signal efficiency or continuum rejection is reported.

## What was searched (30 Oct 2026)

| Source | Result |
|---|---|
| `IDC409_Project_Topics_2026.pdf`, pages 1-2 (Project 1) | Says only "Signal" and "Background" events, "Provided Belle II sample", "simulated/analysis-level Belle II MC dataset". No code definitions. Project 2 (page 2) lists five labels (uubar, ddbar, ssbar, ccbar, bbbar) for what it calls "Belle II MC data with labels", but gives no numeric encoding and the file has six codes. |
| Task notes supplied with the request | State only that the `type` mapping was not established. |
| `data_hep.csv` itself | Columns: unnamed index, 59 features, `type`. No header comments, no metadata, no generation script. |
| Public student repository for this course project (`ms24149/belle-ii-continuum-suppression`) | README and an empty notebook only; no data folder, no scripts, no mapping. |
| Belle II software documentation / training pages (software.belle2.org, training.belle2.org) | Describe the physics variables and say Belle II MC continuum samples are named uubar, ddbar, ssbar, ccbar; the `isContinuumEvent` truth variable is mentioned. They do **not** describe this file's custom numeric `type` encoding. |
| Web search for the column names / dataset | No public source for this specific file found. |

## What can and cannot be said

* Observation (not evidence of mapping): the file has six codes, and Belle II generic MC has six standard categories (charged B+B-, mixed B0B0bar, uubar, ddbar, ssbar, ccbar). The count matches, but the numbering and which codes would be "signal" is **not** documented anywhere I could access. The feature-distribution pattern (codes 0 and 1 similar, code 2 different, codes 3-5 similar) is *consistent with* many assignments and is not used to choose one.
* Code 0 is **not** assumed to be signal because it resembles code 1.
* No grouping is chosen for giving a higher AUC.

## Missing information and the single question for the dataset provider

> **For each value of the `type` column (0, 1, 2, 3, 4, 5), which physical event category does it denote (for example: charged BB̄, mixed BB̄, uū, dd̄, ss̄, cc̄, or something else), which of these does the project define as "signal" and which as "continuum", and should any category be excluded?**

A secondary detail that would help but is not required: whether the first (unnamed) column is a running index from separate per-category source files.

## What changes when the answer arrives

Edit `label_mapping` in `configs/experiment.json` (`positive_codes`, `negative_codes`, names, status, evidence), then run `python -m src.run_all`. Models are retrained from scratch, selection and evaluation repeat, and the report, README tables and notebook are regenerated; a stale-output guard (config fingerprint) fails the build if any dependent output was produced under a different configuration. Signal efficiency and continuum rejection would then be reported with the verified definitions.
