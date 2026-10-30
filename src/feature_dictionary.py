"""Writes reports/feature_dictionary.md from the actual CSV columns."""
import pandas as pd
from . import data
from .paths import ROOT, MET

DESC = {
 "R2": "Ratio H2/H0 of Fox-Wolfram moments from the continuum-suppression (CS) module; near 0 for isotropic (B-pair-like) events, larger for jet-like events.",
 "thrustBm": "Thrust magnitude of the B candidate (CS module).", "thrustOm": "Thrust magnitude of the rest of event, ROE (CS module).",
 "cosTBTO": "Cosine of the angle between the thrust axes of the B candidate and the ROE; expected uniform for B-pair events and peaked near 1 for jet-like events.",
 "cosTBz": "Cosine of the angle between the B-candidate thrust axis and the beam (z) axis.",
 "thrust": "Event thrust (EventShape framework, whole event).", "thrustAxisCosTheta": "Cosine of the polar angle of the event thrust axis (EventShape framework).",
 "foxWolframR1": "Normalised Fox-Wolfram moment R1 = H1/H0 (EventShape framework).", "foxWolframR2": "R2 from the EventShape framework (same idea as `R2`, computed differently).",
 "foxWolframR3": "Normalised Fox-Wolfram moment R3.", "foxWolframR4": "Normalised Fox-Wolfram moment R4.",
 "KSFWVariables__boet__bc": "KSFW `et`: a transverse-energy-type term of the Kakuno-Super-Fox-Wolfram set.", "KSFWVariables__bomm2__bc": "KSFW `mm2`: missing-mass-squared term of the KSFW set.",
}
FAM = {
 "CLEO cones": "CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS).",
 "CLEO cones (cmROE)": "CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`).",
 "KSFW moments": "KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW).",
 "KSFW moments (cmFS1)": "Same KSFW moment computed from the B final-state particles (`FS1` in basf2).",
}


def main():
    cfg = data.load_config(); raw = data.load_raw(cfg); feats = data.feature_columns(raw, cfg)
    s = pd.read_csv(MET / "univariate_separation_train.csv", index_col=0)
    L = ["# Feature dictionary", "",
         "Sources: Belle II software documentation (software.belle2.org: *Continuum suppression* page and the online-book lesson *Continuum Suppression (CS)*) and the variable names in the CSV. "
         "Definitions marked *(naming-based)* were inferred from the basf2 variable names and were **not** independently verified against the code; precise definitions are in Chapter 9 of *The Physics of the B Factories* (arXiv:1406.6311), which I did not reproduce here. "
         "The suffix `__bc` on most column names comes from the data export; its meaning is not documented in the supplied material (*unverified*).", "",
         "Non-feature columns: the unnamed first column (index-like; repeats values and its range depends on the label; origin undocumented; excluded as a possible label leak) and `type` (code 0-5, the label source).", "",
         "| # | Column | Family | Meaning | Train single-feature separation (AUC-0.5, abs) |", "|---|---|---|---|---|"]
    for i, f in enumerate(feats, 1):
        fam = data.feature_family(f); d = DESC.get(f) or (FAM.get(fam, "") + " *(naming-based)*")
        L.append(f"| {i} | `{f}` | {fam} | {d} | {s.loc[f, 'abs_sep']:.3f} |")
    (ROOT / "reports/feature_dictionary.md").write_text("\n".join(L) + "\n"); print("written", len(feats))


if __name__ == "__main__":
    main()
