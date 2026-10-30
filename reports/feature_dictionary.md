# Feature dictionary

Sources: Belle II software documentation (software.belle2.org: *Continuum suppression* page and the online-book lesson *Continuum Suppression (CS)*) and the variable names in the CSV. Definitions marked *(naming-based)* were inferred from the basf2 variable names and were **not** independently verified against the code; precise definitions are in Chapter 9 of *The Physics of the B Factories* (arXiv:1406.6311), which I did not reproduce here. The suffix `__bc` on most column names comes from the data export; its meaning is not documented in the supplied material (*unverified*).

Non-feature columns: the unnamed first column (index-like; repeats values and its range depends on the label; origin undocumented; excluded as a possible label leak) and `type` (code 0-5, the label source).

| # | Column | Family | Meaning | Train single-feature separation (AUC-0.5, abs) |
|---|---|---|---|---|
| 1 | `R2` | Fox-Wolfram | Ratio H2/H0 of Fox-Wolfram moments from the continuum-suppression (CS) module; near 0 for isotropic (B-pair-like) events, larger for jet-like events. | 0.058 |
| 2 | `thrustBm` | Thrust / angular | Thrust magnitude of the B candidate (CS module). | 0.055 |
| 3 | `thrustOm` | Thrust / angular | Thrust magnitude of the rest of event, ROE (CS module). | 0.062 |
| 4 | `cosTBTO` | Thrust / angular | Cosine of the angle between the thrust axes of the B candidate and the ROE; expected uniform for B-pair events and peaked near 1 for jet-like events. | 0.047 |
| 5 | `cosTBz` | Thrust / angular | Cosine of the angle between the B-candidate thrust axis and the beam (z) axis. | 0.036 |
| 6 | `KSFWVariables__boet__bc` | KSFW (et, mm2) | KSFW `et`: a transverse-energy-type term of the Kakuno-Super-Fox-Wolfram set. | 0.038 |
| 7 | `KSFWVariables__bomm2__bc` | KSFW (et, mm2) | KSFW `mm2`: missing-mass-squared term of the KSFW set. | 0.011 |
| 8 | `CleoConeCS__bo1__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.006 |
| 9 | `CleoConeCS__bo2__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.041 |
| 10 | `CleoConeCS__bo3__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.033 |
| 11 | `CleoConeCS__bo4__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.001 |
| 12 | `CleoConeCS__bo5__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.002 |
| 13 | `CleoConeCS__bo6__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.008 |
| 14 | `CleoConeCS__bo7__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.001 |
| 15 | `CleoConeCS__bo8__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.018 |
| 16 | `CleoConeCS__bo9__bc` | CLEO cones | CLEO cone `i` (i=1..9): momentum flow in the i-th angular cone around the thrust axis, from all final-state particles (CleoConeCS). *(naming-based)* | 0.045 |
| 17 | `CleoConeCS__bo1__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.053 |
| 18 | `CleoConeCS__bo2__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.025 |
| 19 | `CleoConeCS__bo3__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.030 |
| 20 | `CleoConeCS__bo4__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.005 |
| 21 | `CleoConeCS__bo5__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.005 |
| 22 | `CleoConeCS__bo6__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.016 |
| 23 | `CleoConeCS__bo7__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.001 |
| 24 | `CleoConeCS__bo8__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.027 |
| 25 | `CleoConeCS__bo9__cmROE__bc` | CLEO cones (cmROE) | CLEO cone `i` built only from rest-of-event (ROE) particles (`cmROE`). *(naming-based)* | 0.043 |
| 26 | `KSFWVariables__bohso00__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.005 |
| 27 | `KSFWVariables__bohso02__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.057 |
| 28 | `KSFWVariables__bohso04__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.044 |
| 29 | `KSFWVariables__bohso10__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.020 |
| 30 | `KSFWVariables__bohso12__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.032 |
| 31 | `KSFWVariables__bohso14__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.009 |
| 32 | `KSFWVariables__bohso20__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.044 |
| 33 | `KSFWVariables__bohso22__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.029 |
| 34 | `KSFWVariables__bohso24__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.016 |
| 35 | `KSFWVariables__bohoo0__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.003 |
| 36 | `KSFWVariables__bohoo1__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.032 |
| 37 | `KSFWVariables__bohoo2__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.050 |
| 38 | `KSFWVariables__bohoo3__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.035 |
| 39 | `KSFWVariables__bohoo4__bc` | KSFW moments | KSFW moment `hso/hoo` + indices (Kakuno et al.), computed from the B primary daughters (default KSFW). *(naming-based)* | 0.022 |
| 40 | `KSFWVariables__bohso00__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.005 |
| 41 | `KSFWVariables__bohso02__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.047 |
| 42 | `KSFWVariables__bohso04__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.039 |
| 43 | `KSFWVariables__bohso10__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.020 |
| 44 | `KSFWVariables__bohso12__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.033 |
| 45 | `KSFWVariables__bohso14__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.047 |
| 46 | `KSFWVariables__bohso20__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.044 |
| 47 | `KSFWVariables__bohso22__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.023 |
| 48 | `KSFWVariables__bohso24__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.017 |
| 49 | `KSFWVariables__bohoo0__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.003 |
| 50 | `KSFWVariables__bohoo1__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.032 |
| 51 | `KSFWVariables__bohoo2__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.050 |
| 52 | `KSFWVariables__bohoo3__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.035 |
| 53 | `KSFWVariables__bohoo4__cmFS1__bc` | KSFW moments (cmFS1) | Same KSFW moment computed from the B final-state particles (`FS1` in basf2). *(naming-based)* | 0.022 |
| 54 | `foxWolframR1` | Fox-Wolfram | Normalised Fox-Wolfram moment R1 = H1/H0 (EventShape framework). | 0.038 |
| 55 | `foxWolframR2` | Fox-Wolfram | R2 from the EventShape framework (same idea as `R2`, computed differently). | 0.012 |
| 56 | `foxWolframR3` | Fox-Wolfram | Normalised Fox-Wolfram moment R3. | 0.059 |
| 57 | `foxWolframR4` | Fox-Wolfram | Normalised Fox-Wolfram moment R4. | 0.007 |
| 58 | `thrust` | Thrust / angular | Event thrust (EventShape framework, whole event). | 0.013 |
| 59 | `thrustAxisCosTheta` | Thrust / angular | Cosine of the polar angle of the event thrust axis (EventShape framework). | 0.018 |
