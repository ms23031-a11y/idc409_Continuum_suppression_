## Table A - required metrics at the validation-selected operating point (test: 5,297 signal, 69 continuum)

| Model | Accuracy | Precision (signal) | Recall (signal) | F1 (signal) | ROC-AUC [95% CI] | Features used | Threshold | Final fit (s) |
|---|---|---|---|---|---|---|---|---|
| Majority class | 0.987 | 0.987 | 1.000 | 0.994 | 0.500 [0.500, 0.500] | 0 | 1.000 | 0.0 |
| Single-feature cut | 0.899 | 0.988 | 0.909 | 0.947 | 0.567 [0.504, 0.630] | 1 (thrustOm; 59 examined) | 0.001 | 0.5 |
| Logistic regression | 0.894 | 0.987 | 0.904 | 0.944 | 0.530 [0.460, 0.598] | 59 | 0.979 | 0.1 |
| Decision tree | 0.923 | 0.987 | 0.934 | 0.960 | 0.522 [0.459, 0.588] | 59 | 0.254 | 1.2 |
| Random forest | 0.922 | 0.987 | 0.933 | 0.959 | 0.505 [0.434, 0.573] | 59 | 0.970 | 14.1 |
| Gradient boosting (HistGB) | 0.897 | 0.988 | 0.907 | 0.946 | 0.545 [0.475, 0.617] | 59 | 0.994 | 11.7 |


Precision/recall/F1 are for the signal class. ROC-AUC uses the continuous scores of the same test rows, so it does not depend on the threshold.


## Table B - signal efficiency and continuum rejection at the operating point

| Model | Signal efficiency [95% exact CI], n=5,297 | Continuum rejection [95% exact CI], n=69 | Continuum acceptance | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Majority class | 100.0% [99.9%, 100.0%] | 0.0% [0.0%, 5.2%] | 100.0% | 0.500 | 0.497 |
| Single-feature cut | 90.9% [90.1%, 91.6%] | 13.0% [6.1%, 23.3%] | 87.0% | 0.520 | 0.489 |
| Logistic regression | 90.4% [89.5%, 91.2%] | 11.6% [5.1%, 21.6%] | 88.4% | 0.510 | 0.485 |
| Decision tree | 93.4% [92.7%, 94.1%] | 5.8% [1.6%, 14.2%] | 94.2% | 0.496 | 0.489 |
| Random forest | 93.3% [92.6%, 94.0%] | 5.8% [1.6%, 14.2%] | 94.2% | 0.495 | 0.489 |
| Gradient boosting (HistGB) | 90.7% [89.9%, 91.5%] | 13.0% [6.1%, 23.3%] | 87.0% | 0.519 | 0.489 |


## Table C - the same models at the fixed threshold 0.5

| Model | Accuracy | Precision (signal) | Recall (signal) | F1 (signal) | ROC-AUC | Signal efficiency | Continuum rejection (n=69) | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|---|---|---|---|---|
| Majority class | 0.987 | 0.987 | 1.000 | 0.994 | 0.500 | 100.0% | 0.0% | 0.500 | 0.497 |
| Single-feature cut | 0.428 | 0.991 | 0.425 | 0.595 | 0.567 | 42.5% | 71.0% | 0.567 | 0.313 |
| Logistic regression | 0.986 | 0.987 | 0.999 | 0.993 | 0.530 | 99.9% | 0.0% | 0.500 | 0.497 |
| Decision tree | 0.605 | 0.988 | 0.607 | 0.752 | 0.522 | 60.7% | 44.9% | 0.528 | 0.390 |
| Random forest | 0.987 | 0.987 | 1.000 | 0.994 | 0.505 | 100.0% | 0.0% | 0.500 | 0.497 |
| Gradient boosting (HistGB) | 0.987 | 0.987 | 1.000 | 0.993 | 0.545 | 100.0% | 0.0% | 0.500 | 0.497 |


## Table D - per-class precision, recall, F1 at the operating point

| Model | Signal P | Signal R | Signal F1 | Continuum P | Continuum R | Continuum F1 |
|---|---|---|---|---|---|---|
| Majority class | 0.987 | 1.000 | 0.994 | 0.000 | 0.000 | 0.000 |
| Single-feature cut | 0.988 | 0.909 | 0.947 | 0.018 | 0.130 | 0.032 |
| Logistic regression | 0.987 | 0.904 | 0.944 | 0.015 | 0.116 | 0.027 |
| Decision tree | 0.987 | 0.934 | 0.960 | 0.011 | 0.058 | 0.019 |
| Random forest | 0.987 | 0.933 | 0.959 | 0.011 | 0.058 | 0.019 |
| Gradient boosting (HistGB) | 0.988 | 0.907 | 0.946 | 0.018 | 0.130 | 0.032 |


## Table E - full vs reduced feature sets (same protocol)

| Model | Feature set | Features | ROC-AUC | AUC diff vs full [95% paired CI] | Signal eff. | Continuum rej. (n=69) | Balanced acc. |
|---|---|---|---|---|---|---|---|
| Logistic regression | full | 59 | 0.530 | - | 90.4% | 11.6% | 0.510 |
| Logistic regression | corr_filtered | 47 | 0.526 | -0.0030 [-0.0201, +0.0140] | 90.0% | 10.1% | 0.501 |
| Logistic regression | reduced | 3 | 0.565 | +0.0360 [-0.0307, +0.1029] | 89.6% | 15.9% | 0.528 |
| Decision tree | full | 59 | 0.522 | - | 93.4% | 5.8% | 0.496 |
| Decision tree | corr_filtered | 47 | 0.513 | -0.0084 [-0.0737, +0.0500] | 96.0% | 1.4% | 0.487 |
| Decision tree | reduced | 3 | 0.568 | +0.0444 [-0.0308, +0.1240] | 90.4% | 17.4% | 0.539 |
| Random forest | full | 59 | 0.505 | - | 93.3% | 5.8% | 0.495 |
| Random forest | corr_filtered | 47 | 0.497 | -0.0072 [-0.0872, +0.0683] | 93.6% | 4.3% | 0.490 |
| Random forest | reduced | 3 | 0.548 | +0.0428 [-0.0275, +0.1121] | 91.1% | 7.2% | 0.491 |
| Gradient boosting (HistGB) | full | 59 | 0.545 | - | 90.7% | 13.0% | 0.519 |
| Gradient boosting (HistGB) | corr_filtered | 47 | 0.537 | -0.0094 [-0.0699, +0.0484] | 91.4% | 10.1% | 0.508 |
| Gradient boosting (HistGB) | reduced | 3 | 0.553 | +0.0063 [-0.0759, +0.0884] | 90.7% | 7.2% | 0.490 |


## Table F - training-only class weighting compared on validation data (full features)

| Model | best validation AUC, no weighting | best validation AUC, training-only class weighting | weighting chosen |
|---|---|---|---|
| Logistic regression | 0.505 | 0.505 | no |
| Decision tree | 0.518 | 0.543 | yes |
| Random forest | 0.508 | 0.562 | yes |
| Gradient boosting (HistGB) | 0.539 | 0.548 | yes |
