# Ridge decoding results

## Summary

The simulated new day tests improved with one minute of output correction on every tested date. The table shows mean R² across dates; each date has equal weight. Gains are calculated from the listed daily scores before rounding the overall means. Scores from different tasks should be read separately because they use different datasets and movement outputs.

| Task | History | No calibration | Output correction | Gain |
|---|---:|---:|---:|---:|
| H1 | 0 | 0.1380 | 0.1501 | +0.0121 |
| H1 | 5 | 0.1574 | 0.1749 | +0.0175 |
| M1 | 0 | 0.2280 | 0.3870 | +0.1590 |
| M1 | 5 | 0.1791 | 0.3991 | +0.2200 |
| M2 | 0 | 0.0471 | 0.0985 | +0.0513 |
| M2 | 5 | 0.0770 | 0.1329 | +0.0559 |

These simulated new day scores are exploratory local results. They are separate from the FALCON local minival evaluation and are not EvalAI held out scores.

## Evaluation method

- **Training CV fit score:** Reported by the FALCON starter decoder during training. It does not measure performance on the simulated new day.
- **Local minival:** Runs the FALCON local evaluator on the released held in minival split using a saved model. The results are not EvalAI held out scores.
- **Simulated new day:** Trains a separate ridge model for each test date and history setting using only earlier recording dates. The first 3,000 bins (one minute) of the first test file are reserved for calibration and excluded from both test scores. The rest of that file and any other files from the same date are scored.
- **Output correction:** Fits a ridge model with `alpha=100` to the difference between the target and the original prediction using eligible bins in the reserved minute. The no calibration score uses the same test bins without this correction.
- **R² aggregation:** R² is variance weighted across movement outputs. A date with multiple test files uses the simple mean of their file R² scores. Overall means give each date equal weight. The number of eligible calibration bins can be less than 3,000.

## H1

### Local minival

The history=0 and history=5 models were trained on November 9, 2026, with training CV fit scores of **0.24** and **0.27**, respectively. The November 11 evaluations restored these saved models from Google Drive; they did not retrain them or calculate new CV scores.

| Evaluation date | Run | History | Held In R² mean | Held In R² std. | Normalized latency | Saved model |
|---|---:|---:|---:|---:|---:|
| Nov 9, 2026 | 1 | 0 | 0.1945 | 0.0350 | 0.3586 |
| Nov 9, 2026 | 1 | 5 | 0.2071 | 0.0368 | 0.3710 |
| Nov 11, 2026 | 2 | 0 | 0.1937 | 0.0381 | 0.2454 | `sklearn_h1_history0_baseline.pkl` |
| Nov 11, 2026 | 2 | 5 | 0.2265 | 0.0426 | 0.2460 | `sklearn_m1_history5.pkl` |
| Nov 11, 2026 | 3 | 5 | 0.2265 | 0.0426 | 0.2449 |
| Nov 11, 2026 | 4 | 0 | 0.1937 | 0.0381 | 0.2476 |
| Nov 11, 2026 | 4 | 5 | 0.2265 | 0.0426 | 0.2394 |
| Nov 11, 2026 | 5 | 0 | 0.1937 | 0.0381 | 0.2350 |
| Nov 11, 2026 | 5 | 5 | 0.2265 | 0.0426 | 0.2428 |

The difference in R² between the November 9 and November 11 evaluations has not been explained.

### Simulated new day tests (January dates)

Each row uses a model trained only on earlier dates. January 8 has two earlier training files, both from January 1.

| Test date | Earlier training files | History | Training CV fit | Eligible calibration bins | No calibration R² | Output correction R² |
|---|---:|---:|---:|---:|---:|---:|
| Jan 08 | 2 | 0 | 0.25 | 2,889 | 0.1390 | 0.1500 |
| Jan 08 | 2 | 5 | 0.28 | 2,889 | 0.1730 | 0.1812 |
| Jan 13 | 5 | 0 | 0.27 | 2,862 | 0.1196 | 0.1362 |
| Jan 13 | 5 | 5 | 0.29 | 2,862 | 0.1196 | 0.1526 |
| Jan 15 | 7 | 0 | 0.27 | 2,881 | 0.1358 | 0.1422 |
| Jan 15 | 7 | 5 | 0.29 | 2,876 | 0.1598 | 0.1678 |
| Jan 19 | 9 | 0 | 0.25 | 2,895 | 0.1372 | 0.1568 |
| Jan 19 | 9 | 5 | 0.26 | 2,895 | 0.1555 | 0.1809 |
| Jan 20 | 11 | 0 | 0.24 | 2,919 | 0.1585 | 0.1654 |
| Jan 20 | 11 | 5 | 0.31 | 2,919 | 0.1790 | 0.1919 |
| **Mean across dates** | | **0** | | | **0.1380** | **0.1501** |
| **Mean across dates** | | **5** | | | **0.1574** | **0.1749** |

Output correction improved the mean file score on all five dates for both histories. After correction, history=5 scored higher than history=0 on every tested date.

In a separate January 20 history=0 check, channel normalization gave a mean R² of 0.1558, compared with 0.1585 without calibration and 0.1654 with output correction.

## M1

### Local minival

| History | Training CV fit | Held In R² mean | Held In R² std. | Normalized latency | Saved model |
|---:|---:|---:|---:|---:|---|
| 0 | 0.47 | 0.3799 | 0.0644 | 0.1650 | `sklearn_m1_history0_baseline.pkl` |
| 5 | 0.53 | 0.4606 | 0.0848 | 0.1365 | `sklearn_m1_history5.pkl` |

### Simulated new day tests (September 2012)

Each test date has one file. Its first 3,000 bins were reserved for calibration, and the remaining eligible bins were used for both scores.

| Test date | History | Earlier training files | Eligible calibration bins | Scored bins | Training CV fit | No calibration R² | Output correction R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sep 26 | 0 | 1 | 1,405 | 53,071 | 0.51 | 0.2098 | 0.3880 |
| Sep 26 | 5 | 1 | 1,405 | 53,071 | 0.58 | 0.0604 | 0.3417 |
| Sep 27 | 0 | 2 | 792 | 48,436 | 0.51 | 0.1315 | 0.3912 |
| Sep 27 | 5 | 2 | 792 | 48,436 | 0.59 | 0.1676 | 0.4061 |
| Sep 28 | 0 | 3 | 1,198 | 53,585 | 0.47 | 0.3427 | 0.3817 |
| Sep 28 | 5 | 3 | 1,198 | 53,585 | 0.52 | 0.3092 | 0.4495 |
| **Mean across dates** | **0** | | | | | **0.2280** | **0.3870** |
| **Mean across dates** | **5** | | | | | **0.1791** | **0.3991** |

Output correction improved R² on all three dates for both histories. After correction, history=5 scored higher on two of the three dates.

## M2

### Local minival

| History | Training CV fit | Held In R² mean | Held In R² std. | Normalized latency | Saved model |
|---:|---:|---:|---:|---:|---|
| 0 | 0.15 | 0.0922 | 0.0355 | 0.3041 | `sklearn_m2_history0_baseline.pkl` |
| 5 | 0.21 | 0.1602 | 0.0364 | 0.1837 | `sklearn_m2_history5.pkl` |

### Simulated new day tests (October 2020)

October 20 and 27 each have two test files; October 28 has one. A separate model was trained for each date and history setting using earlier files only.

| Test date | History | Earlier training files | Test files | Eligible calibration bins | Training CV fit | No calibration R² | Output correction R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oct 20 | 0 | 2 | 2 | 3,000 | 0.36 | -0.0551 | 0.0396 |
| Oct 20 | 5 | 2 | 2 | 2,995 | 0.39 | -0.0283 | 0.0854 |
| Oct 27 | 0 | 4 | 2 | 3,000 | 0.13 | 0.0891 | 0.1134 |
| Oct 27 | 5 | 4 | 2 | 2,995 | 0.21 | 0.1261 | 0.1493 |
| Oct 28 | 0 | 6 | 1 | 3,000 | 0.16 | 0.1074 | 0.1424 |
| Oct 28 | 5 | 6 | 1 | 2,995 | 0.23 | 0.1331 | 0.1640 |
| **Mean across dates** | **0** | | | | | **0.0471** | **0.0985** |
| **Mean across dates** | **5** | | | | | **0.0770** | **0.1329** |

Output correction improved R² on all three dates for both histories. October 20's negative uncorrected scores became positive after correction. After correction, history=5 scored higher on every tested date.