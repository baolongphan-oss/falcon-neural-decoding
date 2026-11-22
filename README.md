# FALCON Neural Decoding

DSCI 550 group project studying how movement decoders generalize across recording days, and how a small amount of labeled new-day data can restore performance.

We compare ridge regression with a causal temporal convolutional network (TCN) on the movement tasks in [FALCON](https://snel-repo.github.io/falcon/). The repository contains ridge experiments for H1, M1, and M2, and TCN development experiments currently focused on M1.

**11/22/2026 finding:** with stronger regularization, a two-epoch TCN followed by anchored output-head refitting gives our best calibrated M1 development result. Longer training improves uncalibrated predictions but reduces calibration's benefit. Chronological validation selects earlier checkpoints than same-day minival.

These are simulated cross-day experiments using public held-in recordings, **not official held-out EvalAI results**.

## Tasks and data

Recordings use 20 ms spike-count bins paired with continuous movement targets.

| Task | Prediction target | Neural channels | Outputs | DANDI dataset |
| --- | --- | ---: | ---: | --- |
| M1 | Monkey arm/hand muscle activity | 64 | 16 | [000941](https://dandiarchive.org/dandiset/000941) |
| M2 | Monkey finger movement velocities | 96 | 2 | [000953](https://dandiarchive.org/dandiset/000953) |
| H1 | Human robotic-arm control velocities | 176 | 7 | [000954](https://dandiarchive.org/dandiset/000954) |

NWB loading and spike binning use the official `falcon_challenge` package. Raw recordings are downloaded separately and are not committed.

## Repository structure

```text
TCN/
  model.py                 Causal convolution layers and TCN output head
  data.py                  Original EMA preprocessing and history windows
  data_long.py             Extended smoothing and target standardization
  train.py                 Original command-line trainer
  train_refactored.py      Reusable training, normalization, and evaluation
  adaptations.py           Anchored head refitting and lambda selection
  cross_day_tcn.py          Cross-day and rolling experiment functions/CLI
  multiseed_calibration.py  Multi-seed calibration or validation runner
  run_tcn_cross_day.ipynb   Colab experiment notebook
Ridge regression/
  Code/                    Download, train, cross-day evaluation, and plot scripts
  Cache/                   Saved ridge models
  results.md               Ridge experiment notes
  *.png                    Ridge comparison figures
results/
  TCN Progress Report.docx  Dated TCN findings and experiment history
  experiments/2026-11-22/   Archived multi-seed results, configs, and summaries
  *.json                   Earlier and single-seed TCN results
scripts/
  download_m1.py           Local downloader for M1 held-in data
  requirements-local.txt   Core versions verified on the local Mac setup
  run_local_m1.sh          Launcher for the original parent-workspace layout
tests/                    Experiment dispatch, split, and summary checks
requirements.txt           General project dependencies
Project Proposal.docx      Project scope and proposed methods
```

## Local setup

Python 3.12 is recommended for a new environment. The recorded local experiments used Python 3.9.6, PyTorch 2.8.0, NumPy 1.26.4, and scikit-learn 1.6.1 on Apple Silicon CPU.

```bash
git clone https://github.com/baolongphan-oss/falcon-neural-decoding.git
cd falcon-neural-decoding

python3 -m venv falcon-env
source falcon-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r scripts/requirements-local.txt
```

The additional requirements file includes the official FALCON package and pins the core versions used locally. NumPy 1.26.4 avoids reproducible matrix-multiplication warnings observed with NumPy 2.0.2 on that Mac.

Run all commands below from the repository root with the environment activated. Windows users should activate their environment using the corresponding Windows command.

## Download M1 data

```bash
python scripts/download_m1.py --out data/000941
export FALCON_DATA="$PWD/data/000941"
```

This downloads eight public M1 files, approximately 289 MB: four held-in calibration recordings and four held-in minival recordings. It preserves DANDI's directory structure, records a download manifest, checks byte sizes, and skips complete downloads. It does not download held-out calibration files.

Expected layout:

```text
data/000941/
  download_manifest.json
  sub-MonkeyL-held-in-calib/
    ...ses-20120924_behavior+ecephys.nwb
    ...ses-20120926_behavior+ecephys.nwb
    ...ses-20120927_behavior+ecephys.nwb
    ...ses-20120928_behavior+ecephys.nwb
  sub-MonkeyL-held-in-minival/
    ... corresponding minival files ...
```

Set `FALCON_DATA` again in each new terminal, or point it to an existing download. Without that variable, the TCN experiments look under `~/falcon_data/000941`.

## TCN method

- Causal neural smoothing: EMA or the finite exponential filter in `data_long.py`.
- Training-only channel normalization and target standardization in the refactored trainer.
- History windows of 30 bins for M1/H1 and 7 bins for M2; windows include the current bin.
- Four causal kernel-3 convolutions with dilations 1, 2, 4, and 8, ReLU, and dropout.
- A linear head predicts movement from the final time step's hidden features.
- Standardized-target MSE training with AdamW, weight decay, gradient clipping, and early stopping by validation MSE; R² is calculated after restoring target units.

History remains continuous across trials and resets between files. Masked targets are excluded from training and scoring, while their neural observations remain available as history.

The current regularized configuration uses **hidden width 32, dropout 0.3, weight decay 0.001, and EMA alpha 0.2**. The architecture can represent the three movement tasks, but the cross-day runner is currently configured for M1; switching tasks requires checking dates, paths, and task settings in the code.

## Experiment procedures

### Rolling chronological validation

Train a fresh model on all earlier days and use the next day for checkpoint selection:

```text
Train Sep 24             -> validate Sep 26
Train Sep 24 and 26      -> validate Sep 27
Train Sep 24, 26, and 27 -> validate Sep 28
```

All files from a day stay together. Normalization is fitted only on each fold's training data. The selected next-day score is a **model-selection score**, not an independent test score.

### Rolling calibration

Use the same expanding training days, but select the checkpoint using **minival files from those earlier days**. On the next day:

1. Reserve the first 3,000 bins (60 seconds) of the first recording for calibration.
2. Fit calibration using eligible target bins within that interval.
3. Compare all methods on the same remaining eligible bins.

The methods are:

- **No calibration:** unchanged pretrained decoder.
- **Output correction:** ridge regression maps initial predictions to their residual errors.
- **Anchored refit:** freeze the temporal network and refit its output weights and bias with a penalty toward the pretrained head. Select lambda on a chronological 70/30 split within calibration, then refit using all calibration examples.

The calibration interval is 60 seconds of elapsed recording, not necessarily 60 seconds of eligible movement targets. Calibration is not full-network fine-tuning.

## Run TCN experiments

### Single seed

```bash
# Chronological validation, regularized configuration
python -m TCN.cross_day_tcn --mode rolling-validation --epochs 50 --patience 8

# Earlier-day minival checkpoint selection plus next-day calibration
python -m TCN.cross_day_tcn --mode rolling-calibration --epochs 2

# Original cross-day experiment and original model-size defaults
python -m TCN.cross_day_tcn --mode cross-day --epochs 2
```

`cross-day` is the default mode and uses width 64, dropout 0.1, and weight decay 0.0001. Both rolling modes default to the regularized configuration. Override settings with `--hidden`, `--dropout`, `--weight-decay`, `--smoothing`, and `--seed`. The original cross-day split strategy still reads `VALIDATION=minival` or `VALIDATION=next_day` from the environment.

### Multiple seeds

```bash
# Two-epoch calibrated reference
python -m TCN.multiseed_calibration --mode calibration --epochs 2 --seeds 0 1 2

# Longer training with earlier-day minival early stopping
python -m TCN.multiseed_calibration --mode calibration --epochs 50 --patience 8 --seeds 0 1 2

# Longer training with next-day chronological early stopping
python -m TCN.multiseed_calibration --mode validation --epochs 50 --patience 8 --seeds 0 1 2
```

The multi-seed runner defaults to one PyTorch CPU thread. The trainer automatically uses CUDA when available; otherwise it uses CPU. A 50-epoch cap does not force 50 epochs: patience 8 stops training after eight epochs without validation-MSE improvement.

New runs create timestamped folders under ignored `runs/`. Single-seed runs save `config.json` and `results.json`; multi-seed runs save configuration, per-seed results, and a running aggregate summary. `--out` accepts a custom **new directory**; existing directories are rejected to prevent overwriting. Copy selected completed results into `results/` when archiving an experiment for Git.

## Current M1 findings

All figures below use width 32, dropout 0.3, weight decay 0.001, EMA smoothing, and seeds 0/1/2. Days are weighted equally within each seed; uncertainty is the **sample standard deviation across seed averages**, not across independent future days.

### Calibration with earlier-day minival checkpoint selection

| Method | 2-epoch cap R² | 50-epoch cap R² |
| --- | ---: | ---: |
| No calibration | 0.286 ± 0.123 | 0.393 ± 0.027 |
| Output correction | 0.448 ± 0.024 | 0.409 ± 0.021 |
| Anchored refit | **0.473 ± 0.019** | **0.430 ± 0.028** |

- With the two-epoch cap, both calibration methods improve all nine seed/day pairs; anchored refit beats output correction in eight of nine.
- With the 50-epoch cap, each calibration method improves seven of nine pairs.
- Longer training improves the uncalibrated decoder, but the two-epoch model remains better after calibration.

### Chronological checkpoint selection

The 50-epoch-cap rolling-validation experiment gives **R² = 0.414 ± 0.047**. Selected epochs are earlier than earlier-day minival selection in all nine matched comparisons:

| Next day | Chronological epochs, seeds 0/1/2 | Earlier-day minival epochs, seeds 0/1/2 |
| --- | --- | --- |
| Sep 26 | 2, 18, 17 | 47, 47, 32 |
| Sep 27 | 4, 3, 8 | 25, 36, 26 |
| Sep 28 | 9, 9, 1 | 23, 33, 24 |

Do not directly rank the rolling-validation score against the calibration table: the checkpoint-selection data and scoring intervals differ. These development days have already informed configuration choices; a separate future-day evaluation is needed to assess the selected procedure independently.

See [archived experiment results](results/experiments/2026-11-22/), their [manifest and reproduction commands](results/experiments/2026-11-22/manifest.json), and the [TCN progress report](results/TCN%20Progress%20Report.docx).

## Ridge experiments

The ridge scripts currently assume **Google Colab**, `/content/` paths, and Google Drive storage. They are not standalone local CLI tools without changes to those dependencies and paths.

In Colab, review and run the scripts in this order:

1. `Ridge regression/Code/load_data.py`: select the task and download its recordings.
2. `Ridge regression/Code/train_ridge.py`: prepare the official FALCON repository and train/evaluate history settings 0 and 5.
3. `Ridge regression/Code/cross_day_ridge.py`: train on earlier dates and compare no calibration with output correction on later dates.
4. `Ridge regression/Code/summarize_results.py`: plot manually recorded scores; this script does not retrain models or read newly generated metrics automatically.

Ridge history 5 means five lagged bins plus the current bin. It is not the same history length as the TCN's 30-bin M1 window. Existing ridge comparisons are useful development references; a controlled model comparison should match preprocessing, temporal history, splits, and calibration examples.

## Checks

```bash
python -m unittest discover -s tests -v
python -m TCN.cross_day_tcn --help
python -m TCN.multiseed_calibration --help
```

The lightweight checks cover chronological day grouping, earlier-day-only minival selection, experiment-mode dispatch, output protection, and seed-summary calculations. They do not replace a real-data run or official benchmark evaluation.

## Remaining work

- Select a training-duration rule using earlier development folds and assess it on a later day outside those decisions.
- Extend the validated TCN procedure to M2 and H1.
- Evaluate calibration-data efficiency and normalized latency.
- Explore weighted recalibration, per-day channel normalization, latent-factor alignment, and full-network fine-tuning; these proposed methods are not implemented in the current TCN experiment runner.
- Package the selected decoder for official FALCON evaluation and EvalAI submission.

## References

- [FALCON benchmark](https://snel-repo.github.io/falcon/)
- [Official FALCON repository](https://github.com/snel-repo/falcon-challenge)
- [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2319/overview)
- [DANDI Archive](https://dandiarchive.org/)

The official repository supplies the data-loading/evaluation interfaces and the ridge reference used by the group. Cite the benchmark and dataset sources when reporting project results.
