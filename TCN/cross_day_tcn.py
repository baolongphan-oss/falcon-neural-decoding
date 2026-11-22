"""Simulated new-day test for the TCN, mirroring cross_day_ridge.py.

For each test day: train on earlier held-in days, use the first minute of the
test day for calibration, and score the rest (no calibration vs. output correction).
Run from the folder above TCN: python -m TCN.cross_day_tcn
"""

from .train_refactored import train_model, load_sessions, DEVICE, BATCH_SIZE
from .data_long import Windows
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np
import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from .adaptations import anchored_refit, choose_lambda

# settings
TASK = 'm1'

DATA_ROOT = Path(os.environ.get("FALCON_DATA", "~/falcon_data/000941")).expanduser()
TEST_DAYS = ("20120926", "20120927", "20120928")
CALIBRATION_BINS = 3000         # first minute (3000 x 20ms)
CORRECTION_ALPHA = 100.0        # ridge penalty for output correction (same as ridge team)
EPOCH_RUNS = int(os.environ.get("EPOCHS", 2))
VALIDATION = os.environ.get("VALIDATION", "minival")   # or "next_day"
# minival: early stopping checks held-out minutes of the training days (default; keeps all training days)
# next_day: early stopping checks a different, unseen earlier day (costs one training day; no gain on M1)
SMOOTHING = os.environ.get("SMOOTHING", "ema")     # or "falcon"
# applied to train, val, and test sessions alike


def recording_day(path):
    """Return the eight digit recording date from an NWB filename."""
    date = path.stem.split("_ses-", 1)[1]
    return date[:10].replace("-", "") if TASK == "m2" else date[:8]


def split_files(day):
    """Return train/val/test files for one test day; train and val use only earlier days."""
    training_files = sorted(DATA_ROOT.glob("*held-in-calib/*.nwb"))
    validation_files = sorted(DATA_ROOT.glob("*held-in-minival/*.nwb"))
    test = [f for f in training_files if recording_day(f) == day]  # test is always the day itself

    # one entry per day (a day can have several files)
    earlier_days = sorted(set(recording_day(f) for f in training_files if recording_day(f) < day))
    if VALIDATION == "next_day" and len(earlier_days) >= 2:
        # hold out the latest earlier day (closest to current day) for validation
        # train on the rest
        val_day = earlier_days[-1]
        train = [f for f in training_files if recording_day(f) < day and recording_day(f) != val_day]
        # validation on the calibration files -- main recordings for the day, instead of minival
        val = [f for f in training_files if recording_day(f) == val_day]
    else:
        train = [f for f in training_files if recording_day(f) < day]
        val = [f for f in validation_files if recording_day(f) < day]

    # every split must have files, otherwise later steps fail
    if not train or not val or not test:
        raise ValueError(f"{day}: train={len(train)}, val={len(val)}, test={len(test)}")

    # cannot have files in both train and val, else validation is meaningless
    train_val_overlaps = list(set(train) & set(val))
    if len(train_val_overlaps) > 0:
        raise ValueError(f"Found overlaps between train and val: {train_val_overlaps}")
    return train, val, test

def make_rolling_splits(files, min_train_days=1):
    """Create chronological splits that preserve time order; training data grows on each split while validation
    always performed on a later day. All recordings from a single day are kept together."""
    if min_train_days < 1:
        raise ValueError("min_train_days must be at least 1")

    from collections import defaultdict
    files_by_day = defaultdict(list)

    for path in files:
        day = recording_day(path)
        files_by_day[day].append(path)

    days = sorted(files_by_day)
    if len(days) <= min_train_days:
        raise ValueError("Not enough recording days to create a validation split")

    splits = []

    for i in range(min_train_days, len(days)):
        train_days = days[:i]
        val_day = days[i]

        # Collect files from every day in train_days.
        train_files = [
            path
            for day in train_days
            for path in files_by_day[day]
        ]

        # Collect all files from val_day.
        val_files = list(files_by_day[val_day])

        # Append a dictionary describing this fold.
        splits.append({
            "train_days": train_days,
            "val_day": val_day,
            "train_files": train_files,
            "val_files": val_files,
        })

    return splits

def run_rolling_validation(splits, task, **train_kwargs):
    """Train on expanding sets of earlier days and validate on the next day.

    Each fold starts with a fresh model and training-only normalization.
    Collect metrics from the epoch selected by lowest validation MSE.

    Scores reflect both the validation day's difficulty and the amount of
    training data. These are model-selection scores, not independent test scores."""
    if not splits:
        raise ValueError("No rolling-validation splits were provided")

    results = []

    for fold_id, split in enumerate(splits, start=1):
        print(
            f"Fold {fold_id}: "
            f"train {split['train_days']} → validate {split['val_day']}"
        )

        # Train a fresh model for this fold.
        model, stats, records = train_model(
            train_files=split["train_files"],
            val_files=split["val_files"],
            task=task,
            **train_kwargs,
        )

        # Find the metrics for the selected epoch.
        best_record = min(records, key=lambda row: row["val_mse"])

        # Record this fold's result.
        results.append({
            "fold": fold_id,
            "train_days": split["train_days"],
            "val_day": split["val_day"],
            "best_epoch": best_record["epoch"],
            "val_mse": best_record["val_mse"],
            "val_r2_mean": best_record["val_r2_mean"],
            "val_r2_sessions": best_record["val_r2_sessions"],
        })

    return results

def predict(model, sessions, stats, batch_size, device):
    """Run a trained model on sessions; return predictions, targets (mV), (session, t) per row, and features."""
    y_mean, y_scale = stats["y_mean"], stats["y_scale"]
    dataset = Windows(sessions, stats["x_mean"], stats["x_scale"], y_mean, y_scale, stats["history"])
    loader = DataLoader(dataset, batch_size=batch_size)
    # no shuffle: rows stay aligned with dataset.index

    model.eval()
    with torch.no_grad():
        predictions, targets, features = [], [], []
        for X, y in loader:
            # same as model(X), split in two so we can keep the features
            # last time step features: (batch, 64)
            h = model.temporal(X.to(device))[:, :, -1]
            predictions.append(model.head(h).cpu().numpy())
            targets.append(y.numpy())
            # input to the head; used by anchored refit
            features.append(h.cpu().numpy())

    pred, target, features = np.concatenate(predictions), np.concatenate(targets), np.concatenate(features)
    # features stay as is (not muscle values)
    # convert values back to mV (for muscles data) for r^2 to match Ridge regression
    pred = pred * y_scale + y_mean
    target = target * y_scale + y_mean
    # dataset.index: list of (session_id, time_bins)
    return pred, target, dataset.index, features


def score_day(pred, target, index, features, model, stats):
    """Score one test day: first minute of the first session = calibration, the rest = scored.
    Only actual movement (masked bins) is included.
    R²: 1 = perfect, 0 = no better than guessing the average, < 0 = worse.

    Output correction: keep the model as is;
    learn its mistakes from the new day’s first minute
    and add a correction to every prediction afterward.

    Anchored refit: freeze the whole network except the last layer;
    retrain that layer on the new day’s first minute,
    staying close to the old version.

    """
    session_ids = np.array([s for s, _ in index])
    time_bins = np.array([t for _, t in index])
    # first minute of first session
    calibration_rows = (session_ids == 0) & (time_bins < CALIBRATION_BINS)
    # get the inverse: non-calibration rows
    scored = ~calibration_rows
    if calibration_rows.sum() < 2:
        raise ValueError("Too few calibration bins")
    # 1) no calibration: the trained model as-is
    # how well does the model survive drift without any help
    no_calibration_r2 = r2_score(target[scored], pred[scored], multioutput="variance_weighted")

    # baseline adaptation method
    # 2) output correction: learn a small fix from calibration errors, apply (i.e. add) it to scored rows (predictions)
    base = pred[calibration_rows]
    residual = target[calibration_rows] - base
    correction = Ridge(alpha=CORRECTION_ALPHA).fit(base, residual)
    corr_preds = pred[scored] + correction.predict(pred[scored])
    output_correction_r2 = r2_score(target[scored], corr_preds, multioutput="variance_weighted")

    # proposed calibration methods
    # 3) anchored refit
    W_old = model.head.weight.detach().cpu().numpy().T
    b_old = model.head.bias.detach().cpu().numpy()
    y_mean, y_scale = stats["y_mean"], stats["y_scale"]
    Y_cal = (target[calibration_rows] - y_mean) / y_scale
    H_cal = features[calibration_rows]
    lambda_val = choose_lambda(H_cal, Y_cal, W_old, b_old)
    W, b = anchored_refit(H_cal, Y_cal, W_old, b_old, lambda_val)
    refit_pred = features[scored] @ W + b
    # convert values back to mV (for muscles data) for r^2 calculation
    refit_pred = refit_pred * y_scale + y_mean
    anchored_refit_r2 = r2_score(target[scored], refit_pred, multioutput="variance_weighted")

    return {
        "no_calibration": float(no_calibration_r2),
        "output_correction": float(output_correction_r2),
        "anchored_refit": float(anchored_refit_r2),
        "anchored_lambda": lambda_val,
        "calibration_bins": int(calibration_rows.sum()), # number of movement bins used to calibrate (x 20 ms = seconds of movement)
        "scored_bins": int(scored.sum()) # number of movement bins graded (rest of the day)
    }

def run_rolling_calibration(splits, task, **train_kwargs):
    results = []

    for fold_id, split in enumerate(splits, start=1):
        early_stop_files = sorted(
            path
            for path in DATA_ROOT.glob("*held-in-minival/*.nwb")
            if recording_day(path) in split["train_days"]
        )

        if not early_stop_files:
            raise ValueError("No earlier-day minival files found")

        model, stats, records = train_model(
            train_files=split["train_files"],
            val_files=early_stop_files,
            task=task,
            **train_kwargs,
        )

        smoothing = train_kwargs.get("smoothing", "ema")
        alpha = train_kwargs.get("alpha", 0.2)

        sessions = load_sessions(
            split["val_files"],
            task,
            smoothing=smoothing,
            alpha=alpha,
        )

        pred, target, index, features = predict(
            model,
            sessions,
            stats,
            batch_size=train_kwargs.get("batch_size", BATCH_SIZE),
            device=next(model.parameters()).device,
        )

        scores = score_day(
            pred, target, index, features, model, stats
        )

        results.append({
            "fold": fold_id,
            "train_days": split["train_days"],
            "calibration_day": split["val_day"],
            "best_epoch": min(
                records, key=lambda row: row["val_mse"]
            )["epoch"],
            **scores,
        })

    return results

def run_cross_day(**train_kwargs):
    """Original experiment: earlier days train/validate, then calibrate a later day."""
    smoothing = train_kwargs.get("smoothing", SMOOTHING)
    results = {"task": TASK, "validation": VALIDATION, "smoothing": smoothing,
               "epochs": train_kwargs.get("epochs", EPOCH_RUNS), "days": [],
               "no_calibration": [], "output_correction": [], "anchored_refit": [],
               "details": []}
    for day in TEST_DAYS:
        train, val, test = split_files(day)
        print(day, "train", [recording_day(f) for f in train],
              "validate", [recording_day(f) for f in val], flush=True)
        model, stats, _ = train_model(train, val, TASK, **train_kwargs)
        sessions = load_sessions(test, TASK, smoothing=smoothing,
                                 alpha=train_kwargs.get("alpha", 0.2))
        pred, target, index, features = predict(
            model, sessions, stats, batch_size=BATCH_SIZE,
            device=next(model.parameters()).device)
        scores = score_day(pred, target, index, features, model, stats)
        results["days"].append(day)
        for method in ("no_calibration", "output_correction", "anchored_refit"):
            results[method].append(scores[method])
        results["details"].append(scores)
    return results


def main(argv=None):
    """Select an experiment without editing the module's __main__ block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cross-day", "rolling-validation", "rolling-calibration"),
                        default="cross-day")
    parser.add_argument("--epochs", type=int, default=EPOCH_RUNS)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoothing", choices=("ema", "falcon"), default=SMOOTHING)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--out", type=Path, help="New output directory; existing directories are protected")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    rolling = args.mode != "cross-day"
    settings = dict(epochs=args.epochs, patience=args.patience, seed=args.seed,
                    smoothing=args.smoothing,
                    hidden=args.hidden if args.hidden is not None else (32 if rolling else 64),
                    dropout=args.dropout if args.dropout is not None else (0.3 if rolling else 0.1),
                    weight_decay=args.weight_decay if args.weight_decay is not None else (1e-3 if rolling else 1e-4))
    if settings["hidden"] < 1 or not 0 <= settings["dropout"] < 1 or settings["weight_decay"] < 0:
        parser.error("Require hidden > 0, 0 <= dropout < 1, and weight-decay >= 0")
    out = args.out or Path("runs") / (
        f"{args.mode}_epochs_{args.epochs}_seed_{args.seed}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if out.exists():
        parser.error(f"Output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(
        mode=args.mode, task=TASK, data_root=str(DATA_ROOT.resolve()),
        cross_day_validation=VALIDATION, settings=settings), indent=2))
    print(f"Saving results to: {out}", flush=True)
    if args.mode == "cross-day":
        results = run_cross_day(**settings)
    else:
        splits = make_rolling_splits(sorted(DATA_ROOT.glob("*held-in-calib/*.nwb")))
        runner = run_rolling_validation if args.mode == "rolling-validation" else run_rolling_calibration
        results = runner(splits, task=TASK, **settings)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    if args.mode == "rolling-validation":
        print("Mean rolling-validation R²:", np.mean([r["val_r2_mean"] for r in results]))
    else:
        for method in ("no_calibration", "output_correction", "anchored_refit"):
            values = results[method] if args.mode == "cross-day" else [r[method] for r in results]
            print(f"Mean {method} R²:", np.mean(values))


if __name__ == "__main__":
    main()
