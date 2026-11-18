"""Simulated new-day test for the TCN, mirroring cross_day_ridge.py.

For each test day: train on earlier held-in days, use the first minute of the
test day for calibration, and score the rest (no calibration vs. output correction).
Run from the folder above TCN: python -m TCN.cross_day_tcn
"""

from .train_refactored import train_model, load_sessions, DEVICE, BATCH_SIZE
from .data_long import Windows
from pathlib import Path 
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
VALIDATION = os.environ.get("VALIDATION", "next_day")   # or "minival"
# next_day: early stopping checks a different, unseen earlier day (mimics a new day)
# minival: early stopping checks held-out minutes of the training days (old behavior)


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

    # 2) output correction: learn a small fix from calibration errors, apply (i.e. add) it to scored rows (predictions)
    base = pred[calibration_rows]
    residual = target[calibration_rows] - base
    correction = Ridge(alpha=CORRECTION_ALPHA).fit(base, residual)
    corr_preds = pred[scored] + correction.predict(pred[scored])
    output_correction_r2 = r2_score(target[scored], corr_preds, multioutput="variance_weighted")

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

# for each test day: split files, train on earlier days, predict, score
if __name__ == "__main__":
    results = {"task": TASK, "validation": VALIDATION, "epochs": EPOCH_RUNS, 
               "days": [], "no_calibration": [], "output_correction": [], 
               "anchored_refit": [], "details": []}
    for day in TEST_DAYS:
        train, val, test = split_files(day)
        print(day, [recording_day(f) for f in train],
                [recording_day(f) for f in val],
                [recording_day(f) for f in test], "\n")
        model, stats, _ = train_model(train, val, TASK, epochs=EPOCH_RUNS)  # short run to test the pipeline; full runs on GPU
        test_sessions = load_sessions(test, TASK)
        pred, target, index, features = predict(model, test_sessions, stats, batch_size=BATCH_SIZE, device=DEVICE)
        
        score_dict = score_day(pred, target, index, features, model, stats)
        results["days"].append(day)
        results["no_calibration"].append(score_dict["no_calibration"])
        results["output_correction"].append(score_dict["output_correction"])
        results["anchored_refit"].append(score_dict["anchored_refit"])
        results["details"].append(score_dict)
        print(day, score_dict, "\n")

    out = Path("results") / f"tcn_{TASK}_{VALIDATION}_cross_day_epochs_{EPOCH_RUNS}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out}")

    print("Mean no calibration:", np.mean(results["no_calibration"]))
    print("Mean output correction:", np.mean(results["output_correction"]))
    print("Mean anchored refit:", np.mean(results["anchored_refit"]))
