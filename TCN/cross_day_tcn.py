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

# settings
TASK = 'm1'

DATA_ROOT = Path(os.environ.get("FALCON_DATA", "~/falcon_data/000941")).expanduser()
TEST_DAYS = ("20120926", "20120927", "20120928")
CALIBRATION_BINS = 3000         # first minute (3000 x 20ms)
CORRECTION_ALPHA = 100.0        # ridge penalty for output correction (same as ridge team)
EPOCH_RUNS = int(os.environ.get("EPOCHS", 2))


def recording_day(path):
    """Return the eight digit recording date from an NWB filename."""
    date = path.stem.split("_ses-", 1)[1]
    return date[:10].replace("-", "") if TASK == "m2" else date[:8]


def split_files(day):
    """Return train/val/test files for one test day; train and val use only earlier days."""
    training_files = sorted(DATA_ROOT.glob("*held-in-calib/*.nwb"))
    train = [f for f in training_files if recording_day(f) < day]
    test = [f for f in training_files if recording_day(f) == day]

    validation_files = sorted(DATA_ROOT.glob("*held-in-minival/*.nwb"))
    val = [f for f in validation_files if recording_day(f) < day]

    # every split must have files, otherwise later steps fail
    if not train or not val or not test:
        raise ValueError(f"{day}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def predict(model, sessions, stats, batch_size, device):
    """Run a trained model on sessions; return predictions, targets (mV), and (session, t) per row."""
    y_mean, y_scale = stats["y_mean"], stats["y_scale"]
    dataset = Windows(sessions, stats["x_mean"], stats["x_scale"], y_mean, y_scale, stats["history"])
    loader = DataLoader(dataset, batch_size=batch_size)  # no shuffle: rows stay aligned with dataset.index

    model.eval()
    with torch.no_grad():
        predictions, targets = [], []
        for X, y in loader:
            predictions.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())

    pred, target = np.concatenate(predictions), np.concatenate(targets)
    # convert values back to mV (for muscles data) for r^2 to match Ridge regression
    pred = pred * y_scale + y_mean 
    target = target * y_scale + y_mean
    return pred, target, dataset.index # list of (session_id, time_bins)


def score_day(pred, target, index):
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

    return {
        "no_calibration": float(no_calibration_r2),
        "output_correction": float(output_correction_r2),
        "calibration_bins": int(calibration_rows.sum()), # number of movement bins used to calibrate (x 20 ms = seconds of movement)
        "scored_bins": int(scored.sum()) # number of movement bins graded (rest of the day)
    }

# for each test day: split files, train on earlier days, predict, score
if __name__ == "__main__":
    results = {"task": TASK, "epochs": EPOCH_RUNS, "days": [], "no_calibration": [], "output_correction": [], "details": []}
    for day in TEST_DAYS:
        train, val, test = split_files(day)
        print(day, [recording_day(f) for f in train],
                [recording_day(f) for f in val],
                [recording_day(f) for f in test], "\n")
        model, stats, _ = train_model(train, val, TASK, epochs=EPOCH_RUNS)  # short run to test the pipeline; full runs on GPU
        test_sessions = load_sessions(test, TASK)
        pred, target, index = predict(model, test_sessions, stats, batch_size=BATCH_SIZE, device=DEVICE)
        
        score_dict = score_day(pred, target, index)
        results["days"].append(day)
        results["no_calibration"].append(score_dict["no_calibration"])
        results["output_correction"].append(score_dict["output_correction"])
        results["details"].append(score_dict)
        print(day, score_dict, "\n")

    out = Path("results") / f"tcn_{TASK}_cross_day_epochs_{EPOCH_RUNS}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out}")

    print("Mean no calibration:", np.mean(results["no_calibration"]))
    print("Mean output correction:", np.mean(results["output_correction"]))
