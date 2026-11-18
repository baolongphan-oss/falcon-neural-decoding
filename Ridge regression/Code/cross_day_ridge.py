# %%
"""Test ridge decoders on later H1 or M1 recording days in Colab.

For each test day, train only on earlier files. Reserve the first minute
of the first test file for output correction, then score the remaining data.
Run load_data.py and the repository setup in train_ridge.py first.
"""

from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
from google.colab import drive
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb

# %% Select a task and check the data
TASK = "h1"  # Choose "h1", "m1", or "m2"

TASKS = {
    "h1": {
        "folder": "/content/data/000954/sub-HumanPitt-held-in-calib",
        "files": 13,
        "channels": 176,
        "days": ("19250108", "19250113", "19250115", "19250119", "19250120"),
    },
    "m1": {
        "folder": "/content/data/000941/sub-MonkeyL-held-in-calib",
        "files": 4,
        "channels": 64,
        "days": ("20120926", "20120927", "20120928"),
    },
    "m2": {
    "folder": "/content/data/000953/sub-MonkeyN-held-in-calib",
    "files": 7,
    "channels": 96,
    "days": ("20201020", "20201027", "20201028"),
    },
}

REPO = Path("/content/falcon-challenge")
MODEL_FOLDER = Path("/content/drive/MyDrive/DSCI550_FALCON/models")
CALIBRATION_BINS = 3000  # One minute at 20 ms per bin.
CORRECTION_ALPHA = 100.0

if TASK not in TASKS:
    raise ValueError(f"Unknown task: {TASK}")

data = TASKS[TASK]
DAYS_TO_RUN = data["days"]
HISTORIES_TO_RUN = (0, 5)
data_folder = Path(data["folder"])
active_model = REPO / "local_data" / f"sklearn_FalconTask.{TASK}.pkl"

if not (REPO / "decoder_demos/sklearn_decoder.py").is_file():
    raise FileNotFoundError("FALCON repository is missing. Run train_ridge.py setup first.")

files = sorted(data_folder.glob("*.nwb"))
if len(files) != data["files"]:
    raise RuntimeError(
        f"{TASK.upper()}: found {len(files)} training files; "
        f"expected {data['files']}. Run load_data.py first."
    )

drive.mount("/content/drive")
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

for module_dir in (REPO, REPO / "decoder_demos"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from filtering import apply_exponential_filter
from decoder_demos.decoding_utils import generate_lagged_matrix

# %% Reuse or train a model from earlier dates
def recording_day(path):
    """Return the eight digit recording date from an NWB filename."""

    date = path.stem.split("_ses-", 1)[1]
    if TASK == "m2":
        return date[:10].replace("-", "")
    return date[:8]

def training_folders(day, earlier_files):
    """Link only earlier files into a folder accepted by the starter trainer."""

    work = Path("/content") / f"{TASK}_cross_day"
    train_dir = work / f"before_{day}"
    empty_dir = work / "no_calibration"
    train_dir.mkdir(parents=True, exist_ok=True)
    empty_dir.mkdir(parents=True, exist_ok=True)

    expected = {path.name for path in earlier_files}
    present = {path.name for path in train_dir.iterdir() if path.suffix == ".nwb"}
    if present - expected:
        raise RuntimeError(f"Unexpected training files in {train_dir}: {present - expected}")
    if list(empty_dir.glob("*.nwb")):
        raise RuntimeError(f"Calibration folder must be empty: {empty_dir}")

    for source in earlier_files:
        link = train_dir / source.name
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"{link} points to the wrong file")
        else:
            link.symlink_to(source)

    if {path.name for path in train_dir.glob("*.nwb")} != expected:
        raise RuntimeError("Training folder does not contain exactly the earlier files")

    return train_dir, empty_dir

def load_or_train(day, history, earlier_files):
    """Keep a separate Drive backup for every task, day, and history."""

    backup = MODEL_FOLDER / f"sklearn_{TASK}_before_{day}_history{history}.pkl"

    if backup.is_file():
        print("Using saved model:", backup.name)
    else:
        train_dir, empty_dir = training_folders(day, earlier_files)
        command = [
            sys.executable, "decoder_demos/sklearn_decoder.py",
            "--training_dir", str(train_dir),
            "--calibration_dir", str(empty_dir),
            "--mode", "all", "--task", TASK, "--history", str(history),
        ]
        run = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        if run.returncode:
            raise RuntimeError(f"Training failed:\n{run.stdout}\n{run.stderr}")

        print(run.stdout.strip())  # Includes CV fit score for a newly trained model.
        if not active_model.is_file():
            raise RuntimeError("Training finished without a model file")
        shutil.copy2(active_model, backup)
        print("Saved model to Drive:", backup.name)

    # Read only models created by this project; pickle files can execute code.
    with backup.open("rb") as file:
        model = pickle.load(file)

    if (
        model["task"] != getattr(FalconTask, TASK)
        or model["history"] != history
        or model["decoder"].n_features_in_ != data["channels"] * (history + 1)
        or len(model["x_mean"]) != len(earlier_files)
    ):
        raise RuntimeError(f"{backup.name} does not match this training split")
    return model

# %% Predict and score the test files
def predict_file(path, model, mean, std, history):
    """Use the starter model's filter and history feature layout."""

    neural, targets, _, mask = load_nwb(path, getattr(FalconTask, TASK))
    features = (apply_exponential_filter(neural) - mean) / std

    if history:
        lagged = generate_lagged_matrix(features, history)
        if len(lagged) != len(targets) - history:
            raise RuntimeError("History features and targets do not align")
        predictions = np.full(targets.shape, np.nan)
        predictions[history:] = model["decoder"].predict(lagged)
    else:
        predictions = model["decoder"].predict(features)

    return path, targets, np.asarray(mask, dtype=bool), predictions

def valid_rows(targets, mask, predictions):
    """Rows with an evaluation label and finite targets and predictions."""

    return mask & np.isfinite(targets).all(axis=1) & np.isfinite(predictions).all(axis=1)

def test_new_day(day, history=0):
    """Fit on earlier dates and score before and after one minute correction."""

    if day not in data["days"] or history not in (0, 5):
        raise ValueError(f"Unsupported day or history for {TASK.upper()}: {day}, {history}")

    earlier = [path for path in files if recording_day(path) < day]
    test_files = [path for path in files if recording_day(path) == day]
    if not earlier or not test_files:
        raise RuntimeError(f"{day}: found {len(earlier)} earlier and {len(test_files)} test files")

    print(f"\n{TASK.upper()} test day {day}, history={history}")
    print(f"Earlier training files: {len(earlier)}; test files: {len(test_files)}")
    model = load_or_train(day, history, earlier)

    # Reference statistics come only from earlier recordings.
    mean = np.mean(np.stack(list(model["x_mean"].values())), axis=0)
    std = np.mean(np.stack(list(model["x_std"].values())), axis=0)
    std[std == 0] = 1
    test_data = [predict_file(path, model, mean, std, history) for path in test_files]

    # The first minute of the first file is calibration data, not test data.
    _, targets, mask, predictions = test_data[0]
    if len(targets) <= CALIBRATION_BINS:
        raise RuntimeError("First test file is too short for calibration")
    calibration = valid_rows(
        targets[:CALIBRATION_BINS], mask[:CALIBRATION_BINS],
        predictions[:CALIBRATION_BINS],
    )
    if calibration.sum() < 2:
        raise RuntimeError("Too few valid calibration bins")

    base = predictions[:CALIBRATION_BINS][calibration]
    residual = targets[:CALIBRATION_BINS][calibration] - base
    correction = Ridge(alpha=CORRECTION_ALPHA).fit(base, residual)
    print("Calibration bins used:", int(calibration.sum()))

    file_scores = []
    for index, (path, targets, mask, predictions) in enumerate(test_data):
        start = CALIBRATION_BINS if index == 0 else 0
        scored = valid_rows(targets[start:], mask[start:], predictions[start:])
        if scored.sum() < 2:
            raise RuntimeError(f"Too few test bins in {path.name}")

        actual = targets[start:][scored]
        original = predictions[start:][scored]
        adjusted = original + correction.predict(original)
        before = float(r2_score(actual, original, multioutput="variance_weighted"))
        after = float(r2_score(actual, adjusted, multioutput="variance_weighted"))
        file_scores.append({"file": path.name, "bins": int(scored.sum()),
                            "no_calibration": before, "output_correction": after})
        print(f"File {index + 1}: {int(scored.sum())} scored bins; "
              f"no calibration={before:.4f}; output correction={after:.4f}")

    summary = {
        "task": TASK, "day": day, "history": history,
        "training_files": len(earlier), "calibration_bins_used": int(calibration.sum()),
        "no_calibration_mean": float(np.mean([s["no_calibration"] for s in file_scores])),
        "output_correction_mean": float(np.mean([s["output_correction"] for s in file_scores])),
        "file_scores": file_scores,
    }
    print(f"Mean no calibration: {summary['no_calibration_mean']:.4f}")
    print(f"Mean output correction: {summary['output_correction_mean']:.4f}")

    return summary

# %% Run the selected dates and history settings
results = {}
for day in DAYS_TO_RUN:
    for history in HISTORIES_TO_RUN:
        results[(day, history)] = test_new_day(day, history)
