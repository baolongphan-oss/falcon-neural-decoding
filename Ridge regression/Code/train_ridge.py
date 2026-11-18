# %%
"""Train and locally evaluate a FALCON ridge decoder."""

from pathlib import Path
import shutil
import subprocess
import sys

from google.colab import drive

# %% Choose one dataset for this run
TASK = "h1"  # Choose "h1", "m1", or "m2"

DATASET_IDS = {
    "h1": "000954",
    "m1": "000941",
    "m2": "000953",
}

# Counts we have verified from the downloaded H1 and M1 datasets.
EXPECTED_COUNTS = {
    "h1": {"held_in_calib": 13, "held_out_calib": 14, "minival": 13},
    "m1": {"held_in_calib": 4, "held_out_calib": 3, "minival": 4},
    "m2": {"held_in_calib": 7, "held_out_calib": 6, "minival": 7},
}

if TASK not in DATASET_IDS:
    raise ValueError(f"Unknown task: {TASK}")

DATA = Path("/content/data") / DATASET_IDS[TASK]
REPO = Path("/content/falcon-challenge")

# Keep the established H1 backup names. Other tasks follow the same pattern.
MODEL_NAMES = {
    0: f"sklearn_{TASK}_history0_baseline.pkl",
    5: f"sklearn_{TASK}_history5.pkl",
}

# %% Check the data and prepare the FALCON repository
if not (REPO / "decoder_demos/sklearn_decoder.py").is_file():
    if REPO.exists():
        raise RuntimeError(f"{REPO} exists but the decoder script is missing.")

    subprocess.run(
        [
            "git", "clone",
            "https://github.com/snel-repo/falcon-challenge.git",
            str(REPO),
        ],
        check=True,
    )

if not DATA.is_dir():
    raise FileNotFoundError(
        f"{DATA} is missing. Download the {TASK.upper()} data first."
    )

# The downloaded DANDI folder names differ between tasks.
# Find them by their shared endings instead of guessing the subject name.
folder_endings = {
    "held_in_calib": "held-in-calib",
    "held_out_calib": "held-out-calib",
    "minival": "held-in-minival",
}

source_folders = {}
task_dir = REPO / "data" / TASK
task_dir.mkdir(parents=True, exist_ok=True)

for link_name, ending in folder_endings.items():
    matches = [
        path for path in DATA.iterdir()
        if path.is_dir() and path.name.endswith(ending)
    ]

    if len(matches) != 1:
        raise RuntimeError(f"Expected one {ending} folder in {DATA}; found {len(matches)}.")

    source = matches[0]
    count = len(list(source.glob("*.nwb")))

    if count == 0:
        raise RuntimeError(f"No NWB files found in {source}.")

    expected = EXPECTED_COUNTS.get(TASK, {}).get(link_name)
    if expected is not None and count != expected:
        raise RuntimeError(f"{source.name}: found {count} NWB files, expected {expected}. Check load_data.py.")

    link = task_dir / link_name
    if link.exists() or link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"{link} points to the wrong folder.")
    else:
        link.symlink_to(source, target_is_directory=True)

    source_folders[link_name] = source
    print(f"{TASK.upper()} {link_name}: {count} files")

print("Starter decoder:", REPO / "decoder_demos/sklearn_decoder.py")

# %% Mount Drive and define reusable helpers
drive.mount("/content/drive")

MODEL_DIR = REPO / "local_data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_MODELS = Path("/content/drive/MyDrive/DSCI550_FALCON/models")
DRIVE_MODELS.mkdir(parents=True, exist_ok=True)

ACTIVE_MODEL = MODEL_DIR / f"sklearn_FalconTask.{TASK}.pkl"

def run_falcon(command):
    """Run a FALCON command and show its output if it fails."""
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FALCON command failed (exit {result.returncode}).\n{result.stdout}\n{result.stderr}")

    return result.stdout

def prepare_model(history, retrain=False):
    """Restore a backup, or train and save a new model."""
    saved_model = DRIVE_MODELS / MODEL_NAMES[history]

    if retrain and saved_model.exists():
        raise FileExistsError(
            f"{saved_model} already exists. Keep or rename this backup before intentionally training a replacement."
        )

    if saved_model.is_file() and not retrain:
        shutil.copy2(saved_model, ACTIVE_MODEL)
        print(f"Restored {TASK.upper()} history={history} model from Drive.")
        return saved_model

    command = [
        "decoder_demos/sklearn_decoder.py",
        "--training_dir", str(source_folders["held_in_calib"]),
        "--calibration_dir", str(source_folders["held_out_calib"]),
        "--mode", "all",
        "--task", TASK,
        "--history", str(history),
    ]

    training_output = run_falcon(command)
    print(training_output)

    if not ACTIVE_MODEL.is_file():
        raise RuntimeError("Training finished but no model file was found.")

    shutil.copy2(ACTIVE_MODEL, saved_model)
    print("Saved model to Drive:", saved_model)

    log_path = DRIVE_MODELS / f"sklearn_{TASK}_history{history}_training.txt"
    log_path.write_text(training_output)
    print("Saved training log:", log_path)

    return saved_model

def evaluate_model(history):
    """Evaluate the saved model on this task's local minival files."""
    saved_model = DRIVE_MODELS / MODEL_NAMES[history]

    if not saved_model.is_file():
        raise FileNotFoundError(f"{saved_model} is missing. Prepare the model first.")

    # FALCON reads a fixed active filename for each task.
    shutil.copy2(saved_model, ACTIVE_MODEL)

    output = run_falcon([
        "decoder_demos/sklearn_sample.py",
        "--evaluation", "local",
        "--phase", "minival",
        "--split", TASK,
        "--model-path", str(ACTIVE_MODEL),
    ])

    result_lines = [line for line in output.splitlines() if "Returning result from phase:" in line]
    if not result_lines:
        raise RuntimeError(
            "Evaluation completed, but its result line was not found.\n"
            + output[-2000:]
        )

    print(f"{TASK.upper()} history={history}")
    print(result_lines[-1])

# %% Baseline: all tasks
prepare_model(history=0)
evaluate_model(history=0)

# %% Temporal history experiment: H1 only for now
prepare_model(history=5)
evaluate_model(history=5)
