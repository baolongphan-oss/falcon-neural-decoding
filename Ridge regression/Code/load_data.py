# %%
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "falcon-challenge"], check=True)

# %%
"""Download and inspect FALCON data in a Colab runtime."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb


# %% Configuration
DATASETS = {
    "h1": "000954",
    "m1": "000941",
    "m2": "000953"
}

task = "h1" # Choose "h1", "m1", or "m2"
dandiset_id = DATASETS[task]
data_root = Path("/content/data") / dandiset_id

ASSETS_URL = (f"https://api.dandiarchive.org/api/dandisets/{dandiset_id}/versions/draft/assets/")

session = requests.Session()

# %% List every page of assets
def list_assets():
    assets = []
    url = ASSETS_URL
    params = {"page_size": 100}

    while url:
        response = session.get(url, params=params, timeout=(10, 60))
        response.raise_for_status()
        page = response.json()

        assets.extend(page["results"])
        url = page.get("next")
        params = None  # The next URL already contains its page information.

    return [asset for asset in assets if asset["path"].endswith(".nwb")]

assets = list_assets()
print(f"Found {len(assets)} {task.upper()} NWB files")

groups = defaultdict(lambda: {"count": 0, "bytes": 0})

for asset in assets:
    group = asset["path"].split("/")[0]
    groups[group]["count"] += 1
    groups[group]["bytes"] += asset["size"]

for group, info in sorted(groups.items()):
    size_mb = info["bytes"] / 1024**2
    print(f"{group}: {info['count']} files, {size_mb:.1f} MB")

# %% Download files, keeping DANDI's folder structure
def download_asset(asset):
    destination = data_root / asset["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_size = asset["size"]

    if destination.is_file() and destination.stat().st_size == expected_size:
        return destination, False

    download_url = (f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/")
    temporary = destination.with_name(destination.name + ".part")

    # A failed download stays in .part instead of appearing complete.
    try:
        with session.get(download_url, stream=True, timeout=(10, 120)) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Size mismatch for {asset['path']}: expected {expected_size}, got {actual_size}")

        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination, True

downloaded = 0

for asset in assets:
    path, was_downloaded = download_asset(asset)
    if was_downloaded:
        downloaded += 1
        print("Downloaded:", asset["path"])

print(f"Done: {downloaded} downloaded, {len(assets) - downloaded} already complete")

# %% Verify local files and inspect the smallest training example
missing_or_incomplete = []

for asset in assets:
    path = data_root / asset["path"]
    if not path.is_file() or path.stat().st_size != asset["size"]:
        missing_or_incomplete.append(asset["path"])

if missing_or_incomplete:
    raise RuntimeError(f"{len(missing_or_incomplete)} files are missing or incomplete: {missing_or_incomplete[:3]}")

print("Verified files:", len(assets))

training_assets = [asset for asset in assets if "held-in-calib" in asset["path"]]
example_asset = min(training_assets, key=lambda asset: asset["size"])
example_path = data_root / example_asset["path"]

neural, targets, trial_change, eval_mask = load_nwb(example_path, getattr(FalconTask, task))

print("Example:", example_path.name)
print("Neural:", neural.shape, neural.dtype)
print("Targets:", targets.shape)
print("Trial markers:", trial_change.shape)
print("Evaluation mask:", eval_mask.shape)
print("Targets finite:", np.isfinite(targets).all())

time_seconds = np.arange(len(neural)) * 0.02

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(time_seconds, neural[:, 0])
axes[0].set(title="Electrode channel 0", ylabel="Spike count")

axes[1].plot(time_seconds, targets[:, 0])
axes[1].set(
    title="Movement output 0",
    xlabel="Time (seconds)",
    ylabel="Movement value",
)

fig.tight_layout()
plt.show()
