# %%
"""Plot recorded H1, M1, and M2 simulated new day results.

Scores are copied from cross_day_ridge.py. This script does not retrain or evaluate models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive

# %% Recorded scores
# Each score is the mean across test files for that day.
# Overall means give each day equal weight.
RESULTS = {
    "h1": {
        "days": ("Jan 08", "Jan 13", "Jan 15", "Jan 19", "Jan 20"),
        0: {
            "no_calibration": (0.1390, 0.1196, 0.1358, 0.1372, 0.1585),
            "output_correction": (0.1500, 0.1362, 0.1422, 0.1568, 0.1654),
        },
        5: {
            "no_calibration": (0.1730, 0.1196, 0.1598, 0.1555, 0.1790),
            "output_correction": (0.1812, 0.1526, 0.1678, 0.1809, 0.1919),
        },
    },
    "m1": {
        "days": ("Sep 26", "Sep 27", "Sep 28"),
        0: {
            "no_calibration": (0.2098, 0.1315, 0.3427),
            "output_correction": (0.3880, 0.3912, 0.3817),
        },
        5: {
            "no_calibration": (0.0604, 0.1676, 0.3092),
            "output_correction": (0.3417, 0.4061, 0.4495),
        },
    },
    "m2": {
        "days": ("Oct 20", "Oct 27", "Oct 28"),
        0: {
            "no_calibration": (-0.0551, 0.0891, 0.1074),
            "output_correction": (0.0396, 0.1134, 0.1424),
        },
        5: {
            "no_calibration": (-0.0283, 0.1261, 0.1331),
            "output_correction": (0.0854, 0.1493, 0.1640),
        },
    },
}

drive.mount("/content/drive")
FIGURE_FOLDER = Path("/content/drive/MyDrive/DSCI550_FALCON")
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

# %% Plot one task
def plot_task(task, scores):
    """Compare both histories before and after output correction."""

    days = scores["days"]
    positions = np.arange(len(days))
    width = 0.38

    panels = (
        ("no_calibration", "No calibration"),
        ("output_correction", "1 minute output correction"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    lowest = min(
        min(scores[history][method])
        for history in (0, 5)
        for method, _ in panels
    )
    highest = max(
        max(scores[history][method])
        for history in (0, 5)
        for method, _ in panels
    )

    for ax, (method, title) in zip(axes, panels):
        for history, offset in ((0, -width / 2), (5, width / 2)):
            values = np.array(scores[history][method])

            if len(values) != len(days):
                raise ValueError(f"{task} history={history}: day and score counts differ")

            ax.bar(positions + offset, values, width, label=f"History={history}")

        ax.set_title(title)
        ax.set_xticks(positions, days)
        ax.set_ylim(min(0, lowest * 1.15), highest * 1.15)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()

    axes[0].set_ylabel("R² (mean across test files on each day)")
    fig.suptitle(f"{task.upper()} ridge: simulated new day tests")
    fig.supxlabel("Test recording day")
    fig.tight_layout()

    output = FIGURE_FOLDER / f"{task}_new_day_history_comparison.png"
    fig.savefig(output, dpi=200)
    plt.show()
    plt.close(fig)

    print("Saved figure:", output)

    for history in (0, 5):
        before = np.mean(scores[history]["no_calibration"])
        after = np.mean(scores[history]["output_correction"])

        print(
            f"{task.upper()} history={history}: "
            f"no calibration={before:.4f}, "
            f"output correction={after:.4f}, "
            f"gain={after - before:+.4f}"
        )

# %% Create the H1, M1, and M2 figures
for task_name, task_scores in RESULTS.items():
    plot_task(task_name, task_scores)
