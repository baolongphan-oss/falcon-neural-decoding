"""
data.py: session loading, smoothing, and lazy window datasets for the TCN.

Change log (newest first):
- 2026-11-15 (Long): added falcon_smooth (240 ms, matches FALCON's
  apply_exponential_filter); Windows now standardizes targets
  (y_mean, y_scale); typo fixes.
- 2026-11-14 (Uma): initial version (EMA smoothing, lazy Windows).
"""


from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# task dimensions
SPECS = {'m1': (64, 16, 30), 'm2': (96, 2, 7), 'h1': (176, 7, 30)}

# reduce noise using past observations
# smaller alpha = stronger smoothing, alpha = 1 means counts are unchanged
def smooth(x, alpha):
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be in (0, 1]')
    result = np.empty_like(x, dtype=np.float32)
    state = np.zeros(x.shape[1], dtype=np.float32)
    for t, row in enumerate(x):
        state = alpha * row + (1 - alpha) * state
        result[t] = state
    return result


# implement the falcon smoothing method
def falcon_smooth(x, fade_ms=240, bin_ms=20):
    """Causal exponential smoothing, matching FALCON's apply_exponential_filter.

    x: spike counts, shape (time, electrodes)
    fade_ms: time constant (tau); how fast older bins fade, in ms
    bin_ms: length of one time bin, in ms
    Returns smoothed firing-rate estimates, same shape as x, float32.
    """
    # ages of the bins in the kernel
    t = np.arange(0, fade_ms, bin_ms)
    # provide weighting for each age; newest bin weighs most, then fade exponentially
    kernel = np.exp(-t / fade_ms)
    # normalize the weights
    kernel = kernel / kernel.sum()
    result = np.empty_like(x, dtype=np.float32)
    for channel in range(x.shape[1]):
        # smooth each electrode's signal and crop to len(x) to remove spillover
        result[:, channel] = np.convolve(x[:, channel], kernel, mode="full")[:len(x)]
    return result


# load and validate a single recording
def load_session(path, task):
    path = Path(path)
    
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            x, y = z['x'].copy(), z['y'].copy()
            mask = z['mask'].astype(bool) if 'mask' in z else np.ones(len(x), bool)
    else:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        x, y, _, mask = load_nwb(path, dataset=FalconTask[task])
    
    channels, outputs, _ = SPECS[task]
    x, y, mask = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(mask, dtype=bool)
    
    if x.ndim != 2 or x.shape[1] != channels or y.shape != (len(x), outputs) or mask.shape != (len(x),):
        raise ValueError(f'Unexpected shapes in {path}: {x.shape}, {y.shape}, {mask.shape}')
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f'Empty or nonfinite neural data: {path}')
    mask = mask & np.isfinite(y).all(axis=1)
    if not mask.any():
        raise ValueError(f'No labeled, evaluable bins: {path}')
    return x, y, mask

# convert sessions into training examples
class Windows(Dataset):
    """Lazy windows avoid materializing time x channels x history arrays.

    History resets at file/session boundaries, remains continuous across trials.
    Masked bins still contribute neural history but never training targets.
    """
    def __init__(self, sessions, x_mean, x_scale, y_mean, y_scale, history):
        self.sessions = []
        self.index = []
        self.history = history
        for sid, (x, y, mask) in enumerate(sessions):
            normalized_x = (x - x_mean) / x_scale
            padded = np.pad(normalized_x, ((history - 1, 0), (0, 0)))
            normalized_y = (y - y_mean) / y_scale
            self.sessions.append((padded, normalized_y))
            self.index.extend((sid, int(t)) for t in np.flatnonzero(mask))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        x, y = self.sessions[sid]
        return torch.from_numpy(x[t:t+self.history].T.copy()), torch.from_numpy(y[t])
