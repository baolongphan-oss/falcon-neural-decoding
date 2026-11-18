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

# load and validate singla recording
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
    def __init__(self, sessions, mean, scale, history):
        self.sessions = []
        self.index = []
        self.history = history
        for sid, (x, y, mask) in enumerate(sessions):
            normalized = (x - mean) / scale
            padded = np.pad(normalized, ((history - 1, 0), (0, 0)))
            self.sessions.append((padded, y))
            self.index.extend((sid, int(t)) for t in np.flatnonzero(mask))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        x, y = self.sessions[sid]
        return torch.from_numpy(x[t:t+self.history].T.copy()), torch.from_numpy(y[t])
