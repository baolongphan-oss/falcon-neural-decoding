"""
train.py: TCN training with early stopping on held-in sessions.

Change log (newest first):
- 2026-11-15 (Long): refactored into functions (load_sessions, compute_stats,
  evaluate, train_model); added --smoothing {falcon, ema} (default falcon);
  target standardization (MSE in standardized units, R² in mV);
  restore best weights after training; settings as module constants.
- 2026-11-14 (Uma): initial version.
"""

import argparse
import json
import random
import copy
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from .data_long import SPECS, Windows, load_session, smooth, falcon_smooth
from .model import TCN

# Default training settings, shared by the command line and train_model()
EPOCHS = 50          # maximum passes over the training data
PATIENCE = 8         # epochs without validation improvement before stopping early
BATCH_SIZE = 256     # windows per training step
HIDDEN = 64          # filters per convolution layer
DROPOUT = 0.1        # fraction of features randomly zeroed during training
ALPHA = 0.2          # EMA smoothing strength (only used when SMOOTHING = 'ema')
LR = 1e-3            # learning rate
WEIGHT_DECAY = 1e-4  # penalty that keeps weights small
SEED = 0             # random seed for reproducibility
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOOTHING = 'falcon' # or 'ema'


def load_sessions(files, task, smoothing=SMOOTHING, alpha=ALPHA):
    sessions = []
    for f in files:
        x, y, mask = load_session(f, task)
        if smoothing == "falcon":
            smoothed_x = falcon_smooth(x)
        elif smoothing == "ema":
            smoothed_x = smooth(x, alpha)
        else:
            raise ValueError(f"Unknown smoothing: {smoothing}")
        sessions.append((smoothed_x, y, mask))
    return sessions 


def compute_stats(train):
    if not train:
        raise ValueError("Training session is missing!")
    x_array = np.concatenate([x for x, _, _ in train])
    x_mean = x_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = x_array.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)

    # stack muscle targets from training sessions; keep only scored/masked bins
    y_array = np.concatenate([y[mask] for _, y, mask in train])
    # calculate each muscle's average and spread; use clip to avoid division by 0
    y_mean = y_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = y_array.std(axis=0, dtype=np.float64).clip(1e-6).astype(np.float32)
    return x_mean, x_scale, y_mean, y_scale


def evaluate(model, val_sets, batch_size, device, y_mean, y_scale):
    model.eval() # evaluate without changing the weights
    squared, n_values, scores = 0., 0, []

    with torch.no_grad():
        for dataset in val_sets:
            predictions, targets = [], []
            for x, y in DataLoader(dataset, batch_size=batch_size):
                predictions.append(model(x.to(device)).cpu().numpy())
                targets.append(y.numpy())

            pred, target = np.concatenate(predictions), np.concatenate(targets)
            # calculate mean square error in standardized units
            squared += float(((pred - target)**2).sum())
            n_values += target.size

            # convert values back to mV (for muscles data) for r^2 to match Ridge regression
            pred = pred * y_scale + y_mean 
            target = target * y_scale + y_mean
            scores.append(float(r2_score(target, pred, multioutput='variance_weighted')))

    mse = squared / n_values
    return mse, scores
    

def train_model(train_files, val_files, task, *, 
                out=None, alpha=ALPHA, smoothing=SMOOTHING, # out=None means don't save anything to disk, unless prompted
                dropout=DROPOUT, lr=LR, weight_decay=WEIGHT_DECAY,
                device=DEVICE, epochs=EPOCHS, patience=PATIENCE, 
                batch_size=BATCH_SIZE, hidden=HIDDEN, seed=SEED):

    config = dict(task=task, alpha=alpha, smoothing=smoothing, 
                  hidden=hidden, dropout=dropout, lr=lr,
                  weight_decay=weight_decay, batch_size=batch_size, seed=seed)
    
    # Same-day files must be grouped by the caller; filenames alone cannot prove day independence.
    if min(epochs, patience, batch_size, hidden) < 1:
        raise ValueError('epochs, patience, batch-size, hidden must be positive')
    if set(map(lambda f: Path(f).resolve(), train_files)) & set(map(lambda f: Path(f).resolve(), val_files)):
        raise ValueError('Train and validation files overlap')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Load & smooth each session
    train = load_sessions(train_files, task, smoothing, alpha)
    val = load_sessions(val_files, task, smoothing, alpha)

    # Compute normalization from the training data
    x_mean, x_scale, y_mean, y_scale = compute_stats(train)

    # Constructing windows & batches
    channels, outputs, history = SPECS[task]
    train_loader = DataLoader(Windows(train, x_mean, x_scale, y_mean, y_scale, history), batch_size=batch_size, shuffle=True)
    val_sets = [Windows([s], x_mean, x_scale, y_mean, y_scale, history) for s in val]
    
    # Create the model and optimizer
    model = TCN(channels, outputs, hidden, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

    best, stale, records = float('inf'), 0, []
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train() # Train for one epoch, backpropagation, update weights
        total, n = 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(x); n += len(x)

        mse, scores = evaluate(model, val_sets, batch_size, device, y_mean, y_scale)
        row = dict(epoch=epoch, train_mse=total/n, val_mse=mse,
                    val_r2_mean=float(np.mean(scores)), val_r2_std=float(np.std(scores)), val_r2_sessions=scores)
        records.append(row)
        print(json.dumps(row), flush=True)
        if mse < best: # save improvements and stop early
            best, stale = mse, 0
            best_state = copy.deepcopy(model.state_dict())
            if out is not None:
                torch.save(dict(state_dict=best_state, config=config, 
                                x_mean=x_mean.tolist(), x_scale=x_scale.tolist(), 
                                y_mean=y_mean.tolist(), y_scale=y_scale.tolist(),
                                history=history, best_epoch=epoch), out / 'best.pt')
        else:
            stale += 1

        if out is not None:
            (out / 'metrics.json').write_text(json.dumps(records, indent=2))
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    stats = dict(x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale, history=history)
    return model, stats, records

def main():
    p = argparse.ArgumentParser(description='Train only on explicit held-in session lists.')
    p.add_argument('--task', choices=SPECS, required=True)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--val', nargs='+', required=True)
    p.add_argument('--out', default='runs/first')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--hidden', type=int, default=HIDDEN)
    p.add_argument('--dropout', type=float, default=DROPOUT)
    p.add_argument('--alpha', type=float, default=ALPHA)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--device', default=DEVICE)
    p.add_argument('--smoothing', choices=['falcon', 'ema'], default=SMOOTHING)
    a = p.parse_args()
    
    model, stats, records = train_model(a.train, a.val, a.task, 
                                        out=a.out, alpha=a.alpha, smoothing=a.smoothing,
                                        dropout=a.dropout, lr=a.lr, weight_decay=a.weight_decay, 
                                        device=a.device, epochs=a.epochs, patience=a.patience, 
                                        batch_size=a.batch_size, hidden=a.hidden, seed=a.seed)

if __name__ == '__main__':
    main()
