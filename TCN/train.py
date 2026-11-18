import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from .data import SPECS, Windows, load_session, smooth
from .model import TCN


def main():
    p = argparse.ArgumentParser(description='Train only on explicit held-in session lists.')
    p.add_argument('--task', choices=SPECS, required=True)
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--val', nargs='+', required=True)
    p.add_argument('--out', default='runs/first')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()

    # Same-day files must be grouped by the caller; filenames alone cannot prove day independence.
    if min(a.epochs, a.patience, a.batch_size, a.hidden) < 1:
        p.error('epochs, patience, batch-size, hidden must be positive')
    if set(map(lambda f: Path(f).resolve(), a.train)) & set(map(lambda f: Path(f).resolve(), a.val)):
        p.error('Train and validation files overlap')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    # Load & smooth each session
    train = []
    for filename in a.train:
        x, y, mask = load_session(filename, a.task)
        train.append((smooth(x, a.alpha), y, mask))

    val = []
    for filename in a.val:
        x, y, mask = load_session(filename, a.task)
        val.append((smooth(x, a.alpha), y, mask))

    # Compute normalization from the training dat
    count = sum(len(x) for x, _, _ in train)
    mean = sum(x.sum(axis=0, dtype=np.float64) for x, _, _ in train) / count
    variance = sum(((x - mean)**2).sum(axis=0) for x, _, _ in train) / count
    mean, scale = mean.astype('float32'), np.sqrt(variance).clip(1e-6).astype('float32')
    
    # Constructing windows & batches
    channels, outputs, history = SPECS[a.task]
    train_loader = DataLoader(Windows(train, mean, scale, history), batch_size=a.batch_size, shuffle=True)
    val_sets = [Windows([s], mean, scale, history) for s in val]
    
    # Create the model and optimizer
    model = TCN(channels, outputs, a.hidden, a.dropout).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best, stale, records = float('inf'), 0, []

    for epoch in range(1, a.epochs + 1):
        model.train() # Train for one epoch, backpropagation, update weights
        total, n = 0., 0
        for x, y in train_loader:
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(x); n += len(x)
        model.eval() # evaluate without changing the weights
        squared, bins, scores = 0., 0, []
        with torch.no_grad():
            for dataset in val_sets:
                predictions, targets = [], []
                for x, y in DataLoader(dataset, batch_size=a.batch_size):
                    predictions.append(model(x.to(a.device)).cpu().numpy()); targets.append(y.numpy())
                pred, target = np.concatenate(predictions), np.concatenate(targets)
                squared += float(((pred - target)**2).sum()); bins += target.size
                scores.append(float(r2_score(target, pred, multioutput='variance_weighted')))
        mse = squared / bins
        row = dict(epoch=epoch, train_mse=total/n, val_mse=mse,
                   val_r2_mean=float(np.mean(scores)), val_r2_std=float(np.std(scores)), val_r2_sessions=scores)
        records.append(row); print(json.dumps(row), flush=True)
        if mse < best: # save improvements and stop early
            best, stale = mse, 0
            torch.save(dict(state_dict=model.state_dict(), config=vars(a), mean=mean.tolist(),
                            scale=scale.tolist(), history=history, best_epoch=epoch), out / 'best.pt')
        else:
            stale += 1
        (out / 'metrics.json').write_text(json.dumps(records, indent=2))
        if stale >= a.patience:
            break


if __name__ == '__main__':
    main()
