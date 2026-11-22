"""Repeat rolling calibration or validation across training seeds.

Run: FALCON_DATA=/path/to/000941 python -m TCN.multiseed_calibration --epochs 50
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from .cross_day_tcn import (DATA_ROOT, TASK, make_rolling_splits,
                            run_rolling_calibration, run_rolling_validation)

METHODS = ('no_calibration', 'output_correction', 'anchored_refit')


def summarize(runs):
    """Equal-weight days within each seed; sample SD across seed means."""
    seeds = sorted(runs)
    days = [fold['calibration_day'] for fold in runs[seeds[0]]]
    values = {}
    for seed in seeds:
        if [fold['calibration_day'] for fold in runs[seed]] != days:
            raise ValueError('Day order differs across seeds')
    for method in METHODS:
        values[method] = np.array([[fold[method] for fold in runs[seed]] for seed in seeds])
        if not np.isfinite(values[method]).all():
            raise ValueError('Nonfinite R²')

    def stats(array):
        return dict(mean=float(array.mean()),
                    std_across_seeds=float(array.std(ddof=1)) if len(array) > 1 else None)

    summary = dict(seeds=seeds, days=days, std_definition='sample SD across seeds (ddof=1)',
                   methods={}, paired_gains={}, per_day={})
    for method, array in values.items():
        means = array.mean(axis=1)
        summary['methods'][method] = dict(stats(means), seed_means=means.tolist())
    comparisons = [('output_correction', 'no_calibration'),
                   ('anchored_refit', 'no_calibration'),
                   ('anchored_refit', 'output_correction')]
    for adapted, baseline in comparisons:
        gains = values[adapted] - values[baseline]
        summary['paired_gains'][f'{adapted}_minus_{baseline}'] = dict(
            stats(gains.mean(axis=1)), seed_mean_gains=gains.mean(axis=1).tolist(),
            positive_seed_day_pairs=int((gains > 0).sum()), total_seed_day_pairs=int(gains.size))
    for i, day in enumerate(days):
        summary['per_day'][day] = {method: stats(array[:, i]) for method, array in values.items()}
    return summary


def summarize_validation(runs):
    """Summarize next-day checkpoint-selection scores, not independent test scores."""
    seeds = sorted(runs)
    days = [fold['val_day'] for fold in runs[seeds[0]]]
    if any([fold['val_day'] for fold in runs[seed]] != days for seed in seeds):
        raise ValueError('Day order differs across seeds')
    values = np.array([[fold['val_r2_mean'] for fold in runs[seed]] for seed in seeds])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite R²')
    means = values.mean(axis=1)
    return dict(seeds=seeds, days=days, mean=float(means.mean()),
                std_across_seeds=float(means.std(ddof=1)) if len(seeds) > 1 else None,
                seed_means=means.tolist(),
                selected_epochs={day: [runs[seed][i]['best_epoch'] for seed in seeds]
                                 for i, day in enumerate(days)},
                score_interpretation='Next-day model-selection scores; not independent test scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('calibration', 'validation'), default='calibration')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--epochs', type=int, default=2, help='Maximum epochs per fold')
    parser.add_argument('--patience', type=int, default=8, help='Epochs without validation improvement before stopping')
    parser.add_argument('--out', type=Path, help='New output directory; defaults to an epoch-tagged, timestamped folder')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error('Provide distinct seeds')
    if args.threads < 1:
        parser.error('threads must be positive')
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    if args.out is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.out = Path('runs') / f'multiseed_{args.mode}_epochs_{args.epochs}_{stamp}'
    if args.out.exists():
        parser.error(f'Output directory already exists: {args.out}. Choose a new --out directory.')
    torch.set_num_threads(args.threads)
    splits = make_rolling_splits(sorted(DATA_ROOT.glob('*held-in-calib/*.nwb')))
    settings = dict(epochs=args.epochs, patience=args.patience,
                    smoothing='ema', hidden=32, dropout=0.3, weight_decay=1e-3)
    args.out.mkdir(parents=True, exist_ok=False)
    print(f'Saving results to: {args.out}', flush=True)
    metadata = dict(task=TASK, mode=args.mode, data_root=str(DATA_ROOT.resolve()), settings=settings,
                    seeds=args.seeds, torch=torch.__version__, numpy=np.__version__,
                    threads=args.threads,
                    early_stopping='earlier-day minival' if args.mode == 'calibration' else 'next chronological day',
                    calibration='first 3000 bins of first session; score remaining valid bins' if args.mode == 'calibration' else None)
    (args.out / 'config.json').write_text(json.dumps(metadata, indent=2))
    runs = {}
    for seed in args.seeds:
        print(f'\n=== Seed {seed} ===', flush=True)
        runner = run_rolling_calibration if args.mode == 'calibration' else run_rolling_validation
        runs[seed] = runner(splits, task=TASK, seed=seed, **settings)
        (args.out / f'seed_{seed}.json').write_text(json.dumps(runs[seed], indent=2))
        summary = summarize(runs) if args.mode == 'calibration' else summarize_validation(runs)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
