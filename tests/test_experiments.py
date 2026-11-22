"""Check chronological separation and entry-point output without training a TCN."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TCN import cross_day_tcn as experiment
from TCN.multiseed_calibration import summarize, summarize_validation


class ExperimentsTest(unittest.TestCase):
    def test_whole_days_stay_together(self):
        files = [Path(f'sub-held-in-calib_ses-{day}_behavior+ecephys.nwb')
                 for day in ('20120927', '20120924', '20120926')]
        files.append(Path('other-held-in-calib_ses-20120924_behavior+ecephys.nwb'))
        splits = experiment.make_rolling_splits(files)
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]['train_files']), 2)
        for split in splits:
            self.assertTrue(all(day < split['val_day'] for day in split['train_days']))
            self.assertFalse(set(split['train_files']) & set(split['val_files']))

    def test_main_dispatch_and_distinct_result_shapes(self):
        for mode, runner, result in (
            ('cross-day', 'run_cross_day', dict(no_calibration=[.1], output_correction=[.2], anchored_refit=[.3])),
            ('rolling-validation', 'run_rolling_validation', [dict(val_r2_mean=.4)]),
            ('rolling-calibration', 'run_rolling_calibration', [dict(no_calibration=.1, output_correction=.2, anchored_refit=.3)])):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'new_run'
                with patch.object(experiment, runner, return_value=result) as run, patch.object(experiment, 'make_rolling_splits', return_value=[]):
                    experiment.main(['--mode', mode, '--epochs', '50', '--out', str(out)])
                self.assertEqual(json.loads((out/'results.json').read_text()), result)
                self.assertEqual(run.call_args.kwargs['epochs'], 50)
                self.assertEqual(run.call_args.kwargs['hidden'], 64 if mode == 'cross-day' else 32)
                with self.assertRaises(SystemExit):
                    experiment.main(['--out', str(out)])

    def test_early_stopping_uses_only_earlier_day_minival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root/'subject-held-in-minival'; group.mkdir()
            earlier = group/'sub_ses-20120924_behavior+ecephys.nwb'; earlier.touch()
            (group/'sub_ses-20120926_behavior+ecephys.nwb').touch()
            split = dict(train_days=['20120924'], val_day='20120926', train_files=[Path('train')], val_files=[Path('nextday')])
            model = __import__('torch').nn.Linear(1, 1)
            records = [dict(epoch=1, val_mse=1.)]
            with patch.object(experiment, 'DATA_ROOT', root), patch.object(experiment, 'train_model', return_value=(model, {}, records)) as train, patch.object(experiment, 'load_sessions', return_value=[]), patch.object(experiment, 'predict', return_value=(None, None, None, None)), patch.object(experiment, 'score_day', return_value={}):
                experiment.run_rolling_calibration([split], 'm1', smoothing='ema')
            self.assertEqual(train.call_args.kwargs['val_files'], [earlier])

    def test_summary_uses_seed_means_and_paired_gains(self):
        runs={seed:[dict(calibration_day='day', no_calibration=float(seed), output_correction=seed+.1, anchored_refit=seed+.2)] for seed in (0,1,2)}
        result=summarize(runs)
        self.assertEqual(result['methods']['no_calibration']['mean'], 1.)
        self.assertEqual(result['methods']['no_calibration']['std_across_seeds'], 1.)
        self.assertAlmostEqual(result['paired_gains']['anchored_refit_minus_output_correction']['mean'], .1)
        validation={seed:[dict(val_day='day', val_r2_mean=float(seed), best_epoch=seed+1)] for seed in (0,1,2)}
        self.assertEqual(summarize_validation(validation)['std_across_seeds'], 1.)


if __name__ == '__main__':
    unittest.main()
