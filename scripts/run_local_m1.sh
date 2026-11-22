#!/usr/bin/env bash
set -euo pipefail
# This checkout uses the virtual environment/data in its parent workspace.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$REPO_ROOT")"
cd "$REPO_ROOT"
mkdir -p runs/local-setup
export FALCON_DATA="${FALCON_DATA:-$WORKSPACE_ROOT/data/000941}"
export EPOCHS="${EPOCHS:-2}"
export VALIDATION="${VALIDATION:-minival}"
export SMOOTHING="${SMOOTHING:-ema}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Isolate generated results from the committed group results.
"$WORKSPACE_ROOT/.venv/bin/python" -B -u - <<'PY' 2>&1 | tee runs/local-setup/m1.log
from pathlib import Path
import runpy
import os
import sys
import torch
repo = Path.cwd()
sys.path.insert(0, str(repo))
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
os.chdir(repo / 'runs' / 'local-setup')
runpy.run_module('TCN.cross_day_tcn', run_name='__main__')
PY
