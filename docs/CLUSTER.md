# Cluster Runs

These instructions assume a SLURM cluster with one GPU per Atari job. If your
cluster uses PBS/LSF instead, keep the Python commands and translate only the
scheduler directives.

## 1. Prepare the Environment

On a login node:

```bash
git clone <repo-url> Epsilon-greedy-to-Boltzmann
cd Epsilon-greedy-to-Boltzmann

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[atari,plots]"
```

Check CUDA:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Atari environments also need ALE ROM access. The project dependency includes
Gymnasium's `accept-rom-license` extra, but the exact ROM installation process
can vary by cluster image.

## 2. Run One Smoke Job

Submit a short job first:

```bash
sbatch scripts/cluster/slurm_atari_smoke.sbatch
```

Watch it:

```bash
squeue -u "$USER"
tail -f logs/slurm/egtb-atari-smoke_*.out
```

Expected output directory:

```text
runs/atari_smoke/breakout_annealed_smoke_seed0
```

If this fails, fix the environment, ROMs, CUDA, or memory request before running
the full array.

## 3. Create the Full Atari Manifest

The default manifest uses 4 Breakout strategies and seeds `0 1 2`, for 12 jobs:

```bash
python3 scripts/cluster/make_atari_manifest.py \
  --seeds 0 1 2 \
  --total-steps 1000000 \
  --output-dir runs/breakout_1m \
  --output cluster_manifests/breakout_1m.tsv
```

The script prints the exact array range. For the default, it is:

```bash
sbatch --array=0-11 scripts/cluster/slurm_atari_array.sbatch
```

## 4. Submit the Array

If you use a virtualenv:

```bash
export VENV_PATH="$PWD/.venv"
export MANIFEST="cluster_manifests/breakout_1m.tsv"
sbatch --array=0-11 scripts/cluster/slurm_atari_array.sbatch
```

If your cluster uses conda:

```bash
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_ENV="egtb"
export MANIFEST="cluster_manifests/breakout_1m.tsv"
sbatch --array=0-11 scripts/cluster/slurm_atari_array.sbatch
```

Change the `#SBATCH --partition=gpu` line if your cluster uses another GPU
partition name.

## 5. Monitor and Resume Safely

Check jobs:

```bash
squeue -u "$USER"
```

Inspect logs:

```bash
tail -f logs/slurm/egtb-breakout_<jobid>_<taskid>.out
```

The array script passes `--skip-existing`, so re-submitting the same manifest is
safe for completed jobs. A job is considered complete when its run directory has
`summary.json`.

The trainer also writes lightweight model checkpoints under each run directory:

```text
runs/breakout_1m/<run-name>/checkpoints/
```

These checkpoints are for inspection/model recovery. They do not store the replay
buffer, so interrupted jobs should normally be restarted from scratch.

## 6. Aggregate Results

After jobs finish:

```bash
python3 -m egtb.analyze \
  runs/breakout_1m \
  --output-dir runs/breakout_1m_analysis \
  --solved-threshold 400
```

Open or copy these files:

```bash
runs/breakout_1m_analysis/learning_curves.png
runs/breakout_1m_analysis/final_returns.png
runs/breakout_1m_analysis/strategy_summary.csv
runs/breakout_1m_analysis/per_seed_summary.csv
```
