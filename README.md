# Epsilon-greedy-to-Boltzmann

This project studies how a DQN agent can move from early epsilon-greedy
exploration to later Boltzmann exploration when its Q-function becomes more
informative.

The implementation is a small, reproducible experiment framework rather than a
single notebook. It includes a DQN baseline, multiple exploration policies,
Gymnasium toy-environment configs, an Atari-ready config path, logging, plotting,
and tests.

## Why Gymnasium

The original project text mentions OpenAI Gym. Current development has moved to
[Gymnasium](https://gymnasium.farama.org/), the maintained successor/fork of Gym,
so the code uses `gymnasium` while keeping the same environment style. Atari
support uses Gymnasium's ALE integration, which requires Atari ROM installation
and license acceptance as described in the
[Gymnasium Atari docs](https://gymnasium.farama.org/environments/atari/).

## Implemented exploration variants

All strategies operate on discrete-action Q-values and can be selected with
`exploration.name` in YAML or `--strategy` on the CLI.

| Strategy | Name | Behavior |
| --- | --- | --- |
| Epsilon-greedy | `epsilon_greedy` | Standard DQN baseline with a linear epsilon schedule. |
| Boltzmann | `boltzmann` | Samples all actions with a softmax over Q-values and a temperature schedule. |
| Annealed switch | `annealed_epsilon_boltzmann` | Smoothly mixes epsilon-greedy into Boltzmann between configured switch steps. |
| Top-k Boltzmann | `topk_boltzmann` | Uses epsilon random exploration, then samples only among the top-k Q actions. |
| Uncertainty-aware switch | `uncertainty_switch` | Switches toward Boltzmann only when the temporal schedule has progressed and normalized Q-value entropy is low. |

## DQN features

- Replay buffer with vectorized random sampling.
- Target network with hard updates, plus optional soft updates via `tau`.
- Double-DQN target selection enabled by default.
- Huber loss and gradient clipping.
- MLP Q-network for vector observations.
- Nature-DQN-style CNN for image observations, used by Atari configs.
- Correct handling of Gymnasium `terminated` vs `truncated`: time-limit truncation resets the episode but does not block bootstrapping.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[classic,dev,plots]"
```

For LunarLander:

```bash
pip install -e ".[box2d]"
```

For Atari:

```bash
pip install -e ".[atari,plots]"
```

Atari ROMs require accepting the ROM license. The `accept-rom-license` extra
installs the helper tooling used by Gymnasium/ALE.

## Run experiments

CartPole baseline:

```bash
egtb-train --config configs/cartpole/epsilon_greedy.yaml
```

CartPole annealed epsilon-to-Boltzmann:

```bash
egtb-train --config configs/cartpole/annealed_boltzmann.yaml
```

CartPole top-k Boltzmann:

```bash
egtb-train --config configs/cartpole/topk_boltzmann.yaml
```

CartPole uncertainty-aware switching:

```bash
egtb-train --config configs/cartpole/uncertainty_switch.yaml
```

LunarLander:

```bash
egtb-train --config configs/lunarlander/annealed_boltzmann.yaml
egtb-train --config configs/lunarlander/topk_boltzmann.yaml
```

Atari scaffold:

```bash
egtb-train --config configs/atari/breakout_annealed.yaml
```

Useful CLI overrides:

```bash
egtb-train \
  --config configs/cartpole/annealed_boltzmann.yaml \
  --total-steps 10000 \
  --seed 3 \
  --device cpu \
  --output-dir runs/cartpole_debug
```

## Outputs

Each run writes a timestamped directory under `runs/`:

- `config.yaml`: exact resolved run configuration.
- `train.csv`: per-episode return, exploration diagnostics, and latest loss.
- `eval.csv`: periodic greedy-policy evaluation.
- `summary.json`: final evaluation and high-level run metadata.
- `agent.pt`: PyTorch checkpoint.

Plot one or more runs:

```bash
egtb-plot runs/<run-a> runs/<run-b> --output runs/comparison.png
```

Aggregate multi-seed results by strategy:

```bash
egtb-analyze runs/cartpole_100k --output-dir runs/cartpole_100k_analysis
```

This writes grouped learning curves, a final-return plot, and CSV summaries.

## Tests and smoke run

Run unit tests:

```bash
python3 -m pytest
```

Short smoke run that exercises environment stepping, replay sampling,
optimization, checkpointing, and final evaluation:

```bash
egtb-train \
  --config configs/cartpole/annealed_boltzmann.yaml \
  --total-steps 300 \
  --learning-starts 64 \
  --eval-interval 0 \
  --eval-episodes 2 \
  --log-interval 100 \
  --device cpu \
  --output-dir runs/smoke
```

## Suggested study protocol

Use the CartPole configs first to validate code changes quickly. Then run
LunarLander with at least three seeds per strategy. Compare `eval.csv` curves
and final returns, not just training episode returns, because exploration policy
changes can affect data collection without directly representing the greedy
policy quality.

For Atari, start with the included Breakout config as a correctness scaffold.
Atari-scale results require substantially longer runs and more seeds than the toy
Gymnasium environments.

For cluster execution, see [docs/CLUSTER.md](docs/CLUSTER.md).
