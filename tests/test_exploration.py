import numpy as np

from egtb.config import ExplorationConfig
from egtb.exploration import (
    boltzmann_distribution,
    build_strategy,
    entropy_uncertainty,
    epsilon_greedy_distribution,
    topk_boltzmann_distribution,
)


def test_epsilon_greedy_distribution_prefers_argmax():
    probabilities = epsilon_greedy_distribution(np.array([1.0, 4.0, 2.0]), 0.3)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities.argmax() == 1
    assert np.isclose(probabilities[1], 0.8)


def test_boltzmann_temperature_controls_sharpness():
    q_values = np.array([0.0, 1.0, 3.0])
    cold = boltzmann_distribution(q_values, 0.1)
    hot = boltzmann_distribution(q_values, 10.0)
    assert cold.argmax() == 2
    assert cold[2] > hot[2]
    assert np.isclose(hot.sum(), 1.0)


def test_topk_distribution_masks_lower_ranked_actions():
    probabilities = topk_boltzmann_distribution(np.array([0.0, 5.0, 1.0, 4.0]), 1.0, 2)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] == 0.0
    assert probabilities[2] == 0.0
    assert probabilities[1] > probabilities[3]


def test_annealed_strategy_moves_weight_to_boltzmann():
    config = ExplorationConfig(
        name="annealed_epsilon_boltzmann",
        switch_start_step=10,
        switch_end_step=20,
        epsilon_start=1.0,
        epsilon_end=1.0,
    )
    strategy = build_strategy(config)
    early = strategy.diagnostics(np.array([0.0, 1.0]), 0)["boltzmann_weight"]
    late = strategy.diagnostics(np.array([0.0, 1.0]), 30)["boltzmann_weight"]
    assert early == 0.0
    assert late == 1.0


def test_uncertainty_switch_respects_entropy():
    config = ExplorationConfig(
        name="uncertainty_switch",
        switch_start_step=0,
        switch_end_step=1,
        low_uncertainty=0.2,
        high_uncertainty=0.8,
    )
    strategy = build_strategy(config)
    confident = strategy.diagnostics(np.array([0.0, 10.0]), 10)["boltzmann_weight"]
    uncertain = strategy.diagnostics(np.array([1.0, 1.0]), 10)["boltzmann_weight"]
    assert confident > uncertain
    assert np.isclose(entropy_uncertainty(np.array([1.0, 1.0])), 1.0)
