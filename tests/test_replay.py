import numpy as np

from egtb.replay import ReplayBuffer


def test_replay_buffer_samples_added_transitions():
    buffer = ReplayBuffer(
        capacity=8,
        observation_shape=(2,),
        observation_dtype=np.float32,
        seed=123,
    )
    for index in range(5):
        obs = np.array([index, index + 1], dtype=np.float32)
        buffer.add(obs, index % 2, float(index), obs + 1.0, done=index == 4)

    batch = buffer.sample(4)
    assert len(buffer) == 5
    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4,)
    assert batch.rewards.dtype == np.float32
