import pytest

from egtb.config import config_from_dict


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown keys"):
        config_from_dict({"dqn": {"learning_rate_typo": 0.1}})


def test_config_converts_hidden_sizes_to_tuple():
    config = config_from_dict({"dqn": {"hidden_sizes": [32, 64]}})
    assert config.dqn.hidden_sizes == (32, 64)
