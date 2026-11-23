"""The suite's supermimic library and env (``tests/support/supermimic.py``) under the names these tests use, and
runners with networks small enough to train on the CPU.
"""

import pytest

from tests.support.supermimic import supermimic_config


def small_learner(cfg):
    """``cfg`` with the learner on the CPU and its networks, batches and horizons cut to a few units."""
    o = cfg.runner_options
    o.device, o.logger, o.num_steps_per_env, o.buffer_size = "cpu", "tensorboard", 4, 32
    o.world_model.hidden_dims, o.world_model.batch_size, o.world_model.num_batches = (32,), 8, 2
    o.policy.actor_hidden_dims, o.policy.critic_hidden_dims, o.policy.batch_size = (32,), (16,), 8
    o.policy.horizon, o.replay_ratio = 4, 64.0
    return cfg


@pytest.fixture(scope="session")
def lafan1_library(supermimic_library):
    return supermimic_library


@pytest.fixture(scope="session")
def env(supermimic_env):
    return supermimic_env


@pytest.fixture(scope="module")
def blinkmimic_runner(supermimic_library, tmp_path_factory):
    """A runner of the task under ``--recipe blinkmimic``: every learner option the recipe turns on, small networks."""
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    cfg = small_learner(supermimic_config(supermimic_library, recipe="blinkmimic"))
    return make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(cfg), log_dir=str(tmp_path_factory.mktemp("blinkmimic")))
