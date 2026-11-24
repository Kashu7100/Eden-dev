"""The shirt fold's scripted demonstrator: it folds both starts it takes, refuses the others, and records."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import torch

from eden.envs.base import RLEnvBase
from eden.extensions.scripted_demos.generate import generate_demos
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY

TASK = "mochi_fold_the_shirt"
# The fold starts with the sleeves across the bench: 90 and 270 degrees.
ACROSS = [1, 3]


def _config(num_envs: int, state_indices: list[int]):
    config = TASK_REGISTRY.build(TASK)
    config.env_options.num_envs = num_envs
    config.event_options.fold_start.state_indices = state_indices
    return config


def test_a_mistake_rate_is_refused_naming_the_task():
    with pytest.raises(ValueError, match=r"'mochi_fold_the_shirt' makes no deliberate mistakes, so p_mistake=0.1"):
        SCRIPTED_DEMO_REGISTRY.get(TASK)(None, seed=0, p_mistake=0.1)


def test_the_episode_fits_the_fold_at_its_slowest_pace():
    from eden.tasks.mochi.scripted_demos import STAGES, ShirtFolder

    slowest = inspect.signature(ShirtFolder).parameters["speed_range"].default[1]
    nominal = sum(move.seconds for _, moves in STAGES for move in moves)
    assert SCRIPTED_DEMO_REGISTRY.get(TASK).episode_length_s >= nominal * slowest


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_task_that_draws_the_sleeves_toward_the_arms_is_refused_with_the_override():
    config = TASK_REGISTRY.build(TASK)
    config.env_options.num_envs = 1
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    with pytest.raises(ValueError, match=r"draws starts \[0, 2\].*state_indices: \[1, 3\]"):
        SCRIPTED_DEMO_REGISTRY.get(TASK)(env)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="folds on CUDA: on the suite's gs.cpu the fold takes hours")
def test_the_fold_folds_both_starts_and_lets_go():
    """One env per start: every stage finishes, both hands end open, and the shirt covers well under half its area.

    The fold runs in a process of its own on CUDA (``tests/support/shirt_fold.py``): the suite holds Genesis on
    gs.cpu, where its ~1400 Mochi steps of the 6888-vertex shirt take hours. On an RTX 3080 it takes about 3 minutes.
    """
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-m", "tests.support.shirt_fold"], cwd=repo, capture_output=True, text=True, timeout=1800
    )
    line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
    assert line is not None, f"shirt_fold printed no RESULT line; stderr tail:\n{result.stderr[-3000:]}"
    fold = json.loads(line.removeprefix("RESULT "))

    assert fold["done"], f"unfinished after the episode: stages {fold['stages']}"
    assert set(fold["stages"].values()) == {2}
    assert not fold["holding"]
    # Minus the log of the footprint's share of the flat shirt's: 0.5 is a footprint 61 % of the flat one's.
    assert min(fold["compression"]) > 0.5, f"footprint compression {fold['compression']}"


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_eden_datagen_records_both_arms(tmp_path):
    """Cut short at 0.5 s (the fold takes 40-57 s): each episode records both arms' actions and state, and the shirt's
    start as its vertices (it has no joints or root pose to record).
    """
    report = generate_demos(_config(2, ACROSS), TASK, num_episodes=2, output_dir=str(tmp_path), episode_length_s=0.5)

    assert report.episodes == 2 and report.successes == 0 and report.cut_short == 2
    with h5py.File(report.dataset_path, "r") as f:
        assert len(f["data"]) == 2
        demo = f["data"][next(iter(f["data"]))]
        steps = demo["action"].shape[0]
        assert demo["action"].shape[1] == 14  # each arm's 6-D IK delta and its gripper
        assert demo["state/qpos"].shape == (steps, 16)  # both YAMs' 8 joints, in scene order
        assert torch.isfinite(torch.as_tensor(demo["action"][:])).all()
        shirt = demo["initial_state/entities/shirt"]
        assert set(shirt) == {"vertex_pos", "vertex_vel"}
        assert shirt["vertex_pos"].shape == (1, 6888, 3)
