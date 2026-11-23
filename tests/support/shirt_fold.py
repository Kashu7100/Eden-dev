"""Fold ``mochi_fold_the_shirt``'s shirt with its scripted demonstrator on CUDA, one env per start it folds.

Run as ``python -m tests.support.shirt_fold`` from the repo root; prints one line, ``RESULT {...}``: whether every env
finished the fold within the demonstrator's episode, the stages finished, whether either hand still holds the shirt,
and each env's footprint compression (ClothDojo's first fold-quality feature, minus the log of the footprint's share
of the flat shirt's). A process of its own because the test suite initialises Genesis on ``gs.cpu``, where the fold's
~1400 Mochi steps of a 6888-vertex shell take hours rather than minutes.
"""

from __future__ import annotations

import json


def main() -> None:
    import eden as en  # before genesis and torch (DDP import order)
    import genesis as gs

    en.init(backend=gs.gpu, logging_level="WARNING")
    import torch

    from eden.envs.base import RLEnvBase
    from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("mochi_fold_the_shirt")
    config.env_options.num_envs = 2
    # The starts with the sleeves across the bench, 90 and 270 degrees: the two the demonstrator folds.
    config.event_options.fold_start.state_indices = [1, 3]
    for name in config.termination_options.term_keys():
        config.termination_options.disable_term(name)
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    # One env per start, rather than two draws that may land on the same one.
    shirt = env.entities["shirt"]
    starts = env.event_manager.get_term("fold_start")._states
    shirt.set_vertex_positions(starts)
    shirt.set_vertex_velocities(torch.zeros_like(starts))

    factory = SCRIPTED_DEMO_REGISTRY.get("mochi_fold_the_shirt")
    folder = factory(env, seed=0)
    folder.reset()
    for _ in range(round(factory.episode_length_s / env.dt)):
        if bool(folder.update_subgoals().all()):
            break
        env.step(folder.act())

    quality = env.metric_manager.get_term("fold_quality")._quality
    compression = quality.features(shirt.get_vertex_positions())[:, 0]
    holding = [bool(env.metric_manager.get_value(f"{arm}_holds").any()) for arm in ("left", "right")]
    result = {
        "done": bool(folder.done.all()),
        "stages": folder.subgoal_counts,
        "holding": any(holding),
        "compression": [round(float(c), 4) for c in compression],
    }
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
