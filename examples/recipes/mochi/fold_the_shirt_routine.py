"""The scripted two-arm shirt fold of ``mochi_fold_the_shirt``, played through ``env.step``, optionally recorded.

---
tier: recipe
capabilities: [solver.mochi, robot.yam, task.manipulation, render.camera]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r
---

The fold is the task's scripted demonstrator, the one ``eden datagen --task mochi_fold_the_shirt`` records
(:class:`~eden.tasks.mochi.scripted_demos.ShirtFolder`; its module docstring has the plan and why it is this fold).
Here it runs in one env from the benchmark's 270-degree fold start, face down, its sleeves pointing at the two arms
and its hem at the robot. Every move goes through the task's policy actions (differential IK on each tool centre,
the jaws turned with the arm, the latching grippers):

1. Sleeves in: each hand takes its sleeve 2 cm in from the cuff and lays it over onto the body, both at once.
2. Up a third: both hands take the hem 8 cm either side of the middle and lay it two thirds of the way up.
3. Over to the top: both hands take the new folded edge and lay it on the shoulders. The shirt is a 15 x 31 cm
   rectangle, folded in thirds along its length.

It prints the task's measures after each stage and at the end. With ``-r`` it records the task's recording camera
(``MochiFoldTheShirtConfig.recording_camera``) to ``<output-dir>/mochi_fold_the_shirt.mp4``, one frame per control
step at the control rate. ``--seed`` draws the episode's pace and grasp jitter; the solve is not deterministic on the
GPU either, so two runs fold a little differently.

Usage:
    uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r
    uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r -o renders/mochi --seed 3
"""

import argparse
import os

import genesis as gs
import torch

import eden as en  # must be imported before genesis / torch
from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY

# The 270-degree fold start: sleeves along y, hem toward the arms.
START = 3


def build(config) -> RLEnvBase:
    """One env of ``config`` (a ``mochi_fold_the_shirt`` config) from :data:`START`, running to the fold's end."""
    config.env_options.num_envs = 1
    config.event_options.fold_start.state_indices = [START]
    # The fold runs to its own end: no time limit, and no reset before the measures are read.
    for name in config.termination_options.term_keys():
        config.termination_options.disable_term(name)
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    return env


def report(env: RLEnvBase, stage: str) -> None:
    quality = env.metric_manager.get_term("fold_quality")
    q = quality._quality.score(quality._quality.features(env.entities["shirt"].get_vertex_positions()))
    speed = env.metric_manager.get_value("settled")
    print(
        f"{stage}: fold quality {q.tolist()} (succeeds at {quality.success_threshold:+.3f}), p95 speed {speed.tolist()}"
    )


def play(env: RLEnvBase, camera=None, *, seed: int = 0, still_s: float = 0.0) -> None:
    """Fold the shirt with the task's demonstrator, rendering ``camera`` every step, after ``still_s`` of stillness."""
    from eden.tasks.mochi.scripted_demos import ShirtFolder

    def step(action: torch.Tensor) -> None:
        env.step(action)
        if camera is not None:
            camera.render_rgb()

    if en.is_smoke():
        for _ in range(5):
            step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=gs.device))
        return
    folder = ShirtFolder(env, seed=seed)
    report(env, "start")
    # Still: no move from either hand (a zero command holds the hand where it is), the jaws open.
    for _ in range(round(still_s / env.dt)):
        step(torch.zeros(env.num_envs, folder.action_dim, device=gs.device))
    counts = folder.subgoal_counts
    while not bool(folder.update_subgoals().all()):
        step(folder.act())
        for stage, count in folder.subgoal_counts.items():
            if count > counts[stage] and stage not in ("start", "rest"):
                report(env, stage.replace("_", " "))
        counts = folder.subgoal_counts
    metrics = env.metric_manager
    readings = {name: [round(v, 3) for v in metrics.get_value(name).tolist()] for name in metrics.active_terms}
    print(f"after {int(env.episode_length_buf[0]) * env.dt:.1f} s: {readings}")
    print(f"goal met: {metrics.success_buf.tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the task's recording camera")
    parser.add_argument("-o", "--output-dir", type=str, default="renders/mochi", help="Directory the video goes to")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the fold's pace and grasp jitter")
    args = parser.parse_args()
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    config = TASK_REGISTRY.build("mochi_fold_the_shirt")
    if args.record:
        config.add_recording_camera()
    env = build(config)
    camera = env.cameras["rec"] if args.record else None
    if camera is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        camera.start_recording(os.path.join(args.output_dir, "mochi_fold_the_shirt.mp4"), fps=1.0 / env.dt)
    try:
        play(env, camera, seed=args.seed)
    finally:
        if camera is not None:
            camera.stop_recording()


if __name__ == "__main__":
    main()
