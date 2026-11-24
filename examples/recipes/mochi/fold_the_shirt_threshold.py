"""Settle ClothDojo's expert shirt folds on ``mochi_fold_the_shirt``'s Mochi cloth and score them again.

ClothDojo fitted its fold-quality threshold to expert folds of its own cloth (Newton's Style3D). Whether the threshold
carries over to Mochi is whether a fold keeps its score on Mochi's cloth. This takes the one expert fold trajectory
ClothDojo publishes (``examples/fold`` of the ``clothdojo/assets`` Hub dataset: garment ``rand_004NVCOKV7``, a 40 s
policy rollout), picks ``--frames`` frames from its last quarter with both grippers open and the cloth slow, and lays
each into its own env of ``mochi_fold_the_shirt`` with that garment swapped in (same material, bench, collider and
size: each frame is scaled in the plane by the task's ``SHIRT_SCALE``, heights kept, as the task scales its starts;
the arms out of contact). It scores each fold as laid (ClothDojo's own score of that geometry), lets Mochi settle it
for ``--seconds``, and scores it again.

A frame whose cloth ClothDojo left interpenetrating can make Mochi's implicit solve diverge on its first step; the
env is then reset, and the frame is reported as diverged rather than scored.

The trajectory file is 280 MB; it is fetched once into the Hugging Face cache. Needs USD to read the garment.

Usage:
    uv run --with usd-core python examples/recipes/mochi/fold_the_shirt_threshold.py
    uv run --with usd-core python examples/recipes/mochi/fold_the_shirt_threshold.py -t 6 --frames 8
"""

import argparse
import os
import tempfile

import numpy as np

import eden as en  # must be imported before genesis / torch

GARMENT_ID = "rand_004NVCOKV7"
TRAJECTORY = f"examples/fold/trajectories/{GARMENT_ID}_000000_r000"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-t", "--seconds", type=float, default=4.0, help="Simulated seconds each fold settles for")
    parser.add_argument("--frames", type=int, default=6, help="How many expert frames to settle, one env each")
    args = parser.parse_args()

    import genesis as gs
    from huggingface_hub import hf_hub_download

    en.init(backend=gs.cpu if args.cpu else gs.gpu)
    import torch
    from convert_clothdojo_shirt import GARMENT_SCALE, REPO_ID, REVISION, read_mesh

    from eden.envs.base import RLEnvBase
    from eden.tasks import TASK_REGISTRY
    from eden.tasks.mochi.custom_terms.events import scale_in_plane
    from eden.tasks.mochi.fold_the_shirt.config import FOLD_QUALITY_THRESHOLD, SHIRT_OFFSET, SHIRT_SCALE

    def fetch(path: str) -> str:
        return hf_hub_download(REPO_ID, path, repo_type="dataset", revision=REVISION)

    points, triangles = read_mesh(fetch(f"benchmark/assets/garments/converted/{GARMENT_ID}/{GARMENT_ID}.usdc"))
    flat = np.load(fetch(f"benchmark/conditions/rgb-fold/templates/{GARMENT_ID}.npz"))["particle_q"]
    signals = np.load(fetch(f"{TRAJECTORY}/selection_signals.npz"))
    with np.load(fetch(f"{TRAJECTORY}/states.npz")) as states:
        gripper_open, trajectory = states["gripper_open_state"], states["particle_q"]

    # Frames from the last quarter with both grippers open and the cloth slow, spread evenly.
    n = len(trajectory)
    calm = np.flatnonzero((gripper_open.min(axis=1) > 0.9) & (signals["speed_rms_m_s"] < 0.02))
    calm = calm[calm >= 3 * n // 4]
    frames = calm[np.linspace(0, len(calm) - 1, min(args.frames, len(calm))).round().astype(int)]
    if en.is_smoke():
        frames = frames[:1]

    workdir = tempfile.mkdtemp(prefix="clothdojo_expert_")
    shirt_file, states_file = os.path.join(workdir, "shirt.obj"), os.path.join(workdir, "frames.npz")
    with open(shirt_file, "w") as f:
        f.write("\n".join([f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in points * GARMENT_SCALE]))
        f.write("\n" + "\n".join(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles) + "\n")
    np.savez(states_file, starts=trajectory[frames].astype(np.float32), flat_reference=flat.astype(np.float32))

    config = TASK_REGISTRY.build("mochi_fold_the_shirt")
    config.env_options.num_envs = len(frames)
    config.scene_options.shirt.file = shirt_file
    config.event_options.fold_start.states_file = states_file
    config.metric_options.fold_quality.reference_file = states_file
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    shirt = env.entities["shirt"]
    for arm in ("left", "right"):
        shirt.set_contact_enabled(env.entities[arm], False)
    quality = env.metric_manager.get_term("fold_quality")._quality

    # The frames' own xy centres, as the task scales its starts: one scaled about the flat garment's centre instead
    # would land a few centimetres off, which no feature sees.
    term = env.event_manager.get_term("fold_start")
    laid = torch.as_tensor(scale_in_plane(term, trajectory[frames], SHIRT_SCALE), device=env.device)
    laid = laid + torch.tensor(SHIRT_OFFSET, device=env.device)
    shirt.set_vertex_positions(laid.to(shirt.get_vertex_positions().dtype))
    shirt.set_vertex_velocities(torch.zeros_like(laid, dtype=shirt.get_vertex_positions().dtype))
    before = quality.score(quality.features(shirt.get_vertex_positions()))

    n_steps = 2 if en.is_smoke() else round(args.seconds / env.dt)
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    start = env.episode_length_buf.clone()
    for _ in range(n_steps):
        env.step(actions)
    # A diverged env is truncated and reset, which restarts its episode count.
    diverged = (env.episode_length_buf - start) != n_steps
    after = quality.score(quality.features(shirt.get_vertex_positions()))

    print(f"ClothDojo threshold {FOLD_QUALITY_THRESHOLD:+.4f}; {GARMENT_ID}, settled {n_steps * env.dt:.1f} s")
    print(f"{'frame':>6s} {'Q laid':>8s} {'Q settled':>10s} {'change':>8s}")
    changes = []
    for i, frame in enumerate(frames):
        if diverged[i]:
            print(f"{frame:>6d} {float(before[i]):+8.3f} {'diverged':>10s}")
            continue
        changes.append(float(after[i] - before[i]))
        print(f"{frame:>6d} {float(before[i]):+8.3f} {float(after[i]):+10.3f} {changes[-1]:+8.3f}")
    if changes:
        print(f"mean change over {len(changes)} settled folds: {np.mean(changes):+.3f}")


if __name__ == "__main__":
    main()
