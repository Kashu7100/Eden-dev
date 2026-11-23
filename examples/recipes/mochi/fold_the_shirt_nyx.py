"""The scripted shirt fold of ``fold_the_shirt_routine.py``, path traced by Nyx in a kitchen.

---
tier: recipe
capabilities: [solver.mochi, robot.yam, task.manipulation, render.nyx]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/mochi/fold_the_shirt_nyx.py
---

The task, the routine and every move are ``fold_the_shirt_routine.py``'s: two YAMs fold the ``mochi_fold_the_shirt``
T-shirt in thirds through the task's own actions. Read that one for what the arms do. What differs is the look. The
workcell stands on the table of LIBERO's kitchen (its floor, walls, counter, stove and fridge), and the recording
camera is a Nyx path-tracer sensor lit by a ceiling lamp and a window light.

The kitchen is scenery: a fixed Mochi body with collision off, so it never moves and nothing touches it. Its table top
lies at the bench's height and stands in for the bench in the picture; the bench plane still carries the arms and the
shirt, drawn invisible. A ``KinematicMaterialOptions`` body would be the usual way to add scenery, but the Nyx plugin
exports only rigid, FEM, PBD, MPM and SPH entities and drops a kinematic one without a word. Nyx reaches Genesis one
solver at a time and does not know the Mochi solver either, so a Mochi scene would export at its rest pose and never
move: ``_nyx.py`` bridges the two and is installed before the scene is built.

Rendering costs per frame, not per step: one frame per control step at the control rate, about 1400 frames for the
whole fold. ``--spp`` trades noise for time.

Usage:
    uv run python examples/recipes/mochi/fold_the_shirt_nyx.py
    uv run python examples/recipes/mochi/fold_the_shirt_nyx.py --spp 128 --res 1920 1080
"""

import argparse
import os
from typing import ClassVar

import genesis as gs

import eden as en  # must be imported before genesis / torch

# Where the camera stands and what it looks at, and its vertical field of view (deg): in front of the table and to
# the right, the shirt in the middle of the frame, the arms behind it and the kitchen's counter behind them.
CAM_POS, CAM_LOOKAT, CAM_FOV = (0.60, -0.58, 0.62), (-0.18, 0.0, 0.12), 40.0
# The kitchen's table under the workcell: its top at the bench's height, the arms' bases 0.25 m in from its far edge.
TABLE_CENTRE_X = -0.10
# ... less a millimetre: the arms' base plates stand on the bench, and two surfaces at one height flicker.
TABLE_DROP = 0.001
# A warm ceiling lamp over the table, a cool window light from the camera's side, and a dim fill behind the arms.
# The intensities are lumens.
LIGHTS = [
    {
        "type": "spot",
        "pos": (0.2, -0.3, 1.6),
        "dir": (-0.2, 0.15, -1.0),
        "color": (1.0, 0.93, 0.82),
        "intensity": 16.0,
        "inner_angle": 35.0,
        "outer_angle": 70.0,
    },
    {"type": "point", "pos": (1.6, -1.8, 1.2), "color": (0.80, 0.88, 1.0), "intensity": 6.0},
    {"type": "point", "pos": (-1.2, 1.0, 1.0), "color": (1.0, 0.90, 0.80), "intensity": 3.0},
]
# A cotton shirt: matte, no sheen.
SHIRT = gs.surfaces.Rough(color=(0.16, 0.30, 0.55))


def kitchen():
    """LIBERO's kitchen as scenery, its table top at the bench's height: a fixed Mochi body with collision off."""
    from eden.options.materials import MochiRigidMaterialOptions
    from eden.options.objects.libero import LiberoKitchenTabletopWorkspace

    class Kitchen(LiberoKitchenTabletopWorkspace):
        # 1.0 m deep and 1.2 m wide under the workcell (x = -0.60 to +0.40), as LIBERO's own table.
        table_full_size: ClassVar[tuple[float, float, float]] = (1.0, 1.2, 0.05)

    top = Kitchen.table_top_pos_local[2]
    return Kitchen(
        default_root_pos=(TABLE_CENTRE_X, 0.0, -top - TABLE_DROP),
        collision=False,
        material=MochiRigidMaterialOptions(),
    )


def nyx_camera(resolution: tuple[int, int], spp: int):
    """The ``rec`` camera, backed by a Nyx path-tracer sensor, over a dim constant environment."""
    import _nyx
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.envs.nyx_splat import write_nyx_camera_profile
    from eden.options import CameraOptions

    return CameraOptions(
        cam_pos=CAM_POS,
        cam_lookat=CAM_LOOKAT,
        cam_resolution=resolution,
        cam_fov=CAM_FOV,
        sensor_options=NyxCameraOptions(
            spp=spp,
            denoise=True,
            camera_profile=write_nyx_camera_profile(CAM_FOV, "ACES_1_3"),
            lights=LIGHTS,
            env_maps=(_nyx.constant_env_map(0.18),),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output-dir", type=str, default="renders/mochi", help="Directory the video goes to")
    parser.add_argument("--spp", type=int, default=64, help="Nyx samples per pixel (noise vs. render time)")
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720), help="Render resolution, width height")
    args = parser.parse_args()
    # No --cpu: Nyx renders on CUDA, so a CPU run has no outcome but a crash inside the first render.
    en.init(backend=gs.gpu)

    import _nyx
    from fold_the_shirt_routine import START, RecordingArms, fold, report

    from eden.envs.base import RLEnvBase
    from eden.options import CamerasOptions
    from eden.tasks import TASK_REGISTRY

    _nyx.install()  # teach the plugin about the Mochi solver, before the scene is built

    config = TASK_REGISTRY.build("mochi_fold_the_shirt")
    config.env_options.num_envs = 1
    config.event_options.fold_start.state_indices = [START]
    for name in config.termination_options.term_keys():
        config.termination_options.disable_term(name)
    # The kitchen's table top is the bench in the picture; the bench plane still carries the arms and the shirt.
    config.scene_options.plane.visualization = False
    config.scene_options.kitchen = kitchen()
    config.scene_options.shirt.surface = SHIRT
    config.cameras_options = CamerasOptions(rec=nyx_camera(tuple(args.res), args.spp))

    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    camera = env.cameras["rec"]
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "mochi_fold_the_shirt_nyx.mp4")
    camera.start_recording(path, fps=1.0 / env.dt)
    arms = RecordingArms(env, camera)
    try:
        if en.is_smoke():
            arms.wait(5 * arms.dt)
            return
        report(env, "start")
        arms.wait(1.0)  # a still beat before the arms move
        fold(env, arms)
        arms.wait(1.5)
    finally:
        camera.stop_recording()
        print(f"Saved video to {path}")


if __name__ == "__main__":
    main()
