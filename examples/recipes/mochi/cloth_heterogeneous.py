"""Mochi heterogeneous cloth: four parallel envs, each draping a sheet of a different size and resolution.

---
tier: recipe
capabilities: [solver.mochi, scene.heterogeneous]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth_heterogeneous.py -r
---

A ``GroupedEntityOptions`` with a Mochi shell material builds one soft body that holds a different mesh per env.
Per-vertex arrays span the vertex count of the largest variant: an env's own vertices come first, the rows past
them read zero, and ``vertices_mask`` marks the real rows. Vertex indices passed to ``set_vertices_fixed`` are
local to each env's own variant.

Every variant exists in every env's solver arrays (fixed at rest and out of contact where another variant is
held), so memory grows with the summed size of the variants times the number of envs.

    uv run python examples/recipes/mochi/cloth_heterogeneous.py -r
"""

import argparse

import genesis as gs
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, GroupedEntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions
from eden.utils.procedural_meshes import grid_sheet_file

ENV_SPACING = 1.6
# The envs sit on a 2 x 2 grid whose first env is at the origin
CAM_POS, CAM_LOOKAT = (0.5 * ENV_SPACING, -3.6, 3.6), (0.5 * ENV_SPACING, 0.5 * ENV_SPACING, 0.2)
# (size in m, cells) of the sheet each variant drapes over the box
SHEETS = (((0.8, 0.8), (12, 12)), ((1.0, 0.5), (20, 10)), ((0.5, 0.5), (16, 16)), ((0.4, 1.0), (6, 15)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=240, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=args.num_envs,
            env_spacing=(ENV_SPACING, ENV_SPACING),
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            box=BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=(0.0, 0.0, 0.15),
                is_fixed_base=True,
                material=MochiRigidMaterialOptions(),
            ),
            cloth=GroupedEntityOptions(
                # The one material of every variant; see `cloth.py` for the contact stiffness of light cloth.
                material=MochiShellMaterialOptions(
                    E=2e4,
                    nu=0.3,
                    rho=200.0,
                    thickness=2e-3,
                    friction=0.6,
                    collider_radius=0.02,
                    penalty_coefficient=1e7,
                ),
                default_root_pos=(0.0, 0.0, 0.6),
                surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
                grouped_entities=[EntityOptions(file=grid_sheet_file(size, cells)) for size, cells in SHEETS],
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT, res=(960, 720)),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    cloth = env.entities["cloth"]
    n_own = cloth.vertices_mask.sum(dim=1)
    print(f"variant of each env: {cloth.env_variant_idx.tolist()}")
    print(f"vertex axis: {cloth.n_vertices} rows, of which each env's own: {n_own.tolist()}")

    recorder = VideoRecorder(env, args, "cloth_heterogeneous")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            # The padding rows read zero, so a minimum over the vertex axis takes the real rows only.
            z = cloth.get_vertex_positions()[..., 2].masked_fill(~cloth.vertices_mask, float("inf"))
            print(
                f"step {i_step:4d}: zmin per env={[round(v, 4) for v in z.min(dim=1).values.tolist()]} "
                f"centroid z={[round(v, 4) for v in cloth.get_pos()[:, 2].tolist()]} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
