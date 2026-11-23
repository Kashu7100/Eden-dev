"""A scripted two-arm shirt fold played through ``env.step`` on ``mochi_fold_the_shirt``, optionally recorded.

---
tier: recipe
capabilities: [solver.mochi, robot.yam, task.manipulation, render.camera]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r
---

The shirt starts from the benchmark's 270-degree fold start, face down, its sleeves pointing at the two arms and its
hem at the robot. Every move goes through the task's policy actions (differential IK on each tool centre, the jaws
turned with the arm, the latching grippers), with ``rss_demo_routines.Arms`` driving them:

1. Sleeves in: each hand takes its sleeve 2 cm in from the cuff and lays it over onto the body, both at once.
2. Hem up a third: both hands take the hem 8 cm either side of the middle and lay it two thirds of the way up.
3. Over to the top: both hands take the new folded edge and lay it on the shoulders. The shirt is a 15 x 31 cm
   rectangle, folded in thirds along its length.

wikiHow's T-shirt fold (each side and its sleeve to the middle, then in half) needs a side's whole 35 cm edge carried
over: taken at one point, by the one hand that reaches it, it bunches instead of folding, and a YAM reaches only its
own half of the shirt, up to about 2 cm short of the middle. Folds across the hem, which both hands take together,
fold cleanly, so the routine folds the length.

It prints the task's measures after each stage and at the end. With ``-r`` it records the task's recording camera
(``MochiFoldTheShirtConfig.recording_camera``) to ``<output-dir>/mochi_fold_the_shirt.mp4``, one frame per control
step at the control rate.

Usage:
    uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r
    uv run python examples/recipes/mochi/fold_the_shirt_routine.py -r -o renders/mochi
"""

import argparse
import os

import genesis as gs
import torch
from rss_demo_routines import Arms

import eden as en  # must be imported before genesis / torch
from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.tasks.mochi.common import YAM_BASES

# The 270-degree fold start: sleeves along y, hem toward the arms.
START = 3
# The tool centre's height with the fingertips on the bench (as in the towel routine).
ON_BENCH = 0.0155
# The tool centre's height laying the cloth down.
LAY = 0.03
# Grasps are taken this far in from an edge (m).
INSET = 0.02
# ... and this far in from a folded or hemmed edge, where all its layers lie (m).
EDGE_IN = 0.008
# The strip of vertices searched for an edge (m).
BAND = 0.01
# How far in from its cuff's edge a sleeve is laid (m): over onto the body, past its seam.
SLEEVE_IN = 0.17
# The across folds take and lay the shirt this far either side of the middle (m).
HEM_Y = 0.08
# The farthest x the hands lay the shirt at (m): about as far as a YAM settles 8 cm from the middle.
TOP_REACH = 0.06
# Where a resting hand waits (x, |y|, z): over its own side, outside the shirt.
PARK = (-0.12, 0.32, 0.12)
SIDES = (("left", 1.0), ("right", -1.0))


def boundary_loops(triangles: torch.Tensor) -> list[torch.Tensor]:
    """The mesh's boundary loops (vertex index tensors), largest first."""
    edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]).sort(dim=1).values
    unique, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary = unique[counts == 1].tolist()
    neighbours: dict[int, list[int]] = {}
    for a, b in boundary:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)
    loops, seen = [], set()
    for v in neighbours:
        if v in seen:
            continue
        loop, stack = [], [v]
        seen.add(v)
        while stack:
            u = stack.pop()
            loop.append(u)
            for w in neighbours[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        loops.append(torch.tensor(loop))
    return sorted(loops, key=len, reverse=True)


class RecordingArms(Arms):
    """``Arms`` that turns each hand with its arm and renders the recording camera after every control step.

    Each hand's jaws turn as its arm's base does (by the heading of the tool centre's target from the base), so the
    wrist keeps its roll and never reaches its stops.
    """

    def __init__(self, env: RLEnvBase, camera=None):
        super().__init__(env)
        self.camera = camera
        self.base = {side: torch.tensor(YAM_BASES[side][:2], device=gs.device) for side in self.sides}
        self.ready_yaw = {side: self.jaw_angle(side) - self.heading(side, self.tcp(side)) for side in self.sides}

    def heading(self, side: str, point: torch.Tensor) -> torch.Tensor:
        offset = point[:, :2] - self.base[side]
        return torch.atan2(offset[:, 1], offset[:, 0])

    def step(self) -> None:
        for side in self.sides:
            if self.target[side] is not None:
                self.yaw[side] = self.ready_yaw[side] + self.heading(side, self.target[side])
        super().step()
        if self.camera is not None:
            self.camera.render_rgb()


def glide_both(arms: Arms, paths: dict, seconds: float) -> None:
    """Move both tool centres along ``paths[side](f)`` together, f from 0 to 1, eased."""
    n = max(1, round(seconds / arms.dt))
    for k in range(1, n + 1):
        s = k / n
        f = s * s * (3.0 - 2.0 * s)
        for side, path in paths.items():
            arms.target[side] = path(f)
        arms.step()


def line(start: torch.Tensor, end: torch.Tensor):
    return lambda f: start + (end - start) * f


def arc(start: torch.Tensor, end: torch.Tensor, height: float):
    """A quadratic arc from ``start`` to ``end`` whose apex is ``height`` over the higher end."""
    apex = (start + end) / 2.0
    apex[:, 2] = torch.maximum(start[:, 2], end[:, 2]) + height
    bend = 2.0 * apex - (start + end) / 2.0
    return lambda f: (1 - f) ** 2 * start + 2 * f * (1 - f) * bend + f**2 * end


def take(arms: Arms, grasp: dict) -> None:
    """Both hands from where they are to 6 cm over their grasp points, down onto them, and close."""
    up = torch.tensor([0.0, 0.0, 0.06], device=gs.device)
    starts = {side: arms.tcp(side).clone() for side in grasp}
    glide_both(arms, {s: line(starts[s], grasp[s] + up) for s in grasp}, 2.0)
    glide_both(arms, {s: line(grasp[s] + up, grasp[s]) for s in grasp}, 1.0)
    for side in grasp:
        arms.close[side] = True
    arms.wait(1.0)


def let_go(arms: Arms, place: dict) -> None:
    """Open both hands where they are and lift them 5 cm clear."""
    for side in place:
        arms.close[side] = False
    arms.wait(0.8)
    up = torch.tensor([0.0, 0.0, 0.05], device=gs.device)
    glide_both(arms, {s: line(place[s], place[s] + up) for s in place}, 1.0)
    arms.wait(0.5)


def report(env: RLEnvBase, stage: str) -> None:
    metrics = env.metric_manager
    term = metrics.get_term("fold_quality")
    q = term._quality.score(term._quality.features(env.entities["shirt"].get_vertex_positions()))
    settled = metrics.get_value("settled")
    print(
        f"{stage}: fold quality {q.tolist()} (succeeds at {term.success_threshold:+.3f}), p95 speed {settled.tolist()}"
    )


def park(arms: Arms, side: str) -> None:
    """Take one hand up and out over its own side of the bench, clear of the shirt and of the other arm."""
    sign = 1.0 if side == "left" else -1.0
    goal = arms.point(PARK[0], sign * PARK[1], PARK[2])
    glide_both(arms, {side: line(arms.tcp(side).clone(), goal)}, 1.5)


def loop_centre(env: RLEnvBase, loop: torch.Tensor) -> torch.Tensor:
    return env.entities["shirt"].get_vertex_positions()[:, loop.to(gs.device)].mean(dim=1)


def side_edge(env: RLEnvBase, sign: float, x: torch.Tensor) -> torch.Tensor:
    """Where the shirt's edge on one side (``sign`` +1 left, -1 right) is now, at ``x`` ``(B,)``: its xy ``(B, 2)``."""
    verts = env.entities["shirt"].get_vertex_positions()
    near = (verts[..., 0] - x[:, None]).abs() < BAND
    outward = (sign * verts[..., 1]).masked_fill(~near, -torch.inf)
    return verts[torch.arange(len(verts)), outward.argmax(dim=1), :2]


def hem_edge(env: RLEnvBase, y: float) -> torch.Tensor:
    """How near the arms the shirt reaches now, at ``y``: the edge's x ``(B,)``."""
    verts = env.entities["shirt"].get_vertex_positions()
    near = (verts[..., 1] - y).abs() < BAND
    return verts[..., 0].masked_fill(~near, torch.inf).amin(dim=1)


def fold_across(arms: Arms, grasp_x: torch.Tensor, place_x: torch.Tensor) -> None:
    """Both hands take the shirt at ``grasp_x`` (``HEM_Y`` either side of the middle) and lay it at ``place_x``."""
    grasp = {s: arms.point(grasp_x, torch.full_like(grasp_x, sign * HEM_Y), ON_BENCH) for s, sign in SIDES}
    place = {s: arms.point(place_x, torch.full_like(place_x, sign * HEM_Y), LAY) for s, sign in SIDES}
    take(arms, grasp)
    glide_both(arms, {s: arc(grasp[s], place[s], 0.07) for s in grasp}, 4.0)
    arms.wait(0.7)
    let_go(arms, place)
    for side, _ in SIDES:
        park(arms, side)


def fold(env: RLEnvBase, arms: Arms) -> None:
    shirt = env.entities["shirt"]
    cuffs = boundary_loops(torch.as_tensor(shirt.elems, device=gs.device).long())[2:]  # after the hem and collar
    top_x = shirt.get_vertex_positions()[..., 0].amax(dim=1)  # the shoulders
    for side, _ in SIDES:
        park(arms, side)

    # 1. Sleeves in: each hand 2 cm in from its cuff, the sleeve laid over onto the body.
    grasp, place = {}, {}
    for loop in cuffs:
        cuff = loop_centre(env, loop)
        side, sign = SIDES[0] if float(cuff[0, 1]) > 0.0 else SIDES[1]
        edge = side_edge(env, sign, cuff[:, 0])
        grasp[side] = arms.point(cuff[:, 0], edge[:, 1] - sign * INSET, ON_BENCH)
        place[side] = arms.point(cuff[:, 0], edge[:, 1] - sign * SLEEVE_IN, LAY)
    take(arms, grasp)
    glide_both(arms, {s: arc(grasp[s], place[s], 0.06) for s in grasp}, 3.0)
    let_go(arms, place)
    for side, _ in SIDES:
        park(arms, side)
    report(env, "sleeves in")

    # 2. Hem up a third: the hem laid two thirds of the way to the shoulders.
    hem_x = (hem_edge(env, HEM_Y) + hem_edge(env, -HEM_Y)) / 2.0
    third = (top_x - hem_x) / 3.0
    fold_across(arms, hem_x + EDGE_IN, hem_x + 2.0 * third - 0.01)
    report(env, "hem up a third")

    # 3. Over to the top: the folded edge, now a third of the way up, laid on the shoulders (or as far as the hands
    # lay it reliably).
    edge_x = (hem_edge(env, HEM_Y) + hem_edge(env, -HEM_Y)) / 2.0
    fold_across(arms, edge_x + EDGE_IN, torch.clamp(edge_x + 2.0 * third - 0.01, max=TOP_REACH))
    arms.wait(2.0)
    report(env, "over to the top")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the task's recording camera")
    parser.add_argument("-o", "--output-dir", type=str, default="renders/mochi", help="Directory the video goes to")
    args = parser.parse_args()
    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    config = TASK_REGISTRY.build("mochi_fold_the_shirt")
    config.env_options.num_envs = 1
    config.event_options.fold_start.state_indices = [START]
    for name in config.termination_options.term_keys():
        config.termination_options.disable_term(name)
    if args.record:
        config.add_recording_camera()
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    camera = env.cameras["rec"] if args.record else None
    if camera is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        camera.start_recording(os.path.join(args.output_dir, "mochi_fold_the_shirt.mp4"), fps=1.0 / env.dt)
    arms = RecordingArms(env, camera)
    try:
        if en.is_smoke():
            arms.wait(5 * arms.dt)
            return
        report(env, "start")
        fold(env, arms)
        metrics = env.metric_manager
        readings = {name: [round(v, 3) for v in metrics.get_value(name).tolist()] for name in metrics.active_terms}
        print(f"after {arms.steps * arms.dt:.1f} s: {readings}")
        print(f"goal met: {metrics.success_buf.tolist()}")
    finally:
        if camera is not None:
            camera.stop_recording()


if __name__ == "__main__":
    main()
