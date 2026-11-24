"""Scripted demonstrators for the Mochi tasks (``eden datagen --task mochi_fold_the_shirt``).

:class:`ShirtFolder` folds ``mochi_fold_the_shirt``'s T-shirt with both YAMs, batched over envs, through the task's
own actions: the differential-IK delta of each tool centre (its turn about the vertical included) and each latching
gripper. It reads the shirt's vertices (where its cuffs and edges lie now), never moves them.

The fold, in three stages, each a fixed schedule of moves whose grasp and lay points are read off the shirt as it
lies when the stage starts:

1. Sleeves in: each hand takes its sleeve 2 cm in from the cuff and lays it over onto the body, both at once.
2. Up a third: both hands take the near end (the one toward the arms) 8 cm either side of the middle, and lay it
   two thirds of the way to the far end.
3. Over to the top: both hands take the new folded edge and lay it on the far end, or as far as they lay it reliably.

A take is: over the grasp point, down onto it with the fingertips on the bench, close (the latch seizes the shirt
when the jaws arrive), carry it in an arc, open, lift clear, and wait over the hand's own side of the bench. Every
move eases in and out (smoothstep) over a fixed time. Each hand's jaws turn as its arm's base turns, so the wrist
keeps its roll: with the turn held, the wrist rolls against every turn of the base and reaches its stops well short
of the shirt's middle.

wikiHow's T-shirt fold (each side and its sleeve to the middle, then in half) is out of this rig's reach: a side's
whole edge has to be carried over, and taken at one point by the one hand that reaches it, the side bunches instead
of folding. Folds across the near end, which both hands take together, fold cleanly.

The fold does not reach the task's success: its final fold quality runs about -0.9 to -2.3 against the 0.142 cut,
held back by a lumpy top (ClothDojo's planar-top feature) and by sleeve folds that sometimes slip. An ``eden datagen``
episode therefore ends on the demonstrator's own end, not on success.

It folds the two fold starts with the sleeves across the bench (90 and 270 degrees, the near end the collar or the
hem); with the sleeves pointing at and away from the arms (0 and 180 degrees) the far sleeve is out of reach. A task
that can draw those is refused, naming the override that leaves them out. ``seed`` draws each episode's pace (every
move's time scaled by a factor in ``speed_range``) and a jitter of each grasp point (up to ``grasp_jitter`` along the
bench); there are no deliberate mistakes, so ``p_mistake`` must be 0.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.tasks.mochi.common import YAM_BASES, YAM_TCP_OFFSET, link_point
from eden.utils.geom import transform_by_quat

#: The tool centre's height with the fingertips on the bench.
ON_BENCH = 0.0155
#: The tool centre's height laying the cloth down.
LAY = 0.03
#: Where a resting hand waits (x, |y|, z): over its own side of the bench, outside the shirt.
PARK = (-0.12, 0.32, 0.12)
#: How far a hand lifts after letting go (m).
LIFT = 0.05
#: A take starts this far over its grasp point (m).
APPROACH = 0.06
#: A sleeve is taken this far in from its cuff's edge, and laid this far in from it (m).
SLEEVE_TAKE, SLEEVE_LAY = 0.02, 0.17
#: The across folds take and lay the shirt this far either side of the middle (m).
ACROSS_Y = 0.08
#: A folded or hemmed edge is taken this far in from it, where all its layers lie (m).
EDGE_IN = 0.008
#: The farthest x the hands lay the shirt at (m): about as far as a YAM settles 8 cm from the middle.
FAR_REACH = 0.06
#: Width of the strip of vertices searched for an edge (m).
BAND = 0.01
#: Each arm's sign across the bench: the left arm stands at +y.
SIDES = {"left": 1.0, "right": -1.0}


class _Move(NamedTuple):
    """One move of both hands: where to (a waypoint name, or None to hold), for how long, the jaws, the path."""

    to: str | None
    seconds: float
    closed: bool
    arc: float = 0.0  # the arc's apex over the higher end (m); 0 is a straight line


def _take(carry_s: float, arc: float, settle_s: float) -> list[_Move]:
    """A take: over the grasp point, down, close, carry in an arc, (settle,) open, lift, wait, and back to the side."""
    moves = [
        _Move("over_grasp", 2.0, False),
        _Move("grasp", 1.0, False),
        _Move(None, 1.0, True),
        _Move("place", carry_s, True, arc),
    ]
    if settle_s > 0.0:
        moves.append(_Move(None, settle_s, True))
    return [*moves, _Move(None, 0.8, False), _Move("over_place", 1.0, False), _Move(None, 0.5, False)] + [
        _Move("park", 1.5, False)
    ]


#: The fold: each stage's moves, run after its waypoints are read off the shirt.
STAGES = (
    ("start", [_Move("park", 1.5, False)]),
    ("sleeves", _take(carry_s=3.0, arc=0.06, settle_s=0.0)),
    ("up_a_third", _take(carry_s=4.0, arc=0.07, settle_s=0.7)),
    ("over_the_top", _take(carry_s=4.0, arc=0.07, settle_s=0.7)),
    ("rest", [_Move(None, 2.0, False)]),
)


def boundary_loops(triangles: torch.Tensor) -> list[torch.Tensor]:
    """The mesh's boundary loops (vertex index tensors), largest first. For the shirt: hem, collar, two cuffs."""
    edges = torch.cat([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]).sort(dim=1).values
    unique, counts = torch.unique(edges, dim=0, return_counts=True)
    neighbours: dict[int, list[int]] = {}
    for a, b in unique[counts == 1].tolist():
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
        loops.append(torch.tensor(loop, device=triangles.device))
    return sorted(loops, key=len, reverse=True)


class ShirtFolder:
    """Batched scripted fold of ``mochi_fold_the_shirt``'s shirt (the module docstring has the plan).

    For ``eden datagen`` it is a demonstrator: :meth:`update_subgoals` returns which envs have finished the fold,
    :meth:`act` the ``(num_envs, action_dim)`` action, and :attr:`subgoal_counts` counts the stages finished. Call
    :meth:`reset` for every env the env resets.

    Parameters
    ----------
    env : RLEnvBase
        A built ``mochi_fold_the_shirt`` env.
    seed : int
        Seeds each episode's pace and grasp jitter.
    speed_range : tuple[float, float]
        Every move's time of an episode is scaled by one factor drawn uniformly from this range (1 is the nominal
        pace, about 47 s for the fold).
    grasp_jitter : float
        Each grasp point moves by up to this much along the bench (m), drawn per episode and stage.
    shirt_name : str
        The shirt's entity.
    """

    def __init__(
        self,
        env,
        *,
        seed: int = 0,
        speed_range: tuple[float, float] = (0.85, 1.2),
        grasp_jitter: float = 0.004,
        shirt_name: str = "shirt",
    ) -> None:
        if not 0.0 < speed_range[0] <= speed_range[1]:
            raise ValueError(f"speed_range must be positive and ordered (low, high), got {speed_range}.")
        self.env = env
        self.device = env.device
        self.n = env.num_envs
        self.dt = env.dt
        self.shirt = env.entities[shirt_name]
        self.speed_range, self.grasp_jitter = speed_range, grasp_jitter
        self.rng = torch.Generator(device=self.device).manual_seed(seed)

        self._find_actions()
        self.arms = list(SIDES)
        self.signs = torch.tensor([SIDES[a] for a in self.arms], device=self.device)
        self.links = {arm: env.entities[arm].get_link("link_6") for arm in self.arms}
        self.offset = torch.tensor(YAM_TCP_OFFSET, device=self.device)
        self.bases = torch.tensor([YAM_BASES[a][:2] for a in self.arms], device=self.device)  # (2, 2)

        loops = boundary_loops(torch.as_tensor(self.shirt.elems, device=self.device).long())
        if len(loops) < 4:
            raise ValueError(
                f"ShirtFolder folds a T-shirt: a mesh with a hem, a collar and two cuffs as its boundary loops; "
                f"'{shirt_name}' has {len(loops)} boundary loops ({[len(loop) for loop in loops]} vertices)."
            )
        self.cuffs = loops[2:4]  # the hem and collar are the two longest loops
        self._check_starts(env)

        # The schedule, flattened: (stage index, move) per entry.
        self.schedule = [(i, move) for i, (_, moves) in enumerate(STAGES) for move in moves]
        self.first_of_stage = {
            i: next(k for k, (s, _) in enumerate(self.schedule) if s == i) for i in range(len(STAGES))
        }
        self.entry = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.step_in_move = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.move_steps = torch.ones(self.n, dtype=torch.long, device=self.device)
        self.start = torch.zeros(self.n, 2, 3, device=self.device)  # each hand's position as the move starts
        self.goal = torch.zeros(self.n, 2, 3, device=self.device)
        self.arc = torch.zeros(self.n, device=self.device)
        self.closed = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.pace = torch.ones(self.n, device=self.device)
        self.ready_yaw = torch.zeros(self.n, 2, device=self.device)
        self.far_x = torch.zeros(self.n, device=self.device)
        self.third = torch.zeros(self.n, device=self.device)
        self.waypoints = {
            name: torch.zeros(self.n, 2, 3, device=self.device)
            for name in ("park", "over_grasp", "grasp", "place", "over_place")
        }
        self.waypoints["park"][:] = torch.tensor(
            [[PARK[0], SIDES[a] * PARK[1], PARK[2]] for a in self.arms], device=self.device
        )
        self._stages_finished = torch.zeros(len(STAGES), dtype=torch.long, device=self.device)
        self._started = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.reset()

    # ---- set-up ----------------------------------------------------------------------------------------------

    def _find_actions(self) -> None:
        """Each arm's IK term and gripper term, and where they sit in the action vector."""
        manager = self.env.action_manager
        self.slots, start = {}, 0
        hands, grips = {}, {}
        for name in manager.active_terms:
            term = manager.get_term(name)
            self.slots[name] = (start, term.action_dim)
            start += term.action_dim
            if isinstance(term, DifferentialIKController):
                hands[term.entity_name] = name
            elif isinstance(term, BinaryJointController):
                grips[term.entity_name] = name
        self.action_dim = start
        layout = {name: type(manager.get_term(name)).__name__ for name in manager.active_terms}
        if set(hands) != set(SIDES) or set(grips) != set(SIDES) or len(layout) != 4:
            raise TypeError(
                f"ShirtFolder drives two YAMs, 'left' and 'right', each through one DifferentialIKController and one "
                f"gripper (a BinaryJointController); the env's action terms are {layout}."
            )
        self.hand_terms = {arm: hands[arm] for arm in SIDES}
        self.grip_terms = {arm: grips[arm] for arm in SIDES}
        self.scale = {arm: manager.get_term(hands[arm]).effective_scale for arm in SIDES}
        if any(float(scale[5]) <= 0.0 for scale in self.scale.values()):
            raise ValueError(
                "ShirtFolder turns each hand with its arm, so each hand's IK term needs a turn about the vertical: "
                f"orientation_scale's z above 0 (mochi_fold_the_shirt's is 0.1); the scales are "
                f"{ {arm: scale.tolist() for arm, scale in self.scale.items()} }."
            )

    def _check_starts(self, env) -> None:
        """Refuse a task that can draw a fold start with the sleeves pointing at or away from the arms."""
        term = env.event_manager.get_term("fold_start")
        states = term._states  # (n_states, n_vertices, 3), as drawn
        indices = list(term.state_indices) or list(range(len(states)))
        cuffs = torch.stack([states[:, loop].mean(dim=1) for loop in self.cuffs], dim=1)  # (n_states, 2, 3)
        between = (cuffs[:, 0] - cuffs[:, 1]).abs()
        along_x = [index for index, d in zip(indices, between.tolist()) if d[0] > d[1]]
        if along_x:
            across = [index for index in indices if index not in along_x] or [1, 3]
            raise ValueError(
                f"ShirtFolder folds the shirt with its sleeves across the bench, but the task's fold_start draws "
                f"starts {along_x} with the sleeves pointing at and away from the arms, and the far sleeve is out of "
                f"their reach. Leave those out: --override 'event_options: {{fold_start: {{state_indices: "
                f"{across}}}}}' (or set config.event_options.fold_start.state_indices = {across})."
            )

    # ---- the demonstrator interface ---------------------------------------------------------------------------

    @property
    def done(self) -> torch.Tensor:
        return self.entry >= len(self.schedule)

    @property
    def subgoal_counts(self) -> dict[str, int]:
        """The stages finished so far, over every env and episode (``eden datagen``'s report)."""
        return {name: int(count) for (name, _), count in zip(STAGES, self._stages_finished.tolist())}

    def update_subgoals(self) -> torch.Tensor:
        """``eden datagen``'s per-step check: :attr:`done` (the fold advances in :meth:`act`)."""
        return self.done

    def reset(self, envs_idx=None) -> None:
        """Start the fold over in ``envs_idx`` (a mask or indices; every env by default), the arms at their ready pose."""
        idx = torch.arange(self.n, device=self.device)[slice(None) if envs_idx is None else envs_idx]
        if len(idx) == 0:
            return
        low, high = self.speed_range
        self.pace[idx] = low + (high - low) * torch.rand(len(idx), generator=self.rng, device=self.device)
        tcp = self.tcp()[idx]
        self.ready_yaw[idx] = self.jaw_angle()[idx] - self.heading(tcp)
        self.far_x[idx] = self.vertices()[idx, :, 0].amax(dim=1)
        self.entry[idx] = 0
        self._started[idx] = False

    def act(self) -> torch.Tensor:
        """The action of every env this step, advancing each env's fold by one control step."""
        if not bool(self._started.all()):
            self._begin(~self._started)
        f = (self.step_in_move.float() / self.move_steps.float()).clamp(max=1.0)
        target = self.path(f)
        action = self.command(target)
        self.step_in_move += 1
        ended = (self.step_in_move >= self.move_steps) & ~self.done
        if bool(ended.any()):
            self.entry[ended] += 1
            self._stages_finished[-1] += int((ended & self.done).sum())
            self._begin(ended)
        return action

    # ---- the plan ---------------------------------------------------------------------------------------------

    def _begin(self, envs: torch.Tensor) -> None:
        """Start the current move of ``envs`` (an ``(N,)`` mask), reading a stage's waypoints off the shirt first."""
        self._started |= envs
        envs = envs & ~self.done
        if not bool(envs.any()):
            return
        for entry in self.entry[envs].unique().tolist():
            mask = envs & (self.entry == entry)
            stage, move = self.schedule[entry]
            if entry == self.first_of_stage[stage]:
                if entry > 0:
                    self._stages_finished[stage - 1] += int(mask.sum())
                self._read_waypoints(STAGES[stage][0], mask)
            # A move starts where the last one ended up (the very first, from the hand itself).
            self.start[mask] = self.tcp()[mask] if entry == 0 else self.goal[mask]
            self.goal[mask] = self.start[mask] if move.to is None else self.waypoints[move.to][mask]
            self.arc[mask] = move.arc
            self.closed[mask] = move.closed
            self.move_steps[mask] = (move.seconds * self.pace[mask] / self.dt).round().long().clamp(min=1)
            self.step_in_move[mask] = 0

    def _read_waypoints(self, stage: str, envs: torch.Tensor) -> None:
        """Read stage ``stage``'s grasp and lay points off the shirt as it lies now, for ``envs``."""
        if stage not in ("sleeves", "up_a_third", "over_the_top"):
            return
        verts = self.vertices()
        if stage == "sleeves":
            # Each hand takes the sleeve on its own side.
            centres = torch.stack([verts[:, loop].mean(dim=1) for loop in self.cuffs], dim=1)  # (N, 2 cuffs, 3)
            middle = verts[..., 1].mean(dim=1, keepdim=True)  # (N, 1)
            grasp, place = [], []
            for sign in self.signs.tolist():
                first_is_mine = (centres[:, 0, 1:2] - middle) * sign > 0  # (N, 1): the first cuff lies on this side
                cuff = torch.where(first_is_mine, centres[:, 0], centres[:, 1])  # (N, 3)
                edge_y = self._side_edge(verts, sign, cuff[:, 0])
                grasp.append(self.point(cuff[:, 0], edge_y - sign * SLEEVE_TAKE, ON_BENCH))
                place.append(self.point(cuff[:, 0], edge_y - sign * SLEEVE_LAY, LAY))
            grasp, place = torch.stack(grasp, dim=1), torch.stack(place, dim=1)
        else:
            near_x = torch.stack([self._near_edge(verts, sign * ACROSS_Y) for sign in self.signs.tolist()], 1).mean(1)
            if stage == "up_a_third":
                self.third[envs] = ((self.far_x - near_x) / 3.0)[envs]
                lay_x = near_x + 2.0 * self.third - 0.01
            else:
                lay_x = torch.clamp(near_x + 2.0 * self.third - 0.01, max=FAR_REACH)
            ys = self.signs * ACROSS_Y
            grasp = torch.stack([self.point(near_x + EDGE_IN, torch.full_like(near_x, y), ON_BENCH) for y in ys], 1)
            place = torch.stack([self.point(lay_x, torch.full_like(near_x, y), LAY) for y in ys], 1)
        jitter = self.grasp_jitter * (2.0 * torch.rand(self.n, 2, 2, generator=self.rng, device=self.device) - 1.0)
        grasp[..., :2] += jitter
        up = torch.tensor([0.0, 0.0, APPROACH], device=self.device)
        lift = torch.tensor([0.0, 0.0, LIFT], device=self.device)
        for name, value in (
            ("grasp", grasp),
            ("over_grasp", grasp + up),
            ("place", place),
            ("over_place", place + lift),
        ):
            self.waypoints[name][envs] = value[envs]

    def path(self, f: torch.Tensor) -> torch.Tensor:
        """Each hand's target ``(N, 2, 3)`` at the eased fraction of its move ``f`` ``(N,)``: a line or an arc."""
        s = (f * f * (3.0 - 2.0 * f))[:, None, None]
        start, goal = self.start, self.goal
        apex = (start + goal) / 2.0
        apex[..., 2] = torch.maximum(start[..., 2], goal[..., 2]) + self.arc[:, None]
        bend = 2.0 * apex - (start + goal) / 2.0  # the quadratic's control point, so that it passes the apex
        arc = (1 - s) ** 2 * start + 2 * s * (1 - s) * bend + s**2 * goal
        line = start + (goal - start) * s
        return torch.where((self.arc > 0.0)[:, None, None], arc, line)

    def command(self, target: torch.Tensor) -> torch.Tensor:
        """The action that moves each hand toward ``target`` ``(N, 2, 3)``, its jaws turned with its arm."""
        action = torch.zeros(self.n, self.action_dim, device=self.device)
        tcp, yaw = self.tcp(), self.jaw_angle()
        want = self.ready_yaw + self.heading(target)
        for k, arm in enumerate(self.arms):
            start, _ = self.slots[self.hand_terms[arm]]
            scale = self.scale[arm]
            action[:, start : start + 3] = torch.atanh(((target[:, k] - tcp[:, k]) / scale[:3]).clamp(-0.97, 0.97))
            # The jaws look the same turned half round.
            error = torch.remainder(want[:, k] - yaw[:, k] + math.pi / 2, math.pi) - math.pi / 2
            action[:, start + 5] = torch.atanh((error / scale[5]).clamp(-0.97, 0.97))
            action[:, self.slots[self.grip_terms[arm]][0]] = torch.where(self.closed, 1.0, -1.0)
        return action

    # ---- reading the scene ------------------------------------------------------------------------------------

    def vertices(self) -> torch.Tensor:
        return self.shirt.get_vertex_positions()

    def tcp(self) -> torch.Tensor:
        """Each hand's tool centre ``(N, 2, 3)``."""
        return torch.stack([link_point(self.links[arm], self.offset) for arm in self.arms], dim=1)

    def jaw_angle(self) -> torch.Tensor:
        """Each hand's heading ``(N, 2)``: its jaws' closing axis (the hand's x) in the bench plane."""
        x = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.n, 3)
        axes = torch.stack([transform_by_quat(x, self.links[arm].get_quat()) for arm in self.arms], dim=1)
        return torch.atan2(axes[..., 1], axes[..., 0])

    def heading(self, points: torch.Tensor) -> torch.Tensor:
        """The heading ``(N, 2)`` of each hand's ``points`` ``(N, 2, 3)`` from its arm's base."""
        offset = points[..., :2] - self.bases
        return torch.atan2(offset[..., 1], offset[..., 0])

    def point(self, x: torch.Tensor, y: torch.Tensor, z: float) -> torch.Tensor:
        return torch.stack([x, y, torch.full_like(x, z)], dim=-1)

    @staticmethod
    def _side_edge(verts: torch.Tensor, sign: float, x: torch.Tensor) -> torch.Tensor:
        """How far out the shirt reaches on one side (``sign`` +1 left, -1 right) at ``x`` ``(N,)``: its y ``(N,)``."""
        near = (verts[..., 0] - x[:, None]).abs() < BAND
        return sign * (sign * verts[..., 1]).masked_fill(~near, -torch.inf).amax(dim=1)

    @staticmethod
    def _near_edge(verts: torch.Tensor, y: float) -> torch.Tensor:
        """How near the arms the shirt reaches at ``y``: the edge's x ``(N,)``."""
        near = (verts[..., 1] - y).abs() < BAND
        return verts[..., 0].masked_fill(~near, torch.inf).amin(dim=1)


@SCRIPTED_DEMO_REGISTRY.register(name="mochi_fold_the_shirt")
def fold_the_shirt(env, *, seed: int = 0, p_mistake: float = 0.0) -> ShirtFolder:
    """Sleeves in, then the length in thirds, with both YAMs (the starts with the sleeves across the bench)."""
    if p_mistake != 0.0:
        raise ValueError(
            f"The scripted demonstrator of 'mochi_fold_the_shirt' makes no deliberate mistakes, so p_mistake={p_mistake} "
            "cannot be honoured. Generate with --p_mistake 0 (or leave it out)."
        )
    return ShirtFolder(env, seed=seed)


# The fold takes about 47 s at the nominal pace, up to 57 s at the slowest; the task's own episode is 40 s.
fold_the_shirt.episode_length_s = 75.0
