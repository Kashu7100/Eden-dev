"""Reset events for Mochi soft bodies."""

from __future__ import annotations

import numpy as np
import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.utils.geom import quat_from_euler_xyz
from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import term_subject


@EVENT_TERM_REGISTRY.register()
class PinSoftBodyVertices(EventTerm):
    """Pin vertices of a Mochi soft body where they stand (a Dirichlet condition).

    Fire it at STARTUP or RESET. A pin, and the position it holds the vertex to, are part of the state a reset
    restores, which the env takes at the end of its build, after the STARTUP events: a pin made at STARTUP holds
    through every reset, and one made later, from code run after ``env.build()``, lasts until its env next
    resets. Writing a pinned vertex's position moves the hold with it, so a RESET event that places the body
    carries a STARTUP pin along. At RESET, the pin holds the vertices where the reset events before it left them.
    A rod is clamped by pinning its first two nodes (the second fixes the twist).

    Parameters
    ----------
    entity_name : str
        The Mochi soft body.
    vertices : list[int]
        Local indices of the vertices to pin.
    """

    entity_name: str = ""
    vertices: list[int] = []

    def build(self) -> None:
        body = self.resolve_entity(self.entity_name)
        if not hasattr(body, "set_vertices_fixed"):
            raise ValueError(
                f"{term_subject(self)} pins a Mochi soft body's vertices, but the entity is a {type(body).__name__}."
            )
        bad = [v for v in self.vertices if not 0 <= v < body.n_vertices]
        if not self.vertices or bad:
            raise ValueError(
                f"{term_subject(self)} needs vertices of the entity (0..{body.n_vertices - 1}); got "
                f"{self.vertices}{f', of which {bad} are out of range' if bad else ''}."
            )

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        self.resolve_entity(self.entity_name).set_vertices_fixed(self.vertices, is_fixed=True, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class ResetSoftBodyPlanarPose(EventTerm):
    """Place a Mochi soft body at its rest shape, shifted in x / y and turned about z by uniform draws.

    The rest geometry (as the scene was built) is turned by the drawn yaw about its own centroid and shifted by
    the drawn offset, and every vertex velocity is zeroed — a rigid placement of the undeformed body, which
    ``set_pos`` / ``set_quat`` cannot express for a vertex-based body.

    Parameters
    ----------
    entity_name : str
        The Mochi soft body.
    pos_x_range, pos_y_range : tuple[float, float]
        Offset ranges (m) from the rest position.
    yaw_range : tuple[float, float]
        Rotation range (rad) about the vertical axis through the rest centroid.
    """

    entity_name: str = ""
    pos_x_range: tuple[float, float] = (0.0, 0.0)
    pos_y_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)

    def build(self) -> None:
        body = self.resolve_entity(self.entity_name)
        if not hasattr(body, "set_vertex_positions"):
            raise ValueError(
                f"{term_subject(self)} places a Mochi soft body, but the entity is a "
                f"{type(body).__name__}; place rigid bodies with events.reset_base_state_uniform."
            )
        rest = torch.as_tensor(body.init_positions, dtype=torch.float, device=self.device)
        self._centroid = rest.mean(dim=0)
        self._rest_centred = rest - self._centroid

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        draws = torch.rand(n_envs, 3, device=self.device)
        low = torch.tensor([self.pos_x_range[0], self.pos_y_range[0], self.yaw_range[0]], device=self.device)
        high = torch.tensor([self.pos_x_range[1], self.pos_y_range[1], self.yaw_range[1]], device=self.device)
        offset_x, offset_y, yaw = (low + (high - low) * draws).unbind(dim=-1)
        zero = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zero, zero, yaw)[:, None, :].expand(-1, len(self._rest_centred), -1)
        verts = transform_by_quat(self._rest_centred.expand(n_envs, -1, -1), quat) + self._centroid
        verts[..., 0] += offset_x[:, None]
        verts[..., 1] += offset_y[:, None]
        body = self.resolve_entity(self.entity_name)
        body.set_vertex_positions(verts, envs_idx=envs_idx)
        body.set_vertex_velocities(torch.zeros_like(verts), envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class ResetSoftBodyFromStates(EventTerm):
    """Place a Mochi soft body at one of a set of saved vertex states, drawn uniformly per env, at rest.

    The states are an ``(n_states, n_vertices, 3)`` array in an ``.npz`` of the Eden asset dataset, in the body's
    vertex order, shifted by ``offset`` into the scene (a benchmark's saved starts, recorded in its own world frame).
    ``planar_scale`` first shrinks or grows each state in the plane about the centre of its own xy bounding box,
    heights kept: the saved starts of a garment that the entity loads at that ``scale``. Every vertex velocity is
    zeroed.

    Parameters
    ----------
    entity_name : str
        The Mochi soft body.
    states_file : str
        The ``.npz`` file's path in the Eden asset dataset (fetched like an entity's file).
    states_key : str
        The array in it.
    offset : tuple[float, float, float]
        Added to every saved position (m), after the scaling.
    planar_scale : float
        The states' x and y scaled by this about each state's own xy centre (see above).
    state_indices : list[int]
        The states to draw from (indices into the array); empty draws from all of them.
    """

    entity_name: str = ""
    states_file: str = ""
    states_key: str = "starts"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    planar_scale: float = 1.0
    state_indices: list[int] = []

    def build(self) -> None:
        body = self.resolve_entity(self.entity_name)
        if not hasattr(body, "set_vertex_positions"):
            raise ValueError(
                f"{term_subject(self)} places a Mochi soft body, but the entity is a {type(body).__name__}."
            )
        states = load_vertex_states(self, self.states_file, self.states_key, body.n_vertices)
        states = scale_in_plane(self, states, self.planar_scale)
        if states.ndim == 2:
            states = states[None]
        bad = [i for i in self.state_indices if not 0 <= i < len(states)]
        if bad:
            raise ValueError(
                f"{term_subject(self)}: state_indices {bad} are not states of {self.states_file!r}[{self.states_key!r}], "
                f"which holds {len(states)} (indices 0..{len(states) - 1})."
            )
        if self.state_indices:
            states = states[list(self.state_indices)]
        self._states = torch.as_tensor(states, dtype=torch.float, device=self.device) + torch.tensor(
            self.offset, device=self.device
        )

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        verts = self._states[torch.randint(len(self._states), (n_envs,), device=self.device)]
        body = self.resolve_entity(self.entity_name)
        body.set_vertex_positions(verts, envs_idx=envs_idx)
        body.set_vertex_velocities(torch.zeros_like(verts), envs_idx=envs_idx)


def scale_in_plane(term, states: np.ndarray, scale: float) -> np.ndarray:
    """``states`` ``(..., n_vertices, 3)`` with each state's x and y scaled by ``scale`` about its own xy box centre.

    Heights are kept: a garment lying on a table stays on it, its layers as far apart as before.
    """
    if not scale > 0.0:
        raise ValueError(f"{term_subject(term)}: planar_scale must be positive, got {scale}.")
    lo, hi = states[..., :2].min(axis=-2, keepdims=True), states[..., :2].max(axis=-2, keepdims=True)
    centre = 0.5 * (lo + hi)
    out = np.array(states, dtype=np.float64)
    out[..., :2] = centre + scale * (states[..., :2] - centre)
    return out


def load_vertex_states(term, states_file: str, states_key: str, n_vertices: int) -> np.ndarray:
    """Read ``states_key`` of the asset ``states_file``: vertex positions ``(..., n_vertices, 3)`` of a soft body.

    Raises naming the file and its arrays when the key is missing or the shape does not index ``n_vertices``.
    """
    from eden.options.entities import EntityOptions

    if not states_file:
        raise ValueError(
            f"{term_subject(term)} needs states_file, the .npz of vertex states in the Eden asset dataset."
        )
    path = EntityOptions.fetch_family_asset(states_file)
    with np.load(path, allow_pickle=False) as data:
        if states_key not in data.files:
            raise KeyError(f"{term_subject(term)}: {states_file!r} has no array {states_key!r}; it has {data.files}.")
        states = np.asarray(data[states_key])
    if states.ndim not in (2, 3) or states.shape[-2:] != (n_vertices, 3):
        raise ValueError(
            f"{term_subject(term)}: {states_file!r}[{states_key!r}] has shape {states.shape}, but the soft body has "
            f"{n_vertices} vertices, so it must be ({n_vertices}, 3) or (n_states, {n_vertices}, 3). Was it saved for "
            f"another mesh?"
        )
    return states


@EVENT_TERM_REGISTRY.register()
class DisableMochiContact(EventTerm):
    """Switch off contact between two Mochi bodies, in every env. Fire it at STARTUP.

    Both bodies keep every other contact, so this lets one pass through a surface that still stops the rest:
    ``mochi_route_the_cable`` takes its cable off the bench plane the arms stand on, and lays it on a bed of its own.

    Parameters
    ----------
    entity_name, other_name : str
        The two Mochi bodies (rigid or soft).
    """

    entity_name: str = ""
    other_name: str = ""

    def build(self) -> None:
        for name in (self.entity_name, self.other_name):
            if not hasattr(self.resolve_entity(name), "set_contact_enabled"):
                raise ValueError(
                    f"{term_subject(self)} switches contact between Mochi bodies, but '{name}' is a "
                    f"{type(self.resolve_entity(name)).__name__}."
                )

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        self.resolve_entity(self.entity_name).set_contact_enabled(self.resolve_entity(self.other_name), False)
