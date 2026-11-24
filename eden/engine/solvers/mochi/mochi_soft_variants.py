"""The heterogeneous deformable entity of the MochiSolver: one deformable body per variant, each present in the
environments assigned to it.

The topology tables of the deformable bodies (vertices, elements, contact samples, the Hessian sparsity) carry no
environment axis, so every variant is an ordinary `MochiSoftEntity` of the solver, present in every environment's
arrays. The kernels read a per-environment activity table (``MochiSoftInfo.entities_is_active``) and leave a variant out
of the environments that hold another one: there its vertices stay fixed at their rest positions, it takes part in no
contact on either side and no element assembly, and it is not drawn. A variant that no environment holds is absent
everywhere.

`MochiSoftVariantsEntity` is the scene entity of such a body. Its per-vertex reads and writes act on the variant each
environment holds, laid out over the vertex count of the largest variant: a read pads the rows past the vertices of an
environment's own variant with zeros (``vertices_mask`` tells them apart), and a write ignores them.
"""

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch
from genesis.engine.entities.base_entity import Entity
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.options.surfaces import Surface

from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.mochi_entity import filter_entity_contacts
from eden.engine.solvers.mochi.states import MochiSoftEntityState

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

    from eden.engine.solvers.mochi.mochi_soft_entity import MochiSoftEntity
    from eden.engine.solvers.mochi.mochi_solver import MochiSolver
    from eden.engine.solvers.mochi.specs import MochiSoftSpec


def _per_variant(name: str, doc: str) -> property:
    """A property of one mesh, which a heterogeneous entity holds once per variant: reading it on the entity raises."""

    def getter(self: "MochiSoftVariantsEntity"):
        self._raise_per_variant(name)

    return property(getter, doc=f"{doc} Defined per variant: read `entity.variants[k].{name}`.")


class MochiSoftVariantsEntity(Entity):
    """
    Heterogeneous deformable body of the MochiSolver: a solid, shell or rod whose mesh differs between the parallel
    environments. It is created by a `MochiSoftSpec` (or ``scene.add_entity``) given a sequence of morphs, one per
    variant, all of them taking the spec's one material.

    Each environment holds one variant, assigned at build by Genesis' rule for heterogeneous entities (contiguous,
    balanced blocks of environments in variant order; see `envs_variant_idx`). Every variant is a `MochiSoftEntity` of
    the solver (`variants`), which reports this entity's scene index in contact records. Per-vertex arrays are laid out
    over `n_vertices`, the vertex count of the largest variant, with the vertices of an environment's own variant first
    and padding after (`vertices_mask`).

    Memory and per-step loop extents of the solver grow with the summed size of the variants times the number of
    environments: every environment stores every variant, present or not.
    """

    def __init__(
        self,
        scene: "Scene",
        solver: "MochiSolver",
        surface: Surface,
        spec: "MochiSoftSpec",
        idx: int,
    ):
        # Read by the name generated from the morph, in the base constructor
        self._is_shell = isinstance(spec.material, mochi_materials.Shell)
        self._is_rod = isinstance(spec.material, mochi_materials.Rod)
        super().__init__(idx, scene, solver, surface, spec)
        self._variants: "gs.List[MochiSoftEntity]" = gs.List()
        # Assigned at build (`_assign_envs`, `_build`)
        self._envs_variant_idx_np = None
        self._envs_variant_idx = None
        self._envs_v_start = None
        self._envs_n_vertices = None
        self._vertices_mask = None
        self._rest_padded = None

    def _get_morph_identifier(self) -> str:
        prefix = "rod" if self._is_rod else "shell" if self._is_shell else "soft"
        return f"{prefix}_variants"

    def _add_variant(self, body: "MochiSoftEntity") -> None:
        """Make a body of the solver the next variant of this entity."""
        body._variants_owner = self
        body._variant_idx = len(self._variants)
        self._variants.append(body)

    def _assign_envs(self, n_envs_: int) -> None:
        """Assign a variant to each of the solver's `n_envs_` environment rows, as Genesis does for the variants of a
        heterogeneous rigid entity and as Eden records them (`eden.entities.base.Entity.env_variant_idx`).
        """
        self._envs_variant_idx_np = _balanced_variant_mapping(self.n_variants, n_envs_).astype(gs.np_int)

    def _build(self) -> None:
        """Lay out the per-environment tables of the padded vertex arrays, once the solver placed the variants."""
        n_max = self.n_vertices
        envs_variant_idx = torch.as_tensor(self._envs_variant_idx_np, dtype=torch.long, device=gs.device)
        variants_v_start = torch.tensor([body.v_start for body in self._variants], dtype=gs.tc_int, device=gs.device)
        variants_n_vertices = torch.tensor(self.variants_n_vertices, dtype=gs.tc_int, device=gs.device)
        self._envs_variant_idx = envs_variant_idx
        self._envs_v_start = variants_v_start[envs_variant_idx].contiguous()
        self._envs_n_vertices = variants_n_vertices[envs_variant_idx].contiguous()
        self._vertices_mask = torch.arange(n_max, device=gs.device)[None, :] < self._envs_n_vertices[:, None]
        rest_padded = np.zeros((self.n_variants, n_max, 3), dtype=gs.np_float)
        for k, body in enumerate(self._variants):
            rest_padded[k, : body.n_vertices] = body.init_positions
        self._rest_padded = torch.as_tensor(rest_padded, device=gs.device)
        self._is_built = True

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def variants(self) -> "gs.List[MochiSoftEntity]":
        """The deformable body of every variant, in the order of the spec's morphs."""
        return self._variants

    @property
    def n_variants(self) -> int:
        return len(self._variants)

    @property
    def morphs(self):
        """The morph of every variant."""
        return self._spec.morphs

    @property
    def envs_variant_idx(self) -> torch.Tensor:
        """The variant every environment holds, shape (n_envs,), or () for an unparallelized scene. Fixed at build."""
        self._require_built("envs_variant_idx")
        return self._envs_variant_idx[0] if self._solver.n_envs == 0 else self._envs_variant_idx

    @property
    def variants_n_vertices(self) -> list[int]:
        """The vertex count of every variant."""
        return [body.n_vertices for body in self._variants]

    @property
    def n_vertices(self) -> int:
        """The vertex count of the largest variant, the length of the vertex axis of every per-vertex array."""
        return max(self.variants_n_vertices)

    @property
    def vertices_mask(self) -> torch.Tensor:
        """Which rows of the vertex axis are vertices of the environment's own variant (the others are padding), shape
        (n_envs, n_vertices), or (n_vertices,) for an unparallelized scene.
        """
        self._require_built("vertices_mask")
        return self._vertices_mask[0] if self._solver.n_envs == 0 else self._vertices_mask

    @property
    def is_shell(self) -> bool:
        return self._is_shell

    @property
    def is_rod(self) -> bool:
        return self._is_rod

    init_positions = _per_variant("init_positions", "Rest vertex positions.")
    elems = _per_variant("elems", "Elements: tetrahedra, shell triangles or rod segments.")
    n_elements = _per_variant("n_elements", "Number of elements.")
    n_dofs = _per_variant("n_dofs", "Number of vertex degrees of freedom.")
    surface_triangles = _per_variant("surface_triangles", "Boundary triangles.")
    n_surfaces = _per_variant("n_surfaces", "Number of boundary triangles.")
    mass = _per_variant("mass", "Total mass.")
    volume = _per_variant("volume", "Rest volume.")
    area = _per_variant("area", "Total area of the surface triangles.")
    length = _per_variant("length", "Total length of the rod centerline.")
    contact_skin = _per_variant("contact_skin", "Contact skin.")
    vgeoms = _per_variant("vgeoms", "Visual geoms.")
    idx_in_solver = _per_variant("idx_in_solver", "Index among the deformable bodies of the solver.")
    v_start = _per_variant("v_start", "First vertex in the solver.")
    v_end = _per_variant("v_end", "End of the vertex range in the solver.")

    def _raise_per_variant(self, name: str):
        gs.raise_exception(
            f"`{name}` differs between the {self.n_variants} variants of heterogeneous entity '{self.name}': read it "
            f"per variant, `entity.variants[k].{name}`, where `entity.envs_variant_idx[i_b]` is the variant that "
            "environment i_b holds."
        )

    def _raise_unsupported(self, method: str):
        if method in ("set_visual_mesh", "set_contact_skin"):
            gs.raise_exception(
                f"`{method}` places one mesh, and heterogeneous entity '{self.name}' has {self.n_variants} variants: "
                f"call it per variant before the build, `entity.variants[k].{method}(...)`, giving each the mesh made "
                "for that variant."
            )
        gs.raise_exception(
            f"`{method}` is not supported on heterogeneous entity '{self.name}': an attachment acts in every "
            "environment, also those where its variant is absent, and pulls the link toward a body that is not "
            "there. Add the attached body as an entity of its own, created from one morph."
        )

    def _require_built(self, what: str) -> None:
        if not self._is_built:
            gs.raise_exception(
                f"`{what}` of heterogeneous entity '{self.name}' is assigned when the scene is built: read it after "
                "`scene.build()`."
            )

    # ------------------------------------------------------------------------------------
    # ---------------------------------------- io ----------------------------------------
    # ------------------------------------------------------------------------------------

    def get_state(self):
        """Vertex positions and velocities of every environment's variant, shape (n_envs, n_vertices, 3), padded."""
        state = MochiSoftEntityState(self, self._solver.sim.cur_step_global)
        envs_idx = self._solver._envs_idx
        state.pos[:] = self._solver.get_soft_variants_field(self, self._solver.soft_state.verts_pos, envs_idx)
        state.vel[:] = self._solver.get_soft_variants_field(self, self._solver.soft_state.verts_vel, envs_idx)
        return state

    def _field(self, field, envs_idx):
        envs_idx = self._solver._sim.sanitize_envs_idx(envs_idx)
        out = self._solver.get_soft_variants_field(self, field, envs_idx)
        return out[0] if self._solver.n_envs == 0 else out

    def get_vertices_position(self, envs_idx=None):
        """World positions of the vertices of every environment's variant, shape (n_envs, n_vertices, 3) or
        (n_vertices, 3), zero past the variant's own vertices.
        """
        return self._field(self._solver.soft_state.verts_pos, envs_idx)

    def get_vertices_velocity(self, envs_idx=None):
        """Velocities of the vertices of every environment's variant, padded as `get_vertices_position`."""
        return self._field(self._solver.soft_state.verts_vel, envs_idx)

    def get_vertices_contact_force(self, envs_idx=None):
        """Net contact force on the vertices of every environment's variant, padded as `get_vertices_position`."""
        self._solver._record_contacts()
        return self._field(self._solver.soft_state.verts_contact_force, envs_idx)

    def get_contacts(self, with_entity=None, exclude_self_contact=False, is_padded=False):
        """Contact points involving this entity, whichever variant made them (see `MochiSoftEntity.get_contacts`). The
        vertices of a contact point are local to the variant of the environment it belongs to.
        """
        return filter_entity_contacts(self._solver, self, with_entity, exclude_self_contact, is_padded)

    def _padded_per_env(self, values, envs_idx, what):
        """Lay out vertex values as (len(envs_idx), n_vertices, 3): a (n_vertices, 3) array is shared by every
        environment, a (len(envs_idx), n_vertices, 3) one has a row per environment.
        """
        n_max = self.n_vertices
        if values.shape == (n_max, 3):
            return values[None].expand(len(envs_idx), -1, -1)
        if values.shape != (len(envs_idx), n_max, 3):
            gs.raise_exception(
                f"Expected {what} of shape ({len(envs_idx)}, {n_max}, 3) or ({n_max}, 3) for heterogeneous entity "
                f"'{self.name}' (the vertex count of its largest variant, padding ignored), got {tuple(values.shape)}."
            )
        return values

    def set_position(self, pos, envs_idx=None):
        """Move every targeted environment's variant: a (3,) or (n_envs, 3) offset of its rest positions, or vertex
        positions of shape (n_vertices, 3) or (n_envs, n_vertices, 3) whose rows past the variant's vertices are
        ignored.
        """
        envs_idx = self._solver._sim.sanitize_envs_idx(envs_idx)
        pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        if pos.shape[-2:] != (self.n_vertices, 3):
            rest = self._rest_padded[self._envs_variant_idx[envs_idx]]
            if pos.shape == (3,):
                pos = rest + pos
            elif pos.shape == (len(envs_idx), 3):
                pos = rest + pos[:, None, :]
            else:
                gs.raise_exception(
                    f"Expected an offset of shape (3,) or ({len(envs_idx)}, 3), or vertex positions of shape "
                    f"({self.n_vertices}, 3) or ({len(envs_idx)}, {self.n_vertices}, 3), for heterogeneous entity "
                    f"'{self.name}'; got {tuple(pos.shape)}."
                )
        pos = self._padded_per_env(pos, envs_idx, "vertex positions")
        self._solver.set_soft_variants_position(self, pos.contiguous(), envs_idx)

    def set_velocity(self, vel, envs_idx=None):
        """Set the vertex velocities of every targeted environment's variant: one (3,) or (n_envs, 3) velocity for all
        its vertices, or per vertex of shape (n_vertices, 3) or (n_envs, n_vertices, 3), padding ignored.
        """
        envs_idx = self._solver._sim.sanitize_envs_idx(envs_idx)
        vel = torch.as_tensor(vel, dtype=gs.tc_float, device=gs.device)
        if vel.shape[-2:] != (self.n_vertices, 3):
            if vel.shape == (3,):
                vel = vel.expand(len(envs_idx), self.n_vertices, 3)
            elif vel.shape == (len(envs_idx), 3):
                vel = vel[:, None, :].expand(-1, self.n_vertices, -1)
            else:
                gs.raise_exception(
                    f"Expected a velocity of shape (3,) or ({len(envs_idx)}, 3), or vertex velocities of shape "
                    f"({self.n_vertices}, 3) or ({len(envs_idx)}, {self.n_vertices}, 3), for heterogeneous entity "
                    f"'{self.name}'; got {tuple(vel.shape)}."
                )
        vel = self._padded_per_env(vel, envs_idx, "vertex velocities")
        self._solver.set_soft_variants_velocity(self, vel.contiguous(), envs_idx)

    def _sanitize_vertices_idx(self, verts_idx_local, envs_idx):
        """Lay out a vertex set as one row per targeted environment, of indices local to the environment's own
        variant: a set shared by every environment, shape (n,), or one per environment, shape (n_envs, n) padded with
        -1. An index past the vertices of some environment's variant raises, naming the environment.
        """
        verts_idx = torch.atleast_1d(torch.as_tensor(verts_idx_local, dtype=gs.tc_int, device=gs.device))
        if verts_idx.ndim == 1:
            rows = verts_idx[None].expand(len(envs_idx), -1)
            verts_idx_min = 0
        elif verts_idx.ndim == 2 and verts_idx.shape[0] == len(envs_idx):
            rows = verts_idx
            verts_idx_min = -1
        else:
            gs.raise_exception(
                "Pass one vertex set shared by every environment, of shape (n,), or one set per environment, of shape "
                f"({len(envs_idx)}, n) padded with -1. Got an array of shape {tuple(verts_idx.shape)}."
            )
        n_verts = self._envs_n_vertices[envs_idx][:, None]
        is_bad = (rows < verts_idx_min) | (rows >= n_verts)
        if is_bad.any():
            i_row, i_col = (int(i) for i in torch.nonzero(is_bad)[0])
            env = int(envs_idx[i_row])
            variant = int(self._envs_variant_idx[env])
            gs.raise_exception(
                f"Vertex index {int(rows[i_row, i_col])} is out of range for environment {env}, which holds variant "
                f"{variant} of heterogeneous entity '{self.name}', of {self.variants_n_vertices[variant]} vertices "
                f"(local indices [{verts_idx_min}, {self.variants_n_vertices[variant]})). Index each environment's "
                f"own variant: pass one row per environment, shape ({len(envs_idx)}, n) padded with -1, built from "
                "`entity.envs_variant_idx` and `entity.variants_n_vertices`."
            )
        return rows.contiguous(), verts_idx.ndim == 1

    def set_vertices_fixed(self, verts_idx_local, is_fixed=True, envs_idx=None):
        """Fix the given vertices of every targeted environment's variant at their current position, or release them.

        `verts_idx_local` holds indices local to each environment's own variant: one set for every environment, shape
        (n,), or one set per environment, shape (n_envs, n), whose -1 entries are skipped.
        """
        envs_idx = self._solver._sim.sanitize_envs_idx(envs_idx)
        verts_idx, _ = self._sanitize_vertices_idx(verts_idx_local, envs_idx)
        self._solver.set_soft_variants_vertices_fixed(self, verts_idx, is_fixed, envs_idx)

    def set_vertices_target(self, verts_idx_local, pos, envs_idx=None):
        """Prescribe the positions the given vertices of every targeted environment's variant must reach at the end
        of the next step (see `MochiSoftEntity.set_vertices_target`).

        `verts_idx_local` is laid out as in `set_vertices_fixed`; `pos` is of shape (n, 3) or (n_envs, n, 3) for a
        shared set and (n_envs, n, 3) for a set per environment.
        """
        envs_idx = self._solver._sim.sanitize_envs_idx(envs_idx)
        verts_idx, is_shared = self._sanitize_vertices_idx(verts_idx_local, envs_idx)
        n_verts = verts_idx.shape[1]
        pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        if is_shared and pos.shape == (n_verts, 3):
            pos = pos[None].expand(len(envs_idx), -1, -1)
        if pos.shape != (len(envs_idx), n_verts, 3):
            gs.raise_exception(
                f"Expected an array of shape ({len(envs_idx)}, {n_verts}, 3)"
                + (f" or ({n_verts}, 3)" if is_shared else "")
                + f", got {tuple(pos.shape)}."
            )
        self._solver.set_soft_variants_vertices_target(self, verts_idx, pos.contiguous(), envs_idx)

    def set_contact_params(
        self,
        *,
        penalty_coefficient=None,
        friction=None,
        penalty_smoothing_half_distance=None,
        penalty_threshold=None,
        friction_falloff_vel=None,
        viscous_friction=None,
        normal_viscous_damping=None,
    ):
        """Update the contact parameters of every variant (see `eden.engine.solvers.mochi.materials.Elastic`)."""
        for body in self._variants:
            self._solver.set_soft_entity_contact_params(
                body,
                penalty_coefficient=penalty_coefficient,
                friction=friction,
                penalty_smoothing_half_distance=penalty_smoothing_half_distance,
                penalty_threshold=penalty_threshold,
                friction_falloff_vel=friction_falloff_vel,
                viscous_friction=viscous_friction,
                normal_viscous_damping=normal_viscous_damping,
            )

    def attach_to_link(self, link, verts_idx, stiffness=1e6, damping=0.0):
        self._raise_unsupported("attach_to_link")

    def set_visual_mesh(self, file, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), scale=1.0):
        self._raise_unsupported("set_visual_mesh")

    def set_contact_skin(self, file, pos=(0.0, 0.0, 0.0), euler=(0.0, 0.0, 0.0), scale=1.0):
        self._raise_unsupported("set_contact_skin")
