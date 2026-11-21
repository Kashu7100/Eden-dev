"""MochiSoftVariantsEntity: a deformable body whose mesh differs between the parallel environments.

Every variant is a body of the solver, present in the environments assigned to it and fixed at rest, out of contact
and out of the element assembly in the others. The acceptance tests compare a heterogeneous scene, environment by
environment, with a scene that holds only that environment's variants.

Those comparisons are exact. On the CPU with one quadrants thread every loop of the solver runs in order, the variants
an environment does not hold contribute only exact zeros to the sums it takes (residual, dot products, norms), and
their contact candidates are never formed, so each environment repeats the floating-point operations of its own
scene. Fast-math is off (``strict_math``): the heterogeneous scene compiles its kernels with the activity checks, and
with fast-math LLVM may contract and reassociate the arithmetic of those kernels differently, which measured 4.5e-14 m
apart after 30 steps of a falling shell. The options the solver derives from the total size of the system
(`n_pcg_iterations`, and whether the dense matrix fits `dense_matrix_max_dofs`) are pinned, so both scenes take the
same solver path.
"""

import genesis as gs
import numpy as np
import pytest
import torch
import trimesh
from genesis.engine.solvers.kinematic_solver import _balanced_variant_mapping
from genesis.utils.misc import qd_to_numpy

from eden.engine.solvers.mochi import MochiOptions, MochiSoftSpec, MochiSoftVariantsEntity, Rod, create_mochi_scene
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.mochi_soft_entity import MochiSoftEntity

N_STEPS = 30
# Pinned in every scene of a comparison: the solver otherwise derives them from the size of the whole system, which
# counts every variant (see the module docstring).
SOLVER_PATH = {"n_pcg_iterations": 200, "dense_matrix_max_dofs": 1024}
CONTACT = {"penalty_coefficient": 1e7}
# A tetgen box refined past its corners, so that solid variants differ in vertex count while the whole system stays
# small enough for the dense solver
REFINED = {"maxvolume": 2e-4, "nobisect": False}


def _sheet_obj(path, n_cells, size):
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([X.reshape(-1), Y.reshape(-1), np.zeros(X.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces += [[a, a + 1, a + n_cells + 2], [a, a + n_cells + 2, a + n_cells + 1]]
    trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False).export(path)
    return str(path)


def _scene(n_envs, add_bodies, **mochi_kwargs):
    """A built scene of a ground plane, a fixed rigid sphere at the origin and the bodies `add_bodies(scene)` adds."""
    scene = create_mochi_scene(
        sim_options=gs.options.SimOptions(dt=1.0 / 60.0),
        mochi_options=MochiOptions(n_newton_iterations=8, **{**SOLVER_PATH, **mochi_kwargs}),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=mochi_materials.Rigid(**CONTACT))
    scene.add_entity(gs.morphs.Sphere(radius=0.1, pos=(0.0, 0.0, 0.1), fixed=True), material=mochi_materials.Rigid())
    bodies = add_bodies(scene)
    scene.build(n_envs=n_envs)
    return scene, bodies


def _assert_envs_match_their_own_variants(bodies_variants, n_envs, n_steps=N_STEPS, **mochi_kwargs):
    """Step a scene of heterogeneous bodies and, for every environment, a scene of the variants it holds, and require
    the same vertex positions. `bodies_variants` holds, per body, a list of variants, each a function adding the body
    of that variant to a scene from one morph or (as a sequence) from several.
    """

    def add_all(scene):
        return [add(scene, list(range(len(variants)))) for add, variants in bodies_variants]

    scene, bodies = _scene(n_envs, add_all, **mochi_kwargs)
    for _ in range(n_steps):
        scene.step()
    positions = [body.get_vertices_position() for body in bodies]
    mappings = [_balanced_variant_mapping(len(variants), n_envs) for _, variants in bodies_variants]
    for body, mapping in zip(bodies, mappings):
        if isinstance(body, MochiSoftVariantsEntity):
            assert body.envs_variant_idx.tolist() == mapping.tolist()

    for i_b in range(n_envs):

        def add_own(scene, i_b=i_b):
            return [add(scene, [int(mapping[i_b])]) for (add, _), mapping in zip(bodies_variants, mappings)]

        scene_b, bodies_b = _scene(1, add_own, **mochi_kwargs)
        for _ in range(n_steps):
            scene_b.step()
        for i_body, (body_b, pos) in enumerate(zip(bodies_b, positions)):
            expected = body_b.get_vertices_position()[0]
            n = len(expected)
            # Exact: see the module docstring
            torch.testing.assert_close(
                pos[i_b, :n], expected, rtol=0.0, atol=0.0, msg=lambda m, i_body=i_body, i_b=i_b: f"{i_body}/{i_b}: {m}"
            )
            assert (pos[i_b, n:] == 0.0).all()
        scene_b.destroy()
    return scene, bodies


def _adder(morphs, material):
    """A function adding the body made of the chosen ones of `morphs` (factories) and `material` to a scene."""

    def add(scene, chosen):
        variants = [morphs[k]() for k in chosen]
        return scene.add_entity(variants[0] if len(variants) == 1 else variants, material=material)

    return add


def _cloth_material(**kwargs):
    return mochi_materials.Shell(
        E=2e4, nu=0.3, rho=200.0, thickness=1e-3, collider_radius=0.01, self_contact=True, **CONTACT, **kwargs
    )


def _cloths(tmp_path, n_cells=(3, 4, 5), height=0.3, x=0.0):
    """Square sheets of different resolutions, held over the sphere (or over the point at `x` on the ground)."""
    return [
        (
            lambda n=n: gs.morphs.Mesh(
                file=_sheet_obj(tmp_path / f"sheet_{n}.obj", n, 0.4 - 0.02 * n), pos=(x, 0.01 * n, height)
            )
        )
        for n in n_cells
    ]


# ------------------------------------------------------------------------------------------------------------------
# equivalence with the scenes of one variant
# ------------------------------------------------------------------------------------------------------------------


@pytest.mark.strict_math
@pytest.mark.parametrize("step_kernel", ["monolith", "pipeline"])
def test_shells_falling_on_a_sphere_match_their_own_scenes(tmp_path, step_kernel):
    """Three sheets of 16, 25 and 36 vertices with self-contact over four environments (the first sheet in two),
    draped over a rigid sphere and the ground.
    """
    cloths = _cloths(tmp_path)
    scene, (cloth,) = _assert_envs_match_their_own_variants(
        [(_adder(cloths, _cloth_material()), cloths)], n_envs=4, step_kernel=step_kernel
    )
    assert cloth.variants_n_vertices == [16, 25, 36]
    # The sheets reached the sphere
    pos = cloth.get_vertices_position()
    assert (pos[:, :16, 2].amin(dim=-1) < 0.25).all()


@pytest.mark.strict_math
def test_solids_falling_on_a_sphere_match_their_own_scenes():
    solids = [
        lambda: gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.02, 0.0, 0.35)),
        lambda: gs.morphs.Box(size=(0.16, 0.1, 0.08), pos=(-0.02, 0.0, 0.33), **REFINED),
    ]
    _, (solid,) = _assert_envs_match_their_own_variants(
        [(_adder(solids, mochi_materials.Elastic(E=5e4, **CONTACT)), solids)], n_envs=2
    )
    assert solid.variants_n_vertices[0] < solid.variants_n_vertices[1]


def _rods():
    def rod(n_nodes, y):
        points = np.stack([np.linspace(-0.25, 0.25, n_nodes), np.full(n_nodes, y), np.zeros(n_nodes)], axis=-1)
        return Rod(points=points, radius=0.01, pos=(0.0, 0.0, 0.3))

    return [lambda: rod(6, 0.0), lambda: rod(9, 0.02)]


@pytest.mark.strict_math
@pytest.mark.parametrize("linear_solver", ["pcg", "ldlt"])
def test_rods_falling_on_a_sphere_match_their_own_scenes(linear_solver):
    """Open rods of 6 and 9 nodes: the iterative solver preconditions them with the per-rod band, the direct one
    factors their twist rows, which the absent variant keeps as identities in both.
    """
    rods = _rods()
    _assert_envs_match_their_own_variants(
        [(_adder(rods, mochi_materials.Rod(E=1e7, **CONTACT)), rods)], n_envs=2, linear_solver=linear_solver
    )


@pytest.mark.strict_math
def test_a_cloth_on_soft_colliders_matches_its_own_scenes(tmp_path):
    """A heterogeneous cloth of two sheets falls onto a solid of its own and a heterogeneous solid of three variants,
    both resting on the ground, which collide with it through their tetrahedra. Over three environments the cloth's
    variants go 0, 0, 1 and the solid's 0, 1, 2, so each environment holds a different pair.
    """
    cloths = _cloths(tmp_path, n_cells=(3, 4), height=0.2, x=0.8)
    solid_material = mochi_materials.Elastic(E=5e4, **CONTACT)
    block = [lambda: gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.68, 0.0, 0.05))]
    solids = [
        lambda: gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.92, 0.0, 0.05)),
        lambda: gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.92, 0.0, 0.05), **REFINED),
        lambda: gs.morphs.Box(size=(0.14, 0.08, 0.1), pos=(0.92, 0.0, 0.05), **REFINED),
    ]
    scene, (cloth, _, solid) = _assert_envs_match_their_own_variants(
        [
            (_adder(cloths, _cloth_material()), cloths),
            (_adder(block, solid_material), block),
            (_adder(solids, solid_material), solids),
        ],
        n_envs=3,
        n_steps=40,
    )
    assert scene.mochi_solver.mochi_config.has_soft_colliders
    # The cloth lies on the heterogeneous solid in every environment, a contact through its tetrahedra
    contacts = cloth.get_contacts(with_entity=solid, is_padded=True)
    assert contacts["valid_mask"].any(dim=-1).all()


@pytest.mark.strict_math
def test_sleeping_bodies_match_their_own_scenes():
    """With sleeping on, an absent variant sleeps on its own island and leaves the others to sleep as they would."""
    solids = [
        lambda: gs.morphs.Box(size=(0.12, 0.12, 0.12), pos=(0.4, 0.0, 0.08)),
        lambda: gs.morphs.Box(size=(0.16, 0.1, 0.08), pos=(0.4, 0.0, 0.06), **REFINED),
    ]
    scene, (solid,) = _assert_envs_match_their_own_variants(
        [(_adder(solids, mochi_materials.Elastic(E=5e4, **CONTACT)), solids)],
        n_envs=2,
        n_steps=60,
        use_sleeping=True,
        sleep_min_steps=4,
    )
    # Every body of the scene sleeps at the end, the absent variants included
    n_rigid = len(scene.mochi_solver.entities)
    assert qd_to_numpy(scene.mochi_solver.island_state.nodes_is_asleep)[n_rigid:].all()


# ------------------------------------------------------------------------------------------------------------------
# the entity
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def cloth_scene(tmp_path):
    """Three sheets over four environments, stepped into contact with the sphere."""
    cloths = _cloths(tmp_path)
    scene, (cloth,) = _scene(4, lambda scene: [scene.add_entity([m() for m in cloths], material=_cloth_material())])
    for _ in range(N_STEPS):
        scene.step()
    return scene, cloth


@pytest.mark.precision("64")
def test_an_absent_variant_stays_at_rest_out_of_contact_and_out_of_sight(cloth_scene):
    scene, cloth = cloth_scene
    solver = scene.mochi_solver
    assert cloth.envs_variant_idx.tolist() == [0, 0, 1, 2]
    is_active = solver.soft_entities_is_active
    vverts = qd_to_numpy(solver.get_soft_state_render()[0], transpose=True)
    envs_offset = np.asarray(scene.envs_offset)
    for k, body in enumerate(cloth.variants):
        assert isinstance(body, MochiSoftEntity) and body.variants_owner is cloth and body.variant_idx == k
        holds = cloth.envs_variant_idx == k
        assert (is_active[body.idx_in_solver] == holds.numpy()).all()
        rest = torch.as_tensor(body.init_positions, dtype=gs.tc_float)
        pos = body.get_vertices_position()
        assert (pos[~holds] == rest).all()
        assert (pos[holds] != rest).any(dim=-1).any(dim=-1).all()
        assert (body.get_vertices_velocity()[~holds] == 0.0).all()
        assert (body.get_vertices_contact_force()[~holds] == 0.0).all()
        assert (body.get_vertices_contact_force()[holds] != 0.0).any()
        # Its render vertices collapse onto one point, which draws nothing
        for vgeom in body.vgeoms:
            for i_b in np.flatnonzero(~holds.numpy()):
                drawn = vverts[i_b, vgeom.vvert_start : vgeom.vvert_end]
                assert np.allclose(drawn, body.init_positions[0] + envs_offset[i_b], atol=1e-6)

    # A contact record names the entity, and its vertices are local to the environment's own variant
    contacts = cloth.get_contacts(is_padded=True)
    for i_b in range(4):
        n_verts = cloth.variants_n_vertices[int(cloth.envs_variant_idx[i_b])]
        valid = contacts["valid_mask"][i_b]
        assert valid.any()
        for side in ("a", "b"):
            mine = valid & (contacts[f"entity_{side}"][i_b] == cloth.idx)
            verts = contacts[f"verts_{side}"][i_b][mine]
            assert (verts < n_verts).all() and (verts[:, 0] >= 0).all()


@pytest.mark.precision("64")
def test_the_vertex_arrays_are_padded_to_the_largest_variant(cloth_scene):
    scene, cloth = cloth_scene
    counts = torch.tensor([16, 16, 25, 36])
    assert cloth.n_vertices == 36
    assert (cloth.vertices_mask.sum(dim=-1) == counts).all()
    for getter in (cloth.get_vertices_position, cloth.get_vertices_velocity, cloth.get_vertices_contact_force):
        values = getter()
        assert values.shape == (4, 36, 3)
        assert (values[~cloth.vertices_mask] == 0.0).all()
    state = cloth.get_state()
    torch.testing.assert_close(state.pos, cloth.get_vertices_position(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(state.vel, cloth.get_vertices_velocity(), rtol=0.0, atol=0.0)
    assert cloth.get_vertices_position(envs_idx=[2]).shape == (1, 36, 3)
    for i_b in range(4):
        body = cloth.variants[int(cloth.envs_variant_idx[i_b])]
        torch.testing.assert_close(
            cloth.get_vertices_position()[i_b, : body.n_vertices],
            body.get_vertices_position()[i_b],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.precision("64")
def test_the_setters_write_each_environment_s_own_variant(cloth_scene):
    scene, cloth = cloth_scene
    rests = [torch.as_tensor(body.init_positions, dtype=gs.tc_float) for body in cloth.variants]
    absent_before = [body.get_vertices_position().clone() for body in cloth.variants]
    mask = cloth.vertices_mask

    def own_rest(i_b):
        return rests[int(cloth.envs_variant_idx[i_b])]

    # An offset of the rest positions, one for all environments, then one per environment
    cloth.set_position(torch.tensor([0.0, 0.0, 0.5]))
    pos = cloth.get_vertices_position()
    for i_b in range(4):
        torch.testing.assert_close(pos[i_b, : len(own_rest(i_b))], own_rest(i_b) + torch.tensor([0.0, 0.0, 0.5]))
    offsets = torch.arange(4, dtype=gs.tc_float)[:, None] * torch.tensor([0.1, 0.0, 0.0])
    cloth.set_position(offsets)
    pos = cloth.get_vertices_position()
    for i_b in range(4):
        torch.testing.assert_close(pos[i_b, : len(own_rest(i_b))], own_rest(i_b) + offsets[i_b])
    # Full padded positions of some environments: the padding rows are ignored
    padded = torch.full((2, 36, 3), 7.0)
    padded[:, :16] = rests[0] + 1.0
    cloth.set_position(padded, envs_idx=[0, 1])
    torch.testing.assert_close(cloth.get_vertices_position()[:2, :16], rests[0][None].expand(2, -1, -1) + 1.0)
    assert (cloth.get_vertices_position()[:2, 16:] == 0.0).all()
    # Velocities: one for all vertices of every environment, padding untouched
    cloth.set_velocity(torch.tensor([0.0, 0.0, -1.0]))
    vel = cloth.get_vertices_velocity()
    assert (vel[mask] == torch.tensor([0.0, 0.0, -1.0])).all() and (vel[~mask] == 0.0).all()

    # Every absent variant kept its rest positions through all of it
    for k, body in enumerate(cloth.variants):
        holds = cloth.envs_variant_idx == k
        assert (body.get_vertices_position()[~holds] == absent_before[k][~holds]).all()

    # Fixed vertices: indices local to every environment's own variant, padded per environment
    fixed = torch.tensor([[0, 15, -1], [1, -1, -1], [24, 3, -1], [35, 0, 2]])
    cloth.set_vertices_fixed(fixed)
    is_fixed = qd_to_numpy(scene.mochi_solver.soft_state.verts_is_fixed)
    for i_b in range(4):
        body = cloth.variants[int(cloth.envs_variant_idx[i_b])]
        expected = {int(v) for v in fixed[i_b] if v >= 0}
        held = set(np.flatnonzero(is_fixed[body.v_start : body.v_end, i_b]).tolist())
        assert held == expected
    target = cloth.get_vertices_position()[:, 0] + torch.tensor([0.0, 0.0, 0.01])
    cloth.set_vertices_target(torch.zeros((4, 1), dtype=torch.int64), target[:, None, :])
    scene.step()
    torch.testing.assert_close(cloth.get_vertices_position()[:, 0], target, rtol=0.0, atol=1e-12)
    cloth.set_vertices_fixed(fixed, is_fixed=False)
    # The absent variants are still fixed at rest
    is_fixed = qd_to_numpy(scene.mochi_solver.soft_state.verts_is_fixed)
    for k, body in enumerate(cloth.variants):
        holds = (cloth.envs_variant_idx == k).numpy()
        assert is_fixed[body.v_start : body.v_end][:, ~holds].all()

    # Contact parameters reach every variant
    cloth.set_contact_params(friction=0.2)
    friction = qd_to_numpy(scene.mochi_solver.soft_info.entities_friction)
    assert all(friction[body.idx_in_solver] == pytest.approx(0.2) for body in cloth.variants)


@pytest.mark.precision("64")
def test_the_setters_refuse_what_they_cannot_write(cloth_scene):
    _, cloth = cloth_scene
    with pytest.raises(
        gs.GenesisException,
        match=r"Vertex index 20 is out of range for environment 0, which holds variant 0 of heterogeneous entity "
        r"'.*', of 16 vertices",
    ):
        cloth.set_vertices_fixed([20])
    # The same index is in range for the environments holding a larger variant
    cloth.set_vertices_fixed([20], envs_idx=[2, 3])
    with pytest.raises(gs.GenesisException, match=r"one set per environment, of shape \(4, n\)"):
        cloth.set_vertices_fixed(torch.zeros((3, 2), dtype=torch.int64))
    with pytest.raises(gs.GenesisException, match=r"Expected an offset of shape \(3,\) or \(4, 3\)"):
        cloth.set_position(torch.zeros(2))
    with pytest.raises(gs.GenesisException, match=r"Expected an array of shape \(4, 1, 3\) or \(1, 3\)"):
        cloth.set_vertices_target([0], torch.zeros((2, 3)))

    # A variant body writes every environment, so its setters point at the entity
    with pytest.raises(gs.GenesisException, match=r"`set_vertices_fixed` on variant 1 .*`variant\.variants_owner"):
        cloth.variants[1].set_vertices_fixed([0])
    with pytest.raises(gs.GenesisException, match=r"`set_position` on variant 0"):
        cloth.variants[0].set_position(torch.zeros(3))
    with pytest.raises(gs.GenesisException, match=r"recorded under that entity"):
        cloth.variants[0].get_contacts()
    # A property of one mesh is read per variant
    with pytest.raises(gs.GenesisException, match=r"`init_positions` differs between the 3 variants.*variants\[k\]"):
        cloth.init_positions
    with pytest.raises(gs.GenesisException, match=r"`mass` differs"):
        cloth.mass


def test_the_spec_takes_variants_for_a_deformable_body_only():
    box = gs.morphs.Box(size=(0.1, 0.1, 0.1))
    sphere = gs.morphs.Sphere(radius=0.05)
    spec = MochiSoftSpec(morph=(box, sphere), material=mochi_materials.Elastic())
    assert spec.morphs == (box, sphere)
    scene = create_mochi_scene(show_viewer=False)
    with pytest.raises(gs.GenesisException, match=r"no heterogeneous rigid entity: pass MochiRigidSpec one morph"):
        scene.add_entity([box, box], material=mochi_materials.Rigid())
    # Every variant is checked as a single morph is
    rod = Rod(points=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    with pytest.raises(gs.GenesisException, match=r"Morph Rod is not supported for deformable Mochi bodies"):
        scene.add_entity([box, rod], material=mochi_materials.Shell())
    with pytest.raises(gs.GenesisException, match=r"requires a `Rod` morph"):
        scene.add_entity([rod, box], material=mochi_materials.Rod())
    # A sequence of one morph is a body of its own
    body = scene.add_entity([box], material=mochi_materials.Elastic())
    assert isinstance(body, MochiSoftEntity)


def test_the_unsupported_calls_say_what_to_do_instead(tmp_path):
    scene = create_mochi_scene(show_viewer=False)
    link = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 1.0)), material=mochi_materials.Rigid())
    cloth = scene.add_entity([m() for m in _cloths(tmp_path, n_cells=(2, 3))], material=_cloth_material())
    with pytest.raises(gs.GenesisException, match=r"`attach_to_link` is not supported on heterogeneous entity"):
        cloth.attach_to_link(link.base_link, [0])
    with pytest.raises(gs.GenesisException, match=r"`attach_to_link` is not supported on heterogeneous entity"):
        cloth.variants[0].attach_to_link(link.base_link, [0])
    for method in ("set_visual_mesh", "set_contact_skin"):
        with pytest.raises(gs.GenesisException, match=rf"call it per variant .*`entity\.variants\[k\]\.{method}"):
            getattr(cloth, method)("mesh.obj")
    with pytest.raises(gs.GenesisException, match=r"assigned when the scene is built"):
        cloth.envs_variant_idx


@pytest.mark.parametrize(("n_envs", "expected"), [(0, [0]), (2, [0, 1]), (5, [0, 0, 1, 1, 2])])
def test_the_variants_go_to_the_environments_as_genesis_hands_them_out(tmp_path, n_envs, expected):
    """Contiguous, balanced blocks; with fewer environments than variants, the last variants are in none."""
    scene, (cloth,) = _scene(
        n_envs, lambda scene: [scene.add_entity([m() for m in _cloths(tmp_path)], material=_cloth_material())]
    )
    assert np.atleast_1d(cloth.envs_variant_idx.numpy()).tolist() == expected
    is_active = scene.mochi_solver.soft_entities_is_active
    for k, body in enumerate(cloth.variants):
        assert is_active[body.idx_in_solver].tolist() == [variant == k for variant in expected]
    pos = cloth.get_vertices_position()
    assert pos.shape == ((36, 3) if n_envs == 0 else (n_envs, 36, 3))
    scene.step()


@pytest.mark.precision("64")
def test_a_restored_state_resumes_every_environment_s_own_variant(cloth_scene):
    scene, cloth = cloth_scene
    state = scene.get_state()
    for _ in range(5):
        scene.step()
    expected = cloth.get_vertices_position().clone()

    cloth.set_vertices_fixed([0], is_fixed=True)
    cloth.set_position(torch.tensor([0.0, 0.0, 1.0]))
    scene.reset(state)
    for _ in range(5):
        scene.step()
    torch.testing.assert_close(cloth.get_vertices_position(), expected, rtol=0.0, atol=0.0)
    # The absent variants came back fixed at rest
    is_fixed = qd_to_numpy(scene.mochi_solver.soft_state.verts_is_fixed)
    for k, body in enumerate(cloth.variants):
        holds = (cloth.envs_variant_idx == k).numpy()
        assert is_fixed[body.v_start : body.v_end][:, ~holds].all()
        assert (body.get_vertices_position()[~torch.as_tensor(holds)] == torch.as_tensor(body.init_positions)).all()


@pytest.mark.precision("64")
def test_a_variant_takes_a_visual_mesh_of_its_own(tmp_path):
    """A solid variant carries a render mesh embedded in its tetrahedra, which collapses where the variant is absent."""
    visual = tmp_path / "visual.obj"
    trimesh.creation.box(extents=(0.09, 0.09, 0.09)).export(visual)
    solids = [
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)),
        gs.morphs.Box(size=(0.08, 0.08, 0.08), pos=(0.0, 0.0, 0.5), **REFINED),
    ]

    def add(scene):
        solid = scene.add_entity(solids, material=mochi_materials.Elastic(E=5e4, **CONTACT))
        solid.variants[0].set_visual_mesh(str(visual))
        return [solid]

    scene, (solid,) = _scene(2, add)
    scene.step()
    body = solid.variants[0]
    (vgeom,) = body.vgeoms
    assert vgeom.elems_idx is not None
    vverts = qd_to_numpy(scene.mochi_solver.get_soft_state_render()[0], transpose=True)
    drawn = vverts[:, vgeom.vvert_start : vgeom.vvert_end]
    envs_offset = np.asarray(scene.envs_offset)
    # Drawn around the box in the environment holding it, one point in the other
    assert np.ptp(drawn[0], axis=0) == pytest.approx(0.09, abs=1e-3)
    assert np.allclose(drawn[1], body.init_positions[0] + envs_offset[1], atol=1e-6)
