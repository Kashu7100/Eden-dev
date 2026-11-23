"""Tests for ClothDojo's fold quality (``custom_terms.shirt``) and the ``mochi_fold_the_shirt`` task that scores with it."""

import cv2
import numpy as np
import pytest
import torch

from eden.tasks.mochi.custom_terms.shirt import FOLD_QUALITY_THRESHOLD, FoldQuality, triangle_pixels


def _sheet(nx: int = 30, ny: int = 20, sx: float = 0.6, sy: float = 0.4, z: float = 0.003):
    """A flat ``sx`` x ``sy`` m grid sheet ``z`` over the table: vertices and triangles."""
    x, y = np.meshgrid(np.linspace(-sx / 2, sx / 2, nx + 1), np.linspace(-sy / 2, sy / 2, ny + 1), indexing="ij")
    verts = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=-1)
    i = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    a, b, c, d = i[:-1, :-1].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel(), i[:-1, 1:].ravel()
    return verts, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def _mirror(verts: np.ndarray, axis: int, gap: float = 0.006) -> np.ndarray:
    """The half beyond 0 along ``axis`` folded sharply back onto the other, ``gap`` above the stack below it."""
    out = verts.copy()
    moving = out[:, axis] > 1e-9
    out[moving, axis] = -out[moving, axis]
    out[moving, 2] = 2 * verts[:, 2].max() - out[moving, 2] + gap
    return out


def test_the_raster_fills_the_pixels_opencv_fills():
    """Each triangle covers exactly what ClothDojo's cv2.fillConvexPoly paints for it, edges and rounding included."""
    rng = np.random.default_rng(0)
    corners = rng.integers(1, 11, size=(5000, 3, 2))
    _, tri, rows, cols = triangle_pixels(torch.tensor(corners, dtype=torch.float64)[None])
    ours = np.zeros((len(corners), 12, 12), dtype=bool)
    ours[tri.numpy(), rows.numpy(), cols.numpy()] = True
    for k, triangle in enumerate(corners):
        opencv = np.zeros((12, 12), dtype=np.uint8)
        cv2.fillConvexPoly(opencv, triangle.astype(np.int32), 1)
        np.testing.assert_array_equal(ours[k], opencv.astype(bool), err_msg=f"triangle {triangle.tolist()}")


# Features and Q from ClothDojo's own fold_quality.py on these sheets (its NumPy and OpenCV reference).
_CLOTHDOJO = {
    "flat": ([0.0, 1.0, 1.0, 0.441314, 1.0], -0.710252),
    "half": ([0.686798, 0.997248, 0.997682, 0.797694, 0.992933], -0.292086),
    "quarter": ([1.370466, 0.983749, 0.90566, 1.194788, 0.952526], 0.496285),
    "wrinkled": ([0.0, 0.90396, 0.567698, 0.430232, 0.942594], -1.31838),
}


def test_the_score_is_clothdojos_on_flat_folded_and_wrinkled_sheets():
    flat, triangles = _sheet()
    half = _mirror(flat, 0)
    wrinkled = flat.copy()
    wrinkled[:, 2] += 0.03 * np.sin(flat[:, 0] * 20) * np.cos(flat[:, 1] * 15) ** 2
    states = {"flat": flat, "half": half, "quarter": _mirror(half, 1), "wrinkled": wrinkled}

    quality = FoldQuality(torch.as_tensor(flat), torch.as_tensor(triangles))
    features = quality.features(torch.as_tensor(np.stack(list(states.values()))))
    scores = quality.score(features)
    for i, name in enumerate(states):
        expected_features, expected_q = _CLOTHDOJO[name]
        np.testing.assert_allclose(features[i].numpy(), expected_features, atol=2e-6, err_msg=name)
        assert float(scores[i]) == pytest.approx(expected_q, abs=2e-6), name


def test_a_batch_scores_each_garment_as_it_would_alone():
    """The batch's raster is padded to its largest garment; the rectangle feature must not see that padding."""
    flat, triangles = _sheet()
    quarter = _mirror(_mirror(flat, 0), 1)
    quality = FoldQuality(torch.as_tensor(flat), torch.as_tensor(triangles))
    alone = quality.features(torch.as_tensor(quarter)[None])
    together = quality.features(torch.as_tensor(np.stack([quarter, flat])))
    torch.testing.assert_close(together[:1], alone)


@pytest.fixture(scope="module")
def env():
    from tests.tasks.mochi.conftest import build_task

    return build_task("mochi_fold_the_shirt")


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_reset_lays_one_of_the_benchmark_fold_starts_down_at_rest(env):
    from eden.tasks.mochi.custom_terms.events import load_vertex_states
    from eden.tasks.mochi.fold_the_shirt.config import SHIRT_SCALE

    env.reset()
    shirt = env.entities["shirt"]
    term = env.event_manager.get_term("fold_start")
    starts = term._states  # (4, n, 3), already scaled and on the bench
    verts = shirt.get_vertex_positions()
    distance = (verts[:, None] - starts[None]).norm(dim=-1).amax(dim=-1)  # (num_envs, 4)
    assert bool((distance.amin(dim=-1) < 1e-5).all()), f"no start matches: {distance}"
    assert float(shirt.get_vertex_velocities().abs().max()) == 0.0
    # Each start is the benchmark's, scaled in the plane about its own centre with its heights kept.
    saved = torch.as_tensor(load_vertex_states(term, term.states_file, term.states_key, shirt.n_vertices)).double()
    laid = starts.cpu().double()

    def extent(points: torch.Tensor) -> torch.Tensor:
        return points[..., :2].amax(dim=-2) - points[..., :2].amin(dim=-2)

    torch.testing.assert_close(extent(laid), SHIRT_SCALE * extent(saved), atol=1e-5, rtol=0)
    torch.testing.assert_close(laid[..., 2] - term.offset[2], saved[..., 2], atol=1e-5, rtol=0)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_flat_start_scores_far_under_the_cut(env):
    """A fold start as laid is a flat shirt: Q -1.620 to -1.621, far under the cut.

    ClothDojo scores its full-size starts -1.387 to -1.390. Shrunk in the plane with the layers' heights kept, the same
    cloth is relatively thicker and its wrinkles steeper, which lowers the flatness and planar-top features. Scored on
    the reset state itself: a step of Mochi settling moves it by a few hundredths.
    """
    env.reset()
    term = env.metric_manager.get_term("fold_quality")
    q = term.compute()
    torch.testing.assert_close(q, torch.full_like(q, -1.6205), atol=0.002, rtol=0)
    assert not bool(term.is_success(q).any())
    assert term.success_threshold == FOLD_QUALITY_THRESHOLD


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_score_is_held_between_evaluations(env):
    """Scored on an episode's first step and every ``interval`` steps after, held in between."""
    env.reset()
    term = env.metric_manager.get_term("fold_quality")
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.step(zero)
    shirt = env.entities["shirt"]
    held = env.metric_manager.get_value("fold_quality").clone()
    # Lift every env's shirt 20 cm, off its support: the score must not move until the next evaluation.
    verts = shirt.get_vertex_positions().clone()
    verts[..., 2] += 0.2
    shirt.set_vertex_positions(verts)
    while int(env.episode_length_buf[0]) % term.interval != 0:
        torch.testing.assert_close(term.compute(), held)
        env.step(zero)
    assert not torch.allclose(term.compute(), held)
