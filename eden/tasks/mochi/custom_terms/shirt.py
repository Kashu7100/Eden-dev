"""ClothDojo's fold-quality score of a garment, batched over envs, as a metric of the shirt-folding task.

ClothDojo (clothdojo.github.io, arXiv 2611.01190; ``src/clothdojo/evaluation/fold_quality.py`` and
``folding.py``, Apache-2.0) scores a folded garment from its vertex positions alone with five features and a
fixed linear model:

1. ``log_area_compression_material_frame``: minus the log of the garment's top-down footprint over its flat
   footprint (pixels of a raster at ``scale / 128``, ``scale`` the square root of the flat convex-hull area).
2. ``all_layer_flatness``: the area-weighted mean of every triangle's squared normal z.
3. ``planar_rectangle_material_frame``: the weight of the largest planar top surface over the union of the
   footprint with that surface's minimum-area rectangle; a neat folded stack is one rectangle.
4. ``xyz_overlap_16``: how many other parts of the garment, far apart in the flat garment, now lie within
   0.16 ``scale`` of each of 512 area-sampled triangles.
5. ``support_10``: how close those triangles lie to the table, ``exp(-(h / 0.1)^2)`` in ``scale`` units.

Each feature is squashed by ``3 tanh(z / 3)`` after standardising, and the score is a fixed linear margin; a state
qualifies at ``FOLD_QUALITY_THRESHOLD`` (ClothDojo's success also holds it for 3 s). A flat reference, the
garment lying settled and unfolded in the same frame, fixes the material weights, the length scale and the table
height; it is not a target shape.

What differs from ClothDojo's reference: the top-down raster runs batched on the GPU. A triangle covers, on each
pixel row its integer-rounded corners span, the columns between its edges' crossings of that row, rounded; the
highest triangle centroid owns a pixel, as in ClothDojo's painter's pass with OpenCV's ``fillConvexPoly``. The
planar-top fit, its connected components and the minimum-area rectangles then run per env on the CPU with the
same OpenCV and NumPy calls as ClothDojo's (copied below with their constants).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.utils.string import term_subject

# ClothDojo's packaged model (``fold_quality_weights.json``), in the feature order of the module docstring.
FEATURE_MEAN = (1.3098480593771762, 0.6267125242680155, 0.31545471814448317, 1.8109926430179561, 0.6968638995290651)
FEATURE_SCALE = (
    0.36092560270131213,
    0.10824116887266653,
    0.11207606412526754,
    0.30657315179917094,
    0.14397583252534885,
)
COEFFICIENTS = (0.6937224660578006, -0.31464788613968286, 1.871787485705648, 1.401243689142226, -0.5248700934616752)
INTERCEPT = -0.21027702661829378
SCORE_CENTER = -0.16786147835605053
SCORE_SCALE = 2.644612442908689
# ClothDojo's success cut on its Newton (Style3D) cloth: one state qualifies at or above this.
FOLD_QUALITY_THRESHOLD = 0.14228658677130532

# Raster resolution: pixels per flat length scale (ClothDojo's fold_quality uses 128).
PIXELS_PER_SCALE = 128
# The planar top surface: heights within this fraction of the scale of one plane, normals within acos(0.8) of it.
FLATNESS_TOLERANCE_RATIO = 0.04
NORMAL_Z_MIN = 0.8
# Area-sampled triangles for the overlap and support features, and their kernel widths (in scale units).
N_SAMPLES = 512
OVERLAP_RADIUS = 0.16
SUPPORT_HEIGHT = 0.1
# Rest-distance kernel of the nonlocality weight: parts closer than ~0.1 scale in the flat garment do not overlap.
NONLOCAL_DISTANCE2 = 0.01
# A triangle covering more pixel rows than this is cut off at it (the raster is chunked by triangle extent).
MAX_TRIANGLE_PIXELS = 24


def _normals_and_areas(points: torch.Tensor, triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit normals ``(B, T, 3)`` and areas ``(B, T)`` of the triangles of ``points`` ``(B, N, 3)``."""
    v = points[:, triangles]  # (B, T, 3, 3)
    cross = torch.linalg.cross(v[:, :, 1] - v[:, :, 0], v[:, :, 2] - v[:, :, 0])
    length = cross.norm(dim=-1)
    return cross / length.clamp_min(1e-12)[..., None] * (length > 1e-12)[..., None], 0.5 * length


def material_frame(points: torch.Tensor, triangles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Turn each garment about z so its principal in-plane axis (by the flat reference's area weights) lies on x.

    ``points`` ``(B, N, 3)``; the sign of the axis follows the first vertex clearly off the centre, as ClothDojo's
    ``_material_frame`` does, so the frame does not flip between two nearby states.
    """
    faces = points[:, triangles, :2]  # (B, T, 3, 2)
    center = torch.einsum("t,btd->bd", weights, faces.mean(dim=2))
    offset = faces - center[:, None, None, :]
    covariance = torch.einsum("t,btvi,btvj->bij", weights / 3.0, offset, offset)
    axis = torch.linalg.eigh(covariance)[1][..., -1]  # (B, 2)
    projection = torch.einsum("bnd,bd->bn", points[..., :2] - center[:, None, :], axis)
    spread = torch.sqrt(covariance.diagonal(dim1=-2, dim2=-1).sum(-1))
    off_centre = projection.abs() > 0.01 * spread[:, None]
    first = torch.where(off_centre.any(dim=-1), off_centre.float().argmax(dim=-1), 0)
    flip = projection.gather(1, first[:, None]).squeeze(1) < 0
    axis = torch.where(flip[:, None], -axis, axis)
    basis = torch.stack([axis, torch.stack([-axis[:, 1], axis[:, 0]], dim=-1)], dim=-1)  # columns: x', y'
    result = points.clone()
    result[..., :2] = torch.einsum("bnd,bde->bne", points[..., :2] - center[:, None, :], basis)
    return result


def triangle_pixels(corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pixels OpenCV's ``fillConvexPoly`` fills for each triangle with integer corners ``(B, T, 3, 2)`` (col, row).

    Returns flat ``(env, triangle, row, col)`` index tensors, one entry per covered pixel (a pixel may repeat).
    OpenCV's ``FillConvexPoly`` (``imgproc/src/drawing.cpp``) at ``shift=0`` draws each edge as an 8-connected
    Bresenham line from its left end, then fills the rows from the top corner to the one above the bottom corner
    between its two edges, each edge's x stepped per row in 16.16 fixed point by a slope rounded toward zero, and
    the row's ends rounded half up. Both are reproduced in integer arithmetic. A triangle spanning more than
    ``MAX_TRIANGLE_PIXELS`` pixels is cut off there.
    """
    one = 1 << 16
    k = torch.arange(MAX_TRIANGLE_PIXELS + 1, device=corners.device)
    c = torch.round(corners).long()

    # The edges' lines, each from its left end: k steps along the longer axis, the other coordinate offset by
    # ceil(k * minor / major - 1/2) (Bresenham's error term starting at major - 2 minor).
    a, b = c, c.roll(-1, dims=-2)  # (B, T, 3, 2): edge i from corner i to corner i + 1
    swap = (b[..., 0] < a[..., 0])[..., None]
    start, end = torch.where(swap, b, a), torch.where(swap, a, b)
    dx, dy = end[..., 0] - start[..., 0], end[..., 1] - start[..., 1]  # dx >= 0
    major, minor = torch.maximum(dx, dy.abs()), torch.minimum(dx, dy.abs())
    offset = -torch.div(
        -(2 * k * minor[..., None] - major[..., None]), (2 * major).clamp_min(1)[..., None], rounding_mode="floor"
    )
    offset = torch.where(major[..., None] > 0, offset, 0)  # (B, T, 3, K)
    steep = (dy.abs() > dx)[..., None]
    sign_y = torch.sign(dy)[..., None]
    line_cols = start[..., 0, None] + torch.where(steep, offset, k)
    line_rows = start[..., 1, None] + sign_y * torch.where(steep, k, offset)
    on_line = k <= major[..., None]

    # The scanline fill, rows top .. bottom - 1, between the long edge (top -> bottom) and the short one (top -> mid
    # above the mid corner's row, mid -> bottom from it).
    order = torch.argsort(c[..., 1], dim=-1, stable=True)
    top, mid, bot = (
        c.gather(-2, order[..., i, None, None].expand(*order.shape[:-1], 1, 2)).squeeze(-2) for i in range(3)
    )
    rows = top[..., 1, None] + k  # (B, T, K)

    def edge_x(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """The fixed-point x of the edge p -> q on each row (B, T, K), as FillConvexPoly steps it from p."""
        h = (q[..., 1] - p[..., 1]).clamp_min(1)[..., None]
        slope = torch.div(2 * one * (q[..., 0] - p[..., 0])[..., None] + h, 2 * h, rounding_mode="trunc")
        return p[..., 0, None] * one + (rows - p[..., 1, None]) * slope

    short = torch.where(rows < mid[..., 1, None], edge_x(top, mid), edge_x(mid, bot))
    long = edge_x(top, bot)
    lo = torch.div(torch.minimum(short, long) + one // 2, one, rounding_mode="floor")
    hi = torch.div(torch.maximum(short, long) + one // 2, one, rounding_mode="floor")
    fill_cols = lo[..., None] + k  # (B, T, K, K)
    in_fill = (rows < bot[..., 1, None])[..., None] & (k <= (hi - lo).clamp(0, MAX_TRIANGLE_PIXELS)[..., None])
    fill_rows = rows[..., None].expand_as(fill_cols)

    all_cols = torch.cat([fill_cols.flatten(-2), line_cols.flatten(-2)], dim=-1)
    all_rows = torch.cat([fill_rows.flatten(-2), line_rows.flatten(-2)], dim=-1)
    covered = torch.cat([in_fill.flatten(-2), on_line.flatten(-2)], dim=-1)
    env, tri, _ = torch.nonzero(covered, as_tuple=True)
    return env, tri, all_rows[covered], all_cols[covered]


def rasterize_top(
    points: torch.Tensor, triangles: torch.Tensor, pixel_size: float, chunk: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The top-down raster of each garment: occupancy, top height, top normal, and each env's own extent.

    Returns ``occupancy`` ``(B, H, W)`` bool, ``height`` ``(B, H, W)`` (the covering triangle with the highest
    centroid; ``-inf`` off the garment), ``normal`` ``(B, H, W, 3)`` (that triangle's normal, z up) and ``half``
    ``(B, 2)``. Each env's raster is centred on its own xy bounding box, as ClothDojo's ``_projection_raster`` is,
    and ClothDojo's raster for env ``i`` is the ``2 half[i] + 1`` columns and rows around the centre; ``H`` and
    ``W`` are the batch's largest, so a smaller garment has a wider empty border. Crop to ``half`` before anything
    that clips to the image (a filled rectangle does).
    """
    B, device = points.shape[0], points.device
    xy = points[..., :2]
    lo, hi = xy.amin(dim=1), xy.amax(dim=1)
    center = 0.5 * (lo + hi)
    own_half = torch.ceil((hi - lo) / (2 * pixel_size)).long() + 2  # (B, 2) as (col, row)
    half = own_half.amax(dim=0)  # common to the batch
    W, H = int(2 * half[0] + 1), int(2 * half[1] + 1)
    pixels = torch.round((xy - center[:, None, :]) / pixel_size).long() + half  # (B, N, 2) as (col, row)

    normals, _ = _normals_and_areas(points, triangles)
    normals = torch.where(normals[..., 2:3] < 0, -normals, normals)
    centroid_z = points[:, triangles, 2].mean(dim=-1)  # (B, T)

    occupancy = torch.zeros(B, H * W, dtype=torch.bool, device=device)
    height = torch.full((B, H * W), -torch.inf, dtype=points.dtype, device=device)
    normal = torch.zeros(B, H * W, 3, dtype=points.dtype, device=device)
    for start in range(0, B, chunk):
        b = slice(start, min(start + chunk, B))
        env, tri, rows, cols = triangle_pixels(pixels[b][:, triangles].to(points.dtype))
        keep = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
        env, tri = env[keep], tri[keep]
        flat_idx = (rows[keep] * W + cols[keep]).long()
        z = centroid_z[b][env, tri]
        occupancy[b][env, flat_idx] = True
        h = height[b]
        h.view(-1).scatter_reduce_(0, env * H * W + flat_idx, z, reduce="amax")
        height[b] = h
        won = z >= h[env, flat_idx]
        normal[b].view(-1, 3)[(env * H * W + flat_idx)[won]] = normals[b][env[won], tri[won]]
    return occupancy.view(B, H, W), height.view(B, H, W), normal.view(B, H, W, 3), own_half


# ---- ClothDojo's planar-top and rectangle measures (folding.py), on one env's raster --------------------------


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _fit_height_plane(x: np.ndarray, y: np.ndarray, height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    design = np.stack((x[mask], y[mask], np.ones(np.count_nonzero(mask))), axis=1)
    return np.linalg.lstsq(design, height[mask].astype(np.float64), rcond=None)[0]


def _dominant_planar_top_mask(height, normal_z, occupancy, *, tolerance_m: float) -> np.ndarray:
    """The largest connected patch in the 2-tolerance-wide height band holding the most horizontal top area."""
    candidate = occupancy & np.isfinite(height) & (normal_z >= NORMAL_Z_MIN)
    values = np.sort(np.asarray(height[candidate], dtype=np.float64))
    if not len(values):
        return np.zeros_like(occupancy, dtype=bool)
    right = np.searchsorted(values, values + 2.0 * tolerance_m, side="right")
    left = int(np.argmax(right - np.arange(len(values))))
    center = 0.5 * (values[left] + values[right[left] - 1])
    return _largest_component(candidate & (np.abs(height - center) <= tolerance_m))


def planar_top_weight(
    occupancy: np.ndarray, height: np.ndarray, normal: np.ndarray, *, tolerance_m: float, pixel_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """ClothDojo's ``_dominant_plane_top_region``: the largest top plane's pixels and their soft weights.

    A height-band seed is grown three times by fitting a plane and taking the largest connected patch of pixels
    near it with normals near its normal; residual and normal agreement then weight each pixel in ``[0, 1]``.
    """
    normal_x, normal_y, normal_z = normal[..., 0], normal[..., 1], normal[..., 2]
    seed = _dominant_planar_top_mask(height, normal_z, occupancy, tolerance_m=tolerance_m)
    if np.count_nonzero(seed) < 3:
        return np.zeros_like(occupancy, dtype=bool), np.zeros(height.shape, dtype=np.float32)
    rows, columns = np.indices(height.shape, dtype=np.float64)
    x = (columns - 0.5 * (height.shape[1] - 1)) * pixel_size_m
    y = (rows - 0.5 * (height.shape[0] - 1)) * pixel_size_m
    plane = _fit_height_plane(x, y, height, seed)
    component = seed
    for _ in range(3):
        residual = np.abs(height - (plane[0] * x + plane[1] * y + plane[2]))
        n = np.array((-plane[0], -plane[1], 1.0)) / np.linalg.norm((-plane[0], -plane[1], 1.0))
        alignment = normal_x * n[0] + normal_y * n[1] + normal_z * n[2]
        candidate = occupancy & np.isfinite(height) & (residual <= tolerance_m) & (alignment >= NORMAL_Z_MIN)
        expanded = _largest_component(candidate)
        if np.count_nonzero(expanded) < 3:
            break
        component = expanded
        plane = _fit_height_plane(x, y, height, component)
    # Off the garment the height is -inf, so the residual is inf and the weight 0.
    residual = np.abs(height - (plane[0] * x + plane[1] * y + plane[2])) / tolerance_m
    residual_weight = np.square(np.clip(1.0 - np.square(residual), 0.0, 1.0))
    n = np.array((-plane[0], -plane[1], 1.0)) / np.linalg.norm((-plane[0], -plane[1], 1.0))
    alignment = normal_x * n[0] + normal_y * n[1] + normal_z * n[2]
    saturation = min(1.0, NORMAL_Z_MIN + 0.15)
    weight = (residual_weight * np.clip((alignment - NORMAL_Z_MIN) / (saturation - NORMAL_Z_MIN), 0.0, 1.0)).astype(
        np.float32
    )
    connected = _largest_component(component & (weight > 0.0))
    weight[~connected] = 0.0
    return connected, weight


def minimum_rectangle(mask: np.ndarray) -> np.ndarray:
    """The four corners ``(4, 2)`` (column, row) of the minimum-area rectangle around a mask's pixels."""
    y, x = np.nonzero(mask)
    return cv2.boxPoints(cv2.minAreaRect(np.stack((x, y), axis=1).astype(np.float32)))


def rectangle_union_pixels(mask: np.ndarray, corners: np.ndarray) -> int:
    """Pixels of ``mask`` united with the rectangle ``corners`` filled."""
    rectangle = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(rectangle, np.rint(corners).astype(np.int32), 1)
    return int(np.count_nonzero(mask | rectangle.astype(bool)))


class FoldQuality:
    """ClothDojo's fold-quality score of one garment mesh against its flat reference, batched over envs.

    Parameters
    ----------
    flat : torch.Tensor
        The flat reference ``(N, 3)``: the garment settled and unfolded, in the frame the scored states are in.
    triangles : torch.Tensor
        The mesh's triangles ``(T, 3)``, indexing the same vertex order.
    """

    def __init__(self, flat: torch.Tensor, triangles: torch.Tensor):
        self.device = flat.device
        flat = flat.double()
        self.triangles = triangles.long().to(self.device)
        _, areas = _normals_and_areas(flat[None], self.triangles)
        areas = areas[0]
        area = float(areas.sum())
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError(f"The flat reference has a total triangle area of {area}; it must be positive.")
        self.weights = areas / area
        flat_xy = flat[:, :2].cpu().numpy()
        hull = cv2.convexHull(flat_xy.astype(np.float32))
        self.scale = float(np.sqrt(cv2.contourArea(hull)))
        self.pixel_size = self.scale / PIXELS_PER_SCALE
        flat_material = material_frame(flat[None], self.triangles, self.weights)
        self.flat_pixels = int(rasterize_top(flat_material, self.triangles, self.pixel_size)[0].sum())
        # The area-sampled triangles, deterministic: the midpoints of 512 equal slices of the area's CDF.
        cdf = torch.cumsum(areas, 0) / area
        quantiles = (torch.arange(N_SAMPLES, device=self.device, dtype=cdf.dtype) + 0.5) / N_SAMPLES
        self.sample = torch.searchsorted(cdf, quantiles).clamp_max(len(areas) - 1)
        rest = flat[self.triangles[self.sample]].mean(dim=1) / self.scale
        self.nonlocality = 1.0 - torch.exp(-((torch.cdist(rest, rest).square() / NONLOCAL_DISTANCE2) ** 2))
        self.sample_area = area / N_SAMPLES / self.scale**2
        self.table_height = float(torch.quantile(flat[:, 2], 0.01))
        self._mean = torch.tensor(FEATURE_MEAN, dtype=torch.float64, device=self.device)
        self._std = torch.tensor(FEATURE_SCALE, dtype=torch.float64, device=self.device)
        self._coef = torch.tensor(COEFFICIENTS, dtype=torch.float64, device=self.device)

    def features(self, points: torch.Tensor) -> torch.Tensor:
        """The five features ``(B, 5)`` of the garments ``points`` ``(B, N, 3)``, in the module docstring's order."""
        q = points.double()
        tri = self.triangles
        top = material_frame(q, tri, self.weights)
        occupancy, height, normal, own_half = rasterize_top(top, tri, self.pixel_size)
        occupied = occupancy.sum(dim=(1, 2)).double()
        compression = -torch.log((occupied / self.flat_pixels).clamp(1e-8, 1.0))

        occupancy_np, height_np, normal_np = occupancy.cpu().numpy(), height.cpu().numpy(), normal.cpu().numpy()
        center = (np.array(occupancy_np.shape[1:]) - 1) // 2  # (row, col)
        own_half = own_half.cpu().numpy()[:, ::-1]  # (B, 2) as (row, col)
        planar = np.zeros(len(q))
        tolerance = FLATNESS_TOLERANCE_RATIO * self.scale
        for i in range(len(q)):
            # Back to ClothDojo's own raster for this env: the rectangle below is clipped to it.
            (r0, c0), (r1, c1) = center - own_half[i], center + own_half[i] + 1
            mask = occupancy_np[i, r0:r1, c0:c1]
            if np.count_nonzero(mask) < 3:
                continue
            top_mask, weight = planar_top_weight(
                mask,
                height_np[i, r0:r1, c0:c1],
                normal_np[i, r0:r1, c0:c1],
                tolerance_m=tolerance,
                pixel_size_m=self.pixel_size,
            )
            rectangle = minimum_rectangle(top_mask if np.count_nonzero(top_mask) >= 3 else mask)
            planar[i] = float(weight.sum(dtype=np.float64)) / rectangle_union_pixels(mask, rectangle)
        planar = torch.as_tensor(planar, device=q.device)

        normals, _ = _normals_and_areas(q, tri)
        flatness = (normals[..., 2].square() * self.weights).sum(dim=-1)
        centers = q[:, tri[self.sample]].mean(dim=2) / self.scale  # (B, S, 3)
        kernel = torch.exp(-torch.cdist(centers, centers).square() / OVERLAP_RADIUS**2) * self.nonlocality
        overlap = kernel.sum(dim=-1) * self.sample_area / (np.pi * OVERLAP_RADIUS**2)
        lift = centers[..., 2] - self.table_height / self.scale
        support = torch.exp(-((lift / SUPPORT_HEIGHT) ** 2)).mean(dim=-1)
        return torch.stack([compression, flatness, planar, torch.log1p(overlap).mean(dim=-1), support], dim=-1)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """The fold-quality margin ``(B,)`` of features ``(B, 5)``: ClothDojo's Q (neither a probability nor a %)."""
        z = 3.0 * torch.tanh((features.double() - self._mean) / (3.0 * self._std))
        return (z @ self._coef + INTERCEPT - SCORE_CENTER) / SCORE_SCALE


@METRIC_TERM_REGISTRY.register()
class ShirtFoldQuality(MetricTerm):
    """ClothDojo's fold-quality score Q of a Mochi shell garment (succeeds at or above ``success_threshold``).

    Q is a margin (0 is no boundary of its own): ClothDojo's flat garments score about -1.4 and its expert folds
    +0.2 to +1. It is scored every ``interval`` control steps (and on the first step of an episode) and held
    between: one evaluation costs ~10 ms per env, most of it the planar-top fit on the CPU.

    Parameters
    ----------
    entity_name : str
        The garment (a Mochi shell).
    reference_file : str
        The ``.npz`` in the Eden asset dataset holding the flat reference (the garment settled and unfolded).
    reference_key : str
        The flat reference's array in it, ``(n_vertices, 3)``.
    offset : tuple[float, float, float]
        Added to the flat reference to bring it into the scene (m), as for the starts it came with.
    planar_scale : float
        The flat reference's x and y scaled by this about its xy centre first, heights kept, as
        :class:`~eden.tasks.mochi.custom_terms.events.ResetSoftBodyFromStates` scales the starts: for a garment
        loaded at that ``scale``.
    interval : int
        Control steps between evaluations.
    success_threshold : float
        Q at or above which the garment counts as folded.
    """

    entity_name: str = "shirt"
    reference_file: str = ""
    reference_key: str = "flat_reference"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    planar_scale: float = 1.0
    interval: int = 5
    success_threshold: float = FOLD_QUALITY_THRESHOLD

    def build(self) -> None:
        from eden.tasks.mochi.custom_terms.events import load_vertex_states, scale_in_plane

        if self.interval < 1:
            raise ValueError(f"{term_subject(self)}: interval must be at least 1 control step, got {self.interval}.")
        shirt = self.resolve_entity(self.entity_name)
        if not getattr(shirt, "is_shell", False):
            raise ValueError(
                f"{term_subject(self)} scores a Mochi shell garment, but '{self.entity_name}' is a "
                f"{type(shirt).__name__}. Give it MochiShellMaterialOptions."
            )
        flat = load_vertex_states(self, self.reference_file, self.reference_key, shirt.n_vertices)
        if flat.ndim != 2:
            raise ValueError(
                f"{term_subject(self)}: {self.reference_file!r}[{self.reference_key!r}] holds {len(flat)} states; "
                f"the flat reference is one, shape ({shirt.n_vertices}, 3)."
            )
        flat = scale_in_plane(self, flat, self.planar_scale)
        flat = torch.as_tensor(flat, dtype=torch.float64, device=self.device) + torch.tensor(
            self.offset, dtype=torch.float64, device=self.device
        )
        self._quality = FoldQuality(flat, torch.as_tensor(np.asarray(shirt.elems), device=self.device))
        self._value = torch.zeros(self.num_envs, device=self.device)
        self._stale = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self) -> torch.Tensor:
        due = self._stale | (self._env.episode_length_buf % self.interval == 0)
        if due.any():
            verts = self.resolve_entity(self.entity_name).get_vertex_positions()
            idx = torch.nonzero(due).squeeze(-1)
            self._value[idx] = self._quality.score(self._quality.features(verts[idx])).to(self._value.dtype)
            self._stale[idx] = False
        return self._value

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # The reset events have placed this episode's garment: score it afresh on the next read.
        self._stale[slice(None) if envs_idx is None else envs_idx] = True
