"""Convert one ClothDojo garment, its four fold starts and its flat reference into the files ``mochi_fold_the_shirt`` reads.

ClothDojo (clothdojo.github.io, arXiv 2611.01190) ships each garment as a triangulated USD mesh in centimetres
(GarmentCodeData, CC BY 4.0) and its shirt-fold benchmark condition as plain particle arrays in the order of that
mesh's points: four fold starts (``benchmark/conditions/nocs-fold/states/<id>/states.npz``, the settled shirt at
quarter turns) and a flat reference (``.../templates/<id>.npz``) that the fold-quality score reads its material
weights, length scale and table height from. This script fetches the three files from the ``clothdojo/assets``
Hub dataset at a pinned revision and writes:

* ``shirt.obj``: the mesh in metres, its points in the USD's order (so the arrays index it directly), with no
  texture coordinates (the USD's face-varying UVs would split the vertices at panel seams on load);
* ``fold_starts.npz``: ``starts`` ``(4, n, 3)`` and ``flat_reference`` ``(n, 3)``, float32, in ClothDojo's
  world frame (table top at z = 0.80, the Piper bases at x = 0.25), which the task maps onto its bench.

Upload ``<out_dir>`` as ``objects/clothdojo/<garment_id>`` of the ``Kashu7100/eden_assets`` Hub dataset, or copy
it to that path under ``eden.utils.misc.get_assets_dir()`` to use it without uploading. The fold starts and the
flat reference are ClothDojo's generated data; check ``LICENSE_SCOPE.md`` of ``clothdojo/assets`` before
redistributing them.

Needs USD to read the mesh::

    uv run --with usd-core python examples/recipes/mochi/convert_clothdojo_shirt.py -o <out_dir>
"""

import argparse
import os

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "clothdojo/assets"
# The revision ClothDojo's own download script pins (scripts/download.sh).
REVISION = "a0628f42dde13ebb0505fd9dc255d4bda50edb94"
GARMENT_ID = "rand_ZZCWITH5HY"
# ClothDojo's garment scale (ShirtFlattenSpec.garment_scale): the USD is in centimetres.
GARMENT_SCALE = 1.0e-2
MESH_PRIM = "/root/Mesh"


def read_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Points ``(n, 3)`` and triangles ``(m, 3)`` of ClothDojo's converted garment USD, in its own units."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath(MESH_PRIM)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"{path} has no mesh at {MESH_PRIM}; is it a ClothDojo converted garment?")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError(
            f"{path}: the garment must be triangulated, got faces of {sorted(set(counts.tolist()))} sides."
        )
    return points, np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to write the three files into.")
    parser.add_argument("--garment-id", type=str, default=GARMENT_ID, help="ClothDojo garment id.")
    args = parser.parse_args()

    def fetch(path: str) -> str:
        return hf_hub_download(REPO_ID, path, repo_type="dataset", revision=REVISION)

    gid = args.garment_id
    usd = fetch(f"benchmark/assets/garments/converted/{gid}/{gid}.usdc")
    states = np.load(fetch(f"benchmark/conditions/nocs-fold/states/{gid}/states.npz"), allow_pickle=False)
    template = np.load(fetch(f"benchmark/conditions/nocs-fold/templates/{gid}.npz"), allow_pickle=False)

    points, triangles = read_mesh(usd)
    points = points * GARMENT_SCALE
    starts = np.asarray(states["particle_q"], dtype=np.float32)
    flat = np.asarray(template["particle_q"], dtype=np.float32)
    if starts.shape[1:] != points.shape or flat.shape != points.shape:
        raise ValueError(
            f"{gid}: the mesh has {len(points)} points, but the fold starts are {starts.shape} and the flat "
            f"reference {flat.shape}; they index the mesh's points, so the counts must match."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    lines = [f"# ClothDojo garment {gid} (GarmentCodeData, CC BY 4.0), {REPO_ID}@{REVISION}, in metres"]
    lines += [f"v {x:.7f} {y:.7f} {z:.7f}" for x, y, z in points]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles]
    with open(os.path.join(args.output_dir, "shirt.obj"), "w") as f:
        f.write("\n".join(lines) + "\n")
    np.savez(os.path.join(args.output_dir, "fold_starts.npz"), starts=starts, flat_reference=flat)
    print(f"{gid}: {len(points)} vertices, {len(triangles)} triangles, {len(starts)} fold starts -> {args.output_dir}")


if __name__ == "__main__":
    main()
