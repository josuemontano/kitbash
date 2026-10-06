"""Coordinate frames of the TriFlow geometry pipeline (added for kitbash, not part of upstream).

Two frames exist:

* the *input frame*: the coordinates of the mesh handed to ``process_one_mesh`` (a .glb / .obj / .ply as loaded by trimesh,
  scenes flattened with node transforms applied, no Y-up/Z-up conversion, units unknown);
* the *grid frame* ("voxel units"): what ``process_one_mesh`` returns and what the neural nets, ``robust_remesh`` and
  ``topology_flow2mesh_QEM`` work in. The input is translated so its bounding-box centre is at ``(R/2, R/2, R/2)`` and uniformly
  scaled so its longest side spans ``R - 2 * pad_voxels`` voxels (``R = res_fine``, ``pad_voxels = pad * res_fine / res_coarse``),
  i.e. voxel ``(i, j, k)`` is the cube ``[i, i + 1] x [j, j + 1] x [k, k + 1]``. Axes are never permuted or flipped.

    grid = (input - center) * scale_factor + R / 2        input = (grid - R / 2) / scale_factor + center

``topology_flow2mesh_QEM`` outputs vertices in the grid frame (they are positions of / interpolations between the dense input
mesh's vertices and the voxel-space NVV targets), so ``to_input_frame`` is what turns its result into the original coordinates.

The transform parameters live in the ``metadata`` dict returned by ``process_one_mesh``: ``scale_factor``, ``center`` and
``grid_resolution``.
"""

import numpy as np
import trimesh

REQUIRED_KEYS = ("scale_factor", "center", "grid_resolution")


def _params(metadata):
    missing = [k for k in REQUIRED_KEYS if k not in metadata]
    if missing:
        raise KeyError(f"metadata lacks {missing}; pass the metadata dict returned by process_one_mesh")
    scale = float(metadata["scale_factor"])
    if not scale > 0.0:
        raise ValueError(f"invalid scale_factor {scale!r}")
    center = np.asarray(metadata["center"], dtype=np.float64).reshape(3)
    half = float(metadata["grid_resolution"]) / 2.0
    return scale, center, half


def points_to_grid_frame(points, metadata):
    """Map ``(N, 3)`` points from the input frame to the grid frame (float64)."""
    scale, center, half = _params(metadata)
    return (np.asarray(points, dtype=np.float64) - center) * scale + half


def points_to_input_frame(points, metadata):
    """Map ``(N, 3)`` points from the grid frame back to the input frame (float64)."""
    scale, center, half = _params(metadata)
    return (np.asarray(points, dtype=np.float64) - half) / scale + center


def to_grid_frame(mesh: trimesh.Trimesh, metadata) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` (input frame) in the grid frame. Faces and winding are untouched."""
    return trimesh.Trimesh(vertices=points_to_grid_frame(mesh.vertices, metadata), faces=np.asarray(mesh.faces), process=False)


def to_input_frame(mesh_out: trimesh.Trimesh, metadata) -> trimesh.Trimesh:
    """Map a mesh in the grid frame (e.g. the output of ``topology_flow2mesh_QEM``) into the original input coordinates.

    Args:
        mesh_out: Mesh in voxel units.
        metadata: The metadata dict from ``process_one_mesh`` (needs ``scale_factor``, ``center``, ``grid_resolution``).

    Returns:
        A new float64 ``trimesh.Trimesh`` with the same faces (same winding; the transform is a positive uniform scale plus a
        translation, so orientation is preserved).
    """
    return trimesh.Trimesh(
        vertices=points_to_input_frame(mesh_out.vertices, metadata), faces=np.asarray(mesh_out.faces), process=False
    )
