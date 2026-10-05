"""Conservative triangle/voxel overlap rasterization (numba).

Written for kitbash to replace ``open3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds``, which upstream TriFlow
uses in ``get_precise_occupancy`` (Open3D is not available on this platform). Semantics: a unit voxel ``(i, j, k)`` occupies
the closed box ``[i, i + 1] x [j, j + 1] x [k, k + 1]`` (voxel size 1, grid origin 0, grid extent ``[0, R)^3``); it is
occupied iff that box intersects at least one triangle (separating-axis test of Akenine-Moller, "Fast 3D Triangle-Box
Overlap Testing", 13 axes). Touching counts as overlapping. Voxels outside ``[0, R)`` are dropped.

The output is sorted lexicographically by ``(x, y, z)`` (Open3D returned hash-map order); nothing downstream depends on
the order, only on it being consistent between ``occ_fine`` and ``nvv_fine``.
"""

import numpy as np
from numba import njit, prange

_HALF = 0.5


@njit(cache=True, inline="always")
def _axis_separates(a0, a1, a2, rad):
    """True if the projections of the (centred) triangle onto an axis are all on one side of the box interval."""
    lo = min(a0, min(a1, a2))
    hi = max(a0, max(a1, a2))
    return lo > rad or hi < -rad


@njit(cache=True)
def _tri_box_overlap(v0, v1, v2, cx, cy, cz):
    """SAT test of the triangle ``v0, v1, v2`` against the axis-aligned box of half size 0.5 centred at ``(cx, cy, cz)``."""
    # Triangle in box-centred coordinates.
    a = np.empty(3)
    b = np.empty(3)
    c = np.empty(3)
    for i in range(3):
        o = (cx, cy, cz)[i]
        a[i] = v0[i] - o
        b[i] = v1[i] - o
        c[i] = v2[i] - o

    # 1) The three box face normals (triangle AABB vs box).
    for i in range(3):
        if _axis_separates(a[i], b[i], c[i], _HALF):
            return False

    e0 = b - a
    e1 = c - b
    e2 = a - c

    # 2) Triangle plane vs box.
    n0 = e0[1] * e1[2] - e0[2] * e1[1]
    n1 = e0[2] * e1[0] - e0[0] * e1[2]
    n2 = e0[0] * e1[1] - e0[1] * e1[0]
    d = n0 * a[0] + n1 * a[1] + n2 * a[2]
    r = _HALF * (abs(n0) + abs(n1) + abs(n2))
    if abs(d) > r:
        return False

    # 3) Nine edge x box-axis cross products.
    for k in range(3):
        if k == 0:
            ex, ey, ez = e0[0], e0[1], e0[2]
        elif k == 1:
            ex, ey, ez = e1[0], e1[1], e1[2]
        else:
            ex, ey, ez = e2[0], e2[1], e2[2]
        # axis = e x X : (0, ez, -ey)
        p0 = ez * a[1] - ey * a[2]
        p1 = ez * b[1] - ey * b[2]
        p2 = ez * c[1] - ey * c[2]
        if _axis_separates(p0, p1, p2, _HALF * (abs(ez) + abs(ey))):
            return False
        # axis = e x Y : (-ez, 0, ex)
        p0 = -ez * a[0] + ex * a[2]
        p1 = -ez * b[0] + ex * b[2]
        p2 = -ez * c[0] + ex * c[2]
        if _axis_separates(p0, p1, p2, _HALF * (abs(ez) + abs(ex))):
            return False
        # axis = e x Z : (ey, -ex, 0)
        p0 = ey * a[0] - ex * a[1]
        p1 = ey * b[0] - ex * b[1]
        p2 = ey * c[0] - ex * c[1]
        if _axis_separates(p0, p1, p2, _HALF * (abs(ey) + abs(ex))):
            return False
    return True


@njit(parallel=True, cache=True)
def _rasterize_kernel(verts, faces, resolution, grid):
    """Mark every voxel of ``grid`` (``uint8``, ``R^3``) whose box overlaps a triangle.

    Parallel over triangles; concurrent writes only ever store the value 1, so the race is benign.
    """
    n_faces = faces.shape[0]
    for f in prange(n_faces):
        v0 = verts[faces[f, 0]]
        v1 = verts[faces[f, 1]]
        v2 = verts[faces[f, 2]]
        lo = np.empty(3, dtype=np.int64)
        hi = np.empty(3, dtype=np.int64)
        skip = False
        for i in range(3):
            tmin = min(v0[i], min(v1[i], v2[i]))
            tmax = max(v0[i], max(v1[i], v2[i]))
            if not (np.isfinite(tmin) and np.isfinite(tmax)):
                skip = True
                break
            # Voxel i spans [i, i + 1]: overlap needs tmin <= i + 1 and tmax >= i.
            lo[i] = max(np.int64(np.ceil(tmin)) - 1, 0)
            hi[i] = min(np.int64(np.floor(tmax)), resolution - 1)
            if lo[i] > hi[i]:
                skip = True
        if skip:
            continue
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    if grid[x, y, z] == 0 and _tri_box_overlap(v0, v1, v2, x + 0.5, y + 0.5, z + 0.5):
                        grid[x, y, z] = 1


def triangle_voxel_overlap(vertices, faces, resolution):
    """Voxels of the unit grid ``[0, resolution)^3`` whose box overlaps a triangle of the mesh.

    Args:
        vertices: ``(V, 3)`` float positions in voxel units.
        faces: ``(F, 3)`` int triangle indices.
        resolution: Cubic grid resolution ``R``.

    Returns:
        ``(N, 3)`` int32 voxel indices, sorted lexicographically.
    """
    resolution = int(resolution)
    verts = np.ascontiguousarray(vertices, dtype=np.float64)
    tris = np.ascontiguousarray(faces, dtype=np.int64)
    if tris.size == 0 or verts.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    _rasterize_kernel(verts, tris, resolution, grid)
    return np.argwhere(grid).astype(np.int32)
