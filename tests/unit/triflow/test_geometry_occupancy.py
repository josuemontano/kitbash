"""The numba triangle/voxel voxelizer against an independent pure-python reference."""

import itertools

import numpy as np
import pytest
import trimesh

from kitbash.retopology.triflow.geometry import triangle_voxel_overlap


def _clip(poly, axis, bound, keep_greater):
    """Sutherland-Hodgman clip of a polygon against one half-space (closed)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        sp = (p[axis] - bound) if keep_greater else (bound - p[axis])
        sq = (q[axis] - bound) if keep_greater else (bound - q[axis])
        if sp >= 0:
            out.append(p)
        if (sp >= 0) != (sq >= 0):
            t = sp / (sp - sq)
            out.append(p + t * (q - p))
    return out


def _overlaps_by_clipping(tri, voxel):
    """True iff the triangle intersects the closed unit box at ``voxel`` (clip the triangle against the six planes)."""
    poly = [np.asarray(v, dtype=float) for v in tri]
    for axis in range(3):
        for bound, keep_greater in ((voxel[axis], True), (voxel[axis] + 1.0, False)):
            poly = _clip(poly, axis, bound, keep_greater)
            if not poly:
                return False
    return True


def brute_force_voxels(vertices, faces, resolution):
    found = set()
    for face in faces:
        tri = vertices[face]
        for voxel in itertools.product(range(resolution), repeat=3):
            if _overlaps_by_clipping(tri, voxel):
                found.add(voxel)
    return found


def _as_set(indices):
    return {tuple(int(c) for c in row) for row in indices}


@pytest.mark.parametrize("seed", range(4))
def test_geometry_occupancy_matches_brute_force_on_random_triangles(seed):
    rng = np.random.default_rng(seed)
    resolution = 7
    # Mix of tiny, medium and grid-spanning triangles, some partly outside [0, R).
    n = 12
    centers = rng.uniform(-1.0, resolution + 1.0, size=(n, 3))
    scales = rng.choice([0.2, 1.0, 3.0, 8.0], size=(n, 1))
    vertices = (centers[:, None, :] + rng.normal(size=(n, 3, 3)) * scales[:, None, :]).reshape(-1, 3)
    faces = np.arange(n * 3).reshape(n, 3)

    got = _as_set(triangle_voxel_overlap(vertices, faces, resolution))
    expected = brute_force_voxels(vertices, faces, resolution)
    assert 0 < len(expected) < resolution**3
    assert got == expected


def test_geometry_occupancy_axis_aligned_box_is_hollow_shell():
    box = trimesh.creation.box(bounds=[[2.2, 2.2, 2.2], [5.7, 5.7, 5.7]])
    got = _as_set(triangle_voxel_overlap(box.vertices, box.faces, 8))
    expected = {v for v in itertools.product(range(2, 6), repeat=3) if any(c in (2, 5) for c in v)}
    assert got == expected  # 4^3 - 2^3 = 56 surface voxels, the interior 2x2x2 block is empty


def test_geometry_occupancy_sphere_is_a_thin_conservative_shell():
    resolution, radius = 48, 17.3
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    sphere.apply_translation([resolution / 2] * 3)
    indices = triangle_voxel_overlap(sphere.vertices, sphere.faces, resolution)
    assert indices.dtype == np.int32 and indices.shape[1] == 3
    centers = indices + 0.5
    dist = np.linalg.norm(centers - resolution / 2, axis=1)
    # Every occupied voxel touches the (faceted) surface: its centre is within a voxel half-diagonal of the true sphere.
    assert np.all(np.abs(dist - radius) <= np.sqrt(3) / 2 + 0.2)
    # Every voxel whose centre is within half a voxel of the sphere (minus facet sag) must be occupied.
    grid = np.stack(np.meshgrid(*[np.arange(resolution)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_dist = np.linalg.norm(grid + 0.5 - resolution / 2, axis=1)
    must_have = grid[np.abs(grid_dist - radius) < 0.5 - 0.2]
    assert _as_set(must_have) <= _as_set(indices)
    # Hollow: nothing deep inside.
    assert dist.min() > radius - 2.0


def test_geometry_occupancy_output_sorted_unique_and_in_bounds():
    rng = np.random.default_rng(7)
    vertices = rng.uniform(-5, 25, size=(30, 3))
    faces = rng.integers(0, 30, size=(40, 3))
    indices = triangle_voxel_overlap(vertices, faces, 20)
    assert indices.min() >= 0 and indices.max() < 20
    keys = indices[:, 0].astype(np.int64) * 400 + indices[:, 1] * 20 + indices[:, 2]
    assert np.all(np.diff(keys) > 0)


def test_geometry_occupancy_empty_and_outside():
    assert triangle_voxel_overlap(np.zeros((0, 3)), np.zeros((0, 3), dtype=int), 8).shape == (0, 3)
    far = np.array([[20.0, 20, 20], [21, 20, 20], [20, 21, 20]])
    assert triangle_voxel_overlap(far, np.array([[0, 1, 2]]), 8).shape == (0, 3)
    nan = np.array([[np.nan, 1, 1], [2, 1, 1], [1, 2, 1]])
    assert triangle_voxel_overlap(nan, np.array([[0, 1, 2]]), 8).shape == (0, 3)
