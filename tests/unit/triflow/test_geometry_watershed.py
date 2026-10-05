"""Paper Eq. (5)–(8) target transfer, connected watershed regions, and total coverage."""

import numpy as np
import pytest
import trimesh

from kitbash.retopology.triflow.geometry.nvv import get_target_point_priority_watershed


def _two_components():
    coords = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]])
    mesh = trimesh.Trimesh(coords + 0.5, [[0, 1, 2], [3, 4, 5]], process=False)
    return mesh, coords


def _assert_connected_labels(mesh, roots):
    adjacency = [set() for _ in mesh.vertices]
    for a, b in mesh.edges_unique:
        adjacency[a].add(b)
        adjacency[b].add(a)
    for root in np.unique(roots):
        members = set(np.flatnonzero(roots == root))
        assert root in members
        visited = {root}
        pending = [root]
        while pending:
            for neighbor in adjacency[pending.pop()] & members - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        assert visited == members


@pytest.mark.parametrize("all_unseeded", [False, True])
def test_watershed_seeds_each_unseeded_component_by_minimum_displacement(all_unseeded):
    mesh, coords = _two_components()
    nvv = np.zeros((6, 3))
    nvv[:, 2] = [3, 2, 2, 4, 1, 1] if all_unseeded else [0, 0.8, 0.9, 4, 1, 1]
    expected_roots = np.array([1, 1, 1, 4, 4, 4] if all_unseeded else [0, 0, 0, 4, 4, 4])

    targets, roots = get_target_point_priority_watershed(mesh, coords, nvv, root_threshold=0.5)

    # Both fallback minima tie: the lower vertex index wins within each component.
    np.testing.assert_array_equal(roots, expected_roots)
    np.testing.assert_allclose(targets, (coords + 0.5 + nvv)[expected_roots])
    _assert_connected_labels(mesh, roots)
    reordered = trimesh.Trimesh(mesh.vertices, mesh.faces[::-1], process=False)
    targets_again, roots_again = get_target_point_priority_watershed(reordered, coords, nvv)
    np.testing.assert_array_equal(roots_again, roots)
    np.testing.assert_array_equal(targets_again, targets)


@pytest.mark.parametrize(
    ("mesh_offset", "voxel_vector", "expected_roots"),
    [
        ([0.75, 0, 0], [0.75, 0, 0], [0, 1, 2]),  # Large voxel vectors, zero mesh displacement.
        ([0.75, 0, 0], [0, 0, 0], [0, 0, 0]),  # Zero voxel vectors, no threshold-qualified mesh root.
        ([0.5, 0.5, 0.5], [0, 0, 0], [0, 1, 2]),  # Inclusive infinity-norm threshold, not Euclidean.
    ],
)
def test_watershed_roots_use_transferred_mesh_displacement(mesh_offset, voxel_vector, expected_roots):
    coords = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0]])
    mesh = trimesh.Trimesh(coords + 0.5 + mesh_offset, [[0, 1, 2]], process=False)
    nvv = np.tile(voxel_vector, (3, 1))

    targets, roots = get_target_point_priority_watershed(mesh, coords, nvv, root_threshold=0.5)

    np.testing.assert_array_equal(roots, expected_roots)
    np.testing.assert_allclose(targets, (coords + 0.5 + nvv)[expected_roots])


def test_watershed_normalized_vectors_are_scaled_before_transfer():
    mesh, coords = _two_components()
    nvv = np.zeros((6, 3))
    nvv[:, 2] = [0, 2, 2, 4, 1, 1]

    targets, roots = get_target_point_priority_watershed(mesh, coords, nvv / 64, resolution=64)

    expected_roots = np.array([0, 0, 0, 4, 4, 4])
    np.testing.assert_array_equal(roots, expected_roots)
    np.testing.assert_allclose(targets, (coords + 0.5 + nvv)[expected_roots])


def test_watershed_expands_connected_regions_by_root_target_distance():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]])
    mesh = trimesh.Trimesh(coords + 0.5, [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], process=False)
    ideal_targets = mesh.vertices[[0, 0, 5, 5, 0, 5]].copy()
    ideal_targets[[1, 3], 0] += 0.1
    ideal_targets[4, 0] += 0.2
    nvv = ideal_targets - mesh.vertices

    targets, roots = get_target_point_priority_watershed(mesh, coords, nvv, root_threshold=0.05)

    expected_roots = np.array([0, 0, 5, 5, 0, 5])
    np.testing.assert_array_equal(roots, expected_roots)
    np.testing.assert_allclose(targets, ideal_targets[expected_roots])
    _assert_connected_labels(mesh, roots)


def test_watershed_isolated_vertex_keeps_its_transferred_target():
    coords = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [10, 0, 0]])
    mesh = trimesh.Trimesh(coords + 0.5, [[0, 1, 2]], process=False)
    nvv = np.zeros((4, 3))
    nvv[3] = [2, 3, 4]

    targets, roots = get_target_point_priority_watershed(mesh, coords, nvv)

    np.testing.assert_array_equal(roots, [0, 1, 2, 3])
    np.testing.assert_allclose(targets, coords + 0.5 + nvv)


def test_watershed_rejects_empty_mesh():
    with pytest.raises(ValueError, match="nonempty mesh"):
        get_target_point_priority_watershed(trimesh.Trimesh(), np.zeros((1, 3)), np.zeros((1, 3)))


def test_watershed_rejects_empty_voxel_support():
    mesh, _ = _two_components()
    with pytest.raises(ValueError, match="nonempty voxel"):
        get_target_point_priority_watershed(mesh, np.empty((0, 3)), np.empty((0, 3)))


def test_watershed_rejects_missing_voxel_vectors():
    mesh, coords = _two_components()
    with pytest.raises(ValueError, match="one NVV vector per voxel"):
        get_target_point_priority_watershed(mesh, coords, np.empty((0, 3)))
