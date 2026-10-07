"""The conditioning SDF and extraction proxy represent the same zero surface."""

import numpy as np
import pytest
import trimesh
from meshlib import mrmeshnumpy

from kitbash.retopology.triflow.geometry import (
    compute_sparse_sdf,
    get_coords_coarse2fine,
    points_to_grid_frame,
    process_one_mesh,
    sdf_proxy_mesh,
    sparse_sdf2dense,
    to_input_frame,
)


def _samples(field, resolution=64, ratio=8):
    xyz = np.indices((resolution,) * 3).reshape(3, -1).T
    distances = field(xyz + 0.5)
    coarse = np.unique(xyz[np.abs(distances) < 2] // ratio, axis=0)
    fine = get_coords_coarse2fine(coarse, ratio)
    sdf = field(fine + 0.5) / resolution
    return coarse, fine, sdf.reshape(len(coarse), ratio**3)


def test_sdf_proxy_preserves_enclosed_cavity_and_center_frame():
    center = np.array([29.25, 31.75, 33.25])
    radius, thickness = 18.0, 3.0

    def field(points):
        return np.abs(np.linalg.norm(points - center, axis=1) - radius) - thickness

    coarse, fine, sdf = _samples(field)
    dense = sparse_sdf2dense(fine, sdf.ravel(), 64)[0]
    assert dense[29, 31, 33] > 0  # cavity must not be filled as solid
    assert dense[47, 31, 33] < 0
    np.testing.assert_allclose(dense[tuple(fine.T)], sdf.ravel(), atol=1e-7)

    proxy = sdf_proxy_mesh(coarse, sdf, 64, 8)
    assert proxy.is_watertight and proxy.is_winding_consistent
    assert len(proxy.split(only_watertight=False)) == 2
    assert proxy.nondegenerate_faces().all()
    np.testing.assert_allclose(field(proxy.vertices), 0, atol=0.04)
    expected_volume = 4 * np.pi / 3 * ((radius + thickness)**3 - (radius - thickness)**3)
    assert proxy.volume == pytest.approx(expected_volume, rel=0.02)


def test_sdf_proxy_rejects_surface_crossing_grid_boundary():
    coarse, _, sdf = _samples(lambda points: points[:, 0] - 31.25)
    with pytest.raises(ValueError, match="closed surface"):
        sdf_proxy_mesh(coarse, sdf, 64, 8)


def _prepare(mesh, resolution=96):
    return process_one_mesh(
        mesh, res_fine=resolution, res_coarse=resolution // 8, pad=1.5,
        round_verts=False, decimate_length=0.0, vertex_merge_threshold=0.0,
        get_metadata=False, cast=False, compute_source_field=False, verbose=False,
    )


@pytest.mark.parametrize("missing_face", [False, True])
def test_preparation_encloses_box_without_changing_input_frame(missing_face):
    source = trimesh.creation.box().apply_translation([3, -2, 5])
    if missing_face:
        source.update_faces(source.face_normals[:, 2] < 0.9)
    results, prepared, _, metadata = _prepare(source)
    assert prepared.is_watertight and prepared.is_winding_consistent
    if not missing_face:
        # Already-closed input must not be voxel-remeshed.
        restored = to_input_frame(prepared, metadata)
        np.testing.assert_allclose(restored.bounds, source.bounds, atol=1e-6)
        assert len(restored.faces) == len(source.faces)
    proxy = sdf_proxy_mesh(results["occ_coarse"], results["sdf_coarse2fine"], 96, 12)
    output = to_input_frame(proxy, metadata)
    assert output.is_watertight and output.is_winding_consistent
    assert output.nondegenerate_faces().all()
    np.testing.assert_allclose(output.bounds, source.bounds, atol=0.02)
    assert output.volume == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("mixed_winding", [False, True])
@pytest.mark.parametrize("shape", ["cup", "torus"])
def test_hole_repair_preserves_cup_cavity_and_handle_tunnel(shape, mixed_winding):
    if shape == "cup":
        source = trimesh.creation.revolve(np.array([[0, 0], [2, 0], [2, 3], [1.6, 3], [1.6, 0.4], [0, 0.4]]))
        source = source.subdivide().subdivide()
        puncture = [2, 0, 1.5]
        probes = [[0, 0, 1.5], [1.8, 0, 1.5]]  # cavity, material
        expected_euler = 2
    else:
        source = trimesh.creation.torus(2.0, 0.5)
        puncture = [2.5, 0, 0]
        probes = [[0, 0, 0], [2, 0, 0]]  # tunnel, material
        expected_euler = 0
    expected_volume = source.volume
    missing = np.argmin(np.linalg.norm(source.triangles_center - puncture, axis=1))
    source.update_faces(np.arange(len(source.faces)) != missing)
    if mixed_winding:
        source.faces[::3] = source.faces[::3, ::-1]

    results, prepared, _, metadata = _prepare(source)
    assert prepared.is_watertight and prepared.is_winding_consistent
    coords = get_coords_coarse2fine(results["occ_coarse"], 8)
    dense = sparse_sdf2dense(coords, results["sdf_coarse2fine"].ravel(), 96)[0]
    indices = np.floor(points_to_grid_frame(probes, metadata)).astype(int)
    signs = dense[tuple(indices.T)]
    assert signs[0] > 0 and signs[1] < 0
    proxy = sdf_proxy_mesh(results["occ_coarse"], results["sdf_coarse2fine"], 96, 12)
    output = to_input_frame(proxy, metadata)
    assert output.is_watertight and output.is_winding_consistent
    assert output.euler_number == expected_euler
    assert output.volume == pytest.approx(expected_volume, rel=0.04)


def test_sdf_sign_ignores_inverted_intersecting_fragment_outside_solid():
    solid = trimesh.creation.box((10, 10, 10)).apply_translation([16, 16, 16])
    fragment = trimesh.creation.box((2, 2, 0.5)).apply_translation([16, 16, 21])
    fragment.invert()
    source = trimesh.util.concatenate([solid, fragment])
    mesh = mrmeshnumpy.meshFromFacesVerts(source.faces, source.vertices)
    # The inverted fragment is nearest to all three centers, but must not
    # turn the unbounded exterior negative (nor erase the solid interior).
    sdf = compute_sparse_sdf(mesh, np.array([[16, 16, 16], [16, 16, 21], [16, 16, 26]]), 32, verbose=False)
    np.testing.assert_allclose(sdf[:, 0] * 32, [-4.25, 0.25, 5.25], atol=1e-6)


def test_preparation_rejects_surface_without_enclosed_volume():
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        faces=[[0, 1, 2], [1, 3, 2]], process=False,
    )
    with pytest.raises(ValueError):
        _prepare(sheet)
