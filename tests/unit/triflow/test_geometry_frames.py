"""Coordinate frames of process_one_mesh / to_input_frame, loading, and the exposed NVV / sparse-voxel helpers."""

import numpy as np
import pytest
import torch
import trimesh

from kitbash.retopology.triflow.geometry import (
    coarse_to_fine,
    coords2pos,
    dirnorm2vector,
    find_coords_indices,
    fine_coords2coarse_coords,
    fine_to_coarse,
    get_coords_coarse2fine,
    get_mc_mesh,
    load_mesh,
    process_one_mesh,
    to_grid_frame,
    to_input_frame,
    vector2dirnorm,
    vector2pos,
)

FAST = {
    "res_coarse": 8,
    "res_fine": 64,
    "pad": 1.5,
    "round_verts": False,
    "decimate_length": 1.0,
    "vertex_merge_threshold": 0.0,
    "get_metadata": False,
    "cast": False,
    "verbose": False,
}


def _asymmetric_mesh():
    """Elongated along y, offset and with a bump on +x so any axis permutation / flip is visible."""
    body = trimesh.creation.box(extents=(1.0, 4.0, 2.0))
    bump = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    bump.apply_translation([0.7, 1.5, 0.0])
    mesh = trimesh.util.concatenate([body, bump])
    mesh.apply_translation([10.0, -3.0, 7.5])
    return mesh


def test_geometry_process_one_mesh_metadata_and_grid_frame():
    mesh = _asymmetric_mesh()
    results, tmesh, amesh, md = process_one_mesh(mesh, **FAST)
    res = FAST["res_fine"]
    pad_voxels = FAST["pad"] * FAST["res_fine"] / FAST["res_coarse"]
    assert tmesh.vertices.dtype == np.float64 and amesh is tmesh
    assert md["grid_resolution"] == res and md["pad_voxels"] == pad_voxels
    assert md["scale_factor"] == pytest.approx((res - 2 * pad_voxels) / 4.0, rel=1e-6)  # longest side is y
    np.testing.assert_allclose(md["center"], mesh.bounds.mean(axis=0), atol=1e-5)
    # Grid frame: bbox centred in the grid, longest axis stays y (no permutation), all inside [pad, R - pad].
    lo, hi = tmesh.bounds
    np.testing.assert_allclose((lo + hi) / 2, [res / 2] * 3, atol=1e-3)
    assert np.argmax(hi - lo) == 1
    assert (hi - lo)[1] == pytest.approx(res - 2 * pad_voxels, abs=1e-3)
    # results as inference.py consumes them
    for key in ("occ_coarse", "sdf_coarse2fine", "occ_fine", "nvv_fine", "res_fine", "res_coarse"):
        assert key in results
    assert results["res_fine"] == res and results["res_coarse"] == FAST["res_coarse"]
    ratio = res // FAST["res_coarse"]
    assert results["sdf_coarse2fine"].shape == (len(results["occ_coarse"]), ratio**3)
    assert results["nvv_fine"].shape == (len(results["occ_fine"]), 3)
    assert "decimated_num_faces" in md and "quad_ratio" in md


def test_geometry_frame_round_trip_is_identity():
    mesh = _asymmetric_mesh()
    _, tmesh, _, md = process_one_mesh(mesh, **FAST)
    size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    grid = to_grid_frame(mesh, md)
    back = to_input_frame(grid, md)
    assert back.vertices.dtype == np.float64
    np.testing.assert_allclose(back.vertices, mesh.vertices, atol=1e-6 * size)
    np.testing.assert_array_equal(back.faces, mesh.faces)
    # The mesh process_one_mesh itself returns (float32 inside meshlib, decimated) maps back onto the input surface.
    in_frame = to_input_frame(tmesh, md)
    np.testing.assert_allclose(in_frame.bounds, mesh.bounds, atol=1e-4 * size)
    # Orientation preserved: positive scale, so the winding and the signed volume keep their sign.
    assert in_frame.volume > 0 and in_frame.volume == pytest.approx(mesh.volume, rel=2e-2)


def test_geometry_to_input_frame_requires_metadata():
    mesh = trimesh.creation.box()
    with pytest.raises(KeyError):
        to_input_frame(mesh, {"scale_factor": 1.0})


def test_geometry_process_one_mesh_rejects_augmentation():
    with pytest.raises(NotImplementedError):
        process_one_mesh(trimesh.creation.box(), **{**FAST, "augment": True})


def test_geometry_load_mesh_flattens_glb_scene_with_transforms_and_keeps_axes(tmp_path):
    a = trimesh.creation.box(extents=(1, 2, 3))
    b = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    scene = trimesh.Scene()
    scene.add_geometry(a, node_name="a", transform=trimesh.transformations.translation_matrix([5, 0, 0]))
    scene.add_geometry(b, node_name="b", transform=trimesh.transformations.translation_matrix([0, 0, 9]))
    path = tmp_path / "scene.glb"
    scene.export(path)
    loaded = load_mesh(path)
    assert loaded.vertices.dtype == np.float64
    assert len(loaded.faces) == len(a.faces) + len(b.faces)
    expected = scene.to_mesh() if hasattr(scene, "to_mesh") else scene.dump(concatenate=True)
    np.testing.assert_allclose(loaded.bounds, expected.bounds, atol=1e-5)
    # process_one_mesh takes the path directly
    _, tmesh, _, md = process_one_mesh(path, **FAST)
    np.testing.assert_allclose(md["center"], expected.bounds.mean(axis=0), atol=1e-4)
    assert tmesh.vertices.dtype == np.float64


def test_geometry_load_mesh_obj_and_ply_round_trip(tmp_path):
    mesh = _asymmetric_mesh()
    for ext in ("obj", "ply"):
        path = tmp_path / f"m.{ext}"
        mesh.export(path)
        loaded = load_mesh(path)
        np.testing.assert_allclose(loaded.bounds, mesh.bounds, atol=1e-5)
    with pytest.raises(ValueError):
        load_mesh(trimesh.Trimesh())


# --- NVV / sparse helpers -----------------------------------------------------------------------------------------

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_geometry_nvv_conversions_device_agnostic(device):
    torch.manual_seed(0)
    coords = torch.randint(0, 32, (50, 4), device=device)
    vec = torch.randn(50, 3, device=device) * 0.1
    _, dn = vector2dirnorm(coords, vec, 32)
    assert dn.shape == (50, 4) and dn.device.type == device
    _, back = dirnorm2vector(coords, dn, 32)
    torch.testing.assert_close(back, vec, atol=1e-6, rtol=1e-4)
    _, pos = vector2pos(coords, vec, 32)
    _, centers = coords2pos(coords, vec, 32)
    torch.testing.assert_close(pos - centers, vec, atol=1e-6, rtol=0)
    assert centers.min() >= -0.5 and centers.max() <= 0.5


@pytest.mark.parametrize("device", DEVICES)
def test_geometry_sparse_voxel_index_helpers(device):
    coarse = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0], [1, 4, 4, 4]], device=device)
    fine = get_coords_coarse2fine(coarse, 2)
    assert fine.shape == (3 * 8, 4)
    assert fine.device.type == device
    # children of (b, x, y, z) are (b, 2x + dx, 2y + dy, 2z + dz)
    assert set(map(tuple, fine[:8].tolist())) == {(0, 2 + dx, 4 + dy, 6 + dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)}
    # fine -> coarse recovers the parents in first-occurrence order
    parents = fine_coords2coarse_coords(fine, 2)
    assert set(map(tuple, parents.tolist())) == set(map(tuple, coarse.tolist()))
    # lookups: present -> index, absent -> -1
    queries = torch.cat([fine[[5, 20]], torch.tensor([[0, 63, 63, 63]], device=device)])
    idx = find_coords_indices(queries, fine)
    assert idx.tolist() == [5, 20, -1]
    # numpy in -> numpy out, 3-column form
    out = get_coords_coarse2fine(np.array([[1, 1, 1]]), 4)
    assert isinstance(out, np.ndarray) and out.shape == (64, 3)
    # space-to-depth round trip
    feats = torch.arange(len(fine), dtype=torch.float32, device=device).unsqueeze(1)
    packed, packed_coords = fine_to_coarse(feats, fine, 2)
    assert packed.shape == (3, 8)
    got, got_coords = coarse_to_fine(packed, packed_coords, fine, 2)
    torch.testing.assert_close(got, feats)
    assert torch.equal(got_coords, fine)


def test_geometry_marching_cubes_sign():
    """get_mc_mesh: inside negative, outward-wound faces, vertices in array-index coordinates + 0.5."""
    n, radius = 40, 11.3
    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1).astype(np.float64)
    centre = np.array([18.0, 20.0, 22.0])
    sdf = np.linalg.norm(grid - centre, axis=-1) - radius  # negative inside
    verts, faces = get_mc_mesh(sdf)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    assert mesh.is_watertight
    assert mesh.volume > 0  # outward normals => positive signed volume
    np.testing.assert_allclose(np.linalg.norm(verts - 0.5 - centre, axis=1), radius, atol=0.05)
    np.testing.assert_allclose(mesh.volume, 4 / 3 * np.pi * radius**3, rtol=0.02)
    assert np.abs(mesh.centroid - 0.5 - centre).max() < 0.05  # no axis permutation
    # Inverting the field (inside positive) must not be silently accepted as "inside negative": volume flips sign.
    inv_verts, inv_faces = get_mc_mesh(-sdf)
    assert trimesh.Trimesh(inv_verts, inv_faces, process=False).volume < 0
    # the level argument shifts the isosurface: sdf = r - radius, level = 2 -> radius + 2
    v2, _ = get_mc_mesh(sdf, level=2.0)
    np.testing.assert_allclose(np.linalg.norm(v2 - 0.5 - centre, axis=1), radius + 2.0, atol=0.05)
