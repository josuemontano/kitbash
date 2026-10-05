"""End-to-end non-neural check of the TriFlow geometry pipeline.

The flow model is replaced by ground truth: the NVV field of a *target-topology* mesh, computed on the occupied voxels of the
input mesh's grid. The QEM stage must then rebuild a mesh with that topology from the dense remeshed input, in grid units,
and ``to_input_frame`` must put it back on the original input.
"""

import numpy as np
import pytest
import trimesh
from meshlib import mrmeshnumpy, mrmeshpy

from kitbash.retopology.triflow.geometry import (
    compute_sparse_direction,
    get_precise_occupancy,
    process_one_mesh,
    robust_remesh,
    sdf_proxy_mesh,
    to_grid_frame,
    to_input_frame,
    topology_flow2mesh_QEM,
)

NVV_SMOOTH = {"radius": 3.5, "threshold": 6, "sigma_s": 1.0, "sigma_r": 1.0}  # as in inference.py


def _kwargs(res_fine, get_metadata=False):
    return {
        "res_coarse": res_fine // 8,
        "res_fine": res_fine,
        "pad": 1.5,
        "round_verts": False,
        "decimate_length": 1.0,
        "vertex_merge_threshold": 0.0,
        "get_metadata": get_metadata,
        "cast": False,
        "verbose": False,
    }


def _qem(dense, coords, nvv, res_fine, target_face_count=500):
    return topology_flow2mesh_QEM(
        dense,
        coords,
        nvv,
        res_fine,
        nvv_smooth_kwargs=NVV_SMOOTH,
        root_threshold=0.5,
        merge_threshold=1.0,
        target_face_count=target_face_count,
        max_quadratic_error=12.0,
        target_position_weight=0.1,
        verbose=False,
        debug_output=None,
    )


def surface_distances(a, b, n=20000, seed=0):
    """Exact point-to-surface distances from samples of ``a`` to mesh ``b`` and vice versa (via meshlib)."""
    out = []
    for src, dst in ((a, b), (b, a)):
        points, _ = trimesh.sample.sample_surface(src, n, seed=seed)
        mr = mrmeshnumpy.meshFromFacesVerts(dst.faces, dst.vertices)
        sd = mrmeshpy.findSignedDistances(mr, mrmeshnumpy.fromNumpyArray(points.astype(np.float32)))
        out.append(np.abs(np.array(sd.vec, dtype=np.float64)))
    hausdorff = max(out[0].max(), out[1].max())
    chamfer = 0.5 * (out[0].mean() + out[1].mean())
    return hausdorff, chamfer


def diag(mesh):
    return float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))


def low_poly_sphere():
    return trimesh.creation.icosphere(subdivisions=2, radius=0.3)  # 320 faces


def low_poly_capsule():
    return trimesh.creation.capsule(height=0.8, radius=0.25, count=[8, 16])


def low_poly_box_cylinder():
    box = trimesh.creation.box((1.0, 1.0, 0.6))
    cyl = trimesh.creation.cylinder(radius=0.23, height=1.6, sections=16)
    cyl.apply_translation([0.07, 0.03, 0.11])  # keep the two shapes off each other's vertices / edges
    union = mrmeshpy.boolean(
        mrmeshnumpy.meshFromFacesVerts(box.faces, box.vertices),
        mrmeshnumpy.meshFromFacesVerts(cyl.faces, cyl.vertices),
        mrmeshpy.BooleanOperation.Union,
    ).mesh
    mesh = trimesh.Trimesh(mrmeshnumpy.getNumpyVerts(union), mrmeshnumpy.getNumpyFaces(union.topology), process=True)
    assert mesh.is_watertight
    return mesh


CASES = {"icosphere": low_poly_sphere, "capsule": low_poly_capsule, "box_cylinder": low_poly_box_cylinder}


@pytest.mark.parametrize("target", ["natural", 500])
@pytest.mark.parametrize("name", list(CASES))
def test_geometry_pipeline_ground_truth_nvv_reproduces_input(name, target):
    """NVV computed by process_one_mesh for the (low-poly) input itself, fed to QEM with a face-count target.

    Both natural and larger face-count targets must retain closed, consistently
    oriented geometry: surplus faces may not collapse into degenerate triangles.
    """
    mesh = CASES[name]()
    mesh.apply_translation([3.0, -1.5, 8.0])  # input frame is deliberately far from the origin
    res_fine = 128
    results, _, amesh, md = process_one_mesh(mesh, **_kwargs(res_fine))
    dense, _ = robust_remesh(amesh, remesh_method="adaptive", allow_collapse=False, get_metadata=False, verbose=False)
    assert len(dense.faces) > 10 * len(mesh.faces)  # a dense remesh of a low-poly input
    face_target = len(mesh.faces) if target == "natural" else target
    out_grid = _qem(dense, results["occ_fine"], results["nvv_fine"], res_fine, face_target)

    # still in the grid frame: voxel units
    assert out_grid.bounds.min() >= 0 and out_grid.bounds.max() <= res_fine
    out = to_input_frame(out_grid, md)
    assert out.vertices.dtype == np.float64

    assert 0 < len(out.faces) <= 500
    assert out.is_watertight and out.is_winding_consistent and out.volume > 0
    assert out.nondegenerate_faces().all()
    if target == "natural":
        assert 0.9 * len(mesh.faces) <= len(out.faces) <= 1.25 * len(mesh.faces)
    # in the input frame: same bounding box as the input (within 0.5% of the diagonal)
    np.testing.assert_allclose(out.bounds, mesh.bounds, atol=0.005 * diag(mesh))
    hausdorff, chamfer = surface_distances(out, mesh)
    print(
        f"{name}/{target}: faces {len(mesh.faces)} -> {len(out.faces)}, watertight {out.is_watertight}, hausdorff {hausdorff / diag(mesh):.4f}, chamfer {chamfer / diag(mesh):.4f}"
    )
    assert hausdorff < 0.02 * diag(mesh)
    assert chamfer < 0.005 * diag(mesh)


def test_marching_cubes_proxy_simplifies_without_breaking_closed_geometry():
    source = low_poly_box_cylinder()
    results, target, _, metadata = process_one_mesh(source, **_kwargs(128))
    proxy = sdf_proxy_mesh(results["occ_coarse"], results["sdf_coarse2fine"], 128, 16)
    proxy_native = mrmeshnumpy.meshFromFacesVerts(proxy.faces, proxy.vertices)
    coords = get_precise_occupancy(proxy_native, 128, verbose=False)
    target_native = mrmeshnumpy.meshFromFacesVerts(target.faces, target.vertices)
    nvv, _ = compute_sparse_direction(target_native, coords, 128, get_metadata=False, verbose=False)
    output = to_input_frame(_qem(proxy, coords, nvv, 128), metadata)
    assert output.is_watertight and output.is_winding_consistent
    assert output.nondegenerate_faces().all()
    assert len(output.faces) <= 1000
    hausdorff, chamfer = surface_distances(output, source)
    assert hausdorff < 0.02 * diag(source)
    assert chamfer < 0.005 * diag(source)


def test_geometry_pipeline_dense_input_with_low_poly_prediction():
    """Retopology proper: a smooth 5120-face sphere, NVV predicted for a 400-face version of it -> ~500 faces."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.3)
    sphere.apply_translation([5.0, -2.0, 1.0])
    res_fine = 128
    _, tmesh, amesh, md = process_one_mesh(sphere, **_kwargs(res_fine))

    low = mrmeshnumpy.meshFromFacesVerts(tmesh.faces, tmesh.vertices)  # grid frame
    settings = mrmeshpy.DecimateSettings()
    settings.maxDeletedFaces = len(tmesh.faces) - 400
    settings.maxError = 1e9
    settings.optimizeVertexPos = True
    mrmeshpy.decimateMesh(low, settings)
    low.pack()
    occ = get_precise_occupancy(low, res_fine, verbose=False)
    nvv, _ = compute_sparse_direction(low, occ, res_fine, get_metadata=False, verbose=False)

    dense, _ = robust_remesh(amesh, remesh_method="adaptive", allow_collapse=False, get_metadata=False, verbose=False)
    out = to_input_frame(_qem(dense, occ, nvv, res_fine), md)
    assert 300 <= len(out.faces) <= 500
    assert out.is_watertight and out.volume > 0
    hausdorff, chamfer = surface_distances(out, sphere)
    print(f"sphere: 5120 -> {len(out.faces)}, hausdorff {hausdorff / diag(sphere):.4f}, chamfer {chamfer / diag(sphere):.4f}")
    assert hausdorff < 0.02 * diag(sphere)
    assert chamfer < 0.005 * diag(sphere)


def test_geometry_pipeline_nvv_of_dense_input_keeps_its_topology():
    """Characterisation: the NVV of a dense mesh marks every vertex as a target, so QEM cannot go below ~its face count.

    (Hence the ground-truth test above feeds a low-poly input; with a dense one the 500-face target is unreachable.)
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.3)  # 1280 faces
    res_fine = 128
    results, _, amesh, _ = process_one_mesh(sphere, **_kwargs(res_fine))
    dense, _ = robust_remesh(amesh, remesh_method="adaptive", allow_collapse=False, get_metadata=False, verbose=False)
    out = _qem(dense, results["occ_fine"], results["nvv_fine"], res_fine)
    assert len(out.faces) > 1000


def test_geometry_pipeline_full_resolution_512():
    """Same as the ground-truth test but at the production resolution (res_fine=512, res_coarse=64)."""
    mesh = low_poly_capsule()
    mesh.apply_translation([0.0, 2.0, -1.0])
    results, _, amesh, md = process_one_mesh(
        mesh, res_coarse=64, res_fine=512, pad=1.5, round_verts=False, decimate_length=1.0, vertex_merge_threshold=0.0,
        augment=False, augment_strength=1.0, augment_density=False, cast=False, get_metadata=True, verbose=False,
    )  # fmt: skip
    assert results["res_fine"] == 512 and results["res_coarse"] == 64
    assert results["sdf_coarse2fine"].shape == (len(results["occ_coarse"]), 512)
    assert md["quad_ratio"] is None or 0.0 <= md["quad_ratio"] <= 1.0
    dense, _ = robust_remesh(amesh, remesh_method="adaptive", allow_collapse=False, get_metadata=False, verbose=False)
    out = to_input_frame(_qem(dense, results["occ_fine"], results["nvv_fine"], 512), md)
    assert 0 < len(out.faces) <= 500 and out.is_watertight
    hausdorff, _ = surface_distances(out, mesh)
    assert hausdorff < 0.02 * diag(mesh)


# --- dirty inputs --------------------------------------------------------------------------------------------------


def _dirty_mesh():
    """Two disjoint components + a floater, a hole, duplicated (unmerged) vertices and an unreferenced vertex."""
    ball = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    cube = trimesh.creation.box((0.3, 0.3, 0.3))
    cube.apply_translation([3.0, 0.2, 0.1])
    speck = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    speck.apply_translation([0.0, 2.0, 0.0])
    soup = trimesh.util.concatenate([ball, cube, speck])
    vertices, faces = soup.vertices, soup.faces[14:]  # 14 faces removed -> holes
    vertices = np.vstack([vertices, vertices[faces[:50].ravel()]])
    faces = faces.copy()
    faces[:50] = (len(soup.vertices) + np.arange(150)).reshape(50, 3)  # first 50 triangles detached from their neighbours
    vertices = np.vstack([vertices, [[9.0, 9.0, 9.0]]])  # never referenced
    return trimesh.Trimesh(vertices, faces, process=False)


def test_geometry_dirty_mesh_does_not_crash_and_keeps_both_components():
    dirty = _dirty_mesh()
    assert not dirty.is_watertight
    res_fine = 128
    results, tmesh, amesh, md = process_one_mesh(dirty, **_kwargs(res_fine, get_metadata=True))
    assert len(results["occ_fine"]) > 0 and np.isfinite(results["nvv_fine"]).all()
    assert tmesh.vertices.dtype == np.float64
    # the unreferenced vertex at (9, 9, 9) must not stretch the frame: bbox centre is that of the referenced geometry
    referenced = dirty.vertices[np.unique(dirty.faces)]
    np.testing.assert_allclose(md["center"], (referenced.min(0) + referenced.max(0)) / 2, atol=1e-4)

    for method in ("adaptive", "sdf"):
        remeshed, _ = robust_remesh(tmesh, remesh_method=method, allow_collapse=False, get_metadata=True, verbose=False)
        assert len(remeshed.faces) > 0 and remeshed.vertices.dtype == np.float64

    dense, _ = robust_remesh(amesh, remesh_method="adaptive", allow_collapse=False, get_metadata=False, verbose=False)
    out = to_input_frame(_qem(dense, results["occ_fine"], results["nvv_fine"], res_fine), md)
    assert len(out.faces) > 0 and np.isfinite(out.vertices).all()
    assert len(out.split(only_watertight=False)) >= 2  # the disjoint cube survives next to the ball
    assert out.bounds[1][0] == pytest.approx(referenced.max(0)[0], abs=0.02 * diag(dirty))


@pytest.mark.parametrize(
    "defect", ["nonmanifold_fin", "duplicate_faces", "degenerate_faces", "nan_vertex", "mixed_winding", "two_triangles"]
)
def test_geometry_robustness_cases_do_not_crash(defect):
    ball = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    v, f = ball.vertices.copy(), ball.faces.copy()
    if defect == "nonmanifold_fin":
        v = np.vstack([v, [[2.0, 2.0, 2.0], [2.0, -2.0, 2.0]]])
        f = np.vstack([f, [[f[0, 0], f[0, 1], len(ball.vertices)], [f[0, 1], f[0, 0], len(ball.vertices) + 1]]])
    elif defect == "duplicate_faces":
        f = np.vstack([f, f[:20]])
    elif defect == "degenerate_faces":
        f = np.vstack([f, [[0, 0, 1], [2, 2, 2]]])
    elif defect == "nan_vertex":
        v[5] = np.nan
    elif defect == "mixed_winding":
        f[: len(f) // 2] = f[: len(f) // 2, ::-1]
    elif defect == "two_triangles":
        v, f = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]), np.array([[0, 1, 2], [1, 3, 2]])
    mesh = trimesh.Trimesh(v, f, process=False)
    results, tmesh, _, md = process_one_mesh(mesh, **_kwargs(64, get_metadata=True))
    assert len(results["occ_fine"]) > 0 and np.isfinite(md["scale_factor"])
    for method in ("adaptive", "sdf"):
        remeshed, _ = robust_remesh(tmesh, remesh_method=method, allow_collapse=False, get_metadata=False, verbose=False)
        assert len(remeshed.faces) > 0
    assert to_grid_frame(mesh, md).vertices.dtype == np.float64
