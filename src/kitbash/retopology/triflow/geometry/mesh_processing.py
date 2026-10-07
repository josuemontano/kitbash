# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
# ruff: noqa: B905, SIM108  (vendored code is kept close to upstream; style rules are not applied to it)

# Modified for kitbash: vendored from triflow/utils/mesh_processing.py (inference path only).
#
# dropped: augment_mesh, _interpolate_displacement, _static_falloff, _wendland_c2 (training-time mesh augmentation; the
#   ``augment`` flag of process_one_mesh now raises NotImplementedError, ``augment_density`` / ``augment_strength`` are accepted
#   and ignored), and the ``meshiki.fps`` farthest-point sampling they needed.
# changed: ``open3d`` is replaced by the numba voxelizer in ``occupancy.py`` (get_precise_occupancy); ``meshiki`` is imported
#   lazily and is optional (only needed for ``quad_ratio`` metadata); process_one_mesh accepts a ``trimesh.Trimesh`` / Scene as
#   well as a path, always returns float64 meshes, and records the input-frame -> grid-frame transform in ``metadata`` (see
#   ``frames.py``).

import ctypes
import time

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmesh
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .occupancy import triangle_voxel_overlap
from .sparse_voxel import get_coords_coarse2fine, get_mc_mesh, sparse_sdf2dense


def pack_trimesh(mesh):
    """Clean up a trimesh in place: merge duplicate verts, drop degenerate
    and unreferenced geometry. Returns the same mesh for chaining.
    """
    trimesh.grouping.merge_vertices(
        mesh, merge_tex=True, merge_norm=True, digits_vertex=5
    )
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def extract_point_proj_results(result):
    """Extract face indices and barycentrics from a ``std_vector_MeshProjectionResult``.

    Reads the underlying C++ struct fields directly via ctypes to avoid a
    per-element Python loop. The returned barycentric coordinates satisfy
    ``bary[:, 0] + bary[:, 1] + bary[:, 2] = 1``, with column ``i`` equal
    to 1 when the projection lands on triangle vertex ``i``.

    Args:
        result: A ``mrmesh.std_vector_MeshProjectionResult``.

    Returns:
        ``(face_ids, bary)`` where ``face_ids`` has shape ``(N,)`` and
        ``bary`` has shape ``(N, 3)``.
    """
    n = len(result)
    elem_size = (
        mrmesh.std_vector_MeshProjectionResult.element_type_byte_size
    )  # 32 bytes
    stride_int32 = elem_size // 4  # 8 int32 units per element

    # --- Offsets ---
    face_off = (
        mrmesh.MeshProjectionResult._offsetof_proj + mrmesh.PointOnFace._offsetof_face
    )  # likely 0

    mtp_off = mrmesh.MeshProjectionResult._offsetof_mtp  # 16
    bary_off = mrmesh.MeshTriPoint._offsetof_bary  # 4
    a_off = mrmesh.TriPointf._offsetof_a  # 0
    b_off = mrmesh.TriPointf._offsetof_b  # 4

    a_abs = mtp_off + bary_off + a_off  # 20
    b_abs = mtp_off + bary_off + b_off  # 24

    base = result.data_pointer()

    total_ints = n * stride_int32
    array_type = ctypes.c_int32 * total_ints
    raw = np.ctypeslib.as_array(array_type.from_address(base))

    face_ids = raw[face_off // 4 : total_ints : stride_int32]

    a_vals = raw[a_abs // 4 : total_ints : stride_int32].view(np.float32)
    b_vals = raw[b_abs // 4 : total_ints : stride_int32].view(np.float32)

    c_vals = 1.0 - a_vals - b_vals

    bary = np.empty((n, 3), dtype=np.float32)
    bary[:, 1] = a_vals  # a == 1 -> vertex 1
    bary[:, 2] = b_vals  # b == 1 -> vertex 2
    bary[:, 0] = c_vals  # c == 1 -> vertex 0

    return face_ids, bary


def merge_close_vertices(vertices, threshold=2.0):
    """Snap groups of vertices within ``threshold`` distance to a single point.

    Uses ``trimesh.grouping.group_distance`` to find the groups and replaces
    every vertex in a group with the group's representative. The mesh's face
    connectivity is left untouched; this is typically followed by a
    :func:`pack_trimesh` call to drop the resulting degenerate faces.

    Args:
        vertices: ``(N, 3)`` array of positions.
        threshold: Maximum distance for two vertices to be merged.

    Returns:
        A new ``(N, 3)`` array with merged positions.
    """
    unique, groups_dist = trimesh.grouping.group_distance(vertices, distance=threshold)
    verts = vertices.copy()
    for u, g in zip(unique, groups_dist):
        verts[g] = u
    return verts


def decimate_mrmesh(mesh, min_edge_length=2.0):
    """Collapse short edges of an mrmesh in place.

    Runs ``mrmesh.decimateMesh`` with a pre-collapse callback that only
    allows collapses on edges shorter than ``min_edge_length``. This is used
    after voxel-scale discretization to merge voxel-adjacent vertices that
    would otherwise produce tiny faces.

    Args:
        mesh: A ``mrmesh.Mesh`` (modified in place).
        min_edge_length: Minimum allowed edge length before collapse.
    """
    mesh.packOptimally()

    settings = mrmesh.DecimateSettings()

    settings.tinyEdgeLength = min_edge_length
    settings.maxError = float("inf")
    settings.maxEdgeLen = min_edge_length * 5.0
    settings.optimizeVertexPos = False

    def pre_collapse(edge_id, _):
        org, dst = mesh.topology.org(edge_id), mesh.topology.dest(edge_id)
        p0 = mesh.points[org]
        p1 = mesh.points[dst]
        length = (p0 - p1).length()
        return length < min_edge_length

    settings.preCollapse = pre_collapse

    mrmesh.decimateMesh(mesh, settings)

    mesh.pack()


def discretize_mesh(
    mesh,
    resolution,
    pad_space,
    merge_threshold=2.0,
    round_verts=False,
    verbose=True,
):
    """Scale and translate a mesh into an integer voxel grid.

    The mesh's bounding box is rescaled so its longest side fits within
    ``resolution - 2 * pad_space`` voxels, then centered at
    ``(resolution / 2, resolution / 2, resolution / 2)``.

    Args:
        mesh: ``mrmesh.Mesh`` to discretize (modified in place).
        resolution: Target grid resolution.
        pad_space: Number of voxels of padding to leave around the bounding
            box.
        merge_threshold: If ``> 0``, nearby vertices are snapped together
            via :func:`merge_close_vertices` after discretization.
        round_verts: If ``True``, round each vertex coordinate to the nearest
            voxel center ``(i + 0.5)``.
        verbose: Print progress.

    Returns:
        ``(mesh, metadata)`` where ``metadata`` contains ``scale_factor``.
    """
    metadata = {}
    if verbose:
        print(
            f"Discretizing mesh to resolution {resolution} with padding {pad_space}..."
        )

    bbox = mesh.computeBoundingBox()
    size = bbox.max - bbox.min
    max_dim = max(size.x, size.y, size.z)

    if max_dim == 0:
        scale_factor = 1.0
    else:
        scale_factor = (resolution - 2 * pad_space) / max_dim
    metadata["scale_factor"] = scale_factor
    center = (bbox.min + bbox.max) / 2
    # Modified for kitbash: record the exact (float32, as meshlib stores it) centre used so the transform can be inverted
    # by ``frames.to_input_frame``: grid = (input - center) * scale_factor + resolution / 2.
    metadata["center"] = [float(center.x), float(center.y), float(center.z)]
    metadata["grid_resolution"] = int(resolution)
    metadata["pad_voxels"] = float(pad_space)
    translation_to_origin = mrmesh.Vector3f(0, 0, 0) - center

    scale_val = scale_factor
    col_x = mrmesh.Vector3f(scale_val, 0.0, 0.0)
    col_y = mrmesh.Vector3f(0.0, scale_val, 0.0)
    col_z = mrmesh.Vector3f(0.0, 0.0, scale_val)

    scale_mtx = mrmesh.Matrix3f(col_x, col_y, col_z)

    xform = mrmesh.AffineXf3f.linear(scale_mtx)
    xform = xform * mrmesh.AffineXf3f.translation(translation_to_origin)

    final_offset = mrmesh.Vector3f(resolution / 2, resolution / 2, resolution / 2)
    xform = mrmesh.AffineXf3f.translation(final_offset) * xform

    mesh.transform(xform)
    mesh.invalidateCaches()

    points_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)

    if merge_threshold > 0.0:
        points_np = merge_close_vertices(points_np, threshold=merge_threshold)

    if round_verts:
        points_np = np.round(points_np - 0.5).astype(np.float32) + 0.5

    rounded_trimesh = trimesh.Trimesh(vertices=points_np, faces=faces_np)
    rounded_trimesh = pack_trimesh(rounded_trimesh)

    mesh = mrmeshnumpy.meshFromFacesVerts(
        rounded_trimesh.faces, rounded_trimesh.vertices
    )

    mesh.pack()

    return mesh, metadata


def get_precise_occupancy(mesh, resolution, verbose=True):
    """Find the voxels that intersect the mesh surface.

    Modified for kitbash: upstream used Open3D's
    ``VoxelGrid.create_from_triangle_mesh_within_bounds`` (voxel size 1, bounds ``[0, R]^3``); this is the same conservative
    triangle/voxel-box overlap test implemented with numba (see ``occupancy.py``).

    Args:
        mesh: ``mrmesh.Mesh``.
        resolution: Cubic grid resolution ``R``.
        verbose: Print progress.

    Returns:
        ``np.ndarray`` of shape ``(N, 3)`` int32 voxel indices in
        ``[0, R)``, sorted lexicographically.
    """
    if verbose:
        print(f"Computing precise occupancy for resolution {resolution}...")
    t0 = time.time()

    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    verts_np = mrmeshnumpy.getNumpyVerts(mesh)

    if faces_np.size == 0:
        if verbose:
            print("  Mesh is empty. No occupancy found.")
        return np.array([], dtype=np.int32).reshape(0, 3)

    occupied_indices = triangle_voxel_overlap(verts_np, faces_np, resolution)

    if verbose:
        print(f"  Precise occupancy: {len(occupied_indices)} voxels. Time: {time.time()-t0:.2f}s")
    return occupied_indices


def compute_sparse_sdf(mesh, occupied_indices, resolution, verbose=True):
    """Compute SDF values at voxel centers for a given set of occupied indices.

    Uses mrmesh's ``findSignedDistances`` for distance magnitudes and generalized
    winding numbers for signs. Closest-face normals can misclassify exterior
    points near self-intersections, even after boundary holes are closed.
    Distances are normalized by ``resolution``.

    Args:
        mesh: ``mrmesh.Mesh``.
        occupied_indices: ``(N, 3)`` int array of voxel indices.
        resolution: Cubic grid resolution (used for normalization).
        verbose: Print progress.

    Returns:
        ``(N, 1)`` float array of normalized SDF values.
    """
    if verbose:
        print(
            f"Computing sparse SDF (direct query) for {len(occupied_indices)} voxels..."
        )
    t0 = time.time()

    voxel_centers = occupied_indices.astype(np.float32) + 0.5
    testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(voxel_centers)
    signed_distances_mrmesh = mrmesh.findSignedDistances(mesh, testPoints_mrmesh)
    sdf_values = np.array(signed_distances_mrmesh.vec)
    winding_numbers = mrmesh.std_vector_float()
    mrmesh.FastWindingNumber(mesh).calcFromVector(
        winding_numbers, testPoints_mrmesh, 2.0, mrmesh.FaceId(), mrmesh.func_bool_from_float(),
    )
    np.abs(sdf_values, out=sdf_values)
    sdf_values[np.asarray(winding_numbers) > 0.5] *= -1
    sdf_values = sdf_values.reshape(-1, 1)
    sdf_values /= resolution

    if verbose:
        print(f"  Sparse SDF calculation complete. Time: {time.time()-t0:.2f}s")
    return sdf_values


def compute_sparse_direction(
    mesh,
    occupied_indices,
    resolution,
    augmented=False,
    prev_verts=None,
    post_verts=None,
    get_metadata=True,
    verbose=True,
):
    """Compute the NVV (nearest-vertex vector) at each occupied voxel center.

    For each voxel, finds the closest point on the mesh surface, looks up
    which triangle vertex that point is nearest to (via the largest
    barycentric coordinate), and returns the offset from the voxel center
    to that vertex, normalized by ``resolution``.

    When ``augmented`` is ``True``, the query point used for the projection
    is re-mapped from the augmented mesh back onto the pre-augmentation
    mesh: the nearest post-augmentation vertex is found, and its
    pre-augmentation counterpart is used as the actual query position. This
    keeps the NVV field consistent with the unaugmented topology while
    sampling at the augmented voxel grid.

    Args:
        mesh: ``mrmesh.Mesh`` used for the projection (the unaugmented one
            when ``augmented=True``).
        occupied_indices: ``(N, 3)`` int voxel indices.
        resolution: Cubic grid resolution (used to normalize).
        augmented: Whether to remap query points through the prev/post
            vertex pair.
        prev_verts: Pre-augmentation vertex positions (``(V, 3)``), required
            when ``augmented=True``.
        post_verts: Post-augmentation vertex positions (``(V, 3)``), required
            when ``augmented=True``.
        get_metadata: If ``True``, also return face-sampling statistics in
            the returned metadata dict.
        verbose: Print progress.

    Returns:
        ``(directions, metadata)`` where ``directions`` has shape ``(N, 3)``
        and is normalized to grid-voxel units.
    """
    if verbose:
        print(f"Computing sparse direction for {len(occupied_indices)} voxels...")
    t0 = time.time()

    metadata = {}

    voxel_centers = occupied_indices.astype(np.float32) + 0.5
    if augmented:
        tree_post = cKDTree(post_verts)
        _, idxs = tree_post.query(voxel_centers, k=1)
        query_points = prev_verts[idxs]
    else:
        query_points = voxel_centers

    Points2MeshProjector = mrmesh.PointsToMeshProjector()
    Points2MeshProjector.updateMeshData(mesh)
    voxel_centers_mrmesh = mrmeshnumpy.fromNumpyArray(query_points)
    result = mrmesh.std_vector_MeshProjectionResult()
    objxf = mrmesh.AffineXf3f()
    refobjxf = mrmesh.AffineXf3f()
    up_dist_limit = 10000.0
    low_dist_limit = 0.0
    Points2MeshProjector.findProjections(
        result, voxel_centers_mrmesh, objxf, refobjxf, up_dist_limit, low_dist_limit
    )
    if verbose:
        print(f"  Projections computed in {time.time()-t0:.2f}s")

    face_ids, bary_coords = extract_point_proj_results(result)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)

    if get_metadata:
        triangle_id_counts = np.bincount(face_ids, minlength=len(faces_np))
        metadata["faces_sampled"] = np.sum(triangle_id_counts > 0).item()
        metadata["faces_sampled_3+times"] = np.sum(triangle_id_counts >= 3).item()
        metadata["faces_samples_0.25_quantile"] = np.quantile(
            triangle_id_counts, 0.25
        ).item()
        metadata["faces_samples_0.5_quantile"] = np.quantile(
            triangle_id_counts, 0.5
        ).item()
        metadata["faces_samples_0.75_quantile"] = np.quantile(
            triangle_id_counts, 0.75
        ).item()

    triangles = verts_np[faces_np]
    selected_triangles = triangles[face_ids]

    max_idx = np.argmax(bary_coords, axis=1)
    selected_vertices = selected_triangles[np.arange(len(max_idx)), max_idx]
    directions = selected_vertices - voxel_centers

    directions /= resolution

    if verbose:
        print(f"  Sparse direction calculation complete. Time: {time.time()-t0:.2f}s")
    return directions, metadata


def adaptive_remesh(
    trimesh_mesh,
    target_edge_length=2.0,
    get_metadata=True,
    allow_collapse=True,
    verbose=True,
):
    """Remesh a trimesh to a target uniform edge length using mrmesh's adaptive remesher.

    Args:
        trimesh_mesh: A ``trimesh.Trimesh`` to remesh.
        target_edge_length: Desired uniform edge length of the output.
        get_metadata: If ``True``, also compute chamfer / max distance from
            the input and put them in the returned metadata dict.
        allow_collapse: If ``False``, a pre-collapse hook blocks every
            collapse, preserving the original vertex count.
        verbose: Print progress.

    Returns:
        ``(remeshed_trimesh, metadata)``.
    """
    t0 = time.time()
    metadata = {}

    mesh = mrmeshnumpy.meshFromFacesVerts(
        trimesh_mesh.faces,
        trimesh_mesh.vertices,
    )

    settings = mrmesh.RemeshSettings()
    settings.targetEdgeLen = target_edge_length
    settings.useCurvature = False

    def pre_collapse(edge_id, new_pos):
        return False

    if not allow_collapse:
        settings.preCollapse = pre_collapse

    mrmesh.remesh(mesh, settings)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    remeshed_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    remeshed_trimesh = pack_trimesh(remeshed_trimesh)

    if get_metadata:
        remeshed_mrmesh = mrmeshnumpy.meshFromFacesVerts(
            remeshed_trimesh.faces, remeshed_trimesh.vertices
        )
        orig_points = trimesh_mesh.sample(100000)
        testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(orig_points)
        signed_distances_mrmesh = mrmesh.findSignedDistances(remeshed_mrmesh, testPoints_mrmesh)
        sdf_values = np.array(signed_distances_mrmesh.vec)
        chamfer_dist = np.mean(np.abs(sdf_values))
        max_dist = np.max(np.abs(sdf_values))

        metadata["remesh_chamfer_dist"] = chamfer_dist
        metadata["remesh_max_dist"] = max_dist
        metadata["remesh_num_vertices"] = len(remeshed_trimesh.vertices)
        metadata["remesh_num_faces"] = len(remeshed_trimesh.faces)

    if verbose:
        print(
            f"  Remeshed mesh: {len(remeshed_trimesh.vertices)} vertices, {len(remeshed_trimesh.faces)} faces."
        )
        if get_metadata:
            print(
                f"  Chamfer distance to original mesh: {chamfer_dist:.4f}, max dist: {max_dist:.4f}"
            )
        print(f"  Adaptive Remeshing time: {time.time()-t0:.2f}s")

    return remeshed_trimesh, metadata


def fill_hole_mrmesh(mesh):
    """Close every hole in an mrmesh in place using mrmesh's universal-metric filler.

    Needed before SDF-based remeshing so that the voxelized distance field
    does not leak through open surfaces and produce a thickened shell.
    """
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    for e in hole_edges:
        params = mrmesh.FillHoleParams()
        params.metric = mrmesh.getUniversalMetric(mesh)
        mrmesh.fillHole(mesh, e, params)


def sdf_remesh(trimesh_mesh, voxel_size=1.0, get_metadata=True, verbose=True):
    """Remesh by voxelizing the SDF and re-extracting an iso-surface.

    This is more robust to poorly-triangulated or self-intersecting input
    meshes than :func:`adaptive_remesh` because it goes through a signed
    distance volume in between. Holes are filled first via
    :func:`fill_hole_mrmesh` to keep the SDF well-defined.

    Args:
        trimesh_mesh: A ``trimesh.Trimesh``.
        voxel_size: Voxel size used to build the intermediate SDF volume.
            Smaller values preserve more detail at the cost of memory.
        get_metadata: If ``True``, also compute chamfer / max distance from
            the input.
        verbose: Print progress.

    Returns:
        ``(remeshed_trimesh, metadata)``.
    """
    t0 = time.time()
    metadata = {}

    mesh = mrmeshnumpy.meshFromFacesVerts(trimesh_mesh.faces, trimesh_mesh.vertices)
    fill_hole_mrmesh(mesh)

    params = mrmesh.MeshToVolumeParams()
    params.surfaceOffset = 3
    params.type = mrmesh.MeshToVolumeParams.Type.Signed
    params.voxelSize = mrmesh.Vector3f.diagonal(voxel_size)
    voxelsShift = mrmesh.AffineXf3f()
    params.outXf = voxelsShift
    vdbVolume = mrmesh.meshToDistanceVdbVolume(mesh, params)

    gSettings = mrmesh.GridToMeshSettings()
    gSettings.voxelSize = params.voxelSize
    gSettings.isoValue = 0.0
    remeshed_mrmesh = mrmesh.gridToMesh(vdbVolume.data, gSettings)
    remeshed_mrmesh.transform(voxelsShift)

    verts_np = mrmeshnumpy.getNumpyVerts(remeshed_mrmesh)
    faces_np = mrmeshnumpy.getNumpyFaces(remeshed_mrmesh.topology)
    remeshed_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    remeshed_trimesh = pack_trimesh(remeshed_trimesh)

    if get_metadata:
        orig_points = trimesh_mesh.sample(100000)
        testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(orig_points)
        signed_distances_mrmesh = mrmesh.findSignedDistances(remeshed_mrmesh, testPoints_mrmesh)
        sdf_values = np.array(signed_distances_mrmesh.vec)
        chamfer_dist = np.mean(np.abs(sdf_values))
        max_dist = np.max(np.abs(sdf_values))

        metadata["remesh_chamfer_dist"] = chamfer_dist
        metadata["remesh_max_dist"] = max_dist
        metadata["remesh_num_vertices"] = len(remeshed_trimesh.vertices)
        metadata["remesh_num_faces"] = len(remeshed_trimesh.faces)

    if verbose:
        print(
            f"  Remeshed mesh: {len(remeshed_trimesh.vertices)} vertices, {len(remeshed_trimesh.faces)} faces."
        )
        if get_metadata:
            print(
                f"  Chamfer distance to original mesh: {chamfer_dist:.4f}, max dist: {max_dist:.4f}"
            )
        print(f"  SDF Remeshing time: {time.time()-t0:.2f}s")

    return remeshed_trimesh, metadata


def robust_remesh(
    original_mesh,
    remesh_voxel_size=1.0,
    remesh_method="sdf",
    allow_collapse=True,
    get_metadata=True,
    verbose=True,
):
    """Remesh a trimesh, preferring SDF remeshing and falling back to adaptive.

    Tries :func:`sdf_remesh` first (more robust for irregular inputs). If it
    throws, returns an empty mesh, or produces a mesh that deviates too far
    from the original (``remesh_max_dist > 2.0``), falls back to
    :func:`adaptive_remesh` with the same effective voxel size.

    Args:
        original_mesh: Input ``trimesh.Trimesh``.
        remesh_voxel_size: Voxel size for ``sdf_remesh`` (and doubled as the
            target edge length for the adaptive fallback).
        remesh_method: ``"sdf"`` or ``"adaptive"``. ``"adaptive"`` skips the
            SDF path entirely.
        allow_collapse: Passed to the adaptive remesher; see its docstring.
        get_metadata: If ``True``, compute chamfer / max-distance metadata.
        verbose: Print progress and (if the SDF remesh raised) the exception
            that triggered the fallback.

    Returns:
        ``(remeshed_trimesh, metadata)``; ``metadata["remesh_method"]`` says
        which branch produced the result.
    """
    fallback_to_adaptive = False
    mesh = None
    metadata = {}

    if remesh_method == "sdf":
        try:
            mesh, remesh_metadata = sdf_remesh(
                original_mesh,
                voxel_size=remesh_voxel_size,
                get_metadata=get_metadata,
                verbose=verbose,
            )
            metadata.update(remesh_metadata)
            if get_metadata:
                metadata["remesh_method"] = "sdf"
            fallback_to_adaptive = (
                remesh_metadata["remesh_num_faces"] == 0
                or remesh_metadata["remesh_max_dist"] > 2.0
            )
        except Exception as e:
            fallback_to_adaptive = True
            if verbose:
                print(f"  SDF remeshing raised ({e}); will fall back to adaptive.")

    if verbose and fallback_to_adaptive:
        print("  Falling back to adaptive remeshing due to SDF remeshing issues.")

    if remesh_method == "adaptive" or fallback_to_adaptive:
        mesh, remesh_metadata = adaptive_remesh(
            original_mesh,
            target_edge_length=remesh_voxel_size * 2.0,
            allow_collapse=allow_collapse,
            get_metadata=get_metadata,
            verbose=verbose,
        )
        metadata.update(remesh_metadata)
        if get_metadata:
            metadata["remesh_method"] = "adaptive"

    return mesh, metadata


def _compute_fine_nvv(mesh, res_fine, get_metadata, verbose):
    """Sample source occupancy and nearest-vertex vectors in the fine grid."""
    if verbose:
        print(f"\n--- Processing Fine Resolution ({res_fine}) ---")
    occ_fine = get_precise_occupancy(mesh, res_fine, verbose=verbose)
    dir_fine, dir_metadata = compute_sparse_direction(
        mesh, occ_fine, res_fine, get_metadata=get_metadata, verbose=verbose,
    )
    return occ_fine, dir_fine, dir_metadata


def _compute_coarse_sdf(augmented_mrmesh, res_fine, res_coarse, ratio, verbose):
    """Compute coarse occupancy and its packed coarse-to-fine SDF payload.

    Builds a downscaled copy of ``augmented_mrmesh`` (so one coarse voxel
    occupies a unit cube), finds the coarse occupancy mask, expands every
    coarse voxel to its ``r^3`` fine children, and queries the SDF at each
    child center.

    Returns:
        ``(occ_coarse, sdf_coarse)`` where ``sdf_coarse`` has shape
        ``(N, r^3)``; both are empty arrays when no coarse voxel is
        occupied.
    """
    if verbose:
        print(f"\n--- Processing Coarse Resolution ({res_coarse}) ---")

    # Downscale the augmented mesh so one coarse voxel occupies a unit cube.
    mesh_coarse = mrmesh.Mesh(augmented_mrmesh)
    scale_down_val = 1.0 / ratio
    col_x_down = mrmesh.Vector3f(scale_down_val, 0.0, 0.0)
    col_y_down = mrmesh.Vector3f(0.0, scale_down_val, 0.0)
    col_z_down = mrmesh.Vector3f(0.0, 0.0, scale_down_val)
    scale_down_mtx = mrmesh.Matrix3f(col_x_down, col_y_down, col_z_down)
    scale_down = mrmesh.AffineXf3f.linear(scale_down_mtx)
    mesh_coarse.transform(scale_down)
    mesh_coarse.invalidateCaches()

    occ_coarse = get_precise_occupancy(mesh_coarse, res_coarse, verbose=verbose)
    # Surface-intersecting coarse cells alone omit SDF samples across their
    # boundaries. A one-cell halo covers the paper's 1/128-extent narrow band
    # (four fine voxels at resolution 512) and closes the sign barrier used by MC.
    offsets = np.indices((3, 3, 3), dtype=np.int32).reshape(3, -1).T - 1
    neighbors = (occ_coarse[:, None, :] + offsets).reshape(-1, 3)
    in_grid = np.all((neighbors >= 0) & (neighbors < res_coarse), axis=1)
    occ_coarse = np.unique(neighbors[in_grid], axis=0)
    N = len(occ_coarse)
    ratio_cubed = int(ratio) ** 3

    if verbose:
        print("Generating Coarse-to-Fine SDF mapping...")

    if N > 0:
        occ_coarse2fine = get_coords_coarse2fine(occ_coarse, ratio)
        if verbose:
            print(f"  Mapped {N} coarse voxels to {len(occ_coarse2fine)} fine voxels.")
        sdf_fine = compute_sparse_sdf(
            augmented_mrmesh,
            occ_coarse2fine,
            res_fine,
            verbose=verbose,
        )
        sdf_coarse = sdf_fine.reshape(N, ratio_cubed)
    else:
        if verbose:
            print("  No coarse voxels occupied. Skipping SDF calculation.")
        sdf_coarse = np.array([], dtype=np.float32).reshape(0, ratio_cubed)

    return occ_coarse, sdf_coarse


def sdf_proxy_mesh(occ_coarse, sdf_coarse2fine, res_fine, res_coarse):
    """Paper §3.3: extract the zero surface of the same sampled SDF used by the encoder.

    Samples are at voxel centers in normalized distance units. The returned mesh
    is in grid units, with no adaptive remeshing or post-extraction vertex welding.
    A signed field must enclose its surface inside the padded grid.
    """
    coords = get_coords_coarse2fine(occ_coarse, res_fine // res_coarse)
    sdf = sparse_sdf2dense(coords, np.asarray(sdf_coarse2fine).reshape(-1), res_fine)[0]
    boundary = (sdf[0], sdf[-1], sdf[:, 0], sdf[:, -1], sdf[:, :, 0], sdf[:, :, -1])
    if any(np.any(face <= 0) for face in boundary):
        raise ValueError("The sampled SDF does not enclose a closed surface inside the grid")
    vertices, faces = get_mc_mesh(sdf)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_watertight:
        raise ValueError("The sampled SDF does not enclose a closed surface inside the grid")
    return mesh


def load_mesh(input_file):
    """Load anything ``trimesh.load`` understands as a single float64 ``trimesh.Trimesh``.

    Added for kitbash. Scenes (e.g. .glb with several nodes) are flattened by concatenating all geometry with the node
    transforms applied; vertex coordinates are otherwise untouched (no axis conversion). ``process=False`` keeps duplicated
    vertices and degenerate faces exactly as in the file.
    """
    if isinstance(input_file, trimesh.Trimesh):
        loaded = input_file
    elif isinstance(input_file, trimesh.Scene):
        loaded = input_file.to_mesh() if hasattr(input_file, "to_mesh") else input_file.dump(concatenate=True)
    else:
        loaded = trimesh.load(input_file, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{input_file!s} does not contain a triangle mesh")
    return trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices, dtype=np.float64),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )


def _mesh_counts(mesh, stage):
    """Describe a preparation stage without changing its geometry."""
    return {
        f"{stage}_num_vertices": len(mesh.vertices),
        f"{stage}_num_faces": len(mesh.faces),
        f"{stage}_num_edges": len(mesh.edges),
    }


def _repair_surface(mesh, discretized_mesh, verbose):
    """Close input boundaries before either the encoder or proxy samples the surface."""
    hole_count = mesh.topology.findNumHoles()
    if not hole_count:
        return mesh
    if verbose:
        print(f"Closing {hole_count} boundary holes at one-voxel resolution...")
    # Only correct inconsistent winding: rays through an open boundary can
    # misclassify correctly oriented faces on the opposite side of a hole.
    if not discretized_mesh.is_winding_consistent:
        flipped = mrmeshnumpy.getNumpyBitSet(mrmesh.findDisorientedFaces(mesh))
        if flipped.any():
            faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
            faces[flipped] = faces[flipped, ::-1]
            oriented = pack_trimesh(trimesh.Trimesh(mrmeshnumpy.getNumpyVerts(mesh), faces, process=False))
            mesh = mrmeshnumpy.meshFromFacesVerts(oriented.faces, oriented.vertices)
            del oriented, faces
        del flipped
    # Voxel repair avoids triangulating hundreds of thousands of input loops.
    settings = mrmesh.RebuildMeshSettings()
    settings.voxelSize = 1.0
    settings.signMode = mrmesh.SignDetectionModeShort.HoleWindingNumber
    settings.closeHolesInHoleWindingNumber = True
    settings.preSubdivide = False
    settings.decimate = False
    mesh = mrmesh.rebuildMesh(mesh, settings)
    if mesh.topology.numValidFaces() == 0:
        raise ValueError("Input mesh has no enclosed surface after repair")
    # Voxel extraction can leave a few residual boundary loops.
    fill_hole_mrmesh(mesh)
    if mesh.topology.findNumHoles():
        raise ValueError("Input mesh still has boundary holes after surface repair")
    mesh.pack()
    return mesh


def _measure_preparation(mesh, discretized_mesh, prepared_mesh, remesh_method, verbose):
    """Measure preparation error and optional remesh quality without replacing the input."""
    points = mrmeshnumpy.fromNumpyArray(discretized_mesh.sample(100000))
    distances = np.array(mrmesh.findSignedDistances(mesh, points).vec)
    metadata = {
        "decimated_chamfer_dist": np.mean(np.abs(distances)),
        "decimated_max_dist": np.max(np.abs(distances)),
    }
    # Upstream remeshing is used only for statistics, never as the inference proxy.
    _, remesh_metadata = robust_remesh(
        prepared_mesh, remesh_voxel_size=1, remesh_method=remesh_method,
        get_metadata=True, verbose=verbose,
    )
    metadata.update(remesh_metadata)
    return metadata


def _prepare_mesh(
    input_file, *, res_fine, pad_voxels, round_verts, decimate_length,
    vertex_merge_threshold, remesh_method, get_metadata, verbose,
):
    """Load an input into the grid frame and prepare its surface for field sampling."""
    if verbose:
        print(f"Loading {input_file}...")
    try:
        original = load_mesh(input_file)
    except Exception as exc:
        if verbose:
            print(f"Error loading mesh {input_file}: {exc}")
        raise
    metadata = _mesh_counts(original, "original")
    mesh = mrmeshnumpy.meshFromFacesVerts(original.faces, original.vertices)
    mesh, frame_metadata = discretize_mesh(
        mesh, res_fine, pad_voxels, merge_threshold=vertex_merge_threshold,
        round_verts=round_verts, verbose=verbose,
    )
    metadata.update(frame_metadata)
    discretized = trimesh.Trimesh(
        vertices=mrmeshnumpy.getNumpyVerts(mesh), faces=mrmeshnumpy.getNumpyFaces(mesh.topology),
    )
    if decimate_length > 0.0:
        decimate_mrmesh(mesh, min_edge_length=decimate_length)
    mesh = _repair_surface(mesh, discretized, verbose)
    prepared = trimesh.Trimesh(
        vertices=mrmeshnumpy.getNumpyVerts(mesh), faces=mrmeshnumpy.getNumpyFaces(mesh.topology),
    )
    metadata.update(_mesh_counts(discretized, "discretized"))
    metadata.update(_mesh_counts(prepared, "decimated"))
    if get_metadata:
        metadata.update(_measure_preparation(mesh, discretized, prepared, remesh_method, verbose))
    return mesh, prepared, metadata


def _sample_mesh_fields(mesh, res_fine, res_coarse, compute_source_field, get_metadata, verbose):
    """Build the sparse SDF payload and optional source NVV from one prepared surface."""
    results = {}
    metadata = {}
    if compute_source_field:
        occ_fine, dir_fine, metadata = _compute_fine_nvv(mesh, res_fine, get_metadata, verbose)
        results["occ_fine"] = occ_fine
        results["nvv_fine"] = dir_fine
    results["res_fine"] = res_fine
    occ_coarse, sdf_coarse = _compute_coarse_sdf(
        mesh, res_fine, res_coarse, int(res_fine / res_coarse), verbose,
    )
    results["occ_coarse"] = occ_coarse
    results["sdf_coarse2fine"] = sdf_coarse
    results["res_coarse"] = res_coarse
    return results, metadata


def _cast_mesh_fields(results):
    """Compact sparse fields in place for storage, preserving resolution-dependent index widths."""
    for name in ("nvv_fine", "sdf_coarse2fine"):
        if name in results:
            results[name] = results[name].astype(np.float16)
    for name, resolution_key in (("occ_coarse", "res_coarse"), ("occ_fine", "res_fine")):
        if name not in results:
            continue
        resolution = results[resolution_key]
        if resolution <= 2**8:
            dtype = np.uint8
        elif resolution <= 2**16:
            dtype = np.uint16
        else:
            dtype = np.uint32
        results[name] = results[name].astype(dtype)


def _sample_metadata(mesh, results):
    """Describe sampled support and optional quad pairing for the prepared mesh."""
    metadata = {"num_occ_coarse": len(results["occ_coarse"])}
    if "occ_fine" in results:
        metadata["num_occ_fine"] = len(results["occ_fine"])
    # meshiki is optional; inference supplies an explicit quad-ratio condition.
    try:
        metadata["quad_ratio"] = compute_quad_ratio(mesh)
    except ImportError:
        metadata["quad_ratio"] = None
    return metadata


def process_one_mesh(
    input_file,
    res_coarse=64,
    res_fine=128,
    pad=1.5,
    round_verts=True,
    decimate_length=8.0,
    vertex_merge_threshold=2.0,
    augment=False,
    augment_density=True,
    augment_strength=1.0,
    remesh_method="sdf",
    cast=True,
    get_metadata=True,
    verbose=True,
    compute_source_field=True,
):
    """Coordinate mesh preparation, field sampling, and output packing.

    ``input_file`` accepts a mesh path, ``trimesh.Trimesh``, or scene. Preparation
    preserves its axes, centers it in the fine grid, and closes boundary holes.
    ``res_fine`` must be an integer multiple of ``res_coarse``; ``pad`` is measured
    in coarse voxels. ``round_verts`` snaps to voxel centers, while
    ``vertex_merge_threshold`` and ``decimate_length`` control grid-space merging
    and short-edge collapse (zero disables each).

    ``compute_source_field=False`` omits source occupancy/NVV for inference,
    which instead voxelizes its SDF proxy. ``cast`` compacts the sparse arrays to
    float16 and resolution-appropriate unsigned indices. ``get_metadata`` enables
    distance, face-sampling, and remesh statistics; ``remesh_method`` selects the
    statistics-only remesher. ``verbose`` controls stage progress output.

    Augmentation is not ported: ``augment=True`` raises ``NotImplementedError``;
    ``augment_density`` and ``augment_strength`` are unused.

    Returns ``(results, mesh, mesh, metadata)``. The two mesh entries are the same
    prepared float64 grid-frame mesh. Results contain ``res_fine``, ``res_coarse``,
    ``occ_coarse``, ``sdf_coarse2fine`` and, when requested, ``occ_fine``/``nvv_fine``.
    Metadata includes counts and the frame transform (``scale_factor``, ``center``,
    ``grid_resolution``) consumed by :func:`to_input_frame`.
    """
    started = time.time()
    if augment:
        raise NotImplementedError("mesh augmentation is training-only and was not ported to kitbash")
    mesh, prepared, metadata = _prepare_mesh(
        input_file, res_fine=res_fine, pad_voxels=pad * int(res_fine / res_coarse),
        round_verts=round_verts, decimate_length=decimate_length,
        vertex_merge_threshold=vertex_merge_threshold, remesh_method=remesh_method,
        get_metadata=get_metadata, verbose=verbose,
    )
    results, field_metadata = _sample_mesh_fields(
        mesh, res_fine, res_coarse, compute_source_field, get_metadata, verbose,
    )
    if cast:
        _cast_mesh_fields(results)
    metadata.update(field_metadata)
    metadata.update(_sample_metadata(prepared, results))
    if verbose:
        print(f"Done processing mesh in {time.time() - started:.2f} seconds.")
    return results, prepared, prepared, metadata


def compute_quad_ratio(mesh: trimesh.Trimesh):
    """Return the fraction of faces that can be paired into quads.

    Runs ``meshiki``'s quadrangulation pass on a copy of the mesh and
    reports its ``quad_ratio`` — a scalar in ``[0, 1]`` that is close to 1
    for meshes with largely quadrilateral topology and 0 for purely
    triangle-dominated ones. Used as a conditioning signal for the flow
    model.
    """
    try:
        import meshiki  # optional dependency
    except ImportError as e:
        raise ImportError("compute_quad_ratio needs the optional 'meshiki' package") from e

    meshiki_mesh = meshiki.Mesh(mesh.vertices, mesh.faces)
    meshiki_mesh.quadrangulate(thresh_bihedral=5, thresh_convex=185)
    return meshiki_mesh.quad_ratio
