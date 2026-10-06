"""Non-neural geometry stages of TriFlow, vendored for kitbash.

Everything here runs on CPU (numpy / meshlib / numba / scipy); torch is only used for small index helpers and is
device-agnostic.

Vendored from https://github.com/DerKleineLi/triflow (``triflow/utils/{mesh_processing,mesh_reconstruction,nvv,sparse_voxel}.py``)
under the Automotive Development Public Non-Commercial License v1.0, Copyright (c) 2026 Haoxuan Li. ``frames.py`` and
``occupancy.py`` are kitbash additions. See ``frames.py`` for the coordinate frames.
"""

from .frames import points_to_grid_frame, points_to_input_frame, to_grid_frame, to_input_frame
from .mesh_processing import (
    adaptive_remesh,
    compute_quad_ratio,
    compute_sparse_direction,
    compute_sparse_sdf,
    decimate_mrmesh,
    discretize_mesh,
    get_precise_occupancy,
    load_mesh,
    pack_trimesh,
    process_one_mesh,
    robust_remesh,
    sdf_proxy_mesh,
    sdf_remesh,
)
from .mesh_reconstruction import topology_flow2mesh_QEM
from .nvv import (
    coords2pos,
    dirnorm2vector,
    filter_nvv_geodesic_fast,
    get_target_point_priority_watershed,
    vector2dirnorm,
    vector2nid,
    vector2pos,
)
from .occupancy import triangle_voxel_overlap
from .sparse_voxel import (
    coarse_to_fine,
    coord_hash,
    coords2dense_sdf,
    create_meshgrid,
    dense2sparse,
    find_coords_indices,
    find_indices,
    fine_coords2coarse_coords,
    fine_to_coarse,
    get_coords_coarse2fine,
    get_mc_mesh,
    nested_device_transfer,
    sparse2dense,
    sparse_sdf2dense,
)

__all__ = [
    "adaptive_remesh",
    "coarse_to_fine",
    "compute_quad_ratio",
    "compute_sparse_direction",
    "compute_sparse_sdf",
    "coord_hash",
    "coords2dense_sdf",
    "coords2pos",
    "create_meshgrid",
    "decimate_mrmesh",
    "dense2sparse",
    "dirnorm2vector",
    "discretize_mesh",
    "filter_nvv_geodesic_fast",
    "find_coords_indices",
    "find_indices",
    "fine_coords2coarse_coords",
    "fine_to_coarse",
    "get_coords_coarse2fine",
    "get_mc_mesh",
    "get_precise_occupancy",
    "get_target_point_priority_watershed",
    "load_mesh",
    "nested_device_transfer",
    "pack_trimesh",
    "points_to_grid_frame",
    "points_to_input_frame",
    "process_one_mesh",
    "robust_remesh",
    "sdf_proxy_mesh",
    "sdf_remesh",
    "sparse2dense",
    "sparse_sdf2dense",
    "to_grid_frame",
    "to_input_frame",
    "topology_flow2mesh_QEM",
    "triangle_voxel_overlap",
    "vector2dirnorm",
    "vector2nid",
    "vector2pos",
]
