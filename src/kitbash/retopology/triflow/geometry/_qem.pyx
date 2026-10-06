# distutils: language = c++
# The native implementation is MIT-licensed; see ../licenses/pyfqmr-MIT.txt.
# Keep the GIL for the whole transaction: Fast-QEM owns process-global mesh state.

from libcpp cimport bool
from libcpp.vector cimport vector
cimport cython
import numpy as np

cdef extern from "Simplify.h":
    cdef cppclass vec3f:
        double x, y, z

cdef extern from "Simplify.h" namespace "Simplify":
    cdef cppclass Vertex:
        vec3f p
    cdef cppclass Triangle:
        int v[3]
        int attr, material
    vector[Vertex] vertices
    vector[Triangle] triangles
    void simplify_mesh(int, int, double, void (*)(char*, int), int, double,
                       int, bool, double, bool, double*, int, int*, int, double) except +


@cython.boundscheck(False)
@cython.wraparound(False)
def simplify(positions, faces, quadratics, roots, int target_count,
             double max_error, double merge_threshold):
    """Constrained QEM with orientation/degeneracy guards; returns vertices and faces."""
    cdef const double[:, ::1] points = np.ascontiguousarray(positions, dtype=np.float64)
    cdef const int[:, ::1] indices = np.ascontiguousarray(faces, dtype=np.int32)
    cdef double[:, ::1] q = np.ascontiguousarray(quadratics, dtype=np.float64)
    cdef int[::1] groups = np.ascontiguousarray(roots, dtype=np.int32)
    cdef Py_ssize_t nv = points.shape[0], nf = indices.shape[0], i, j
    if nv == 0 or nf == 0 or points.shape[1] != 3 or indices.shape[1] != 3:
        raise ValueError("QEM requires a nonempty triangle mesh")
    if q.shape[0] != nv or q.shape[1] != 10 or groups.shape[0] != nv:
        raise ValueError("QEM requires one quadric and region id per vertex")
    if not np.isfinite(positions).all() or not np.isfinite(quadratics).all():
        raise ValueError("QEM requires finite positions and quadrics")
    if np.min(faces) < 0 or np.max(faces) >= nv or np.min(roots) < 0 or np.max(roots) >= nv:
        raise ValueError("QEM vertex and region indices must refer to the input mesh")
    vertices.clear()
    triangles.clear()
    vertices.resize(nv)
    triangles.resize(nf)
    for i in range(nv):
        vertices[i].p.x = points[i, 0]
        vertices[i].p.y = points[i, 1]
        vertices[i].p.z = points[i, 2]
    for i in range(nf):
        for j in range(3):
            triangles[i].v[j] = indices[i, j]
        triangles[i].attr = 0
        triangles[i].material = -1
    simplify_mesh(target_count, 5, 7., NULL, 200, 1e-9, 3, False, max_error,
                  False, &q[0, 0], nv * 10, &groups[0], nv, merge_threshold)
    out_vertices = np.empty((vertices.size(), 3), dtype=np.float64)
    out_faces = np.empty((triangles.size(), 3), dtype=np.int32)
    cdef double[:, ::1] output_points = out_vertices
    cdef int[:, ::1] output_indices = out_faces
    for i in range(vertices.size()):
        output_points[i, 0] = vertices[i].p.x
        output_points[i, 1] = vertices[i].p.y
        output_points[i, 2] = vertices[i].p.z
    for i in range(triangles.size()):
        for j in range(3):
            output_indices[i, j] = triangles[i].v[j]
    return out_vertices, out_faces
