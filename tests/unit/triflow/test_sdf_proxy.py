"""The conditioning SDF and extraction proxy represent the same zero surface."""

import numpy as np
import pytest

from kitbash.retopology.triflow.geometry import get_coords_coarse2fine, sdf_proxy_mesh, sparse_sdf2dense


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
