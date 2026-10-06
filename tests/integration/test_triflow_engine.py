"""The real TriFlow engine on the real weights (skipped when the weights are not downloaded)."""

from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from kitbash.errors import PreflightError, RetopologyError
from kitbash.retopology.base import RetopologyMethod
from kitbash.retopology.triflow.engine import TriflowRetopologizer
from kitbash.retopology.triflow.weights import CHECKPOINTS, checkpoint_path

WEIGHTS = Path("~/.cache/kitbash/triflow").expanduser()
have_weights = all(checkpoint_path(WEIGHTS, c.name).is_file() for c in CHECKPOINTS)
needs_weights = pytest.mark.skipif(not have_weights, reason="TriFlow weights not downloaded (kitbash retopology download-weights)")


def engine(weights_dir: Path = WEIGHTS, *, face_count: int = 1500, steps: int = 10, download: bool = True) -> TriflowRetopologizer:
    return TriflowRetopologizer(
        face_count=face_count, qem_threshold=12.0, quad_ratio=0.95, flow_steps=steps, device="auto",
        weights_dir=weights_dir, recorder=None, seed=42, download=download,
    )


def chamfer_over_diagonal(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    pa, _ = trimesh.sample.sample_surface(a, 8000, seed=0)
    pb, _ = trimesh.sample.sample_surface(b, 8000, seed=0)
    diagonal = np.linalg.norm(a.bounds[1] - a.bounds[0])
    return float((cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean()) / 2 / diagonal)


@needs_weights
@pytest.mark.timeout(900)
def test_dense_mesh_becomes_a_clean_low_poly_mesh_in_the_same_frame(tmp_path):
    dense = trimesh.util.concatenate([
        trimesh.creation.torus(0.5, 0.18, 120, 60),
        trimesh.creation.capsule(height=0.9, radius=0.12, count=[60, 60]).apply_translation([0.5, 0, 0.45]),
    ])
    source = tmp_path / "dense.obj"
    dense.export(source)

    result = engine().retopologize(source, tmp_path / "out", "asset")

    assert result.method is RetopologyMethod.TRIFLOW
    assert result.mesh_path == tmp_path / "out" / "asset.obj"
    assert result.faces_in > 15000
    assert 500 <= result.faces_out <= result.faces_in // 3  # compact; face count is a soft conditioning signal
    out = trimesh.load(result.mesh_path, process=False)
    assert len(out.faces) == result.faces_out
    assert out.is_watertight and out.is_winding_consistent
    assert out.nondegenerate_faces().all()
    diagonal = np.linalg.norm(dense.bounds[1] - dense.bounds[0])
    assert np.abs(out.bounds - dense.bounds).max() / diagonal < 0.02  # scale, position and up axis are the input's
    assert chamfer_over_diagonal(dense, out) < 0.02


@needs_weights
def test_garbage_input_raises_retopology_error_not_a_raw_exception(tmp_path):
    (tmp_path / "bad.obj").write_text("this is not a mesh\n")
    with pytest.raises(RetopologyError, match="TriFlow failed"):
        engine().retopologize(tmp_path / "bad.obj", tmp_path / "out", "asset")


def test_check_reports_missing_weights_when_downloads_are_off(tmp_path):
    with pytest.raises(PreflightError, match="weights missing"):
        engine(tmp_path, download=False).check()


def test_check_accepts_a_missing_weights_dir_when_downloads_are_on(tmp_path):
    engine(tmp_path).check()


def test_inference_never_downloads_when_download_is_disabled(tmp_path, monkeypatch):
    from kitbash.retopology.triflow import weights

    def forbidden(*args, **kwargs):
        raise AssertionError("download was attempted")

    monkeypatch.setattr(weights, "_download", forbidden)
    with pytest.raises(RetopologyError, match="weights missing"):
        engine(tmp_path, download=False).retopologize(tmp_path / "mesh.obj", tmp_path / "out", "asset")
