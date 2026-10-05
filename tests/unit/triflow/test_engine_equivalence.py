"""Checks the engine against the upstream procedure it shortcuts, using the real NVV VAE weights."""

from pathlib import Path

import pytest
import torch
import trimesh

from kitbash.retopology.triflow import geometry
from kitbash.retopology.triflow.device import resolve_device
from kitbash.retopology.triflow.engine import _batch, _latent_coords
from kitbash.retopology.triflow.models import build_nvv_vae
from kitbash.retopology.triflow.weights import checkpoint_path

WEIGHTS = Path("~/.cache/kitbash/triflow").expanduser()
pytestmark = pytest.mark.skipif(not checkpoint_path(WEIGHTS, "nvv_vae").is_file(), reason="TriFlow weights not downloaded")


def test_latent_coords_equal_the_nvv_encoder_output(tmp_path):
    """Upstream encodes the input's own NVV field and keeps only the pooled coordinates (features become noise)."""
    from safetensors.torch import load_file

    mesh = trimesh.util.concatenate([trimesh.creation.torus(0.5, 0.18, 40, 20), trimesh.creation.box([0.3, 0.2, 0.9])])
    mesh.export(tmp_path / "m.obj")
    results, *_ = geometry.process_one_mesh(
        tmp_path / "m.obj", res_fine=512, pad=1.5, round_verts=False, decimate_length=1.0, vertex_merge_threshold=0.0,
        augment=False, augment_strength=1.0, augment_density=False, cast=False, get_metadata=False,
    )
    device = resolve_device("auto")
    data = _batch(results, 4000, 0.95, device)

    # upstream PreProcess.__call__
    coords, feats = data["occ_fine"], data["nvv_fine"].float()
    coords, dirnorm = geometry.vector2dirnorm(coords, feats)
    _, target_pos = geometry.vector2pos(coords, feats, data["res_fine"])
    _, coords_pos = geometry.coords2pos(coords, feats, data["res_fine"])
    norm = dirnorm[:, [-1]].clone()
    dirnorm[:, -1].sqrt_()
    merged = torch.cat([dirnorm, norm, feats, target_pos, coords_pos], dim=-1)

    vae = build_nvv_vae()
    vae.load_state_dict(load_file(str(checkpoint_path(WEIGHTS, "nvv_vae")), device="cpu"), strict=True)
    with torch.no_grad():
        latent, _ = vae.to(device).encode({"feats": merged, "coords": coords}, sample_posterior=False)

    shortcut = _latent_coords(data["occ_fine"], data["res_fine"], data["res_coarse"]).int()
    assert torch.equal(latent.coords.cpu(), shortcut.cpu())
