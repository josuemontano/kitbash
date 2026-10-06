"""The vendored networks: strict checkpoint loading (skipped without weights) and tiny forward passes."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kitbash.retopology.triflow import sparse as sp
from kitbash.retopology.triflow.models import (
    Direct3ds2SparseVAE,
    ShapeConditionedSlatFlowModel,
    build_flow_model,
    build_nvv_vae,
    build_sdf_vae,
)

WEIGHTS_DIR = Path(os.environ.get("KITBASH_TRIFLOW_WEIGHTS", Path.home() / ".cache" / "kitbash" / "triflow"))
BUILDERS = {"sdf_vae": build_sdf_vae, "nvv_vae": build_nvv_vae, "flow_model": build_flow_model}


def _need(name: str) -> Path:
    path = WEIGHTS_DIR / f"{name}.safetensors"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return path


@pytest.mark.parametrize("name", list(BUILDERS))
def test_checkpoint_loads_strictly(name):
    path = _need(name)
    model = BUILDERS[name]()
    assert not model.training
    assert all(p.device.type == "cpu" for p in model.parameters())
    result = model.load_state_dict(load_file(path), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_sdf_vae_reconstructs_a_sphere_with_real_weights():
    """Encode + decode of a narrow-band sphere SDF only works if the conv kernel layout was bridged correctly."""
    model = BUILDERS["sdf_vae"]()
    model.load_state_dict(load_file(_need("sdf_vae")), strict=True)
    radius, centre = 24.0, 255.5
    axis = torch.arange(int(centre) - 30, int(centre) + 31)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    dist = (grid.float() - centre).norm(dim=1) - radius
    keep = dist.abs() <= 4.0
    coords = torch.cat([torch.zeros(int(keep.sum()), 1, dtype=torch.int32), grid[keep].int()], dim=1)
    sdf = (dist[keep] / 4.0)[:, None]  # triflow scale: voxel distance / 512 * 128
    with torch.no_grad():
        latent, _ = model.encode({"feats": sdf, "coords": coords}, sample_posterior=False)
        recon = model.decoder(latent, fine_coords=coords)
    assert latent.feats.shape[1] == 16
    assert torch.equal(recon.coords, coords)
    assert (recon.feats - sdf.clamp(-1, 1)).abs().mean() < 0.03


def _tiny_vae(**kw) -> Direct3ds2SparseVAE:
    cfg = {
        "embed_dim": 4,
        "in_channels": 3,
        "out_channels": 2,
        "model_channels_encoder": 512,
        "num_blocks_encoder": 1,
        "num_heads_encoder": 8,
        "model_channels_decoder": 512,
        "num_blocks_decoder": 1,
        "num_heads_decoder": 8,
        "use_nvv_encoder": True,
        "use_nvv_decoder": True,
        "decoder_channel_down_factors": [2, 4, 4],
    }
    return Direct3ds2SparseVAE(**{**cfg, **kw}).eval()


def _shell(n_batches=2, res=32, radius=9):
    rows = []
    axis = torch.arange(res)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    on = ((grid.float() - res / 2 + 0.5).norm(dim=1) - radius).abs() <= 0.9
    for b in range(n_batches):
        xyz = grid[on][b * 7 :]  # different size per batch element
        rows.append(torch.cat([torch.full((xyz.shape[0], 1), b), xyz], dim=1))
    return torch.cat(rows).int()


def test_tiny_nvv_vae_forward():
    torch.manual_seed(0)
    model = _tiny_vae()
    coords = _shell()
    feats = torch.randn(coords.shape[0], 3)
    with torch.no_grad():
        latent, _ = model.encode({"feats": feats, "coords": coords}, sample_posterior=False)
        assert latent.feats.shape == (latent.coords.shape[0], 4)
        assert latent.shape[0] == 2
        parents = torch.cat([coords[:, :1], coords[:, 1:] // 8], dim=1)
        assert torch.equal(latent.coords, torch.unique(parents, dim=0))
        out = model.decoder(latent, fine_coords=coords)
        assert torch.equal(out.coords, coords)
        assert out.feats.shape == (coords.shape[0], 2)
        assert torch.isfinite(out.feats).all()
        assert out.feats.abs().max() <= 1
        recon, post = model(
            {"feats": feats, "coords": coords}, sample_posterior=True
        )  # training-style call: reconstruction + posterior
        assert recon.feats.shape == out.feats.shape
        assert post.mean.shape == (latent.coords.shape[0], 4)


def test_tiny_sdf_style_vae_decodes_without_reference_coords():
    torch.manual_seed(1)
    model = _tiny_vae(use_nvv_encoder=False, use_nvv_decoder=False, in_channels=1, out_channels=1, decoder_channel_down_factors=[4, 8, 16])
    coords = _shell(n_batches=1, res=16, radius=4)
    with torch.no_grad():
        latent, _ = model.encode({"feats": torch.randn(coords.shape[0]), "coords": coords}, sample_posterior=False)
        out = model.decoder(latent)
    # un-referenced decoding keeps every child: 8^3 voxels per latent voxel
    assert out.coords.shape[0] == latent.coords.shape[0] * 512
    assert out.feats.shape[1] == 1


def _tiny_flow() -> ShapeConditionedSlatFlowModel:
    return ShapeConditionedSlatFlowModel(
        slat_flow_config={
            "resolution": 16,
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 64,
            "cond_channels": 64,
            "num_blocks": 2,
            "num_heads": 4,
            "mlp_ratio": 2,
            "patch_size": 2,
            "num_io_res_blocks": 2,
            "io_block_channels": [16],
            "pe_mode": "ape",
            "qk_rms_norm": True,
        },
        cond_face_count=True,
        cond_quad_ratio=True,
        cond_sdf_latent=True,
        sdf_feat_dim=4,
    ).eval()


def test_tiny_flow_forward_batched():
    torch.manual_seed(2)
    model = _tiny_flow()
    # random-init zero-out layers would make the output trivially zero; randomise them
    for p in model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.05)
    coords = _shell(res=16, radius=5)
    cond_coords = _shell(res=16, radius=6)
    x_t = sp.sparse2sparse_tensor(coords, torch.randn(coords.shape[0], 4))
    sdf_latent = sp.sparse2sparse_tensor(cond_coords, torch.randn(cond_coords.shape[0], 4))
    cond = {"sdf_latent": sdf_latent, "face_count": torch.tensor([[1000.0], [4000.0]]), "quad_ratio": torch.tensor([[0.2], [0.9]])}
    with torch.no_grad():
        out = model(x_t, torch.tensor([500.0, 100.0]), cond=cond)
    assert out.feats.shape == (coords.shape[0], 4)
    assert torch.isfinite(out.feats).all()
    assert out.feats.abs().sum() > 0
    assert torch.equal(out.coords, x_t.coords)
    # the conditioning is used: another face count changes the prediction
    cond2 = {**cond, "face_count": torch.tensor([[50.0], [40.0]])}
    with torch.no_grad():
        other = model(x_t, torch.tensor([500.0, 100.0]), cond=cond2)
    assert not torch.allclose(out.feats, other.feats)


def test_fp16_is_rejected():
    with pytest.raises(NotImplementedError):
        _tiny_vae(use_fp16=True)
