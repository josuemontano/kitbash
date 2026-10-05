# Vendored from Direct3D-S2 (DreamTechAI/Direct3D-S2, Copyright (c) 2025 DreamTechAI, MIT License):
# direct3d_s2/models/autoencoders/{base,encoder,decoder,ss_vae}.py.
# Modified for kitbash: imports point at the unified pure-PyTorch ``sparse`` package; the fp16 conversion helpers, the
# chunked / random-crop decoding and the marching-cubes mesh export (``decode_mesh`` / ``sparse2mesh``, which pulled in
# trimesh and scikit-image) are dropped - TriFlow only needs ``encode`` and ``decoder.forward`` with ``chunk_size == 1``.

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .. import sparse as sp
from ..sparse.dense import AbsolutePositionEmbedder
from ..sparse.transformer import SparseTransformerBlock
from .layers import DiagonalGaussianDistribution

AttnMode = Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"]


def _reject_fp16(use_fp16: bool) -> None:
    if use_fp16:
        raise NotImplementedError("use_fp16 is not supported; run float32 (MPS / CPU) or torch.autocast (CUDA)")


def block_attn_config(self):
    """Return the attention configuration of the model."""
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """Sparse Transformer without output layers. Serves as the base class for encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_ratio: float = 4.0,
        attn_mode: AttnMode = "full",
        window_size: int | None = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        _reject_fp16(use_fp16)
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.attn_mode = attn_mode
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.dtype = torch.float32

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode=attn_mode,
                    window_size=window_size,
                    shift_sequence=shift_sequence,
                    shift_window=shift_window,
                    serialize_mode=serialize_mode,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    qk_rms_norm=self.qk_rms_norm,
                )
                for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
            ]
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, factor: float | None = None) -> sp.SparseTensor:
        h = self.input_layer(x)
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(x.coords[:, 1:], factor)
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)
        return h


class SparseDownBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        num_groups: int = 32,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(sp.SparseGroupNorm32(num_groups, channels), sp.SparseSiLU())

        self.down = sp.SparseDownsample(2)
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1)

        self.use_checkpoint = use_checkpoint

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)
        h = self.out_layers(h)
        return h + self.skip_connection(x)

    def forward(self, x: sp.SparseTensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class SparseSDFEncoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_ratio: float = 4,
        attn_mode: AttnMode = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )

        self.input_layer1 = sp.SparseLinear(1, model_channels // 16)

        self.downsample = nn.ModuleList(
            [
                SparseDownBlock3d(channels=model_channels // 16, out_channels=model_channels // 8, use_checkpoint=use_checkpoint),
                SparseDownBlock3d(channels=model_channels // 8, out_channels=model_channels // 4, use_checkpoint=use_checkpoint),
                SparseDownBlock3d(channels=model_channels // 4, out_channels=model_channels, use_checkpoint=use_checkpoint),
            ]
        )

        self.resolution = resolution
        self.out_layer = sp.SparseLinear(model_channels, 2 * latent_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, factor: float | None = None):
        x = self.input_layer1(x)
        for block in self.downsample:
            x = block(x)
        h = super().forward(x, factor)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return self.out_layer(h)


class SparseSubdivideBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.act_layers = nn.Sequential(sp.SparseConv3d(channels, self.out_channels, 3, padding=1), sp.SparseSiLU())

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU(),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.sub(h)
        return self.out_layers(h)

    def forward(self, x: sp.SparseTensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class SparseSDFDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: int | None = None,
        num_head_channels: int | None = 64,
        mlp_ratio: float = 4,
        attn_mode: AttnMode = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict | None = None,
        out_channels: int = 1,
        chunk_size: int = 1,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        if chunk_size != 1:
            raise NotImplementedError("chunked decoding is not ported")
        self.resolution = resolution
        self.rep_config = representation_config
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.upsample = nn.ModuleList(
            [
                SparseSubdivideBlock3d(channels=model_channels, out_channels=model_channels // 4, use_checkpoint=use_checkpoint),
                SparseSubdivideBlock3d(channels=model_channels // 4, out_channels=model_channels // 8, use_checkpoint=use_checkpoint),
                SparseSubdivideBlock3d(channels=model_channels // 8, out_channels=model_channels // 16, use_checkpoint=use_checkpoint),
            ]
        )

        self.out_layer = sp.SparseLinear(model_channels // 16, self.out_channels)
        self.out_active = sp.SparseTanh()

        self.initialize_weights()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, factor: float | None = None, return_feat: bool = False):
        h = super().forward(x, factor)
        for block in self.upsample:
            h = block(h)
        h = h.type(x.dtype)

        if return_feat:
            return self.out_active(self.out_layer(h)), h

        return self.out_active(self.out_layer(h))


class SparseSDFVAE(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 0,
        resolution: int = 64,
        model_channels_encoder: int = 512,
        num_blocks_encoder: int = 4,
        num_heads_encoder: int = 8,
        num_head_channels_encoder: int = 64,
        model_channels_decoder: int = 512,
        num_blocks_decoder: int = 4,
        num_heads_decoder: int = 8,
        num_head_channels_decoder: int = 64,
        out_channels: int = 1,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        chunk_size: int = 1,
        latents_scale: float = 1.0,
        latents_shift: float = 0.0,
        build_encoder: bool = True,
        build_decoder: bool = True,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.resolution = resolution
        self.latents_scale = latents_scale
        self.latents_shift = latents_shift

        # Modified for kitbash: the subclass can skip building the stock encoder / decoder it is about to replace.
        if build_encoder:
            self.encoder = SparseSDFEncoder(
                resolution=resolution,
                in_channels=model_channels_encoder,
                model_channels=model_channels_encoder,
                latent_channels=embed_dim,
                num_blocks=num_blocks_encoder,
                num_heads=num_heads_encoder,
                num_head_channels=num_head_channels_encoder,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
            )

        if build_decoder:
            self.decoder = SparseSDFDecoder(
                resolution=resolution,
                model_channels=model_channels_decoder,
                latent_channels=embed_dim,
                num_blocks=num_blocks_decoder,
                num_heads=num_heads_decoder,
                num_head_channels=num_head_channels_decoder,
                out_channels=out_channels,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
                chunk_size=chunk_size,
            )
        self.embed_dim = embed_dim

    def forward(self, batch):
        z, posterior = self.encode(batch)

        reconst_x = self.decoder(z)
        return {"reconst_x": reconst_x, "posterior": posterior}

    def encode(self, batch, sample_posterior: bool = True):
        feat, xyz, batch_idx = batch["sparse_sdf"], batch["sparse_index"], batch["batch_idx"]
        if feat.ndim == 1:
            feat = feat.unsqueeze(-1)
        coords = torch.cat([batch_idx.unsqueeze(-1), xyz], dim=-1).int()

        x = sp.SparseTensor(feat, coords)
        h = self.encoder(x, batch.get("factor", None))
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
        z = posterior.sample() if sample_posterior else posterior.mode()
        z = h.replace(z)

        return z, posterior
