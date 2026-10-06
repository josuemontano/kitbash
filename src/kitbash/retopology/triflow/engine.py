# Copyright (c) 2026 Haoxuan Li (TriFlow inference.py), adapted for kitbash.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See licenses/TriFlow-ADPNCL-1.0.txt for details.
#
# Modified for kitbash: in-process engine instead of a CLI script; no hydra/accelerate; device selected at run time
# (CUDA, Apple MPS or CPU); deterministic noise from an explicit seed; result mapped back into the input mesh's frame.

"""TriFlow retopology engine: generated mesh in, artist-like low-poly mesh out (same frame, same up axis)."""

import importlib.util
import logging
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from kitbash.errors import PreflightError, RetopologyError
from kitbash.retopology.base import RetopologyMethod, RetopologyResult
from kitbash.retopology.triflow import weights as weight_files
from kitbash.retopology.triflow.device import DEVICES, autocast, resolve_device

log = logging.getLogger(__name__)

RES_FINE = 512
RES_COARSE = 64
SDF_SCALE = 128.0  # SDF values are scaled by this before the SDF VAE encodes them (must match training)
NVV_SMOOTH = {"radius": 3.5, "threshold": 6, "sigma_s": 1.0, "sigma_r": 1.0}
ROOT_THRESHOLD = 0.5
MERGE_THRESHOLD = 1.0
TARGET_POSITION_WEIGHT = 0.1
REQUIRED_MODULES = ("torch", "safetensors", "trimesh", "meshlib", "numba", "scipy", "skimage", "einops")


class TriflowRetopologizer:
    method = RetopologyMethod.TRIFLOW

    def __init__(
        self,
        *,
        face_count: int,
        qem_threshold: float,
        quad_ratio: float,
        flow_steps: int,
        device: str,
        weights_dir: Path,
        recorder: Any,
        seed: int,
        download: bool = True,
    ) -> None:
        self._face_count = face_count
        self._qem_threshold = qem_threshold
        self._quad_ratio = quad_ratio
        self._steps = flow_steps
        self._device_name = device
        self._weights_dir = Path(weights_dir)
        self._recorder = recorder
        self._seed = seed
        self._download = download
        self._lock = threading.Lock()  # one retopology at a time: the networks own the accelerator
        self._runtime: _Runtime | None = None

    # -- preflight ---------------------------------------------------------------------------------

    def check(self) -> None:
        missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
        if missing:
            raise PreflightError(
                f"TriFlow retopology needs Python packages that are not installed: {', '.join(missing)}",
                hint="Run `poetry install`, or choose `--retopology decimate`.",
            )
        try:
            importlib.import_module("kitbash.retopology.triflow.geometry._qem")
        except ImportError as exc:
            raise PreflightError("TriFlow's native QEM extension is unavailable", hint="Rebuild with `poetry install`.") from exc
        if self._device_name not in DEVICES:
            raise PreflightError(f"Unknown retopology.device {self._device_name!r}", hint=f"Use one of {', '.join(DEVICES)}.")
        try:
            resolve_device(self._device_name)
        except ValueError as exc:
            raise PreflightError(str(exc), hint="Set retopology.device = \"auto\" (or \"cpu\").") from exc
        weight_files.check(self._weights_dir, allow_download=self._download)

    # -- API -----------------------------------------------------------------------------------------

    def retopologize(self, mesh_path: Path, out_dir: Path, asset_name: str) -> RetopologyResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{asset_name}.obj"
        started = time.monotonic()
        with self._lock:
            try:
                runtime = self._load()
                faces_in, faces_out = self._run(runtime, Path(mesh_path), output)
            except RetopologyError:
                raise
            except Exception as exc:  # the research code raises everything from RuntimeError to meshlib's own errors
                log.exception("TriFlow failed on %s", mesh_path)
                raise RetopologyError(f"TriFlow failed on {Path(mesh_path).name}: {type(exc).__name__}: {exc}") from exc
        return RetopologyResult(
            method=self.method, mesh_path=output, faces_in=faces_in, faces_out=faces_out,
            duration_s=time.monotonic() - started, device=str(runtime.device),
        )

    # -- internals -----------------------------------------------------------------------------------

    def _span(self, name: str, **meta: Any):
        return self._recorder.span("step", name, **meta) if self._recorder is not None else nullcontext()

    def _load(self) -> _Runtime:
        if self._runtime is not None:
            return self._runtime
        import torch
        from safetensors import safe_open

        from kitbash.retopology.triflow.models import build_flow_model, build_nvv_vae, build_sdf_vae
        device = resolve_device(self._device_name)

        if weight_files.missing(self._weights_dir) and self._download:
            with self._span("triflow.download_weights", directory=str(self._weights_dir)):
                paths = weight_files.ensure(self._weights_dir, allow_download=True)
        else:
            paths = weight_files.ensure(self._weights_dir, allow_download=self._download)
        with self._span("triflow.load_models", device=str(device)):
            models = {}
            for name, build in (("sdf_vae", build_sdf_vae), ("nvv_vae", build_nvv_vae), ("flow_model", build_flow_model)):
                # Build shapes without allocating or initializing weights that
                # inference never uses. Strict-load only the retained modules.
                with torch.device("meta"):
                    model = build()
                unused = {"sdf_vae": "decoder", "nvv_vae": "encoder"}.get(name)
                if unused is not None:
                    delattr(model, unused)
                with safe_open(paths[name], framework="pt", device="cpu") as checkpoint:
                    checkpoint_keys = checkpoint.keys()
                    state = {
                        key: checkpoint.get_tensor(key) for key in checkpoint_keys
                        if unused is None or not key.startswith(unused + ".")
                    }
                model.load_state_dict(state, strict=True, assign=True)
                for module in model.modules():
                    if hasattr(module, "freq_dim") and hasattr(module, "freqs") and module.freqs.device.type == "meta":
                        frequencies = torch.arange(module.freq_dim, dtype=torch.float32, device=device) / module.freq_dim
                        module.freqs = 1.0 / (10000**frequencies)
                models[name] = model.eval().to(device)
        self._runtime = _Runtime(device=device, **models)
        log.info("TriFlow models loaded on %s", device)
        return self._runtime

    def _run(self, rt: _Runtime, mesh_path: Path, output: Path) -> tuple[int | None, int | None]:
        import torch

        from kitbash.retopology.triflow import geometry
        from kitbash.retopology.triflow.sampling import euler_sample

        device = rt.device
        with self._span("triflow.prepare_mesh"):
            results, _, _, metadata = geometry.process_one_mesh(
                mesh_path, res_fine=RES_FINE, pad=1.5, round_verts=False, decimate_length=0.0, vertex_merge_threshold=0.0,
                augment=False, augment_strength=1.0, augment_density=False, cast=False, get_metadata=False,
                compute_source_field=False,
            )
            faces_in = int(metadata["original_num_faces"])
            proxy = geometry.sdf_proxy_mesh(
                results["occ_coarse"], results["sdf_coarse2fine"], results["res_fine"], results["res_coarse"],
            )
            # Generate the NVF on the surface of G, not on the input triangulation.
            from meshlib import mrmeshnumpy

            proxy_native = mrmeshnumpy.meshFromFacesVerts(proxy.faces, proxy.vertices)
            results["occ_fine"] = geometry.get_precise_occupancy(proxy_native, RES_FINE, verbose=False)
            data = _batch(results, self._face_count, self._quad_ratio, device)
            del results, proxy_native

        with self._span("triflow.encode_sdf"):
            condition = _condition(rt, data)
        del data["sdf_coords"], data["sdf_features"]
        noise = _noise(data, rt, self._seed, device)

        with self._span("triflow.sample", steps=self._steps):
            with torch.no_grad(), autocast(device):
                encoded_condition = rt.flow_model.get_condition(noise, **condition)
                sample = euler_sample(rt.flow_model.flow_model, noise, encoded_condition, steps=self._steps)
            recon = _decode(rt, data, sample, device)
            del sample, noise, encoded_condition, condition

        with self._span("triflow.extract_mesh"):
            mesh = geometry.topology_flow2mesh_QEM(
                proxy, recon["coords"], recon["nvv"], recon["resolution"],
                nvv_smooth_kwargs=NVV_SMOOTH, root_threshold=ROOT_THRESHOLD, merge_threshold=MERGE_THRESHOLD,
                target_face_count=self._face_count, max_quadratic_error=self._qem_threshold,
                target_position_weight=TARGET_POSITION_WEIGHT, verbose=False, debug_output=None,
            )
            mesh = geometry.to_input_frame(mesh, metadata)
        if mesh is None or len(mesh.faces) == 0:
            raise RetopologyError("TriFlow produced an empty mesh")
        mesh.export(output)
        return faces_in, len(mesh.faces)


class _Runtime:
    def __init__(self, *, device, sdf_vae, nvv_vae, flow_model) -> None:
        self.device = device
        self.sdf_vae = sdf_vae
        self.nvv_vae = nvv_vae
        self.flow_model = flow_model


# -- batch construction (mirrors triflow's MeshDataset.collate_fn for a batch of one) -----------------------


def _batch(results: dict, face_count: int, quad_ratio: float, device) -> dict:
    import torch

    from kitbash.retopology.triflow.geometry import get_coords_coarse2fine

    def tensor(value):
        out = torch.as_tensor(value)
        if out.dtype == torch.float64:
            out = out.half()  # upstream rounds float64 arrays to half before they reach the networks
        return out

    def with_batch_index(coords):
        coords = tensor(coords)
        return torch.cat([torch.zeros((coords.shape[0], 1), dtype=coords.dtype), coords], dim=1).int()

    float_dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Prune on CPU before transfer instead of materializing the expanded SDF
    # coordinates, masks and discarded samples on the accelerator.
    ratio = results["res_fine"] // results["res_coarse"]
    sdf = tensor(results["sdf_coarse2fine"]).to(float_dtype).reshape(-1, 1) * SDF_SCALE
    keep = sdf[:, 0].abs() <= 1.0
    sdf_coords = torch.as_tensor(get_coords_coarse2fine(results["occ_coarse"], ratio))[keep]
    return {
        "occ_fine": with_batch_index(results["occ_fine"]).to(device),
        "sdf_coords": with_batch_index(sdf_coords).to(device),
        "sdf_features": sdf[keep].to(device),
        "res_fine": int(results["res_fine"]),
        "res_coarse": int(results["res_coarse"]),
        "face_count": torch.tensor([[float(face_count)]], device=device),
        "quad_ratio": torch.tensor([[float(quad_ratio)]], device=device),
    }


def _condition(rt: _Runtime, data: dict) -> dict:
    """Port of ``GetCondition``: encode the narrow-band SDF into the latent the flow model is conditioned on."""
    import torch

    from kitbash.retopology.triflow.sparse import sparse2sparse_tensor

    with torch.no_grad(), autocast(rt.device):
        latent, _ = rt.sdf_vae.encode({"feats": data["sdf_features"], "coords": data["sdf_coords"]}, sample_posterior=False)
    return {
        "sdf_latent": sparse2sparse_tensor(latent.coords, latent.feats),
        "face_count": data["face_count"],
        "quad_ratio": data["quad_ratio"],
    }


def _latent_coords(occ_fine, res_fine: int, res_coarse: int):
    """Voxels of the flow latent: the fine occupancy pooled to the latent grid.

    Upstream obtains these by running the NVV *encoder* on the input's own NVV field and discarding its features (they
    are replaced by noise); only the pooled coordinates survive. Pooling is a pure coordinate operation, so it is
    done directly: floor-divide, deduplicate, order by (batch, x, y, z) exactly as the sparse downsampler does.
    """
    from kitbash.retopology.triflow.models.voxel import fine_coords2coarse_coords

    return fine_coords2coarse_coords(occ_fine, res_fine // res_coarse)


def _noise(data: dict, rt: _Runtime, seed: int, device):
    import torch

    from kitbash.retopology.triflow.sparse import sparse2sparse_tensor

    coords = _latent_coords(data["occ_fine"], data["res_fine"], data["res_coarse"]).int()
    generator = torch.Generator().manual_seed(seed)  # CPU generator: same noise on every device
    feats = torch.randn((coords.shape[0], 16), generator=generator).to(device)
    return sparse2sparse_tensor(coords, feats)


def _decode(rt: _Runtime, data: dict, sample, device) -> dict:
    """Port of ``PostProcess``: decode the sampled latent into one NVV vector per fine voxel."""
    import torch

    from kitbash.retopology.triflow import geometry
    from kitbash.retopology.triflow.sparse import sparse2sparse_tensor

    latent = sparse2sparse_tensor(sample.coords, sample.feats)
    fine_coords = data["occ_fine"]
    with torch.no_grad(), autocast(device):
        decoded = rt.nvv_vae.decoder(latent, fine_coords=fine_coords)
    # Referenced subdivision emits exactly fine_coords in their supplied order.
    feats = decoded.feats.to(torch.float32)
    feats[:, -1].pow_(2)  # undo the sqrt applied to the magnitude during training
    coords, vectors = geometry.dirnorm2vector(fine_coords, feats)
    return {
        "coords": coords[:, 1:].cpu().numpy(),
        "nvv": vectors.cpu().numpy(),
        "resolution": int(data["res_fine"]),
    }
