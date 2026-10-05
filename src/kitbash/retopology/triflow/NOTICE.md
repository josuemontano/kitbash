# Third-party code in this directory

This directory vendors and adapts **TriFlow** ("TriFlow: Generating Artist-Like 3D Mesh Topology via Nearest-Vertex
Vector Fields", Li et al., ECCV 2026; <https://github.com/DerKleineLi/triflow>) so kitbash can retopologize
generated meshes in-process, on CUDA, Apple MPS or CPU.

| component | upstream | license | file |
|---|---|---|---|
| TriFlow (models, mesh processing, reconstruction) | DerKleineLi/triflow | **Automotive Development Public Non-Commercial License 1.0** (MPL-2.0 based, file-level copyleft, non-commercial use only) | `licenses/TriFlow-ADPNCL-1.0.txt` |
| Sparse tensor / transformer / flow sampler code | microsoft/TRELLIS | MIT | `licenses/TRELLIS-MIT.txt` |
| Sparse VAE encoder/decoder, sparse attention | DreamTechAI/Direct3D-S2 | MIT | `licenses/Direct3D-S2-MIT.txt` |
| Constrained QEM simplification (runtime dependency `pyfqmr-triflow`, not vendored) | DerKleineLi/pyfqmr-triflow | MIT | `licenses/pyfqmr-MIT.txt` |
| Pretrained weights (downloaded at first use, not vendored) | huggingface.co/lihcxr/TriFlow | see the model card | |

## What this means for kitbash

* Files derived from TriFlow keep their ADPNCL header and stay under ADPNCL (a *Larger Work* may be combined with code
  under other terms, but the covered files stay covered). **ADPNCL permits non-commercial use only.** The TRELLIS- and
  Direct3D-S2-derived files are MIT.
* The rest of kitbash does not import this package unless `retopology.method = "triflow"` (the default). Choose
  `--retopology decimate` to avoid using the TriFlow code and weights.
* The runtime dependency **MeshLib** (`meshlib`, used by upstream for remeshing, voxel SDF and projection) is *not*
  open source; review its license before any commercial use.

## Changes relative to upstream

* One pure-PyTorch sparse backend (`sparse/`) replaces spconv / torchsparse / flash-attn / xformers: the sparse
  convolutions, windowed and cross attention run on CUDA, MPS and CPU. Upstream's Triton spatial-sparse-attention kernels
  (Direct3D-S2) are not used by the TriFlow networks and are not vendored (see `sparse/README` notes if present).
* No hard-coded `.cuda()`; float32 on MPS/CPU, fp16 autocast only on CUDA (`device.py`).
* No hydra / omegaconf / accelerate / wandb: hyper-parameters are Python constants (`models/`).
* `open3d` (voxelization) is replaced by a numba implementation; `PyMCubes` by scikit-image.
* Training, dataset-preparation, augmentation and rendering code is dropped; only inference is vendored.
* Checkpoints are fetched from a pinned Hugging Face revision and verified by SHA-256 (`weights.py`).
* Meshes are mapped back into the coordinate frame of the input mesh (`geometry.to_input_frame`), so kitbash's
  scale/orientation handling is unchanged.
