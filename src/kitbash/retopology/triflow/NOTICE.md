# Third-party code in this directory

This directory vendors and adapts **TriFlow** ("TriFlow: Generating Artist-Like 3D Mesh Topology via Nearest-Vertex
Vector Fields", Li et al., ECCV 2026; <https://github.com/DerKleineLi/triflow>) so kitbash can retopologize
generated meshes in-process, on CUDA or Apple MPS (never the CPU).

| component | upstream | license | file |
|---|---|---|---|
| TriFlow (models, mesh processing, reconstruction) | DerKleineLi/triflow | **Automotive Development Public Non-Commercial License 1.0** (MPL-2.0 based, file-level copyleft, non-commercial use only) | `licenses/TriFlow-ADPNCL-1.0.txt` |
| Sparse tensor / transformer / flow sampler code | microsoft/TRELLIS | MIT | `licenses/TRELLIS-MIT.txt` |
| Sparse VAE encoder/decoder, sparse attention | DreamTechAI/Direct3D-S2 | MIT | `licenses/Direct3D-S2-MIT.txt` |
| Constrained QEM simplification (`geometry/Simplify.h`, vendored at `c93a7671ba4a2163dd3c7ea6ab6b52ad0167cc72`) | DerKleineLi/pyfqmr-triflow | MIT | `licenses/pyfqmr-MIT.txt` |
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
  convolutions, windowed and cross attention run on CUDA and MPS. Upstream's Triton spatial-sparse-attention kernels
  (Direct3D-S2) are not used by the TriFlow networks and are not vendored (see `sparse/README` notes if present).
* No hard-coded `.cuda()`; float32 on MPS, fp16 autocast only on CUDA (`device.py`).
* No hydra / omegaconf / accelerate / wandb: hyper-parameters are Python constants (`models/`).
* `open3d` (voxelization) is replaced by a numba implementation; `PyMCubes` by scikit-image.
* The constrained QEM is built in-package through a small Cython binding. Flip/degeneracy rejection is enabled;
  output connectivity is retained instead of welding vertices and deleting faces after QEM.
* Extraction follows the paper's SDF marching-cubes proxy rather than upstream inference's adaptive remesh.
  The sparse SDF includes a full narrow-band halo; watershed roots use transferred mesh displacements.
  Unseeded components receive their deterministic minimum-displacement vertex as a recovery root.
* Before SDF sampling, open inputs have inverted faces corrected by ray parity, then are repaired with
  MeshLib's hole-aware voxel reconstruction at one fine-grid voxel resolution, followed by filling residual
  boundary loops. Closed inputs skip reconstruction.
  SDF signs use generalized winding numbers instead of the closest-face normal, which is unreliable near
  self-intersections. The encoder and extraction proxy share the repaired field.
* `process_one_mesh` only coordinates preprocessing; grid preparation, surface repair, sparse field sampling,
  compact casting, and metadata collection are separate helpers with an unchanged public return contract.
* Sparse pooling and neighbor maps are cached by coordinate set across diffusion steps; coordinate caches
  release without cyclic garbage collection. Inference does not construct source NVF, or retain the unused
  SDF decoder and NVF encoder weights on the accelerator. Cached checkpoint hashes are verified before use.
* Training, dataset-preparation, augmentation and rendering code is dropped; only inference is vendored.
* Checkpoints are fetched from a pinned Hugging Face revision and verified by SHA-256 (`weights.py`).
* Meshes are mapped back into the coordinate frame of the input mesh (`geometry.to_input_frame`), so kitbash's
  scale/orientation handling is unchanged.
