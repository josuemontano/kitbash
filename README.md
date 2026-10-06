# kitbash

`kitbash` turns a reference image or a text prompt into a complete, human-editable Blender scene
(plus USD). It breaks the shot down into individual objects, models each one (reference image →
Trellis mesh → Blender build script with node-based PBR materials), lays the scene out for a chosen
style, and exports everything. Every approved asset goes into a reusable global library, the
**backlot**, so later scenes can reuse it instead of modelling it again.

Everything runs headless: Blender (`blender -b`), the `omp` agent (all LLM calls) and Trellis.

```
image or prompt
   │
   ▼
1 Breakdown ── inventory + blockout render ── critic loop ── backlot lookup ── user gate
   │
   ▼
2 Modelling ── per asset, pipelined:  reference → Trellis → retopology → build script → critic loop → review queue
   │           (workers keep generating while you review; nothing enters the backlot until approved)
   ▼
3 Layout ───── placement, camera, lights, world for --style ── critic loop ── user gate
   │
   ▼
4 Assembly ─── self-contained scene.blend + scene.usd + final render + analytics
```

## Requirements

- macOS or Linux, Python 3.14, [Poetry](https://python-poetry.org) 2.x (no uv).
- Blender 4.2 or newer on `PATH` (developed against Blender 5.2.2). MaterialX export is detected at
  startup, never assumed.
- `omp` on `PATH`, already configured with the models you use (kitbash never manages API keys).
- A [trellis-mac](https://github.com/shivampkumar/trellis-mac) checkout with its `.venv` set up and model
  weights downloaded.

## Setup

```sh
poetry env use python3.14
poetry install
poetry run kitbash --help
```

`poetry install` also installs `sentence-transformers` (with torch) for the default local embedding
model. The first run downloads that model (~90 MB).

## Quick start

```sh
# From a reference image (photorealistic by default), fully automatic:
poetry run kitbash build --image room.jpg --output out/ --no-interactive

# From a prompt, in a toon style, reviewing every asset yourself:
poetry run kitbash build --prompt "a cosy reading corner with an armchair and a floor lamp" \
    --style 2d --output out/corner

# See the plan (phases, models, paths, estimated steps) without running anything:
poetry run kitbash build --image room.jpg --output out/ --dry-run

# Continue an interrupted run, or re-open a phase:
poetry run kitbash resume --output out/
poetry run kitbash resume --output out/ --from-phase layout
```

Ctrl+C and fatal model-access errors stop subprocess groups and wait for modelling workers to unwind
before closing HTTP or SQLite. Interrupted assets keep their checkpoints; cancellation does not turn
them into failed assets or start tool retries. A second Ctrl+C during teardown is deferred until cleanup
finishes.

### `build` options

| option | default | meaning |
|---|---|---|
| `--output DIR` | required | scene output directory |
| `--image PATH` / `--prompt TEXT` | exactly one | the input |
| `--style` | `photorealistic` | `photorealistic`, `2d` or `animated-3d` |
| `--threads N` | 2 | parallel modelling workers (and Trellis jobs) |
| `--review-buffer N` | 6 | max assets waiting for your review (backpressure) |
| `--max-cycles N` | 4 | critic cycles per phase and per asset (and per round of feedback) |
| `--retopology` | `triflow` | retopology method for the Trellis mesh: `triflow` or `decimate` (see below) |
| `--rubric PATH` | packaged `rubric.md` | rubric the critics score against |
| `--config PATH` | packaged `config.toml` | config overriding the defaults |
| `--model.<role>=<name>` | from config | model for a role in every phase |
| `--model.<phase>.<role>=<name>` | from config | model for a role in one phase |
| `--no-interactive` | off | auto-approve every gate and review; input requests are skipped and logged |
| `--dry-run` | off | print the plan and exit |

Roles: `code`, `image_analysis`, `reference_selection`, `visual_critic`, `prompt_analysis`,
`technical_critic`. Phases: `breakdown`, `modelling`, `layout`, `assembly`.

```sh
kitbash build --image room.jpg --output out/ --model.code=claude-sonnet-5 --model.layout.visual_critic=gemini-3.1-pro-preview
```

### Library (backlot) commands

```sh
kitbash library search "walnut armchair"   # semantic search (sqlite-vec)
kitbash library list
kitbash library show <id>
kitbash library remove <id> [--yes]
kitbash library reindex                    # after changing the embedding backend
```

## Configuration

Defaults live in [`config.toml`](config.toml) (a link to `src/kitbash/defaults/config.toml`). Pass your
own file with `--config`: only the keys you set are overridden, and unknown keys are rejected with a
clear error. Each run writes the exact configuration it used to `<output>/config.snapshot.toml`, and
`resume` reads it back.

Main sections:

- `[models]`: model per role. `[models.phases.<phase>]` overrides roles for one phase.
  `[models.thinking]` sets the omp thinking level per role. `[models.capabilities].image_roles` lists
  the roles whose models must accept images (checked at startup).
- `[paths]`: `backlot` (default `~/.local/share/backlot`), `trellis`, `downloads` (reference and Poly Haven caches),
  `triflow_weights` (default `~/.cache/kitbash/triflow`).
- `[tools]`: executables for `omp`, `blender`, and the Python that runs Trellis.
- `[critic]`: cycles, pass threshold, stall detection, revert tolerance, patch attempts.
- `[reference]`: deterministic image providers, rights allow-list, quality thresholds, cache TTL and
  optional `vision_fallback`. Defaults: `input_crop`, `wikimedia`, `openverse`; no model search.
  Remove `omp_web` from older config files/run snapshots; model web search is no longer supported.
- `[trellis]`: `steps = 64`, `pipeline_type = "1024"`, `no_texture = true`, retries and timeouts.
  `mesh_up_axis = "Z"`: Trellis writes raw Z-up vertices, even inside its `.glb`.
- `[retopology]`: `method` (`triflow` or `decimate`), `face_count = 4000`, `qem_threshold = 12.0`,
  `quad_ratio = 0.95`, `flow_steps = 50`, `device` (`auto`, `cuda`, `mps` or `cpu`) and `fallback_on_error`.
  `--retopology` overrides `method`; `resume` reuses the method stored in the run's config snapshot.
- `[blender]`, `[usd]`: render sizes and samples, bake resolution, round-trip threshold, MaterialX switch.
- `[embedding]`: `sentence-transformers` (local, default), `http` (an OpenAI-compatible
  `/embeddings` endpoint such as LM Studio) or `hashing` (offline, used by the tests).
- `[naming]`: data-block naming convention used by prompts and checks alike.
- `[styles.<name>]`: render engine and guidance text for each `--style`.

> **Model defaults.** The spec asked for `opus-5.5`, which this omp installation cannot resolve (omp
> fuzzy-matches it to an OpenRouter model with no key). The defaults use `claude-opus-5` from the aigw
> gateway instead. Change `[models]` once `opus-5.5` is available.

### Startup checks

Before doing any work, `build` and `resume` check that `omp`, Blender and the Trellis checkout exist,
and that every configured model (including per-phase overrides) appears in `omp models --json`. They
also check that roles which send images use models that accept them. Blender's USD exporter options
are probed to detect MaterialX support. The Blender, Blender Python and omp versions are recorded in
the analytics. With `omp.preflight_ping = true` (the default), each model also gets one tiny prompt, so rejected
credentials or an exhausted budget stop the run in seconds rather than mid-phase. During a run, those
two errors are fatal and never retried. Any problem stops the run with a list of what to fix.

## The rubric

Critics score against [`rubric.md`](rubric.md), a Markdown table:

```
| criterion | weight | pass condition | applies to | critic |
```

- `applies to` lists phases (`breakdown, modelling, layout, assembly` or `all`).
- `critic` (optional) is `visual`, `technical` or `both`.
- Text in backticks inside the pass condition is a machine check over measured facts, for example
  `` `usd_roundtrip_score >= 0.85 and missing_textures == 0` ``. When every fact it names was measured,
  the check decides pass or fail. Otherwise the critics decide.

With `critic.require_all_pass = true`, every applicable criterion must have an affirmative assessment:
missing evidence is **unassessed**, not an implicit pass or an invented failure. A criterion assigned
to `both` critics needs evidence from both to pass, unless a measured machine check decides it. A
known negative assessment still makes the criterion failed even if the other critic is unavailable.
Threshold-only scoring (`require_all_pass = false`) retains its configured behavior, but never
relabels an unassessed criterion as passed. Final assembly always requires all criteria to pass.

Critic responses must cover every criterion assigned to that role and phase. Omitted criteria use
the existing response-repair path; unavailable evidence must be represented with `score: null`,
`pass: null`, and an explanatory `notes` value. A visual critic with no available render marks each
of its criteria unassessed without asking the model to guess. Scorecard JSON exposes each criterion's
`status` (`passed`, `failed`, or `unassessed`); terminal tables use the same labels. Unknown verdicts
remain `pass: null`, distinct from evidenced failures (`pass: false`), across saved scorecards.

Editing the file changes critic prompts, scoring and pass or fail decisions, with no code changes.
Facts available to checks include `scale_error`, `origin_offset_m`, `up_axis_ok`, `naming_violations`,
`non_principled_materials`, `missing_textures`, `usd_roundtrip_score`, `usd_broken_materials`,
`missing_assets`, `unexpected_assets`, `missing_placeholders`, `unexpected_placeholders`,
`floating_assets`, `has_camera`, `items` and `unrecognized_items`. Final USD scene inspection supplies
the corresponding `usd_`-prefixed instance, placeholder, grounding and camera facts.

## How it works

### Critic loop (breakdown, modelling per asset, layout)

Each cycle evaluates a script in headless Blender, then runs two critics in parallel:

- the **visual critic** sees the renders and the references;
- the **technical critic** sees the script, the inspection report (names, dimensions, origin, node
  graphs, texture paths) and the USD round-trip results.

Each critic returns rubric scores and a structured list of edits. The **code role** turns those edits
into a **unified diff** against the latest kept script. All Python is written by the code role. Every
cycle is stored in `cycles/NN/` as `script.py`, `diff.patch`, `critique.json` and `report.json`.

Only kept cycles with successful evaluation, a score and verdict for every applicable criterion, and
all required artifact files are eligible to return. Breakdown requires inventory, blend and render;
modelling requires blend, USD and preview; layout requires blend and render. Passing cycles outrank
non-passing cycles, regardless of score. `passed` always describes the cycle actually returned.
If no eligible result exists, the loop reports an error instead of returning a broken build.

Resume recognizes a checkpointed pass even if interruption happened before the session was marked
complete or after the final allowed cycle. It does not patch or re-evaluate that pass. Before critic
review, Kitbash seals the evaluated script, artifacts, preview images and complete build tree with
SHA-256 hashes and workspace/phase/subject/cycle ownership. Selection and resume check that evidence
again; changed, missing, empty required, foreign-owned or symlinked artifacts are not eligible.
Legacy checkpoints without this evidence require a new evaluation, not an inferred approval; start
a new build workspace if an old completed session cannot resume. Moving a workspace also invalidates
its ownership evidence. Published backlots remain relocatable.

The loop never repeats itself:

- A patch that breaks an eligible build or drops its score beyond `revert_epsilon` is **reverted**.
  A passing patch is kept even if its score is below a non-passing parent's. The next patch starts
  from the latest kept script, including a failed initial script that still needs repair.
- Patches that do not apply are **rejected**, and the reason goes back to the code writer.
- The full diff history, with each status, goes into every prompt.
- The loop stops early when a new diff repeats an earlier one, or when the best score has not
  improved for 2 cycles. It also stops after `--max-cycles`. It then escalates to you.

User feedback starts a new session of up to `--max-cycles` more cycles. The feedback is the first,
highest-priority edit, and the session's result is never reverted to the result from before the
feedback.

The breakdown uses the same loop. Its "script" holds the inventory as data and builds a labelled
blockout (one proxy box per item, seen through the estimated camera), so critics compare the blockout
render with the reference.

### Modelling pipeline

Each asset follows a checkpointed state machine:

```
queued → referencing → generating → building ⇄ critiquing → awaiting_review → approved | skipped | needs_rework
                 └──────────┴──→ input_needed (no reference found, Trellis failed) → queued | skipped
```

- **Producers:** `--threads` workers take assets through reference search, Trellis
  (`python generate.py <ref> --output <asset> --steps 64 --no-texture --pipeline-type 1024`, with up to 2
  retries), retopology, the build script and the critic loop. A worker never waits on you: a finished asset goes
  onto the review queue and the worker picks up the next one.
- **Consumer:** one review loop shows finished assets in completion order, while generation continues.
  For each asset you can approve it, give feedback, regenerate it (new Trellis seed and a new script)
  or skip it. "Input needed" entries show a ranked reference contact sheet when available: select a
  candidate number, give another search name or image path, or skip. Textual handles input on the main
  thread while the pipeline and modelling workers continue in background threads.
- **Backpressure:** at most `--review-buffer` assets are in flight or waiting for review. When the
  buffer is full, workers finish their current asset and then idle. The idle time is reported.
- **Barrier:** layout starts when every asset is approved or skipped. Skipped assets become labelled
  placeholder boxes.
- **Reuse:** identical inventory items (for example four matching chairs) are modelled once. Backlot
  matches found during the breakdown are proposed for reuse; `--no-interactive` accepts matches above
  `backlot.match_threshold`.

`build` and `resume` use a full-screen [Textual](https://textual.textualize.io/) interface when stdin
and stdout are terminals, including runs with `--no-interactive`:

- A scrollable phase/asset overview keeps states, scores, worker activity and review backpressure visible.
- Each concurrent command has its own labelled, scrollable output pane: asset/phase, worker, tool,
  invocation ID, elapsed time and running/done/failed/timed-out/cancelled state. stdout and stderr stream
  as the child flushes them; stderr is labelled. Live history is bounded, while existing command log
  files retain the output.
- Reviews, phase gates and feedback stay in the same screen; they do not pause job output. Type an
  answer and press **Enter** (an empty answer uses the displayed default). **Tab** moves focus between
  controls; narrow terminals stack the review below the monitor.
- The pinned bottom status bar shows run/input/stopping state, running command count, recent failures
  and elapsed time. **Ctrl+C** stops the run and waits for children and workers to unwind before exit.
- Preview paths appear in the review pane. **F2** or **Open previews** opens them in the system viewer,
  without emitting terminal image escape sequences over the dashboard. `ui.show_previews = false`
  disables previews.

Redirected output or input uses plain tables and stdin prompts instead of a full-screen UI, so scripts
and pipes keep working. Help, `--dry-run`, library commands and the final run summary remain ordinary
CLI output. Interrupted runs can be continued with `kitbash resume --output <dir>`.

### Retopology

After Trellis, each asset's mesh goes through the configured retopology method before the build script
imports it. Both methods hand the build script a Z-up mesh, so `trellis.mesh_up_axis` applies unchanged.

- `triflow` (default): learned retopology that produces a low-poly triangle mesh of about
  `retopology.face_count` faces. Output goes to `phases/02_modelling/<asset>/retopo/attempt_NN/`. The
  preflight checks that it can run (dependencies, weights in `paths.triflow_weights`, device) and the
  `--dry-run` plan shows the weights status.
- `decimate`: the original behaviour. There is no retopology step, and the build script reduces the raw
  Trellis mesh in Blender with `kb.decimate` (collapse decimation, voxel remesh when needed).

For TriFlow output, `kb.decimate` leaves the mesh unchanged even if Blender's face budget is lower.
`kb.clean_mesh` updates normals and shading without welding vertices or removing small components.
The decimate path, including fallback after a TriFlow failure, retains ordinary cleanup and reduction.

#### TriFlow

[TriFlow](https://github.com/DerKleineLi/triflow) (Li et al., ECCV 2026) turns a mesh into one with
artist-like topology: a latent flow-matching model, conditioned on the input's SDF, a face count and a quad
ratio, predicts a nearest-vertex vector field. Watershed clustering plus a constrained quadric-error
simplification then extract the mesh. kitbash vendors and adapts the code (`src/kitbash/retopology/triflow/`),
so it runs in-process and no separate checkout is needed.

The encoder's SDF samples include a halo around surface cells to cover the full narrow band. Marching
cubes extracts the proxy from that same SDF, preserving signed cavities; NVF support is voxelized from
the proxy rather than the original triangulation. Inputs must define a closed signed surface inside
the padded grid: an unbounded or open extracted surface raises a retopology error instead of being
silently replaced by adaptive remeshing.

Face count and quad ratio are conditioning signals, not hard guarantees. Output remains triangular;
quad ratio encourages regular triangle pairings, not native quad faces. Topology and geometry guards
take precedence over reaching the requested count.

The constrained QEM is compiled from the vendored C++ source during `poetry install` (a C++17 compiler is
required; on macOS install Xcode Command Line Tools). It rejects inverted or degenerate contractions,
using safe endpoint/midpoint candidates when the quadric minimizer would invalidate the current face fan;
the final mesh is not welded or stripped of faces after simplification, which would bypass topology checks.

Watershed roots use the transferred displacement at mesh vertices (paper Eq. 6). If a connected
component has no root below the threshold, its minimum-displacement vertex seeds that component
(ties use vertex order). No component is assigned an invalid root or a fabricated origin target.

- **Devices.** Upstream is CUDA-only. Here the sparse convolutions and attention are plain PyTorch
  (`scaled_dot_product_attention`), so TriFlow runs on CUDA, Apple MPS and CPU: `retopology.device = "auto"`
  prefers CUDA, then MPS, then CPU. fp16 is used on CUDA only. There are no Triton or spconv/torchsparse/flash-attn
  dependencies. Sparse pooling and neighbour maps are reused across flow steps; only the SDF encoder,
  NVF decoder and flow model are loaded on the accelerator. Runtime varies with occupied voxels and
  the selected face count; a 21.6k-face synthetic asset takes about 85 seconds on an M5 Pro with MPS.
- **Weights** (about 1.3 GB, pinned to a Hugging Face revision and verified by SHA-256, including cached copies)
  are downloaded on first use into `paths.triflow_weights`. Fetch them ahead of time with
  `kitbash retopology download-weights`.
- **Output frame.** The result is mapped back into the input mesh's scale, position and orientation, so
  the build script's orientation and sizing logic is unchanged.
- **License.** TriFlow is under the Automotive Development Public Non-Commercial License 1.0, and its
  dependency MeshLib is not open source. See `src/kitbash/retopology/triflow/NOTICE.md`. Use
  `--retopology decimate` to avoid both.

With `retopology.fallback_on_error = true` (the default), a failed `triflow` run logs a warning, keeps the
Trellis mesh and continues on the `decimate` path. The reason is stored as `fallback_reason` in the
asset's `retopology` record. With `false`, a retopology failure is handled like a Trellis failure: the
asset asks you for another item name or reference image. The analytics report retopology time next to
Trellis time.

### Reference acquisition: deterministic first

1. An approved backlot reuse decision bypasses reference acquisition and Trellis entirely.
2. An explicit user image wins next (JPEG, PNG or WebP, subject to decode/size limits). Otherwise,
   an input-image crop is reused only with at least `reference.min_crop_side_px = 384` pixels on its
   **original, unpadded shorter side**, plus the automatic quality gates below. Crops are never upscaled.
3. Commons and Openverse queries are cached and ranked deterministically. Rights are checked before
   downloading; title/category overlap and provider rank bound download work. Decoded pixel hashes
   deduplicate identical images across URLs; sharpness, border clipping, approximate foreground
   occupancy/background uniformity and decoded/native dimensions determine the final ranking.
4. Automatic selection requires supported rights, `auto_select_threshold = 0.88` and a lead of
   `ambiguity_margin = 0.08` over the next candidate. Hard gates also require native/decoded shorter
   sides of at least 384 pixels, token overlap ≥ 0.65, normalized sharpness ≥ 0.2, estimated occupancy
   between 0.08 and 0.75, edge clipping ≤ 0.02 and border uniformity ≥ 0.9. A lone candidate is not
   automatically trusted, and does not trigger a model call.

These are **screening heuristics, not proof of the correct object, completeness or lack of occlusion**.
Ambiguous/low-quality results produce `contact_sheet.png` and `review.json`, then `input_needed` before
mesh generation. Interactive review can explicitly accept a rights-eligible, decodable candidate below
the heuristic threshold. `--no-interactive` skips unresolved assets; they remain placeholders/partial
input, subject to final acceptance—not an expensive mesh run on the top lexical hit.
Set `reference.vision_fallback = true` to let OMP judge ambiguous candidates when at least one passes
the automatic quality threshold. Visual fallback cannot override rights or the quality gates; its
choice must still pass them. It is disabled by default.

The supported rights allow-list defaults to CC0, public-domain declarations and configured versions of
CC BY / CC BY-SA. Unknown, noncommercial, no-derivatives, conflicting or incomplete rights metadata
is rejected. Attribution licenses require a creator and a matching license URL. This policy records
provider claims; it is **not legal clearance**. Users remain responsible for attribution, share-alike
and other obligations, and for rights to their supplied images/crops. `selection.json`, asset state
and backlot metadata preserve image URL, provider item ID, page URL, creator, license identifier/URL,
dimensions, query, content hash and quality assessment rather than only a free-form license label.

Persistent caches live under `paths.downloads/references`: `searches/` uses
`reference.search_cache_ttl_s = 86400`; `urls/` indexes content-addressed PNGs in `content/`.
Cache hits are rechecked against current rights, byte/dimension limits and pixel hashes. Delete this
cache to force fresh retrieval. Incoming and normalized files are limited by `max_download_mb`;
decoded images are additionally bounded to 16,384 pixels per side and 40 million pixels total.

Downloads permit only public HTTP(S) destinations on standard ports, without URL credentials or
environment proxies. Every request/redirect validates all DNS answers and connects directly to a
validated numeric address, retaining the original hostname for verified TLS. Private, loopback,
link-local, reserved and mixed public/private destinations are rejected; redirects and bodies are
bounded. HTTP remains supported for provider compatibility, without HTTPS transport guarantees.
The OS resolver, sockets and TLS trust store are trusted. This hardens a risk boundary; it does not
assert that the previous implementation was exploited.

### USD export and material fidelity

The `.blend` is the source of truth. It is fully node based, including procedural nodes. Every asset
and the final scene also get a `.usd`, exported per material with this ladder:

1. **MaterialX**: used when the installed Blender can export it *and* a probe export shows every linked
   Principled input still connected in the MaterialX network.
2. **UsdPreviewSurface, baked**: otherwise, procedural graphs (noise, voronoi, ramps, math and so on)
   feeding supported preview inputs are baked to image textures next to the `.usd`, referenced by
   relative path. Baking cannot add channels that UsdPreviewSurface cannot represent.

Baking happens in a temporary copy, so the original `.blend` is never modified. Blender's own USD
importer only reads UsdPreviewSurface, so by default (`usd.bake_preview_fallback = true`) MaterialX
materials also get a baked preview-surface fallback for supported channels.

**Known preview loss is a failure**, not a successful round trip. Before importing or comparing renders,
the fidelity checker rejects any material with `lost_in_preview` channels, naming the material and
inputs in the error. This includes unsupported linked inputs such as transmission or subsurface,
whether driven by procedural nodes or a direct image, and applies even when MaterialX preserves them.
Asset evaluation reports the error to the critic; final assembly stops. Low-level export reports retain
the loss diagnostics, but those exports are not accepted by the fidelity checker. This guard does not
certify arbitrary shader graphs or every unlinked Principled value; the render comparison remains required.

**Round-trip validation** is part of the critic loop:

1. Re-import the `.usd` into a clean headless Blender session.
2. Check that each material has a Principled BSDF with the expected texture-driven inputs, and that no
   texture path is missing or broken.
3. Render with the same camera and lighting as the `.blend`, and compare the two renders.

The comparison is `score = min(SSIM, 1 − 2·mean color delta)`. The rung used for each material, and
the score, are stored in the backlot (`usd_material_mode`, `usd_roundtrip_score`) and in the analytics.

### Final acceptance

Assembly inspects the **rebuilt, localized `scene.blend`**, not just its earlier layout preview, and
inspects the **actual re-imported USD**. Both must contain the expected number of each approved asset
instance (including inventory `same_as` copies) and skipped-item placeholders, with no missing or
extra placements, a valid camera, available textures and grounded geometry. Inspection includes
EMPTY-root hierarchies and linked collection instances. Support must be external geometry within
3 cm of the placement base; the object's own geometry and an imaginary floor at z=0 do not count.
The USD comparison uses the selected scene camera, not whichever camera imports first.

Automatic publication requires the final rubric to pass with every applicable criterion scored,
and the structural checks above cannot be disabled by omitting them from a custom rubric. A failed
or unmeasured final criterion blocks automatic acceptance. Final assembly has no visual/technical
LLM critic call: custom assembly criteria need measured machine checks to pass automatically.

With `--no-interactive`, failed final acceptance exits nonzero and does not mark assembly complete.
Completed rejected outputs remain in `phases/04_assembly/rejected/` for diagnosis, including when the
user quits the acceptance gate. They never replace `scene/`. Interactive runs show the failures and
default to quitting. Only an explicit `p` choice can **publish degraded**; ordinary approval is not an
override. Both `.blend` and USD outputs must be nonempty, as must any rendered final frame. A scene
without a usable camera can only be published without a render by this explicit degraded override.

Assembly seals the staged file tree before any human acceptance gate and checks it again immediately
before publication. Even an explicit degraded override cannot approve bytes changed after inspection.
`assembly.json` records the canonical workspace owner and final payload hashes, including trusted
report-path rewrites; the journal itself is excluded from its content manifest.

The current attempt's state metadata and analytics record `acceptance.status` (`pending`, `passed`,
`failed` or `overridden`), `automatic_pass`, `published` and failure details. A human override records
`status = "overridden"`, `automatic_pass = false` and the still-failing scorecard; terminal output also
labels it degraded. Rebuilds clear the current attempt's acceptance before work starts, so a failed or
interrupted rerun cannot retain an earlier automatic pass. The previous published scene and its
`scene/assembly.json` remain unchanged; SQLite's `published_assembly` metadata tracks that publication
separately from the latest `assembly` attempt. Fix the scene and use `resume --from-phase layout`, or
resume interactively to explicitly publish a degraded output.

### Performance

The Trellis command from the spec (`--pipeline-type 1024 --steps 64`) is slow: expect roughly 30 to 60
minutes per mesh on Apple Silicon, far more than the `512`/12-step Trellis defaults. `trellis.timeout_s`
defaults to 2 hours. Parallel Trellis jobs share one GPU, so `--threads 2` mostly overlaps Trellis with
the Blender and LLM work of other assets rather than doubling Trellis throughput. Lower
`trellis.pipeline_type` or `trellis.steps` in your config for quicker drafts. Trellis output streams live
into `phases/02_modelling/<asset>/trellis/attempt_NN/trellis_attempt_N.log`.

### Backlot

The backlot is a SQLite database with sqlite-vec for embeddings, stored in `paths.backlot`, never
inside a scene directory. Each asset folder holds the `.blend`, its `textures/`, the `usd/` tree, a
preview and `metadata.json`.

Only an asset awaiting review can be committed, and its item identity, workspace, subject and selected
cycle must match the evaluated result. Approval cannot silently switch to an older eligible cycle.
New bundles carry evaluation-time SHA-256 hashes for the complete build tree and a separate preview.
The source and staged copy must match these hashes; symlinks, escaped paths and collisions with the
publisher's `metadata.json` or normalized `preview.<suffix>` are rejected.

Bundles are copied into hidden `.staging/` directories first. Asset metadata, vectors and a complete
publication manifest commit together with a pending status. The SQLite manifest binds the asset ID,
relative publication directory and every payload file, including generated metadata. Only a verified
bundle renamed into `assets/` becomes available to get/list/search. Opening the library verifies the
same evidence before completing an interrupted publication; changed or incomplete pending entries
are removed along with their vectors and files. Existing explicitly ready legacy entries remain
available, but legacy pending entries without hash evidence are discarded, not promoted. Publication
and recovery share an OS-owned lock so recovery cannot remove a live writer's files.

These checks prevent stale, misrouted or corrupted publication. They do not sandbox generated Python,
authenticate against a process able to rewrite both files and state, or continuously audit ready assets.

Vector reuse requires the same embedding model identity **and** dimensions. After changing the
backlot's model, run `kitbash library reindex`; the old index remains intact if rebuilding fails.
Existing scene state also requires its original embedding backend (or a new output directory).

Search reads vectors, model identity and asset metadata from one SQLite snapshot, including during
concurrent publication or reindexing. Scene inventory rows, scene metadata and vectors are replaced
in one transaction; failed or interrupted embedding/index writes preserve the previous inventory.

Scenes copy the asset folders they use into `scene/assets/`, so the export is self-contained and every
texture path stays relative and valid. Linked image paths resolve against their owning library, not
`scene.blend`; localization leaves those library-owned paths unchanged. External linked libraries must
already be bundled with their textures by assembly; localization does not copy arbitrary library trees.

Asset image copies, external scene images (including HDRIs), and temporary USD source-image copies use
full SHA-256 content names plus the file extension. Different files named `albedo.png` cannot overwrite
or reuse each other's images; identical content can share a copy. USD staging also protects existing
libraries whose texture filenames are not hashed. External local image sequences/UDIMs that require
multi-file copying are not renamed as single textures: localization reports unresolved paths or fails
explicitly rather than silently dropping frames/tiles. Packed images remain packed.

## Output directory

```
<output>/
  config.snapshot.toml  rubric.snapshot.md  state.db
  input/                                   reference image or prompt.txt
  phases/
    01_breakdown/  inventory.json  cycles/NN/{script.py, diff.patch, critique.json, report.json}  renders/
    02_modelling/<asset_id>/  reference/  trellis/  retopo/  script.py  cycles/NN/...  previews/  usd_roundtrip/
    03_layout/     script.py  cycles/NN/...  renders/
    04_assembly/   rejected/ (last rejected scene and its assembly.json, if any)
  scene/  scene.blend  scene.usd  textures/  assets/  renders/  assembly.json
  analytics/  analytics.json  analytics.md
  logs/  kitbash.log  llm/ (every prompt and answer)  *.log (every subprocess)
```

Assembly builds in `.scene-staging/`. Only a validated or explicitly accepted degraded result is
renamed into `scene/`; failed or interrupted rebuilds leave the previous published scene unchanged.
Resume removes incomplete staging and recovers `.scene-previous/` if replacement stopped between
renames. A candidate's journal must have matching workspace ownership, acceptance and payload hashes
before recovery can restore published SQLite metadata or discard the prior scene. An invalid candidate
is rolled back to the prior directory intact. Legacy scene directories without integrity evidence are
preserved during replacement but cannot restore verified publication metadata; reassemble them in a
workspace with sealed evaluation checkpoints. Blender library/texture paths stay relative, and
generated JSON reports point to their final or rejected location rather than staging paths.

Recovery covers process interruption and crashes; it is not a power-loss durability guarantee.

## Analytics

`analytics/analytics.json` and `analytics/analytics.md` report wall time, LLM calls, tokens, cost (from
omp), critic cycles, retries, user interventions, Trellis and retopology time and the USD material mode. They break
these down per step, agent, model, role, asset and phase. They also report:

- **user time apart from compute time**, and how long each asset waited in the review queue;
- **worker idle time** caused by backpressure, per worker;
- a **per-asset timeline** (ASCII Gantt) marking when you reviewed each asset, and a **review overlap**
  table measuring how much generation and critique ran for other assets while you reviewed one.

A summary table is printed at the end of every run.

## Development

```sh
poetry run pytest                  # unit + integration (Blender tests skip when Blender is missing)
poetry run pytest tests/unit       # fast unit tests
poetry run ruff check src tests
```

The integration tests use a fake `omp` (`tests/fakes/fake_omp.py`, which answers each prompt by its
`<!-- kitbash-task -->` marker) and a fake Trellis (`tests/fakes/trellis/generate.py`), with real
headless Blender. They cover:

- the full pipeline, then backlot reuse on a similar image;
- generation continuing during review;
- resume;
- the MaterialX and baked-fallback export paths;
- relative texture paths after an asset is copied into a scene;
- distinct same-basename textures across asset, scene, and USD copies, including relocated linked libraries;
- rejection of known lost preview channels before round-trip validation;
- scene and USD instance/placeholder counts, grounding, textures and active-camera preservation.

### Code map

```
src/kitbash/
  cli.py, app.py            command line; composition root (all wiring happens in app.py)
  config.py, paths.py       configuration; deterministic output layout
  domain/                   inventory, asset state machine, rubric, critiques, phases, roles
  agents/                   scene (orchestrator), breakdown, modelling, layout agents
  phases/                   one class per phase: gates, persistence, coordination
  pipeline/                 asset board, scheduler (backpressure), review queue, workers, per-asset steps
  critique/                 critic loop, resumable sessions, critics, patch writer, diff history
  llm/                      provider-neutral client types, prompt library, typed calls, parsing
  retopology/               retopology methods behind one protocol: decimate (pass-through) and triflow
                            (vendored TriFlow: sparse/ pure-torch backend, models/, geometry/, engine, weights)
  infra/                    omp, Blender runner, Trellis, image search, Poly Haven, imaging, patching, embeddings
  services/                 Blender toolkit, USD fidelity checker, reference finder, preflight, dry-run plan
  store/, backlot/          scene state DB, sqlite-vec index, asset library
  analytics/                spans and events, report builder
  interaction/, ui/         terminal user and autopilot; live dashboard, tables, inline images
  blender/                  scripts that run inside Blender, including the kitbash_bpy helper API
  prompts/                  prompt templates (Markdown)
  defaults/                 config.toml, rubric.md
```

Generated scripts use the `kitbash_bpy` helper module (`import kitbash_bpy as kb`). Its API reference
in every code-writing prompt is generated from the module's own `@api` functions, so it cannot drift.
